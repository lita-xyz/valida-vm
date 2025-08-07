use core::cmp::min;
use core::ops::Mul;
use std::sync::{Arc, Mutex};

use p3_field::{AbstractExtensionField, AbstractField, Field, Powers, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, MatrixRowSlices};
use p3_maybe_rayon::prelude::*;
use smallvec::SmallVec;

use crate::{
    ChipWithPersistence, Interaction, InteractionMap, InteractionMetadata, InteractionType,
    InteractionVec, Machine, PublicTrace, PublicValues, StarkConfig,
};

pub const MAX_PERMUTATION_HEIGHT: u32 = u32::MAX;

pub mod columns;
pub mod stark;
use columns::{sizes_from_interactions, PermutationColsViewMut, MAX_PERMUTATION_CONSTRAINT_DEGREE};
pub use stark::eval_permutation_constraints;

/// Generate the permutation trace for a chip with the provided machine.
///
/// # Overview
/// The permutation argument is used to prove that values such as read/write memory
/// accesses, bus interactions, or other cross-chip or cross-segment relations are
/// consistent across the system. This function builds a permutation trace whose
/// height is set by the degree of the main/preprocessed/public traces, and width
/// is set by the number of ephemeral and persistent relationships.
///
/// For each row in the permutation trace, we include:
/// - Ephemeral columns, which are used for interactions within a segment
/// - Persistent columns, which are used for cross-segment interactions
///
/// # Usage
/// This function should be called after all chip traces have been generated.
pub fn generate_permutation_trace<M, SC, P>(
    machine: &M,
    chip: &dyn ChipWithPersistence<M, SC, Public = P>,
    preprocessed: &Option<RowMajorMatrix<SC::Val>>,
    public: &Option<PublicTrace<SC::Val>>,
    main: &Option<RowMajorMatrix<SC::Val>>,
    height: usize,
    random_elements: Vec<SC::Challenge>,
    global_random_elements: Vec<SC::Challenge>,
    #[cfg(debug_assertions)] interaction_map_guard: Arc<Mutex<InteractionMap<SC::Val>>>,
) -> Option<RowMajorMatrix<SC::Challenge>>
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
    P: PublicValues<SC::Val, SC::Challenge>,
{
    let ephemeral_interactions = chip.ephemeral_interactions(machine);
    let num_ephemeral = ephemeral_interactions.len();
    let persistent_sends = chip.persistent_sends(machine);
    let persistent_receives = chip.persistent_receives(machine);
    let num_persistent_sends = persistent_sends.len();
    let num_persistent_receives = persistent_receives.len();

    let alpha = random_elements[0];

    // Returns an iterator over the base element, i.e. `beta^0, beta^1, beta^2, ...`
    let betas = random_elements[1].powers();

    let global_alpha = global_random_elements[0];

    // Returns an iterator over the base element, i.e. `beta^0, beta^1, beta^2, ...`
    let global_betas = global_random_elements[1].powers();

    // The width of the permutation trace is the sum of the number of ephemeral
    // and persistent columns.
    // The height of the permutation trace is set by the degree of the main/preprocessed/public traces.
    let (num_ephemeral_cols, num_persistent_cols) =
        sizes_from_interactions::<MAX_PERMUTATION_CONSTRAINT_DEGREE>(
            num_ephemeral,
            num_persistent_sends,
            num_persistent_receives,
        );
    let perm_width = num_ephemeral_cols + num_persistent_cols;

    // Bound the height to less than 2**32 (from epsilon_1 term in soundness analysis, see Theorem 4 of LogUp paper).
    if height as u32 >= MAX_PERMUTATION_HEIGHT {
        panic!("height is too large: {}", height);
    }

    // Gather the values we'll need to compute each row in the permutation trace.
    let perm_values = (0..height)
        .into_par_iter()
        .flat_map(|n| -> Vec<SC::Challenge> {
            let main_row = match main {
                Some(main) => main.row_slice(n),
                None => &[],
            };
            let preprocessed_row = match preprocessed {
                Some(preprocessed) => preprocessed.row_slice(n),
                None => &[],
            };
            let public_row = match public {
                Some(public) => public.row_slice(n),
                None => &[],
            };

            interactions_to_row::<SC>(
                &ephemeral_interactions,
                &persistent_sends,
                &persistent_receives,
                (preprocessed_row, public_row, main_row),
                alpha,
                betas.clone(),
                global_alpha,
                global_betas.clone(),
                #[cfg(debug_assertions)]
                n,
                #[cfg(debug_assertions)]
                &chip.name(),
                #[cfg(debug_assertions)]
                interaction_map_guard.clone(),
            )
        })
        .collect::<Vec<_>>();
    let mut perm = RowMajorMatrix::new(perm_values, perm_width);

    // Compute the running sum columns.
    let mut running_ephemeral_sum = SC::Challenge::zero();
    let mut running_persistent_sum = SC::Challenge::zero();

    // Generate the permutation trace, row-by-row, using the values we gathered earlier.
    //
    // The structure of each row is as follows for `n` ephemeral logup columns and `m` persistent logup columns:
    //
    // [ephemeral_0, ephemeral_1, ..., ephemeral_{n-1}, persistent_0, persistent_1, ..., persistent_{m-1}]
    //
    // The _last_ ephemeral column (`ephemeral_{n-1}`) contains the running sum of all ephemeral interactions.
    // The _last_ persistent column (`persistent_{m-1}`) contains the running sum of all persistent interactions.
    for row in perm.rows_mut() {
        let PermutationColsViewMut {
            ephemeral_cols,
            persistent_cols,
        } = PermutationColsViewMut::as_view_mut::<MAX_PERMUTATION_CONSTRAINT_DEGREE>(
            num_ephemeral,
            num_persistent_sends,
            num_persistent_receives,
            row,
        );
        if num_ephemeral > 0 {
            // add the previous row's running sum to each ephemeral column
            ephemeral_cols.iter_mut().for_each(|partial_ephem_sum| {
                *partial_ephem_sum += running_ephemeral_sum;
            });
            // update the running sum
            running_ephemeral_sum = ephemeral_cols
                .last()
                .copied()
                .expect("ephemeral_cols is not empty");
        }
        if num_persistent_sends + num_persistent_receives > 0 {
            // add the previous row's running sum to each persistent running sum
            persistent_cols.iter_mut().for_each(|partial_pers_sum| {
                *partial_pers_sum += running_persistent_sum;
            });
            // update the running sum
            running_persistent_sum = persistent_cols
                .last()
                .copied()
                .expect("persistent_cols is not empty");
        }
    }

    if num_ephemeral + num_persistent_sends + num_persistent_receives > 0 {
        Some(perm)
    } else {
        None
    }
}

/// Computes the random linear combination of the fields in the interaction.
///
/// # Overview
/// The random linear combination is computed as:
///  `RLC = alpha + bus_id * beta + \sum_i \beta^(i+1) * f_i`.
///
/// where:
/// - `alpha` and `beta` are random challenges.
/// - `bus_id` is the identifier for the relevant argument bus, included as the first field
/// - `f_i` are the fields in the interaction
///
/// Returns this sum, as well as the evaluated `count` field of the interaction.
/// If debug assertions are on, also returns the vector of evaluated fields.
fn reduce_interaction<F, Expr, Var, ExprEF>(
    main_row: &[Var],
    preprocessed_row: &[Var],
    public_row: &[Var],
    interaction: &Interaction<F>,
    alpha: ExprEF,
    betas: Powers<ExprEF>,
) -> (ExprEF, Expr, SmallVec<[Expr; 12]>)
where
    F: Field + Into<Expr>,
    Var: Into<Expr> + Copy,
    Expr: AbstractField + Mul<F, Output = Expr>,
    ExprEF: AbstractExtensionField<Expr>,
{
    let mut rlc = ExprEF::zero();

    const SMALLVEC_SIZE: usize = 12;
    #[cfg(not(debug_assertions))]
    let interaction_vec: SmallVec<[_; SMALLVEC_SIZE]> = SmallVec::new();
    #[cfg(debug_assertions)]
    let mut interaction_vec: SmallVec<[_; SMALLVEC_SIZE]> =
        SmallVec::with_capacity(interaction.fields.len());

    let mut betas = betas.clone();

    // Include the identifier of the argument bus to the list of fields, avoiding the need
    // to have separate permutation challenges for each argument bus.
    let bus_field = ExprEF::from_canonical_usize(interaction.argument_index.identifier());
    rlc += bus_field * betas.next().unwrap();

    // For each field in the interaction, compute the value, multiply it by the next power of beta,
    // and add it to the running total.
    for (virtual_column, beta) in interaction.fields.iter().zip(betas) {
        let virtual_column_value =
            virtual_column.apply::<Expr, Var>(preprocessed_row, public_row, main_row);
        #[cfg(debug_assertions)]
        interaction_vec.push(virtual_column_value.clone());
        rlc += beta * virtual_column_value;
    }

    // Evaluate the count field.
    let count_value = interaction
        .count
        .apply::<Expr, Var>(preprocessed_row, public_row, main_row);

    // The running total is the sum of the random linear combination of the fields (including the argument bus ID)
    // and the random challenge alpha.
    rlc += alpha;

    (rlc, count_value, interaction_vec)
}

/// Computes the values for a single row in the permutation trace for a chip.
fn interactions_to_row<SC: StarkConfig>(
    ephemeral: &[(Interaction<SC::Val>, InteractionType)],
    persistent_sends: &[Interaction<SC::Val>],
    persistent_receives: &[Interaction<SC::Val>],
    (preprocessed, public, main): (&[SC::Val], &[SC::Val], &[SC::Val]),
    alpha: SC::Challenge,
    betas: Powers<SC::Challenge>,
    global_alpha: SC::Challenge,
    global_betas: Powers<SC::Challenge>,
    #[cfg(debug_assertions)] row_index: usize,
    #[cfg(debug_assertions)] chip_name: &str,
    #[cfg(debug_assertions)] interaction_map_guard: Arc<Mutex<InteractionMap<SC::Val>>>,
) -> Vec<SC::Challenge> {
    // Determine the number of ephemeral and persistent interactions to determine the width
    // of the permutation trace.
    let num_ephemeral = ephemeral.len();
    let num_persistent_sends = persistent_sends.len();
    let num_persistent_receives = persistent_receives.len();

    let (num_ephemeral_cols, num_persistent_cols) =
        sizes_from_interactions::<MAX_PERMUTATION_CONSTRAINT_DEGREE>(
            ephemeral.len(),
            persistent_sends.len(),
            persistent_receives.len(),
        );
    let perm_width = num_ephemeral_cols + num_persistent_cols;

    // Initialize each row with zeros. These will be populated with the values for each column in the row.
    let mut row = vec![SC::Challenge::zero(); perm_width];
    let PermutationColsViewMut {
        ephemeral_cols,
        persistent_cols,
    } = PermutationColsViewMut::as_view_mut::<MAX_PERMUTATION_CONSTRAINT_DEGREE>(
        ephemeral.len(),
        persistent_sends.len(),
        persistent_receives.len(),
        &mut row,
    );

    // Transform persistent_sends and persistent_receives into a single vector of interactions with the appropriate interaction type.
    let persistent = persistent_sends
        .iter()
        .cloned()
        .map(|interaction| (interaction, InteractionType::PersistentSend))
        .chain(
            persistent_receives
                .iter()
                .cloned()
                .map(|interaction| (interaction, InteractionType::PersistentReceive)),
        )
        .collect::<Vec<(Interaction<SC::Val>, InteractionType)>>();

    // Helper function to compute the partial sum columns for chunks of interactions.
    //
    // Each column equals the prior column plus the sum of D - 1 terms
    // of the form m_i*q_i where:
    // * D = MAX_PERMUTATION_CONSTRAINT_DEGREE
    // * m_i is the multiplicity column for the ith interaction
    // * q_i = \frac{1}{\alpha^i + \sum_j \beta^j * f_{i,j}}
    // * f_{i,j} is the jth trace column for the ith interaction
    let chunk_sum = |chunk| {
        let mut chunk_sum = SC::Challenge::zero();
        for &(ref interaction, interaction_type) in chunk {
            // Get the randomized linear combination and the count.

            // Shadow `alpha`, `betas` based on whether this is a single segment or global interaction
            let (alpha, betas) = match interaction_type {
                InteractionType::LocalSend
                | InteractionType::LocalReceive
                | InteractionType::GlobalSend
                | InteractionType::GlobalReceive => (alpha, betas.clone()),
                InteractionType::PersistentSend | InteractionType::PersistentReceive => {
                    (global_alpha, global_betas.clone())
                }
            };

            let (reduced_interaction, count, fields) = reduce_interaction(
                main,
                preprocessed,
                public,
                interaction,
                alpha,
                betas.clone(),
            );

            // For sends, the contribution is positive, while for receives, it's negative.
            // If they balance, the contribution is zero.
            let numerator = match interaction_type {
                InteractionType::LocalSend
                | InteractionType::GlobalSend
                | InteractionType::PersistentSend => count,
                InteractionType::LocalReceive
                | InteractionType::GlobalReceive
                | InteractionType::PersistentReceive => -count,
            };

            // Add the contribution of this interaction to the sum using the inverse of the interaction hash.
            chunk_sum += Into::<SC::Challenge>::into(numerator) * reduced_interaction.inverse();

            #[cfg(not(debug_assertions))]
            let _ = fields;
            #[cfg(not(debug_assertions))]
            let _ = interaction_type;

            #[cfg(debug_assertions)]
            {
                let interaction_vec = InteractionVec {
                    fields,
                    metadata: InteractionMetadata {
                        chip_name: chip_name.to_string(),
                        row: row_index,
                    },
                };
                let mut interaction_map = interaction_map_guard.lock().unwrap();
                let (sends, receives) = interaction_map
                    .entry(interaction.argument_index)
                    .or_insert_with(|| (vec![], vec![]));
                for _ in 0..(count.as_canonical_u32() as usize) {
                    match interaction_type {
                        InteractionType::LocalSend
                        | InteractionType::GlobalSend
                        | InteractionType::PersistentSend => {
                            sends.push(interaction_vec.clone());
                        }
                        InteractionType::LocalReceive
                        | InteractionType::GlobalReceive
                        | InteractionType::PersistentReceive => {
                            receives.push(interaction_vec.clone());
                        }
                    }
                }
            }
        }
        chunk_sum
    };

    // Populate the ephemeral columns chunk by chunk.
    if num_ephemeral > 0 {
        let chunk_size = MAX_PERMUTATION_CONSTRAINT_DEGREE - 1;
        let first_chunk_size = min(num_ephemeral, chunk_size - 1);
        let (first_chunk, rest) = ephemeral.split_at(first_chunk_size);

        let mut prev_col = SC::Challenge::zero();
        // For each chunk, compute the reciprocal sum and add it to the previous column's value.
        for (chunk, col) in vec![first_chunk]
            .into_iter()
            .chain(rest.chunks(chunk_size))
            .zip(ephemeral_cols.iter_mut())
        {
            *col = chunk_sum(chunk) + prev_col;
            prev_col = *col;
        }
    }

    // Populate the persistent columns chunk by chunk.
    if num_persistent_sends + num_persistent_receives > 0 {
        let chunk_size = MAX_PERMUTATION_CONSTRAINT_DEGREE - 1;
        let first_chunk_size = min(
            num_persistent_sends + num_persistent_receives,
            chunk_size - 1,
        );
        let (first_chunk, rest) = persistent.split_at(first_chunk_size);

        let mut prev_col = SC::Challenge::zero();
        // For each chunk, compute the reciprocal sum and add it to the previous column's value.
        for (chunk, col) in vec![first_chunk]
            .into_iter()
            .chain(rest.chunks(chunk_size))
            .zip(persistent_cols.iter_mut())
        {
            *col = chunk_sum(chunk) + prev_col;
            prev_col = *col;
        }
    }

    row
}
