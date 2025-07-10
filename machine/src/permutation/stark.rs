use core::cmp::min;

use p3_air::{Air, ExtensionBuilder};
use p3_field::{AbstractField, Powers};
use p3_matrix::{Matrix, MatrixRowSlices};

use crate::{
    columns::{PermutationColsView, MAX_PERMUTATION_CONSTRAINT_DEGREE},
    permutation::{reduce_interaction, MAX_PERMUTATION_HEIGHT},
    persistence::ChipWithPersistence,
    Interaction, InteractionType, Machine, StarkConfig, ValidaAirBuilder,
};

pub fn eval_permutation_constraints<M, C, SC, AB>(
    chip: &C,
    builder: &mut AB,
    cumulative_ephemeral_sum: Option<AB::ExprEF>,
    cumulative_persistent_sum: Option<AB::ExprEF>,
) where
    M: Machine<SC::Val>,
    C: ChipWithPersistence<M, SC> + Air<AB>,
    SC: StarkConfig,
    AB: ValidaAirBuilder<Machine = M, F = SC::Val, EF = SC::Challenge>,
{
    let rand_elems = builder.permutation_randomness().to_vec();
    let global_rand_elems = builder.global_permutation_randomness().to_vec();

    let machine = builder.machine();

    let perm = builder.permutation();
    let perm_width = perm.width();
    debug_assert_eq!(perm_width, chip.permutation_width(machine));

    let ephemeral_interactions = chip.ephemeral_interactions(machine);
    let persistent_sends = chip.persistent_sends(machine);
    let persistent_receives = chip.persistent_receives(machine);

    let num_ephemeral = ephemeral_interactions.len();
    let num_persistent_sends = persistent_sends.len();
    let num_persistent_receives = persistent_receives.len();

    let PermutationColsView {
        ephemeral_cols: ephemeral_cols_local,
        persistent_cols: persistent_cols_local,
    } = PermutationColsView::as_view::<MAX_PERMUTATION_CONSTRAINT_DEGREE>(
        num_ephemeral,
        num_persistent_sends,
        num_persistent_receives,
        perm.row_slice(0),
    );

    let PermutationColsView {
        ephemeral_cols: ephemeral_cols_next,
        persistent_cols: persistent_cols_next,
    } = PermutationColsView::as_view::<MAX_PERMUTATION_CONSTRAINT_DEGREE>(
        num_ephemeral,
        num_persistent_sends,
        num_persistent_receives,
        perm.row_slice(1),
    );

    // Transform persistent_sends and persistent_receives into a single vector of interactions with the appropriate interaction type.
    let persistent_interactions = persistent_sends
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

    let trace_height = num_ephemeral + num_persistent_sends + num_persistent_receives;
    // Bound the height to less than 2**32 (from epsilon_1 term in soundness analysis, see Theorem 4 of LogUp paper).
    if trace_height as u32 >= MAX_PERMUTATION_HEIGHT {
        panic!("height is too large: {}", trace_height);
    }

    let alpha = AB::ExprEF::from_f(rand_elems[0]);
    let betas = (AB::ExprEF::from_f(rand_elems[1])).powers();

    let global_alpha = AB::ExprEF::from_f(global_rand_elems[0]);
    let global_betas = (AB::ExprEF::from_f(global_rand_elems[1])).powers();

    if num_ephemeral != 0 {
        eval_logup_constraints::<_, SC, _>(
            builder,
            cumulative_ephemeral_sum
                .expect("cumulative_ephemeral_sum is Some if num_ephemeral > 0"),
            ephemeral_interactions,
            (ephemeral_cols_local, ephemeral_cols_next),
            alpha.clone(),
            betas.clone(),
        );
    }

    if num_persistent_sends + num_persistent_receives != 0 {
        eval_logup_constraints::<_, SC, _>(
            builder,
            cumulative_persistent_sum
                .expect("cumulative_persistent_sum is Some if num_persistent_sends + num_persistent_receives > 0"),
            persistent_interactions,
            (persistent_cols_local, persistent_cols_next),
            global_alpha,
            global_betas,
        );
    }
}

/// The columns for logup are laid out as follows: the `i`-th column holds the sum of the row's interaction hashes
/// for the first `i+1` interactions, with the exception of the final column.
/// The final column holds the running sum of all interaction hashes for the current row and all prior rows.
fn eval_logup_constraints<M, SC, AB>(
    builder: &mut AB,
    cumulative_sum: AB::ExprEF,
    interactions: Vec<(Interaction<AB::F>, InteractionType)>,
    (logup_cols_local, logup_cols_next): (&[AB::VarEF], &[AB::VarEF]),
    alpha: AB::ExprEF,
    betas: Powers<AB::ExprEF>,
) where
    M: Machine<SC::Val>,
    SC: StarkConfig,
    AB: ValidaAirBuilder<Machine = M, F = SC::Val, EF = SC::Challenge>,
{
    let main = builder.main();
    let main_local: &[AB::Var] = main.row_slice(0);
    let main_next: &[AB::Var] = main.row_slice(1);

    let preprocessed = builder.preprocessed();
    let preprocessed_local = preprocessed.row_slice(0);
    let preprocessed_next = preprocessed.row_slice(1);

    let public = builder.public_values();
    let public_local = public.row_slice(0);
    let public_next = public.row_slice(1);

    let num_interactions = interactions.len();

    let chunk_size = MAX_PERMUTATION_CONSTRAINT_DEGREE - 1;
    let first_chunk_size = min(num_interactions, chunk_size - 1);
    let (first_chunk, remaining_interactions) = interactions.split_at(first_chunk_size);

    let running_sum_local: AB::ExprEF =
        (*logup_cols_local.last().expect("num_interactions > 0")).into();

    // Constrain `col_diff` to equal the sum of the values +/- count_i/hash_i for the interactions i in the chunk, with the
    // sign positive if the interaction is a send and negative if it is a receive.
    // This is enforced by the constraint (col_diff) * \prod_{i in chunk} hash_i = \sum_{i in chunk} +/- count_i * \prod_{j in chunk, j != i} hash_j
    // We compute the right hand side of this constraint iteratively by decomposing it as:
    // RHS = count_last * prod_{j in chunk, j < last} hash_j + hash_last * (RHS_prev).
    // e.g. when the chunk size is 3, this is RHS = count_3 * (hash_1 * hash_2) + hash_3 * (count_2 * hash_1 + hash_2 * count_1)
    // note that the degree of the lhs expression is equal to the number of interactions in `chunk` plus the degree of `col_diff`.
    let constrain_to_chunk_sum = |col_diff: AB::ExprEF, chunk, is_next| {
        let mut lhs = col_diff;
        let mut rhs = AB::ExprEF::zero();
        let mut prod_previous_hashes = AB::ExprEF::one();
        for &(ref interaction, interaction_type) in chunk {
            let (interaction_hash, count, _interaction_vec) = if is_next {
                reduce_interaction(
                    main_next,
                    preprocessed_next,
                    public_next,
                    interaction,
                    alpha.clone(),
                    betas.clone(),
                )
            } else {
                reduce_interaction(
                    main_local,
                    preprocessed_local,
                    public_local,
                    interaction,
                    alpha.clone(),
                    betas.clone(),
                )
            };
            let numerator: AB::ExprEF = match interaction_type {
                InteractionType::LocalSend
                | InteractionType::GlobalSend
                | InteractionType::PersistentSend => count.into(),
                InteractionType::LocalReceive
                | InteractionType::GlobalReceive
                | InteractionType::PersistentReceive => (-count).into(),
            };
            lhs *= interaction_hash.clone();
            rhs *= interaction_hash.clone();
            rhs += numerator * prod_previous_hashes.clone();
            prod_previous_hashes *= interaction_hash;
        }
        (lhs, rhs)
    };

    let col_diffs_local = logup_cols_local
        .iter()
        .skip(1)
        .map(|l| (*l).into())
        // The first column should hold the sum of the row's interaction hashes for all interactions in the first chunk.
        .zip(logup_cols_local.iter().map(|r| (*r).into()))
        .map(|(l, r)| l - r);

    // For `i` in 0..num_chunks - 1, the `i + 1`-st column minus the `i`-th column should equal the sum
    // of the interaction hashes for the interactions in the `i + 1`-st chunk.
    for (chunk, diff) in remaining_interactions
        .chunks(chunk_size)
        .zip(col_diffs_local)
    {
        let (lhs, rhs) = constrain_to_chunk_sum(diff, chunk.iter(), false);
        builder.assert_eq_ext(lhs, rhs);
    }

    // In the first row, the first column should hold the sum of the interaction hashes in `first_chunk`.
    let (lhs_first_row, rhs_first_row) = constrain_to_chunk_sum(
        (*logup_cols_local.first().unwrap()).into(),
        first_chunk.iter(),
        false,
    );
    builder
        // this condition adds one to the degree, hence why the `first_chunk` must be smaller
        // than the others
        .when_first_row()
        .assert_eq_ext(lhs_first_row, rhs_first_row);
    // In rows after the first row, the first column should equal the running sum from the prior
    // row plus the sum of the interaction hashes in `first_chunk`.
    let (lhs_transition, rhs_transition) = constrain_to_chunk_sum(
        (*logup_cols_next.first().unwrap()).into() - running_sum_local.clone(),
        first_chunk.iter(),
        true,
    );
    builder
        .when_transition()
        .assert_eq_ext(lhs_transition, rhs_transition);

    // The final column in the final row should equal `cumulative_sum`.
    builder
        .when_last_row()
        .assert_eq_ext(running_sum_local, cumulative_sum);
}
