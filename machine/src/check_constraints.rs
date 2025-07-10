use crate::columns::PermutationColsView;
use crate::columns::MAX_PERMUTATION_CONSTRAINT_DEGREE;
use crate::eval_permutation_constraints;
use crate::persistence::ChipWithPersistence;
use crate::Interaction;
use crate::InteractionType;
use crate::PublicTrace;
use crate::__internal::DebugConstraintBuilder;
use valida_machine::{PersistentInteractionType, StarkConfig};

use crate::Machine;
use p3_air::TwoRowMatrixView;
use p3_field::{AbstractField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_matrix::MatrixRowSlices;
use p3_maybe_rayon::prelude::*;

fn display_interaction<F: Field>(
    index: usize,
    interaction_type: InteractionType,
    interaction: &Interaction<F>,
    preprocessed_local: &[F],
    public_local: &[F],
    main_local: &[F],
) -> String {
    let Interaction {
        fields,
        count,
        argument_index,
    } = interaction;
    let count_res: F = count.apply(preprocessed_local, public_local, main_local);
    // only print if the count is non-zero, i.e. the interaction is actually sent
    if count_res != F::zero() {
        let type_str = match interaction_type {
            InteractionType::LocalSend => "local send",
            InteractionType::LocalReceive => "local receive",
            InteractionType::GlobalSend => "global send",
            InteractionType::GlobalReceive => "global receive",
            InteractionType::PersistentSend => "persistent send",
            InteractionType::PersistentReceive => "persistent receive",
        };
        let mut print = format!("interaction no. {index}: type: {} | ", type_str);
        let fields_applied = fields
            .iter()
            .map(|field| field.apply(preprocessed_local, public_local, main_local))
            .collect::<Vec<F>>();
        print.push_str(&format!("fields: {:?} | ", fields_applied));
        print.push_str(&format!("count: {:?} | ", count_res));
        print.push_str(&format!("argument_index: {:?}", argument_index));
        print.push('\n');
        print
    } else {
        String::new()
    }
}

fn display_persistent_interaction<F: Field>(
    index: usize,
    interaction_type: PersistentInteractionType,
    interaction: &Interaction<F>,
    preprocessed_local: &[F],
    public_local: &[F],
    main_local: &[F],
) -> String {
    let Interaction {
        fields,
        count,
        argument_index,
    } = interaction;
    let count_res: F = count.apply(preprocessed_local, public_local, main_local);
    // only print if the count is non-zero, i.e. the interaction is actually sent
    if count_res != F::zero() {
        let type_str = match interaction_type {
            PersistentInteractionType::PersistentSend => "persistent send",
            PersistentInteractionType::PersistentReceive => "persistent receive",
        };
        let mut print = format!("persistent interaction no. {index}: type: {} | ", type_str);
        let fields_applied = fields
            .iter()
            .map(|field| field.apply(preprocessed_local, public_local, main_local))
            .collect::<Vec<F>>();
        print.push_str(&format!("fields: {:?} | ", fields_applied));
        print.push_str(&format!("count: {:?} | ", count_res));
        print.push_str(&format!("argument_index: {:?}", argument_index));
        print.push('\n');
        print
    } else {
        String::new()
    }
}

/// Check that all constraints vanish on the subgroup. Setting `verbose` to `true` will
/// print all interactions in each row which have non-zero multiplicity.
#[allow(clippy::too_many_arguments)]
pub fn check_constraints<M, A, SC>(
    machine: &M,
    air: &A, // This is just a chip
    preprocessed: &Option<RowMajorMatrix<SC::Val>>,
    main: &Option<RowMajorMatrix<SC::Val>>,
    perm: &Option<RowMajorMatrix<SC::Challenge>>,
    height: usize,
    perm_challenges: &[SC::Challenge],
    global_perm_challenges: &[SC::Challenge],
    public: &Option<PublicTrace<SC::Val>>,
    verbose: bool,
) -> Option<Vec<String>>
where
    M: Machine<SC::Val>,
    A: ChipWithPersistence<M, SC>,
    SC: StarkConfig,
{
    if height == 0 {
        return None;
    }

    if let Some(main) = main {
        assert_eq!(
            height,
            main.height(),
            "height reported as {height} but height of main trace is {}",
            main.height()
        );
        assert_eq!(air.width(), main.width());
    }
    if let Some(preprocessed) = preprocessed {
        assert_eq!(
            air.preprocessed_width(),
            preprocessed.width(),
            "preprocessed width reported as {} but width of air is {}",
            air.preprocessed_width(),
            preprocessed.width()
        );
        assert_eq!(
            height,
            preprocessed.height(),
            "height reported as {} but height of preprocessed trace is {}",
            height,
            preprocessed.height()
        );
    }
    if let Some(public) = public {
        assert_eq!(
            air.public_width(),
            public.width(),
            "public width reported as {} but width of air is {}",
            public.width(),
            air.public_width()
        );
        match public {
            PublicTrace::PublicMatrix(matrix) => {
                assert_eq!(
                    height,
                    matrix.height(),
                    "height reported as {} but height of public trace is {}",
                    height,
                    matrix.height()
                )
            }
            PublicTrace::PublicVector(_) => {}
        }
    }
    if let Some(perm) = perm {
        assert_eq!(
            height,
            perm.height(),
            "height reported as {} but height of permutation trace is {}",
            height,
            perm.height()
        );
    }

    let (cumulative_ephemeral_sum, cumulative_persistent_sum) =
        perm.as_ref().map_or((None, None), |perm| {
            let num_ephemeral = air.ephemeral_interactions(machine).len();
            let num_persistent_sends = air.persistent_sends(machine).len();
            let num_persistent_receives = air.persistent_receives(machine).len();

            let PermutationColsView {
                ephemeral_cols,
                persistent_cols,
            } = PermutationColsView::as_view::<MAX_PERMUTATION_CONSTRAINT_DEGREE>(
                num_ephemeral,
                num_persistent_sends,
                num_persistent_receives,
                perm.row_slice(perm.height() - 1),
            );
            (ephemeral_cols.last(), persistent_cols.last())
        });

    // Check that constraints are satisfied.
    let prints = (0..height)
        .into_par_iter()
        .map(|i| {
            let mut prints_row = String::new();
            let i_next = (i + 1) % height;

            let (main_local, main_next) = match main {
                Some(main) => (main.row_slice(i), main.row_slice(i_next)),
                None => (&[][..], &[][..]),
            };
            let (public_local, public_next) = match public {
                Some(public) => (public.row_slice(i), public.row_slice(i_next)),
                None => (&[][..], &[][..]),
            };
            let (preprocessed_local, preprocessed_next) = match preprocessed {
                Some(preprocessed) => (preprocessed.row_slice(i), preprocessed.row_slice(i_next)),
                None => (&[][..], &[][..]),
            };
            let (perm_local, perm_next) = match perm {
                Some(perm) => (perm.row_slice(i), perm.row_slice(i_next)),
                None => (&[][..], &[][..]),
            };

            if verbose {
                let eph_str = air
                    .ephemeral_interactions(machine)
                    .iter()
                    .enumerate()
                    .map(|(m, (interaction, interaction_type))| {
                        display_interaction(
                            m,
                            *interaction_type,
                            interaction,
                            preprocessed_local,
                            public_local,
                            main_local,
                        )
                    })
                    .collect::<String>();
                let pers_str = air
                    .persistent_interactions(machine)
                    .iter()
                    .enumerate()
                    .map(|(m, (interaction, interaction_type))| {
                        display_persistent_interaction(
                            m,
                            *interaction_type,
                            interaction,
                            preprocessed_local,
                            public_local,
                            main_local,
                        )
                    })
                    .collect::<String>();
                prints_row = prints_row + &eph_str + &pers_str;
            }
            let mut builder = DebugConstraintBuilder {
                machine,
                main: TwoRowMatrixView {
                    local: main_local,
                    next: main_next,
                },
                public_values: TwoRowMatrixView {
                    local: public_local,
                    next: public_next,
                },
                preprocessed: TwoRowMatrixView {
                    local: preprocessed_local,
                    next: preprocessed_next,
                },
                perm: TwoRowMatrixView {
                    local: perm_local,
                    next: perm_next,
                },
                perm_challenges,
                global_perm_challenges,
                is_first_row: SC::Val::zero(),
                is_last_row: SC::Val::zero(),
                is_transition: SC::Val::one(),
                row_index: i,
            };
            if i == 0 {
                builder.is_first_row = SC::Val::one();
            }
            if i == height - 1 {
                builder.is_last_row = SC::Val::one();
                builder.is_transition = SC::Val::zero();
            }

            air.eval(&mut builder);
            eval_permutation_constraints(
                air,
                &mut builder,
                cumulative_ephemeral_sum.copied(),
                cumulative_persistent_sum.copied(),
            );
            prints_row
        })
        .collect::<Vec<_>>();

    // iterate through the prints outside of the parallel iterator so they're in order
    if verbose {
        let mut log_prints = Vec::with_capacity(prints.len());
        for (i, prints_row) in prints.iter().enumerate() {
            let log_print = if !prints_row.is_empty() {
                format!("Interactions in row no. {}:\n{prints_row}", i)
            } else {
                String::new()
            };
            log_prints.push(log_print);
        }
        Some(log_prints)
    } else {
        None
    }
}
