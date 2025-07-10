use crate::__internal::ProverConstraintFolder;
use crate::config::StarkConfig;
use crate::symbolic::symbolic_builder::get_log_quotient_degree;
use crate::{eval_permutation_constraints, persistence::ChipWithPersistence, Machine};
use itertools::Itertools;
use p3_air::TwoRowMatrixView;
use p3_commit::UnivariatePcsWithLde;
use p3_field::{
    cyclic_subgroup_coset_known_order, AbstractExtensionField, AbstractField, Field, PackedField,
    TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{MatrixGet, MatrixRows};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{decompose_and_flatten, ZerofierOnCoset};
use tracing::instrument;

pub fn quotient<M, A, SC, PreprocessedTraceLde, MainTraceLde, PermTraceLde, PublicTraceLde>(
    machine: &M,
    config: &SC,
    air: &A,
    log_degree: usize,
    preprocessed_trace_lde: Option<PreprocessedTraceLde>,
    main_trace_lde: Option<MainTraceLde>,
    perm_trace_lde: Option<PermTraceLde>,
    public_trace_lde: Option<PublicTraceLde>,
    cumulative_sum: Option<SC::Challenge>,
    cumulative_product: Option<SC::Challenge>,
    perm_challenges: &[SC::Challenge],
    global_perm_challenges: &[SC::Challenge],
    alpha: SC::Challenge,
) -> RowMajorMatrix<SC::Val>
where
    M: Machine<SC::Val>,
    A: ChipWithPersistence<M, SC>,
    SC: StarkConfig,
    PreprocessedTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync,
    MainTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync,
    PermTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync,
    PublicTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync,
{
    let pcs = config.pcs();
    let log_quotient_degree = get_log_quotient_degree::<M, SC, A>(machine, air);

    let log_stride_for_quotient = pcs.log_blowup() - log_quotient_degree;
    let preprocessed_trace_lde_for_quotient =
        preprocessed_trace_lde.map(|lde| lde.vertically_strided(1 << log_stride_for_quotient, 0));
    let main_trace_lde_for_quotient =
        main_trace_lde.map(|lde| lde.vertically_strided(1 << log_stride_for_quotient, 0));
    let perm_trace_lde_for_quotient =
        perm_trace_lde.map(|lde| lde.vertically_strided(1 << log_stride_for_quotient, 0));
    let public_trace_lde_for_quotient =
        public_trace_lde.map(|values| values.vertically_strided(1 << log_stride_for_quotient, 0));

    let quotient_values = quotient_values::<M, SC, A, _, _, _, _>(
        machine,
        config,
        air,
        log_degree,
        log_quotient_degree,
        preprocessed_trace_lde_for_quotient,
        main_trace_lde_for_quotient,
        perm_trace_lde_for_quotient,
        public_trace_lde_for_quotient,
        cumulative_sum,
        cumulative_product,
        perm_challenges,
        global_perm_challenges,
        alpha,
    );

    decompose_and_flatten::<SC::Val, SC::Challenge>(
        quotient_values,
        SC::Challenge::from_base(pcs.coset_shift()),
        log_quotient_degree,
    )
}

#[instrument(name = "compute quotient polynomial", skip_all)]
fn quotient_values<M, SC, A, PreprocessedTraceLde, MainTraceLde, PermTraceLde, PublicTraceLde>(
    machine: &M,
    config: &SC,
    air: &A,
    log_degree: usize,
    log_quotient_degree: usize,
    preprocessed_trace_lde: Option<PreprocessedTraceLde>,
    main_trace_lde: Option<MainTraceLde>,
    perm_trace_lde: Option<PermTraceLde>,
    public_values: Option<PublicTraceLde>,
    cumulative_sum: Option<SC::Challenge>,
    cumulative_product: Option<SC::Challenge>,
    perm_challenges: &[SC::Challenge],
    global_perm_challenges: &[SC::Challenge],
    alpha: SC::Challenge,
) -> Vec<SC::Challenge>
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
    A: ChipWithPersistence<M, SC>,
    PublicTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync,
    PreprocessedTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync,
    MainTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync,
    PermTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync,
{
    let degree = 1 << log_degree;
    let log_quotient_size = log_degree + log_quotient_degree;
    let quotient_size = 1 << log_quotient_size;
    let g_subgroup = SC::Val::two_adic_generator(log_degree);
    let g_extended = SC::Val::two_adic_generator(log_quotient_size);
    let subgroup_last = g_subgroup.inverse();
    let coset_shift = config.pcs().coset_shift();
    let next_step = 1 << log_quotient_degree;

    let mut coset: Vec<_> =
        cyclic_subgroup_coset_known_order(g_extended, coset_shift, quotient_size).collect();

    let zerofier_on_coset = ZerofierOnCoset::new(log_degree, log_quotient_degree, coset_shift);

    // Evaluations of L_first(x) = Z_H(x) / (x - 1) on our coset s H.
    let mut lagrange_first_evals = zerofier_on_coset.lagrange_basis_unnormalized(0);
    let mut lagrange_last_evals = zerofier_on_coset.lagrange_basis_unnormalized(degree - 1);

    // We have a few vectors of length `quotient_size`, and we're going to take slices therein of
    // length `WIDTH`. In the edge case where `quotient_size < WIDTH`, we need to pad those vectors
    // in order for the slices to exist. The entries beyond quotient_size will be ignored, so we can
    // just use default values.
    for _ in quotient_size..SC::PackedVal::WIDTH {
        coset.push(SC::Val::default());
        lagrange_first_evals.push(SC::Val::default());
        lagrange_last_evals.push(SC::Val::default());
    }

    (0..quotient_size)
        .into_par_iter()
        .step_by(SC::PackedVal::WIDTH)
        .flat_map_iter(|i_local_start| {
            let wrap = |i| i % quotient_size;
            let i_next_start = wrap(i_local_start + next_step);
            let i_range = i_local_start..i_local_start + SC::PackedVal::WIDTH;

            let x = *SC::PackedVal::from_slice(&coset[i_range.clone()]);
            let is_transition = x - subgroup_last;
            let is_first_row = *SC::PackedVal::from_slice(&lagrange_first_evals[i_range.clone()]);
            let is_last_row = *SC::PackedVal::from_slice(&lagrange_last_evals[i_range]);

            let indices_to_packed_val_main = |start_row: usize, col: usize, lde: &MainTraceLde| {
                SC::PackedVal::from_fn(|offset| {
                    let row = wrap(start_row + offset);
                    lde.get(row, col)
                })
            };
            let indices_to_packed_val_preprocessed =
                |start_row: usize, col: usize, lde: &PreprocessedTraceLde| {
                    SC::PackedVal::from_fn(|offset| {
                        let row = wrap(start_row + offset);
                        lde.get(row, col)
                    })
                };

            let (main_local, main_next): (Vec<_>, Vec<_>) = match &main_trace_lde {
                Some(lde) => (0..lde.width())
                    .map(|col| {
                        (
                            indices_to_packed_val_main(i_local_start, col, lde),
                            indices_to_packed_val_main(i_next_start, col, lde),
                        )
                    })
                    .collect(),
                None => (vec![], vec![]),
            };

            let (preprocessed_local, preprocessed_next): (Vec<_>, Vec<_>) =
                match &preprocessed_trace_lde {
                    Some(lde) => (0..lde.width())
                        .map(|col| {
                            (
                                indices_to_packed_val_preprocessed(i_local_start, col, lde),
                                indices_to_packed_val_preprocessed(i_next_start, col, lde),
                            )
                        })
                        .collect(),
                    None => (vec![], vec![]),
                };

            let ext_degree = <SC::Challenge as AbstractExtensionField<SC::Val>>::D;

            let (perm_local, perm_next): (Vec<_>, Vec<_>) = match &perm_trace_lde {
                Some(ref perm_trace_lde) => {
                    debug_assert_eq!(perm_trace_lde.width() % ext_degree, 0);
                    let perm_width_ext = perm_trace_lde.width() / ext_degree;

                    // Maintain original structure with explicit duplicated code
                    let local = (0..perm_width_ext)
                        .map(|ext_col| {
                            SC::PackedChallenge::from_base_fn(|coeff_idx| {
                                SC::PackedVal::from_fn(|offset| {
                                    let row = wrap(i_local_start + offset);
                                    perm_trace_lde.get(row, ext_col * ext_degree + coeff_idx)
                                })
                            })
                        })
                        .collect();

                    let next = (0..perm_width_ext)
                        .map(|ext_col| {
                            SC::PackedChallenge::from_base_fn(|coeff_idx| {
                                SC::PackedVal::from_fn(|offset| {
                                    let row = wrap(i_next_start + offset);
                                    perm_trace_lde.get(row, ext_col * ext_degree + coeff_idx)
                                })
                            })
                        })
                        .collect();

                    (local, next)
                }
                None => (vec![], vec![]),
            };

            let (public_local, public_next): (Vec<_>, Vec<_>) = match public_values {
                Some(ref public_values) => (
                    (0..public_values.width())
                        .map(|col| {
                            SC::PackedVal::from_fn(|offset| {
                                let row = wrap(i_local_start + offset);
                                public_values.get(row, col)
                            })
                        })
                        .collect(),
                    (0..public_values.width())
                        .map(|col| {
                            SC::PackedVal::from_fn(|offset| {
                                let row = wrap(i_next_start + offset);
                                public_values.get(row, col)
                            })
                        })
                        .collect(),
                ),
                None => (vec![], vec![]),
            };

            let accumulator = SC::PackedChallenge::zero();
            let mut folder = ProverConstraintFolder {
                machine,
                public_values: TwoRowMatrixView {
                    local: &public_local,
                    next: &public_next,
                },
                preprocessed: TwoRowMatrixView {
                    local: &preprocessed_local,
                    next: &preprocessed_next,
                },
                main: TwoRowMatrixView {
                    local: &main_local,
                    next: &main_next,
                },
                perm: TwoRowMatrixView {
                    local: &perm_local,
                    next: &perm_next,
                },
                perm_challenges,
                global_perm_challenges,
                is_first_row,
                is_last_row,
                is_transition,
                alpha,
                accumulator,
            };
            air.eval(&mut folder);
            eval_permutation_constraints(
                air,
                &mut folder,
                cumulative_sum.map(SC::PackedChallenge::from_f),
                cumulative_product.map(SC::PackedChallenge::from_f),
            );

            // quotient(x) = constraints(x) / Z_H(x)
            let zerofier_inv: SC::PackedVal = zerofier_on_coset.eval_inverse_packed(i_local_start);
            let quotient = folder.accumulator * zerofier_inv;

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            let limit = SC::PackedVal::WIDTH.min(quotient_size);
            (0..limit).map(move |idx_in_packing| {
                let quotient_value = (0..<SC::Challenge as AbstractExtensionField<SC::Val>>::D)
                    .map(|coeff_idx| quotient.as_base_slice()[coeff_idx].as_slice()[idx_in_packing])
                    .collect_vec();
                SC::Challenge::from_base_slice(&quotient_value)
            })
        })
        .collect()
}
