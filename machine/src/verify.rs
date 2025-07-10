use p3_air::TwoRowMatrixView;
use p3_field::AbstractExtensionField;
use p3_field::{AbstractField, Field};
use p3_util::reverse_slice_index_bits;

use crate::folding_builder::VerifierConstraintFolder;
use crate::public::PublicValues;
use crate::{
    eval_permutation_constraints, persistence::ChipWithPersistence, proof::OpenedValues, Chip,
    Machine, StarkConfig,
};

#[derive(Debug)]
pub enum ConstraintError<SC>
where
    SC: StarkConfig,
{
    OodEvaluationMismatch {
        expected: <SC as StarkConfig>::Challenge,
        actual: <SC as StarkConfig>::Challenge,
    },
}

#[allow(clippy::too_many_arguments)]
pub fn verify_constraints<M, C, SC>(
    machine: &M,
    chip: &C,
    opened_values: &OpenedValues<SC::Challenge>,
    public_values: &Option<<C as Chip<M, SC>>::Public>,
    cumulative_ephemeral_sum: Option<SC::Challenge>,
    cumulative_persistent_sum: Option<SC::Challenge>,
    log_degree: usize,
    g: SC::Val,
    zeta: <SC as StarkConfig>::Challenge,
    alpha: <SC as StarkConfig>::Challenge,
    permutation_challenges: &[<SC as StarkConfig>::Challenge],
    global_permutation_challenges: &[<SC as StarkConfig>::Challenge],
) -> Result<(), ConstraintError<SC>>
where
    M: Machine<SC::Val>,
    C: ChipWithPersistence<M, SC>,
    SC: StarkConfig,
{
    let z_h = zeta.exp_power_of_2(log_degree) - <SC as StarkConfig>::Challenge::one();
    let is_first_row = z_h / (zeta - SC::Val::one());
    let is_last_row = z_h / (zeta - g.inverse());
    let is_transition = zeta - g.inverse();

    let OpenedValues {
        preprocessed_local,
        preprocessed_next,
        trace_local,
        trace_next,
        permutation_local,
        permutation_next,
        quotient_chunks,
    } = opened_values;

    let (public_local, public_next) = if let Some(ref public_values) = public_values {
        (
            public_values.interpolate(zeta, 0),
            public_values.interpolate(zeta, 1),
        )
    } else {
        (vec![], vec![])
    };

    let monomials = (0..SC::Challenge::D)
        .map(SC::Challenge::monomial)
        .collect::<Vec<_>>();

    let unflatten = |v: &[SC::Challenge]| {
        v.chunks_exact(SC::Challenge::D)
            .map(|chunk| {
                chunk
                    .iter()
                    .zip(monomials.iter())
                    .map(|(x, m)| *x * *m)
                    .sum()
            })
            .collect::<Vec<SC::Challenge>>()
    };

    // Recompute the quotient as extension elements.
    let mut quotient_parts = quotient_chunks
        .chunks_exact(SC::Challenge::D)
        .map(|chunk| {
            chunk
                .iter()
                .zip(monomials.iter())
                .map(|(x, m)| *x * *m)
                .sum()
        })
        .collect::<Vec<SC::Challenge>>();

    let mut folder = VerifierConstraintFolder {
        machine,
        preprocessed: TwoRowMatrixView {
            local: preprocessed_local,
            next: preprocessed_next,
        },
        main: TwoRowMatrixView {
            local: trace_local,
            next: trace_next,
        },
        public_values: TwoRowMatrixView {
            local: &public_local,
            next: &public_next,
        },
        perm: TwoRowMatrixView {
            local: &unflatten(permutation_local),
            next: &unflatten(permutation_next),
        },
        perm_challenges: permutation_challenges,
        global_perm_challenges: global_permutation_challenges,
        is_first_row,
        is_last_row,
        is_transition,
        alpha,
        accumulator: <SC as StarkConfig>::Challenge::zero(),
    };
    chip.eval(&mut folder);
    eval_permutation_constraints(
        chip,
        &mut folder,
        cumulative_ephemeral_sum,
        cumulative_persistent_sum,
    );

    reverse_slice_index_bits(&mut quotient_parts);
    let quotient: <SC as StarkConfig>::Challenge = zeta
        .powers()
        .zip(quotient_parts)
        .map(|(weight, part)| part * weight)
        .sum();

    let folded_constraints = folder.accumulator;

    match folded_constraints == z_h * quotient {
        true => Ok(()),
        false => Err(ConstraintError::OodEvaluationMismatch {
            expected: z_h * quotient,
            actual: folded_constraints,
        }),
    }
}
