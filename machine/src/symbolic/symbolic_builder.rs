use alloc::vec::Vec;

use crate::config::StarkConfig;
use crate::persistence::ChipWithPersistence;
use crate::{Machine, ValidaAirBuilder};
use p3_air::{AirBuilder, PairBuilder, PermutationAirBuilder};
use p3_air::{AirBuilderWithPublicValues, ExtensionBuilder, TwoRowMatrixView};
use p3_field::{AbstractExtensionField, AbstractField};
use p3_util::log2_ceil_usize;
use valida_machine::debug_builder::AirBuilderWithGlobalPermutationChallenges;
use valida_machine::symbolic::symbolic_expression_ext::SymbolicExpressionExt;
use valida_machine::symbolic::symbolic_variable::Trace;

use crate::symbolic::symbolic_expression::SymbolicExpression;
use crate::symbolic::symbolic_variable::SymbolicVariable;

pub fn get_log_quotient_degree<M, SC, C>(machine: &M, chip: &C) -> usize
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
    C: ChipWithPersistence<M, SC>,
{
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = get_max_constraint_degree(machine, chip).max(3);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the zerofier.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

pub fn get_max_constraint_degree<M, SC, C>(machine: &M, chip: &C) -> usize
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
    C: ChipWithPersistence<M, SC>,
{
    get_symbolic_constraints(machine, chip)
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0)
}

pub fn get_symbolic_constraints<M, SC, C>(machine: &M, chip: &C) -> Vec<SymbolicExpression<SC::Val>>
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
    C: ChipWithPersistence<M, SC>,
{
    let mut builder = SymbolicAirBuilder::new(
        machine,
        chip.main_width(),
        chip.preprocessed_width(),
        chip.public_width(),
        chip.permutation_width(machine),
    );
    chip.eval(&mut builder);
    builder.constraints()
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
pub struct SymbolicAirBuilder<'a, M: Machine<SC::Val>, SC: StarkConfig> {
    machine: &'a M,
    preprocessed: TwoRowMatrixView<'a, SymbolicVariable<SC::Val>>,
    main: TwoRowMatrixView<'a, SymbolicVariable<SC::Val>>,
    permutation: TwoRowMatrixView<'a, SymbolicVariable<SC::Challenge>>,
    public_values: TwoRowMatrixView<'a, SymbolicVariable<SC::Val>>,
    constraints: Vec<SymbolicExpression<SC::Val>>,
    perm_challenges: Vec<SC::Challenge>,
    global_perm_challenges: Vec<SC::Challenge>,
}

impl<'a, M: Machine<SC::Val>, SC: StarkConfig> SymbolicAirBuilder<'a, M, SC> {
    const NUM_ROUNDS: usize = 3;
    pub(crate) fn new(
        machine: &'a M,
        main_width: usize,
        preprocessed_width: usize,
        public_width: usize,
        permutation_width: usize,
    ) -> Self {
        Self {
            machine,
            preprocessed: SymbolicVariable::window(Trace::Preprocessed, preprocessed_width),
            main: SymbolicVariable::window(Trace::Main, main_width),
            permutation: SymbolicVariable::window(Trace::Permutation, permutation_width),
            public_values: SymbolicVariable::window(Trace::Public, public_width),
            constraints: vec![],
            // TODO: replace with symbolic challenge variables
            perm_challenges: vec![SC::Challenge::zero(); Self::NUM_ROUNDS],
            global_perm_challenges: vec![SC::Challenge::zero(); Self::NUM_ROUNDS],
        }
    }

    pub(crate) fn constraints(self) -> Vec<SymbolicExpression<SC::Val>> {
        self.constraints
    }
}

impl<'a, M: Machine<SC::Val>, SC: StarkConfig> AirBuilder for SymbolicAirBuilder<'a, M, SC> {
    type F = SC::Val;
    type Expr = SymbolicExpression<SC::Val>;
    type Var = SymbolicVariable<SC::Val>;
    type M = TwoRowMatrixView<'a, Self::Var>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let cons = x.into();
        let d = cons.degree_multiple();
        if d > 3 {
            panic!("attempt to add a constraint of degree {d}; the max supported by valida is 3")
        }
        self.constraints.push(cons);
    }
}

impl<M: Machine<SC::Val>, SC: StarkConfig> PairBuilder for SymbolicAirBuilder<'_, M, SC> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<M: Machine<SC::Val>, SC: StarkConfig> ExtensionBuilder for SymbolicAirBuilder<'_, M, SC> {
    type EF = SC::Challenge;
    type ExprEF = SymbolicExpressionExt<SC::Challenge>;
    type VarEF = SymbolicVariable<SC::Challenge>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        for xb in x.into().as_base_slice().iter().cloned() {
            self.assert_zero::<SymbolicExpression<SC::Val>>(xb);
        }
    }
}

impl<'a, M: Machine<SC::Val>, SC: StarkConfig> PermutationAirBuilder
    for SymbolicAirBuilder<'a, M, SC>
{
    type MP = TwoRowMatrixView<'a, Self::VarEF>;

    fn permutation(&self) -> Self::MP {
        self.permutation
    }

    fn permutation_randomness(&self) -> &[Self::EF] {
        &self.perm_challenges[..]
    }
}

impl<M: Machine<SC::Val>, SC: StarkConfig> ValidaAirBuilder for SymbolicAirBuilder<'_, M, SC> {
    type Machine = M;

    fn machine(&self) -> &Self::Machine {
        self.machine
    }
}

impl<M: Machine<SC::Val>, SC: StarkConfig> AirBuilderWithPublicValues
    for SymbolicAirBuilder<'_, M, SC>
{
    fn public_values(&self) -> Self::M {
        self.public_values
    }
}

impl<M: Machine<SC::Val>, SC> AirBuilderWithGlobalPermutationChallenges
    for SymbolicAirBuilder<'_, M, SC>
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
{
    fn global_permutation_randomness(&self) -> &[SC::Challenge] {
        &self.global_perm_challenges
    }
}
