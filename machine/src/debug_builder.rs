use crate::{Machine, ValidaAirBuilder};
use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
    TwoRowMatrixView,
};
use p3_field::AbstractField;
use valida_machine::StarkConfig;

pub trait AirBuilderWithGlobalPermutationChallenges: ExtensionBuilder {
    fn global_permutation_randomness(&self) -> &[Self::EF];
}

/// An `AirBuilder` which asserts that each constraint is zero, allowing any failed constraints to
/// be detected early.
pub struct DebugConstraintBuilder<'a, M: Machine<SC::Val>, SC: StarkConfig> {
    pub(crate) machine: &'a M,
    pub(crate) main: TwoRowMatrixView<'a, SC::Val>,
    pub(crate) preprocessed: TwoRowMatrixView<'a, SC::Val>,
    pub(crate) perm: TwoRowMatrixView<'a, SC::Challenge>,
    pub(crate) perm_challenges: &'a [SC::Challenge],
    pub(crate) global_perm_challenges: &'a [SC::Challenge],
    pub(crate) is_first_row: SC::Val,
    pub(crate) is_last_row: SC::Val,
    pub(crate) is_transition: SC::Val,
    pub(crate) public_values: TwoRowMatrixView<'a, SC::Val>,
    pub(crate) row_index: usize,
}

impl<'a, M, SC> AirBuilder for DebugConstraintBuilder<'a, M, SC>
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
{
    type F = SC::Val;
    type Expr = SC::Val;
    type Var = SC::Val;
    type M = TwoRowMatrixView<'a, SC::Val>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("only supports a window size of 2")
        }
    }

    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        let x = x.into();
        let y = y.into();
        assert_eq!(
            x, y,
            "constraint {:?} == {:?} failed on row {}",
            x, y, self.row_index
        );
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x = x.into();
        assert_eq!(
            x,
            SC::Val::zero(),
            "constraint {:?} == 0 failed on row {}",
            x,
            self.row_index
        );
    }
}

impl<M, SC> PairBuilder for DebugConstraintBuilder<'_, M, SC>
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
{
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<M, SC> ExtensionBuilder for DebugConstraintBuilder<'_, M, SC>
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
{
    type EF = SC::Challenge;
    type ExprEF = SC::Challenge;
    type VarEF = SC::Challenge;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        assert_eq!(
            x.into(),
            SC::Challenge::zero(),
            "constraints must evaluate to zero"
        );
    }
}

impl<'a, M, SC> PermutationAirBuilder for DebugConstraintBuilder<'a, M, SC>
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
{
    type MP = TwoRowMatrixView<'a, SC::Challenge>;

    fn permutation(&self) -> Self::MP {
        self.perm
    }

    fn permutation_randomness(&self) -> &[Self::EF] {
        self.perm_challenges
    }
}

impl<M: Machine<SC::Val>, SC> ValidaAirBuilder for DebugConstraintBuilder<'_, M, SC>
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
{
    type Machine = M;

    fn machine(&self) -> &Self::Machine {
        self.machine
    }
}

impl<M: Machine<SC::Val>, SC> AirBuilderWithPublicValues for DebugConstraintBuilder<'_, M, SC>
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
{
    fn public_values(&self) -> Self::M {
        self.public_values
    }
}

impl<M: Machine<SC::Val>, SC> AirBuilderWithGlobalPermutationChallenges
    for DebugConstraintBuilder<'_, M, SC>
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
{
    fn global_permutation_randomness(&self) -> &[SC::Challenge] {
        self.global_perm_challenges
    }
}
