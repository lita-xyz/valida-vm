use crate::columns::{OutputCols, NUM_OUTPUT_COLS, NUM_PUBLIC_OUTPUT_COLS};
use crate::OutputChip;
use core::borrow::Borrow;
use core::fmt::Debug;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField};
use p3_matrix::MatrixRowSlices;
use valida_machine::{reduce_word, Word};

impl<F> BaseAir<F> for OutputChip {
    fn width(&self) -> usize {
        NUM_OUTPUT_COLS
    }

    fn public_width(&self) -> usize {
        NUM_PUBLIC_OUTPUT_COLS
    }
}

impl<AB> Air<AB> for OutputChip
where
    AB: AirBuilder,
    AB::F: PrimeField + Debug,
    AB::Expr: Debug,
    AB::Var: Debug,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &OutputCols<AB::Var> = main.row_slice(0).borrow();
        let next: &OutputCols<AB::Var> = main.row_slice(1).borrow();

        let base = Word::from_components_le([1, 1 << 8, 1 << 16, 1 << 24])
            .transform(AB::Expr::from_canonical_u32);
        let diff = reduce_word::<AB>(&base, local.diff);
        builder
            .when_transition()
            .when(next.is_real)
            .assert_eq(diff, next.clk - local.clk);
    }
}
