use super::columns::Add32Cols;
use super::Add32Chip;
use core::borrow::Borrow;
use core::fmt::Debug;

use crate::add::columns::NUM_ADD_COLS;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField};
use p3_matrix::MatrixRowSlices;

impl<F> BaseAir<F> for Add32Chip {
    fn width(&self) -> usize {
        NUM_ADD_COLS
    }
}

impl<AB> Air<AB> for Add32Chip
where
    AB: AirBuilder,
    AB::F: PrimeField + Debug,
    AB::Expr: Debug,
    AB::Var: Debug,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &Add32Cols<AB::Var> = main.row_slice(0).borrow();

        let one = AB::F::one();
        let base = AB::F::from_canonical_u32(1 << 8);

        let carry_1 = local.carry[0];
        let carry_2 = local.carry[1];
        let carry_3 = local.carry[2];

        let overflow_0 =
            *local.input_1.index_be(3) + *local.input_2.index_be(3) - *local.output.index_be(3);

        let overflow_1 = *local.input_1.index_be(2) + *local.input_2.index_be(2)
            - *local.output.index_be(2)
            + carry_1;

        let overflow_2 = *local.input_1.index_be(1) + *local.input_2.index_be(1)
            - *local.output.index_be(1)
            + carry_2;

        let overflow_3 = *local.input_1.index_be(0) + *local.input_2.index_be(0)
            - *local.output.index_be(0)
            + carry_3;

        // Limb constraints
        builder.assert_zero(overflow_0.clone() * (overflow_0.clone() - base));
        builder.assert_zero(overflow_1.clone() * (overflow_1.clone() - base));
        builder.assert_zero(overflow_2.clone() * (overflow_2.clone() - base));
        builder.assert_zero(overflow_3.clone() * (overflow_3 - base));

        // Carry constraints
        builder.assert_zero(overflow_0.clone() * (carry_1 - one) + (overflow_0 - base) * carry_1);
        builder.assert_zero(overflow_1.clone() * (carry_2 - one) + (overflow_1 - base) * carry_2);
        builder.assert_zero(overflow_2.clone() * (carry_3 - one) + (overflow_2 - base) * carry_3);
        builder.assert_bool(carry_1);
        builder.assert_bool(carry_2);
        builder.assert_bool(carry_3);
    }
}
