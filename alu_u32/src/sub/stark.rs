use super::columns::Sub32Cols;
use super::Sub32Chip;
use core::borrow::Borrow;
use core::fmt::Debug;

use crate::sub::columns::NUM_SUB_COLS;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField};
use p3_matrix::MatrixRowSlices;

impl<F> BaseAir<F> for Sub32Chip {
    fn width(&self) -> usize {
        NUM_SUB_COLS
    }
}

impl<AB> Air<AB> for Sub32Chip
where
    AB: AirBuilder,
    AB::F: PrimeField + Debug,
    AB::Expr: Debug,
    AB::Var: Debug,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &Sub32Cols<AB::Var> = main.row_slice(0).borrow();

        let base = AB::Expr::from_canonical_u32(1 << 8);

        for ((((output_byte, input_1_byte), input_2_byte), this_borrow_bit), last_borrow_bit) in
            local
                .output
                .iter_be()
                .zip(local.input_1.iter_be())
                .zip(local.input_2.iter_be())
                .zip(local.borrow.iter_be())
                .zip(local.borrow.iter_be().skip(1))
        {
            builder.assert_eq(
                *output_byte,
                base.clone() * *this_borrow_bit + *input_1_byte - *input_2_byte - *last_borrow_bit,
            );
        }

        builder.assert_eq(
            *local.output.index_be(3),
            base.clone() * *local.borrow.index_be(3) + *local.input_1.index_be(3)
                - *local.input_2.index_be(3),
        );

        for borrow_bit in local.borrow.iter_be() {
            builder.assert_bool(*borrow_bit);
        }
    }
}
