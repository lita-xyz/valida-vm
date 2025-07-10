use super::{columns::Div32Cols, Div32Chip};
use core::{borrow::Borrow, fmt::Debug};

use crate::div::columns::NUM_DIV_COLS;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField};
use p3_matrix::MatrixRowSlices;

impl<F> BaseAir<F> for Div32Chip {
    fn width(&self) -> usize {
        NUM_DIV_COLS
    }
}

impl<AB> Air<AB> for Div32Chip
where
    AB: AirBuilder,
    AB::F: PrimeField + Debug,
    AB::Expr: Debug,
    AB::Var: Debug,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &Div32Cols<AB::Var> = main.row_slice(0).borrow();

        let sign_byte = AB::Expr::from_canonical_u32(0xff);

        builder.assert_bool(local.is_div);
        builder.assert_bool(local.is_sdiv);
        builder.assert_bool(local.is_div + local.is_sdiv);

        // a * c should not overflow; since product_upper is checked to equal mulhu when the opcode is div,
        // it suffices to check that product_upper is all zeros.
        local.product_upper.iter_be().for_each(|byte| {
            builder
                .when(local.is_div)
                .assert_eq(*byte, AB::Expr::zero())
        });

        // a * c should not overflow. since product_upper is checked by lookup to equal mulhs of a and c,
        // it suffices to check that product_upper is either all zeros or all ones.
        local.product_upper.iter_be().for_each(|byte| {
            // as the entries of product_upper are known to be bytes, their sum is zero iff they are all zero
            builder
                .when(local.is_sdiv)
                .when(
                    local
                        .product_upper
                        .iter_be()
                        .map(|b| (*b).into())
                        .sum::<AB::Expr>(),
                )
                .assert_eq(*byte, sign_byte.clone());
        });

        // ensures that lookups depending on the sign are only sent for SDIV32 rows.
        builder
            .when(local.sign_1)
            .assert_eq(local.is_sdiv, AB::Expr::one());
        builder
            .when(local.sign_2)
            .assert_eq(local.is_sdiv, AB::Expr::one());

        // Ensure that `same_sign` is set correctly
        builder.assert_bool(local.same_sign);
        builder.when(local.is_sdiv).assert_eq(
            local.same_sign,
            local.sign_1 * local.sign_2
                + (AB::Expr::one() - local.sign_1) * (AB::Expr::one() - local.sign_2),
        );
        builder
            .when(local.is_div)
            .assert_eq(local.same_sign, AB::Expr::one());

        // Check that, in the case of sdiv, the sign of the remainder is either equal to
        // the sign of input_1, or the remainder is zero.
        local.remainder.iter_be().for_each(|byte| {
            builder
                .when(local.is_sdiv)
                .when(*byte)
                .assert_eq(local.sign_remainder, local.sign_2);
        });
    }
}
