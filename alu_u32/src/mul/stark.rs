use super::columns::Mul32Cols;
use super::{get_partially_reduced_and_reduced_sums, Mul32Chip};
use crate::mul::columns::{CARRY_MAX, LIMB_SIZE, NUM_MUL_COLS};
use crate::mul::Long;
use core::array::from_fn;
use core::borrow::Borrow;
use core::fmt::Debug;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::MatrixRowSlices;
use std::iter;
use valida_machine::{Word, MEMORY_CELL_BYTES};

impl<F> BaseAir<F> for Mul32Chip {
    fn width(&self) -> usize {
        NUM_MUL_COLS
    }
}

impl<AB> Air<AB> for Mul32Chip
where
    AB: AirBuilder,
    AB::F: PrimeField32 + Debug,
    AB::Expr: Debug,
    AB::Var: Debug,
{
    fn eval(&self, builder: &mut AB) {
        // This is the condition needed to ensure that there
        // is no overflow possible in our calculations.
        #[cfg(debug_assertions)]
        {
            use super::columns::PI_MAX;
            assert!((AB::F::ORDER_U32 as usize) > PI_MAX);
        }

        let main = builder.main();
        let local: &Mul32Cols<AB::Var> = main.row_slice(0).borrow();
        let next: &Mul32Cols<AB::Var> = main.row_slice(1).borrow();

        // Limb weights
        let base_m = [1, 1 << 8, 1 << 16].map(AB::Expr::from_canonical_u32);

        debug_assert!(base_m.len() >= LIMB_SIZE);

        let product = Long {
            low: local.lower_word.transform(|var| var.into()),
            high: local.upper_word.transform(|var| var.into()),
        };

        let sign_extension_byte = AB::Expr::from_canonical_u8(u8::MAX);
        let (input_1_upper, input_2_upper) = (
            Word::from_components_le(from_fn(|_| local.sign_1 * sign_extension_byte.clone())),
            Word::from_components_le(from_fn(|_| local.sign_2 * sign_extension_byte.clone())),
        );
        let (input_1_extended, input_2_extended) = (
            Long {
                low: local.input_1.transform(|var| var.into()),
                high: input_1_upper,
            },
            Long {
                low: local.input_2.transform(|var| var.into()),
                high: input_2_upper,
            },
        );

        // The pis each compute a sum of products of the individual input bytes, each formed
        // by collecting all terms in the formal product of the base-2^8 expansions of the inputs
        // whose total "degree" is in a `LIMB_SIZE` chunk of `[0, PRODUCT_LENGTH)`, then factoring
        // out the common lowest power. They are "partially reduced" in that their size is at most PI_MAX/2.
        //
        // The sigmas each compute the reduction LIMB_SIZE-byte limb of the output, reduced in that each is guaranteed
        // to be less than 2^(8 * LIMB_SIZE).
        let (pis, sigmas) = get_partially_reduced_and_reduced_sums::<{ Mul32Chip::MIN_LENGTH }, _, _>(
            &base_m,
            &input_1_extended,
            &input_2_extended,
            &product,
        );

        // Congruence checks
        sigmas
            .iter()
            .zip(pis)
            .zip(local.carry.iter_le())
            .zip(
                iter::once(AB::Expr::zero()).chain(
                    local
                        .carry
                        .into_iter_le()
                        .take(MEMORY_CELL_BYTES - 1)
                        .map(|c| c.into()),
                ),
            )
            .for_each(|(((sigma, pi), carry), prev_carry)| {
                builder.assert_eq(
                    // pi is at most the binomial coefficient `binom(PRODUCT_LIMS, 2)`
                    // assuming the input scalars are all smaller than 2^8, pi_0 is less than 2^24 * 4 = 2^26.
                    // the entries of carry are at most 2^10
                    pi + prev_carry.clone() - sigma.clone(),
                    *carry * base_m[2].clone(),
                );
            });

        // Range check counter
        let last_counter = AB::Expr::from_canonical_usize(CARRY_MAX - 1);
        builder
            .when_first_row()
            .assert_eq(local.counter, AB::Expr::zero());
        let counter_diff = next.counter - local.counter;
        builder
            .when_transition()
            .assert_zero(counter_diff.clone() * (counter_diff - AB::Expr::one()));
        builder
            .when_last_row()
            .assert_eq(local.counter, last_counter);
    }
}
