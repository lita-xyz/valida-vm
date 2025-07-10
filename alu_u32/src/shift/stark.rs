use super::columns::Shift32Cols;
use super::Shift32Chip;
use core::fmt::Debug;
use core::{borrow::Borrow, iter};

use crate::shift::columns::NUM_SHIFT_COLS;
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::MatrixRowSlices;
use valida_machine::MEMORY_CELL_BYTES;

impl<F> BaseAir<F> for Shift32Chip {
    fn width(&self) -> usize {
        NUM_SHIFT_COLS
    }
}

impl<AB> Air<AB> for Shift32Chip
where
    AB: AirBuilder,
    AB::F: AbstractField + Debug,
    AB::Expr: Debug,
    AB::Var: Debug,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &Shift32Cols<AB::Var> = main.row_slice(0).borrow();

        // Check that opcode flags are well-formed
        builder.assert_bool(local.is_shl);
        builder.assert_bool(local.is_shr);
        builder.assert_bool(local.is_sra);
        builder.assert_bool(local.is_shl + local.is_shr + local.is_sra);

        builder.when(local.is_sra).assert_eq(
            local.sign_extension_byte,
            local.sign_1 * AB::Expr::from_canonical_u8(u8::MAX),
        );
        builder
            .when(local.is_shl + local.is_shr)
            .assert_zero(local.sign_extension_byte);

        // First, we perform a byte-wise shift to shift input_1 by `shift_by` bits rounded down to a multiple of 8

        // Check that the flags encoding the number of full bytes to shift by are well-formed
        builder.assert_bool(local.shift_by_zero_full_bytes);
        builder.assert_bool(local.shift_by_one_full_byte);
        builder.assert_bool(local.shift_by_two_full_bytes);
        builder.assert_bool(local.shift_by_three_full_bytes);
        builder.assert_bool(
            local.shift_by_zero_full_bytes
                + local.shift_by_one_full_byte
                + local.shift_by_two_full_bytes
                + local.shift_by_three_full_bytes,
        );

        let extended_bytes_left = local
            .input_1
            .iter_be()
            .chain(iter::repeat(&local.sign_extension_byte).take(MEMORY_CELL_BYTES))
            .collect_vec();

        let extended_bytes_right = local
            .input_1
            .iter_le()
            .chain(iter::repeat(&local.sign_extension_byte).take(MEMORY_CELL_BYTES))
            .collect_vec();

        local
            .byte_shifted_word
            .iter_be()
            .enumerate()
            .for_each(|(i, byte)| {
                builder
                    .when(local.is_shl)
                    .when(local.shift_by_zero_full_bytes)
                    .assert_eq(*byte, *extended_bytes_left[i]);
                builder
                    .when(local.is_shl)
                    .when(local.shift_by_one_full_byte)
                    .assert_eq(*byte, *extended_bytes_left[i + 1]);
                builder
                    .when(local.is_shl)
                    .when(local.shift_by_two_full_bytes)
                    .assert_eq(*byte, *extended_bytes_left[i + 2]);
                builder
                    .when(local.is_shl)
                    .when(local.shift_by_three_full_bytes)
                    .assert_eq(*byte, *extended_bytes_left[i + 3]);
            });
        local
            .byte_shifted_word
            .iter_le()
            .enumerate()
            .for_each(|(i, byte)| {
                builder
                    .when(local.is_shr + local.is_sra)
                    .when(local.shift_by_zero_full_bytes)
                    .assert_eq(*byte, *extended_bytes_right[i]);
                builder
                    .when(local.is_shr + local.is_sra)
                    .when(local.shift_by_one_full_byte)
                    .assert_eq(*byte, *extended_bytes_right[i + 1]);
                builder
                    .when(local.is_shr + local.is_sra)
                    .when(local.shift_by_two_full_bytes)
                    .assert_eq(*byte, *extended_bytes_right[i + 2]);
                builder
                    .when(local.is_shr + local.is_sra)
                    .when(local.shift_by_three_full_bytes)
                    .assert_eq(*byte, *extended_bytes_right[i + 3]);
            });

        for ((output_byte, this_shifted), prev_overflow) in local
            .output
            .iter_be()
            .zip(local.bit_shifted_bytes_left.iter_be())
            .zip(
                local
                    .bit_shifted_bytes_right
                    .iter_be()
                    .skip(1)
                    .chain(iter::once(&local.sign_extension_byte)),
            )
        {
            builder
                .when(local.is_shl)
                .assert_eq(*output_byte, *this_shifted + *prev_overflow);
        }

        for ((output_byte, this_shifted), prev_overflow) in local
            .output
            .iter_le()
            .zip(local.bit_shifted_bytes_right.iter_le())
            .zip(
                local
                    .bit_shifted_bytes_left
                    .iter_le()
                    .skip(1)
                    .chain(iter::once(&local.sign_extension_byte_overflow)),
            )
        {
            builder
                .when(local.is_shr + local.is_sra)
                .assert_eq(*output_byte, *this_shifted + *prev_overflow)
        }
    }
}
