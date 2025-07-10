use crate::columns::{NUM_MEM_COLS, NUM_MEM_PUBLIC_VALUES};
use crate::{MemoryChip, MemoryCols};
use core::borrow::Borrow;
use core::fmt::Debug;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_field::PrimeField;
use p3_matrix::MatrixRowSlices;
use valida_machine::{reduce_word, Word};

impl<F> BaseAir<F> for MemoryChip {
    fn width(&self) -> usize {
        NUM_MEM_COLS
    }
    fn public_width(&self) -> usize {
        NUM_MEM_PUBLIC_VALUES
    }
}

impl<AB> Air<AB> for MemoryChip
where
    AB: AirBuilder,
    AB::F: PrimeField + Debug,
    AB::Expr: Debug,
    AB::Var: Debug,
{
    fn eval(&self, builder: &mut AB) {
        self.eval_main(builder);
    }
}

impl MemoryChip {
    fn eval_main<AB: AirBuilder>(&self, builder: &mut AB)
    where
        <AB as AirBuilder>::Var: Debug,
    {
        let main = builder.main();
        let local: &MemoryCols<AB::Var> = main.row_slice(0).borrow();
        let next: &MemoryCols<AB::Var> = main.row_slice(1).borrow();

        // `0` if the row is a padding row; 1 otherwise.
        let local_is_real =
            local.is_read + local.is_write + local.is_dummy_read + local.is_static_write;
        let next_is_real = next.is_read + next.is_write + next.is_dummy_read + next.is_static_write;

        builder.assert_bool(local.is_read);
        builder.assert_bool(local.is_write);
        builder.assert_bool(local.is_dummy_read);
        builder.assert_bool(local.is_static_write);
        builder.assert_bool(local_is_real.clone());
        builder.assert_bool(local.addr_equal);
        builder.assert_bool(local.is_zero_initialized);
        builder.assert_bool(local.is_final);
        builder.assert_bool(local.skip_persistent_send);

        // check the `is_initial` flag is set correctly: a row is the intitial operation
        // for its address if and only if it is either the first row, or the prior
        // row has `addr_equal` false.
        builder
            .when_first_row()
            .when(local_is_real.clone())
            .assert_one(local.is_initial);

        // If the next row is an initial memory access `is_initial == 1`, then it must refer to a different
        // address than this row
        builder
            .when_transition()
            //.when_ne(local.is_static_write, AB::Expr::one())
            .when(next.is_initial)
            .assert_zero(local.addr_equal);

        // If we look at a static write (i.e. memory cell initialized by static data) and the cell is accessed
        // afterwards, the next access will *NOT* be `is_initial == 1`
        builder
            .when_transition()
            .when(local.is_static_write)
            .when(local.addr_equal)
            .assert_zero(next.is_initial);

        // the first operation to each address should be a read or a dummy read
        builder.when(local.is_initial).assert_zero(local.is_write);
        // dummy reads should always be initial *unless* the previous was a static write (i.e. static data chip)
        builder
            .when_transition()
            .when_ne(local.is_static_write, AB::Expr::one())
            .when(next.is_dummy_read)
            .assert_eq(next.is_initial, AB::Expr::one());

        builder
            .when_transition()
            .when(AB::Expr::one() - local.addr_equal)
            .assert_eq(next.is_initial, next_is_real.clone());

        // dummy reads should always precede a write to the same address and with the same clock
        builder.when(local.is_dummy_read).assert_one(next.is_write);
        builder
            .when(local.is_dummy_read)
            .assert_eq(local.clk, next.clk);
        builder
            .when(local.is_dummy_read)
            .assert_eq(local.addr, next.addr);

        // Constrain the `diff` field to be set to the address difference when the addresses are unequal
        // and the clock difference when they are equal.
        builder
            .when(local.addr_equal)
            .assert_eq(local.diff, next.clk - local.clk);
        builder
            .when_transition()
            // `0` if the next row is a padding row; 1 - `addr_equal` otherwise.
            .when(next_is_real.clone() - local.addr_equal)
            .assert_eq(local.diff, next.addr - local.addr);

        // Constrain the `addr_equal` flag to be set correctly:
        builder
            .when(local.addr_equal)
            .assert_eq(local.addr, next.addr);
        // forces `diff`, and thus `next.addr - local.addr`, to be nonzero if `addr_equal` is `0`.
        builder
            .when_transition()
            .when(next_is_real.clone() - local.addr_equal)
            .assert_eq(AB::Expr::one(), local.diff * local.diff_inv);

        // Constrain the `value` field in the case of a read to be equal to the value stored at that address
        for (local_value_byte, next_value_byte) in local.value.iter_le().zip(next.value.iter_le()) {
            // If the read is not the first operation at an address, its value should
            // be the same as the value at that address in the previous row.
            builder
                .when(next.is_read)
                .when(local.addr_equal)
                .assert_eq(*next_value_byte, *local_value_byte);
        }

        // Edge case: multiple operations at the same address in the same cycle are allowed,
        // but there may only be one write per cycle and it must occur last.
        builder
            // diff == 0 exactly when the current row and the next row have the same address and same clock.
            .when(local_is_real.clone() - local.diff * local.diff_inv)
            .assert_eq(local.is_read + local.is_dummy_read, AB::Expr::one());

        // Invariant: Only the memory operation associated with the first dummy read or read of
        // a particular address cell in the entire execution is allowed to set is_zero_initialized to true.
        //
        // Note: Above we constrain that the initial operation for an address is a read or a dummy read.

        // Constrain: is_zero_initialized is true iff the prior_timestamp is zero
        builder
            .when(local.is_zero_initialized)
            .assert_eq(local.prior_timestamp, AB::Expr::zero());
        builder
            .when_ne(local.prior_timestamp, AB::Expr::zero())
            .assert_eq(local.is_zero_initialized, AB::Expr::zero());

        // Constrain: If zero-initialized is true, then this row must be initial.
        builder
            .when(local.is_zero_initialized)
            .assert_one(local.is_initial);

        // Constrain the byte decompositions.
        let base = Word::from_components_le([1, 1 << 8, 1 << 16, 1 << 24])
            .transform(AB::Expr::from_canonical_u32);

        let local_addr = reduce_word::<AB>(&base, local.addr_bytes);
        let local_diff = reduce_word::<AB>(&base, local.diff_bytes);
        builder.assert_eq(local_addr, local.addr);
        builder.assert_eq(local_diff, local.diff);

        // Constrain: `skip_persistent_send` is true, if:
        // `addr_equal == 1`
        // `is_final == 1`
        // `is_dummy_read == 1`
        // `is_static_write == 1`
        // -> Implies to *not* perform persistent send if any of these true
        builder
            .when(local.addr_equal)
            .assert_one(local.skip_persistent_send);
        builder
            .when(local.is_final)
            .assert_one(local.skip_persistent_send);
        builder
            .when(local.is_dummy_read)
            .assert_one(local.skip_persistent_send);
        builder
            .when(local.is_static_write)
            .assert_one(local.skip_persistent_send);

        builder
            .when_ne(local.skip_persistent_send, AB::Expr::one())
            .assert_zero(local.addr_equal);
        builder
            .when_ne(local.skip_persistent_send, AB::Expr::one())
            .assert_zero(local.is_final);
        builder
            .when_ne(local.skip_persistent_send, AB::Expr::one())
            .assert_zero(local.is_dummy_read);
        builder
            .when_ne(local.skip_persistent_send, AB::Expr::one())
            .assert_zero(local.is_static_write);

        // Constrain: `skip_persistent_receive` is true, if:
        // - `is_initial == 1`
        // - `prior_timestamp == 2`
        // NOTE: The final required condition, namely that the segment number is >0
        // cannot be expressed here, as we have no access to the segment number in the AIR

        // Skip persistent receive is only used for access to static data cells for the first
        // time in the program if segment numbe > 0. As a result `is_initial` is _also_ true.
        builder
            .when(local.skip_persistent_receive)
            .assert_one(local.is_initial);

        builder
            .when(local.skip_persistent_receive)
            .assert_eq(local.prior_timestamp, AB::Expr::one() + AB::Expr::one());
    }
}
