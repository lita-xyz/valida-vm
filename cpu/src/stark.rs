use crate::columns::{CpuCols, CpuPublicVector, NUM_CPU_COLS, NUM_CPU_PUBLIC_VALUES};
use crate::CpuChip;
use core::borrow::Borrow;
use core::fmt::Debug;
use valida_machine::{Word, MEMORY_CELL_BYTES};

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, PrimeField};
use p3_matrix::MatrixRowSlices;
use valida_opcodes::BYTES_PER_INSTR;

impl<F> BaseAir<F> for CpuChip {
    fn width(&self) -> usize {
        NUM_CPU_COLS
    }
    fn public_width(&self) -> usize {
        NUM_CPU_PUBLIC_VALUES
    }
}

impl<AB> Air<AB> for CpuChip
where
    AB: AirBuilderWithPublicValues,
    AB::F: PrimeField + Debug,
    AB::Expr: Debug,
    AB::Var: Debug,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &CpuCols<AB::Var> = main.row_slice(0).borrow();
        let next: &CpuCols<AB::Var> = main.row_slice(1).borrow();

        let public = builder.public_values();
        // since the CPUChip uses a `PublicVector` instead of a `PublicMatrix`,
        // there is only one row, and `row_slice(i)` always returns it.
        let public_vector: &CpuPublicVector<AB::Var> = public.row_slice(0).borrow();

        let base = Word::from_components_le([1, 1 << 8, 1 << 16, 1 << 24])
            .transform(AB::Expr::from_canonical_u32);
        self.eval_pc(builder, local, next, public_vector, &base);
        self.eval_fp(builder, local, next, public_vector, &base);
        self.eval_last_segmt(builder, local, public_vector);
        self.eval_equality(builder, local);
        self.eval_memory_channels(builder, local, next, &base);

        // Check `is_real` constraints
        // - If the next row has `is_real == 1`, then the current one *MUST* also be real
        //   i.e. `is_real` is used to extend the trace to a power of two (empty padding)
        builder.assert_bool(local.is_real);
        builder
            .when_ne(local.is_real, AB::Expr::one()) // -> local == 0
            .assert_eq(local.clk, AB::Expr::zero());
        builder
            .when_transition()
            .when(next.is_real)
            .assert_one(local.is_real);

        // Clock constraints
        builder.when_first_row().assert_zero(local.clk);
        let next_is_real = next.is_real;
        builder
            .when(next_is_real)
            .when_transition()
            .assert_eq(local.clk + AB::Expr::one(), next.clk);

        // Immediate value constraints (TODO: we'd need to range check read_value_2 in
        // this case)
        // this asserts that at most one of `is_imm_op` and `is_left_imm_op` is true.
        builder.assert_bool(local.opcode_flags.is_imm_op + local.opcode_flags.is_left_imm_op);
        builder.when(local.opcode_flags.is_imm_op).assert_eq(
            local.instruction.operands.c(),
            reduce::<AB>(&base, local.read_value_2()),
        );
        builder.when(local.opcode_flags.is_left_imm_op).assert_eq(
            local.instruction.operands.b(),
            reduce::<AB>(&base, local.read_value_1()),
        );

        // "Stop" constraints (to check that program execution was not stopped prematurely)
        builder.assert_bool(local.opcode_flags.is_stop);
        builder
            .when_transition()
            .when(local.opcode_flags.is_stop)
            .assert_eq(next.pc, AB::Expr::zero()); // zero, because padded with zeros
        builder
            .when_ne(local.is_last_segment, AB::Expr::zero())
            .when_last_row()
            .assert_one(local.opcode_flags.is_stop);
    }
}

impl CpuChip {
    fn eval_memory_channels<AB>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        _next: &CpuCols<AB::Var>, // TODO: unused
        base: &Word<AB::Expr>,
    ) where
        AB: AirBuilder,
        AB::F: Debug,
        AB::Expr: Debug,
        AB::Var: Debug,
    {
        let bytes_per_instr_expr = AB::Expr::from_canonical_u32(BYTES_PER_INSTR);
        let is_load = local.opcode_flags.is_load;
        let is_load_u8 = local.opcode_flags.is_load_u8;
        let is_load_s8 = local.opcode_flags.is_load_s8;
        let is_store_u8 = local.opcode_flags.is_store_u8;
        let is_store = local.opcode_flags.is_store;
        let is_jal = local.opcode_flags.is_jal;
        let is_jalv = local.opcode_flags.is_jalv;
        let is_beq = local.opcode_flags.is_beq;
        let is_bne = local.opcode_flags.is_bne;
        let is_imm32 = local.opcode_flags.is_imm32;
        let is_loadfp = local.opcode_flags.is_loadfp;
        let _is_advice = local.opcode_flags.is_advice; // TODO: unused
        let is_imm_op = local.opcode_flags.is_imm_op;
        let is_left_imm_op = local.opcode_flags.is_left_imm_op;
        let is_bus_op = local.opcode_flags.is_bus_op;
        let is_pointer_op = local.opcode_flags.is_pointer_op;
        let is_write = local.opcode_flags.is_write;

        builder.assert_bool(is_load);
        builder.assert_bool(is_load_u8);
        builder.assert_bool(is_load_s8);
        builder.assert_bool(is_store_u8);
        builder.assert_bool(is_store);
        builder.assert_bool(is_jal);
        builder.assert_bool(is_jalv);
        builder.assert_bool(is_beq);
        builder.assert_bool(is_bne);
        builder.assert_bool(is_imm32);
        builder.assert_bool(is_loadfp);
        builder.assert_bool(is_imm_op);
        builder.assert_bool(is_left_imm_op);
        builder.assert_bool(is_bus_op);
        builder.assert_bool(is_pointer_op);
        builder.assert_bool(is_write);

        let addr_a = local.fp + local.instruction.operands.a();
        let addr_b = local.fp + local.instruction.operands.b();
        let addr_c = local.fp + local.instruction.operands.c();

        let addr_offset: AB::Expr = local
            .addr_offset_flags
            .iter_le()
            .enumerate()
            .map(|(i, flag)| *flag * AB::Expr::from_canonical_usize(i))
            .sum();

        let single_byte_to_write: AB::Expr = local
            .read_value_2()
            .iter_le()
            .zip(local.addr_offset_flags.iter_le())
            .map(|(byte, flag)| *byte * *flag)
            .sum();

        // Read (1)
        // note that here we are using the fact that at most one of 'is_imm_op' and 'is_left_imm_op' is ever true.
        builder
            .when(
                is_jalv
                    + is_beq
                    + is_bne
                    + (is_bus_op + is_pointer_op) * (AB::Expr::one() - is_left_imm_op)
                    + is_write,
            )
            .assert_eq(local.read_addr_1(), addr_b.clone());
        builder
            .when(is_load + is_store + is_load_u8 + is_load_s8 + is_store_u8)
            .assert_eq(local.read_addr_1(), addr_c.clone());
        builder
            .when(
                is_load
                    + is_store
                    + is_load_s8
                    + is_load_u8
                    + is_store_u8
                    + is_jalv
                    + is_beq
                    + is_bne
                    + (AB::Expr::one() - is_left_imm_op) * is_bus_op
                    + is_write,
            )
            .assert_one(local.read_1_used());
        builder
            .when(is_jal + is_left_imm_op + is_loadfp + is_imm32)
            .assert_zero(local.read_1_used());

        // Read (2)
        // note that here we are again using the fact that at most one of 'is_imm_op' and 'is_left_imm_op' is ever true.
        builder.when(is_load).assert_eq(
            local.read_addr_2(),
            reduce::<AB>(base, local.read_value_1()),
        );
        builder.when(is_load_s8 + is_load_u8).assert_eq(
            local.read_addr_2() + addr_offset.clone(),
            reduce::<AB>(base, local.read_value_1()),
        );
        builder
            .when(is_store + is_store_u8)
            .assert_eq(local.read_addr_2(), addr_b.clone());
        builder
            .when(is_jalv + (AB::Expr::one() - is_imm_op) * is_bus_op)
            .assert_eq(local.read_addr_2(), addr_c);
        builder
            .when(
                is_load
                    + is_load_u8
                    + is_load_s8
                    + is_store_u8
                    + is_store
                    + is_jalv
                    + (AB::Expr::one() - is_imm_op) * (is_beq + is_bne + is_bus_op),
            )
            .assert_one(local.read_2_used());
        builder
            .when(
                is_jal
                    + is_imm_op * (is_beq + is_bne + is_bus_op)
                    + is_loadfp
                    + is_imm32
                    + is_write,
            )
            .assert_zero(local.read_2_used());

        // Write
        builder
            .when(
                is_load
                    + is_load_u8
                    + is_load_s8
                    + is_jal
                    + is_jalv
                    + is_imm32
                    + is_bus_op
                    + is_loadfp,
            )
            .assert_eq(local.write_addr(), addr_a);
        builder
            .when(is_store)
            .assert_eq(local.write_addr(), reduce::<AB>(base, local.read_value_2()));
        builder.when(is_store_u8).assert_eq(
            local.write_addr() + addr_offset.clone(),
            reduce::<AB>(base, local.read_value_2()),
        );
        builder.when(is_store).assert_zero(
            local
                .read_value_1()
                .into_iter_le()
                .zip(local.write_value().into_iter_le())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<AB::Expr>(),
        );
        for ((write_value_byte, old_value_byte), addr_offset_flag) in local
            .write_value()
            .into_iter_le()
            .zip(local.old_value_for_single_byte_write().into_iter_le())
            .zip(local.addr_offset_flags.into_iter_le())
        {
            builder
                .when(is_store_u8)
                .when(AB::Expr::one() - addr_offset_flag)
                .assert_eq(write_value_byte, old_value_byte);
            builder
                .when(is_store_u8)
                .when(addr_offset_flag)
                .assert_eq(*local.read_value_1().index_le(0), write_value_byte);
        }
        builder.when(is_load).assert_zero(
            local
                .read_value_2()
                .into_iter_le()
                .zip(local.write_value().into_iter_le())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<AB::Expr>(),
        );
        builder
            .when(is_load_u8 + is_load_s8)
            .assert_eq(*local.write_value().index_le(0), single_byte_to_write);
        for write_byte in local.write_value().iter_le().skip(1) {
            builder.when(is_load_u8).assert_zero(*write_byte);
            let sign_extension_byte = local.sign_bit * AB::Expr::from_canonical_u8(0xff);
            builder
                .when(is_load_s8)
                .assert_eq(*write_byte, sign_extension_byte);
        }
        builder.when_transition().when(is_jal + is_jalv).assert_eq(
            bytes_per_instr_expr.clone() * (local.pc + AB::F::one()),
            reduce::<AB>(base, local.write_value()),
        );
        builder.when(is_imm32).assert_zero(
            local
                .write_value()
                .into_iter_le()
                .zip(local.instruction.operands.imm32().into_iter_le())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<AB::Expr>(),
        );
        builder
            .when(is_loadfp)
            .assert_eq(addr_b, reduce::<AB>(base, local.write_value()));
        builder
            .when(is_store + is_load + is_jal + is_jalv + is_imm32 + is_loadfp + is_bus_op)
            .assert_one(local.write_used());
        builder
            .when(is_beq + is_bne + is_write)
            .assert_zero(local.write_used());
    }

    fn eval_pc<AB>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        next: &CpuCols<AB::Var>,
        public: &CpuPublicVector<AB::Var>,
        base: &Word<AB::Expr>,
    ) where
        AB: AirBuilder,
        AB::F: Debug,
        AB::Expr: Debug,
        AB::Var: Debug,
    {
        builder.when_first_row().assert_eq(local.pc, public.pc_init);

        let bytes_per_instr_expr = AB::Expr::from_canonical_u32(BYTES_PER_INSTR);
        let should_not_increment_pc = local.opcode_flags.is_jal
            + local.opcode_flags.is_jalv
            + local.opcode_flags.is_bne
            + local.opcode_flags.is_beq
            + local.opcode_flags.is_stop;
        let should_increment_pc = AB::Expr::one() - should_not_increment_pc;
        let incremented_pc = local.pc + AB::F::one();

        let is_real = local.is_real;
        let next_is_real = next.is_real;
        builder
            .when_transition()
            .when(next_is_real)
            .when(should_increment_pc)
            .assert_eq(next.pc, incremented_pc.clone());

        // Branch manipulation
        let equal = AB::Expr::one() - local.not_equal;
        let next_pc_times_24_if_branching = local.instruction.operands.a();
        let beq_next_pc_times_24 = equal.clone() * next_pc_times_24_if_branching
            + bytes_per_instr_expr.clone() * local.not_equal * incremented_pc.clone();
        let bne_next_pc_times_24 = bytes_per_instr_expr.clone() * equal * incremented_pc
            + local.not_equal * next_pc_times_24_if_branching;
        builder
            .when_transition()
            .when(local.opcode_flags.is_beq)
            .assert_eq(bytes_per_instr_expr.clone() * next.pc, beq_next_pc_times_24);
        builder
            .when_transition()
            .when(local.opcode_flags.is_bne)
            .assert_eq(bytes_per_instr_expr.clone() * next.pc, bne_next_pc_times_24);

        // Jump manipulation
        builder
            .when_transition()
            .when(local.opcode_flags.is_jal)
            .assert_eq(
                bytes_per_instr_expr.clone() * next.pc,
                local.instruction.operands.b(),
            );
        builder
            .when_transition()
            .when(local.opcode_flags.is_jalv)
            .assert_eq(
                bytes_per_instr_expr.clone() * next.pc,
                reduce::<AB>(base, local.read_value_1()),
            );
    }

    fn eval_fp<AB>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        next: &CpuCols<AB::Var>,
        public: &CpuPublicVector<AB::Var>,
        base: &Word<AB::Expr>,
    ) where
        AB: AirBuilder,
        AB::F: Debug,
        AB::Expr: Debug,
        AB::Var: Debug,
    {
        let next_is_real = next.is_real;
        builder.when_first_row().assert_eq(local.fp, public.fp_init);
        builder
            .when_transition()
            .when(local.opcode_flags.is_jal)
            .assert_eq(next.fp, local.fp + local.instruction.operands.c());
        // i.e. 2^32 (mod p)
        let pow_2 = AB::Expr::from_wrapped_u64(1 << (8 * MEMORY_CELL_BYTES));
        builder
            .when_transition()
            .when(local.opcode_flags.is_jalv)
            .assert_eq(
                next.fp,
                // we interpret `local.read_value_2()` as an i32, subtracting 2^32 if the sign bit is set
                local.fp + reduce::<AB>(base, local.read_value_2()) - local.sign_bit * pow_2,
            );
        builder
            .when(next_is_real)
            .when_transition()
            .when(AB::Expr::one() - local.opcode_flags.is_jal - local.opcode_flags.is_jalv)
            .assert_eq(next.fp, local.fp);
    }

    fn eval_equality<AB>(&self, builder: &mut AB, local: &CpuCols<AB::Var>)
    where
        AB: AirBuilder,
        AB::F: Debug,
        AB::Expr: Debug,
        AB::Var: Debug,
    {
        // Check if the first two operand values are equal, in case we're doing a conditional branch.
        // (when is_imm == 1, the second read value is guaranteed to be an immediate value)
        builder.assert_eq(
            local.diff,
            local
                .read_value_1()
                .into_iter_le()
                .zip(local.read_value_2().into_iter_le())
                .map(|(a, b)| (a - b).square())
                .sum::<AB::Expr>(),
        );
        builder.assert_bool(local.not_equal);
        builder.assert_eq(local.not_equal, local.diff * local.diff_inv);
        let equal = AB::Expr::one() - local.not_equal;
        builder.assert_zero(equal * local.diff);
    }

    fn eval_last_segmt<AB>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        public: &CpuPublicVector<AB::Var>,
    ) where
        AB: AirBuilder,
        AB::F: Debug,
        AB::Expr: Debug,
        AB::Var: Debug,
    {
        builder
            .when_first_row()
            .assert_eq(local.is_last_segment, public.is_last_segment);
        builder.assert_bool(local.is_last_segment);
    }
}

pub fn reduce<AB: AirBuilder>(base: &Word<AB::Expr>, input: Word<AB::Var>) -> AB::Expr {
    input
        .into_iter_le()
        .zip(base.iter_le())
        .map(|(i, b)| b.clone() * i)
        .sum()
}
