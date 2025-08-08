extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use columns::{Div32Cols, DIV_COL_MAP, NUM_DIV_COLS};
use core::{borrow::Borrow, mem::transmute};
use p3_field::PrimeField32;
use valida_bus::{MachineWithBytesBus, MachineWithGeneralBus, MachineWithRangeBus8};
use valida_cpu::MachineWithCpuChip;
use valida_machine::SDiv;
use valida_machine::StarkConfig;
use valida_machine::{
    instructions, Chip, ChipTraceHeight, Instruction, Interaction, Operands, Word,
};
use valida_machine::{ChipWithPersistence, PublicTrace, RunningMachine};
use valida_opcodes::LT32;
use valida_opcodes::MUL32;
use valida_opcodes::MULHS32;
use valida_opcodes::MULHU32;
use valida_opcodes::SLT32;
use valida_opcodes::{convert_opcode, ADD32};
use valida_opcodes::{DIV32, SDIV32};
use valida_util::pad_to_power_of_two;

use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::add;
use crate::add::MachineWithAdd32Chip;
use crate::lt;
use crate::lt::MachineWithLt32Chip;
use crate::mul;
use crate::mul::{get_carries, Long, MachineWithMul32Chip};
use valida_bytes::{byte_send_simple, range8_sends_word, ByteOperation};
use valida_bytes::{MachineWithBytesChip, MachineWithRangeCheckeru8};

use valida_memory_footprint::MemoryFootprint;

pub mod columns;
pub mod stark;

#[derive(Clone)]
pub enum Operation {
    Div32(Word<u8>, Word<u8>, Word<u8>), // (quotient, dividend, divisor)
    SDiv32(Word<u8>, Word<u8>, Word<u8>), //signed
}

impl MemoryFootprint for Operation {
    fn memory_footprint(&self) -> usize {
        match self {
            Operation::Div32(a, _b, _c) => 3 * a.memory_footprint(),
            Operation::SDiv32(a, _b, _c) => 3 * a.memory_footprint(),
        }
    }
}

#[derive(Default)]
pub struct Div32Chip {
    pub operations: Vec<Operation>,
}

impl MemoryFootprint for Div32Chip {
    fn memory_footprint(&self) -> usize {
        self.operations.memory_footprint()
    }
}

impl ChipTraceHeight for Div32Chip {
    fn trace_height(&self) -> u32 {
        self.operations.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for Div32Chip
where
    M: MachineWithGeneralBus<SC::Val>
        + MachineWithRangeBus8<SC::Val>
        + MachineWithBytesBus<SC::Val>,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "Div32".to_string()
    }

    fn generate_main_trace(
        &self,
        _machine: &M,
        verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>) {
        let rows = self
            .operations
            .par_iter()
            .map(|op| self.op_to_row(op))
            .collect::<Vec<_>>();

        let log = if verbose {
            let mut log_prints = Vec::with_capacity(rows.len());
            for (index, row) in rows.iter().enumerate() {
                let cols: &Div32Cols<SC::Val> = row[..].borrow();
                log_prints.push(format!("Div32 row {index}: {:?}", cols));
            }
            Some(log_prints)
        } else {
            None
        };

        let mut trace =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_DIV_COLS);

        pad_to_power_of_two::<NUM_DIV_COLS, SC::Val>(&mut trace.values);

        (Some(trace), log)
    }

    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let opcode = VirtualPairCol::new_main(
            vec![
                (
                    DIV_COL_MAP.is_div,
                    SC::Val::from_canonical_u32(convert_opcode(DIV32)),
                ),
                (
                    DIV_COL_MAP.is_sdiv,
                    SC::Val::from_canonical_u32(convert_opcode(SDIV32)),
                ),
            ],
            SC::Val::zero(),
        );
        let input_1 = DIV_COL_MAP.input_1.transform(VirtualPairCol::single_main);
        let input_2 = DIV_COL_MAP.input_2.transform(VirtualPairCol::single_main);
        let output = DIV_COL_MAP.output.transform(VirtualPairCol::single_main);

        let mut fields = vec![opcode];
        fields.extend(input_1.into_iter_le());
        fields.extend(input_2.into_iter_le());
        fields.extend(output.into_iter_le());

        let is_real = VirtualPairCol::sum_main(vec![DIV_COL_MAP.is_div, DIV_COL_MAP.is_sdiv]);

        let receive = Interaction {
            fields,
            count: is_real,
            argument_index: machine.general_bus(),
        };
        vec![receive]
    }

    fn global_sends(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let sign_bit_sends = {
            let input_1_top_byte_col =
                VirtualPairCol::single_main(*DIV_COL_MAP.input_1.index_be(0));
            let sign_1_col = VirtualPairCol::single_main(DIV_COL_MAP.sign_1);
            let is_sdiv_col = VirtualPairCol::single_main(DIV_COL_MAP.is_sdiv);
            let sign_1_send = byte_send_simple(
                machine,
                input_1_top_byte_col,
                Some(sign_1_col),
                is_sdiv_col.clone(),
                ByteOperation::MostSignificantBit,
            );
            let input_2_top_byte_col =
                VirtualPairCol::single_main(*DIV_COL_MAP.input_2.index_be(0));
            let sign_2_col = VirtualPairCol::single_main(DIV_COL_MAP.sign_2);
            let sign_2_send = byte_send_simple(
                machine,
                input_2_top_byte_col,
                Some(sign_2_col),
                is_sdiv_col.clone(),
                ByteOperation::MostSignificantBit,
            );
            let remainder_top_byte_col =
                VirtualPairCol::single_main(*DIV_COL_MAP.remainder.index_be(0));
            let remainder_sign_col = VirtualPairCol::single_main(DIV_COL_MAP.sign_remainder);
            let remainder_sign_send = byte_send_simple(
                machine,
                remainder_top_byte_col,
                Some(remainder_sign_col),
                is_sdiv_col,
                ByteOperation::MostSignificantBit,
            );
            vec![sign_1_send, sign_2_send, remainder_sign_send]
        };

        let is_real = VirtualPairCol::sum_main(vec![DIV_COL_MAP.is_div, DIV_COL_MAP.is_sdiv]);
        let output = DIV_COL_MAP.output.transform(VirtualPairCol::single_main);

        let lt_opcode = VirtualPairCol::new_main(
            vec![
                (
                    DIV_COL_MAP.is_div,
                    SC::Val::from_canonical_u32(convert_opcode(LT32)),
                ),
                (
                    DIV_COL_MAP.is_sdiv,
                    SC::Val::from_canonical_u32(convert_opcode(SLT32)),
                ),
            ],
            SC::Val::zero(),
        );

        // This is one unless the opcode is SDIV32 and input_2 is negative
        let div_or_input_2_positive = VirtualPairCol::new_main(
            vec![
                (DIV_COL_MAP.is_div, SC::Val::one()),
                (DIV_COL_MAP.is_sdiv, SC::Val::one()),
                (DIV_COL_MAP.sign_2, -SC::Val::one()),
            ],
            SC::Val::zero(),
        );

        // Case 1: `input_2` is positive or the opcode is `DIV32`
        // We have `0 <= remainder < input_2`.
        let lt_upper_positive_send = {
            let input_1 = DIV_COL_MAP.remainder.transform(VirtualPairCol::single_main);
            let input_2 = DIV_COL_MAP.input_2.transform(VirtualPairCol::single_main);
            let output = (Word::from(SC::Val::one())).transform(VirtualPairCol::constant);
            let fields = vec![lt_opcode.clone()]
                .into_iter()
                .chain(input_1.into_iter_le())
                .chain(input_2.into_iter_le())
                .chain(output.into_iter_le())
                .collect::<Vec<_>>();

            Interaction {
                fields,
                count: div_or_input_2_positive.clone(),
                argument_index: machine.general_bus(),
            }
        };
        let lt_lower_negative_send = {
            let input_1 = DIV_COL_MAP.input_2.transform(VirtualPairCol::single_main);
            let input_2 = DIV_COL_MAP.remainder.transform(VirtualPairCol::single_main);
            let output = (Word::from(SC::Val::one())).transform(VirtualPairCol::constant);
            let fields = vec![lt_opcode]
                .into_iter()
                .chain(input_1.into_iter_le())
                .chain(input_2.into_iter_le())
                .chain(output.into_iter_le())
                .collect::<Vec<_>>();

            Interaction {
                fields,
                count: VirtualPairCol::single_main(DIV_COL_MAP.sign_2),
                argument_index: machine.general_bus(),
            }
        };

        let mul_input_1 = DIV_COL_MAP.output.transform(VirtualPairCol::single_main);
        let mul_input_2 = DIV_COL_MAP.input_2.transform(VirtualPairCol::single_main);
        let mul_lower_output = DIV_COL_MAP
            .product_lower
            .transform(VirtualPairCol::single_main);
        let mul_lower_opcode =
            VirtualPairCol::constant(SC::Val::from_canonical_u32(convert_opcode(MUL32)));
        let mul_lower_fields = vec![mul_lower_opcode]
            .into_iter()
            .chain(mul_input_1.clone().into_iter_le())
            .chain(mul_input_2.clone().into_iter_le())
            .chain(mul_lower_output.into_iter_le())
            .collect::<Vec<_>>();

        let mul_lower_send = Interaction {
            fields: mul_lower_fields,
            count: is_real.clone(),
            argument_index: machine.general_bus(),
        };

        let mul_output_upper = DIV_COL_MAP
            .product_upper
            .transform(VirtualPairCol::single_main);
        let mul_upper_opcode = VirtualPairCol::new_main(
            vec![
                (
                    DIV_COL_MAP.is_div,
                    SC::Val::from_canonical_u32(convert_opcode(MULHU32)),
                ),
                (
                    DIV_COL_MAP.is_sdiv,
                    SC::Val::from_canonical_u32(convert_opcode(MULHS32)),
                ),
            ],
            SC::Val::zero(),
        );
        let mul_upper_fields = vec![mul_upper_opcode]
            .into_iter()
            .chain(mul_input_1.into_iter_le())
            .chain(mul_input_2.into_iter_le())
            .chain(mul_output_upper.into_iter_le())
            .collect::<Vec<_>>();
        let mul_upper_send = Interaction {
            fields: mul_upper_fields,
            count: is_real.clone(),
            argument_index: machine.general_bus(),
        };

        // We have `input_1 = input_2 * output + R`, where `R` has the same sign as `input_1`,
        // and we defined `remainder = |R| * sgn(input_2)`. Unpacking the cases,
        // `remainder = R = input_1 - input_2 * output` if `input_1` and `input_2` have the same sign
        // or the opcode is DIV32, and
        // `remainder = -R = input_2 * output - input_1` if `input_1` and `input_2` have different signs.
        let add_same_sign_send = {
            let add_input_1 = DIV_COL_MAP
                .product_lower
                .transform(VirtualPairCol::single_main);
            let add_input_2 = DIV_COL_MAP.remainder.transform(VirtualPairCol::single_main);
            let add_output = DIV_COL_MAP.input_1.transform(VirtualPairCol::single_main);
            let add_opcode =
                VirtualPairCol::constant(SC::Val::from_canonical_u32(convert_opcode(ADD32)));
            Interaction {
                fields: vec![add_opcode]
                    .into_iter()
                    .chain(add_input_1.into_iter_le())
                    .chain(add_input_2.into_iter_le())
                    .chain(add_output.into_iter_le())
                    .collect::<Vec<_>>(),
                count: VirtualPairCol::single_main(DIV_COL_MAP.same_sign),
                argument_index: machine.general_bus(),
            }
        };
        let add_different_sign_send = {
            let add_input_1 = DIV_COL_MAP.input_1.transform(VirtualPairCol::single_main);
            let add_input_2 = DIV_COL_MAP.remainder.transform(VirtualPairCol::single_main);
            let add_output = DIV_COL_MAP
                .product_lower
                .transform(VirtualPairCol::single_main);
            let add_opcode =
                VirtualPairCol::constant(SC::Val::from_canonical_u32(convert_opcode(ADD32)));
            let different_sign = VirtualPairCol::new_main(
                vec![
                    (DIV_COL_MAP.is_sdiv, SC::Val::one()),
                    (DIV_COL_MAP.is_div, SC::Val::one()),
                    (DIV_COL_MAP.same_sign, -SC::Val::one()),
                ],
                SC::Val::zero(),
            );
            Interaction {
                fields: vec![add_opcode]
                    .into_iter()
                    .chain(add_input_1.into_iter_le())
                    .chain(add_input_2.into_iter_le())
                    .chain(add_output.into_iter_le())
                    .collect::<Vec<_>>(),
                count: different_sign,
                argument_index: machine.general_bus(),
            }
        };

        sign_bit_sends
            .into_iter()
            // range check on the output of the division
            .chain(range8_sends_word(machine, output, &is_real))
            .chain(vec![
                lt_upper_positive_send,
                lt_lower_negative_send,
                mul_lower_send,
                mul_upper_send,
                add_same_sign_send,
                add_different_sign_send,
            ])
            .collect::<Vec<_>>()
    }
}

impl<M, SC> ChipWithPersistence<M, SC> for Div32Chip
where
    M: MachineWithGeneralBus<SC::Val>
        + MachineWithRangeBus8<SC::Val>
        + MachineWithBytesBus<SC::Val>,
    SC: StarkConfig,
{
}
impl Div32Chip {
    fn op_to_row<F>(&self, op: &Operation) -> [F; NUM_DIV_COLS]
    where
        F: PrimeField,
    {
        let mut row = [F::zero(); NUM_DIV_COLS];
        let cols: &mut Div32Cols<F> = unsafe { transmute(&mut row) };

        match op {
            Operation::Div32(a, b, c) => {
                cols.is_div = F::one();
                cols.same_sign = F::one();
                self.set_cols(a, b, c, cols, false);
            }
            Operation::SDiv32(a, b, c) => {
                cols.is_sdiv = F::one();
                self.set_cols(a, b, c, cols, true);
            }
        }

        row
    }

    fn set_cols<F>(
        &self,
        a: &Word<u8>,
        b: &Word<u8>,
        c: &Word<u8>,
        cols: &mut Div32Cols<F>,
        signed: bool,
    ) where
        F: PrimeField,
    {
        cols.input_1 = b.transform(F::from_canonical_u8);
        cols.input_2 = c.transform(F::from_canonical_u8);
        cols.output = a.transform(F::from_canonical_u8);
        // The definition of `*` on `Word<u8>`'s is the product reduced mod-2^32
        cols.product_lower = (*a * *c).transform(F::from_canonical_u8);

        if signed {
            let b_i32: i32 = (*b).into();
            let c_i32: i32 = (*c).into();
            // b % c has the same sign as b for i32's.
            let b_mod_c = b_i32 % c_i32;
            let (sign_b, sign_c) = ((b_i32 < 0) as u32, (c_i32 < 0) as u32);
            // our remainder should have the same sign as c
            let remainder = if sign_b == sign_c { b_mod_c } else { -b_mod_c };
            cols.remainder = Word::from((remainder) as u32).transform(F::from_canonical_u8);
            let sign_remainder = (remainder < 0) as u32;
            // The product does not overflow, as the absolute value of the product
            // is less than that of `input_1`, which has 32 bits. `
            let sign_bits = 0xffffffffu32;

            // if the remainder is zero, the product is zero, so the upper bits are zero as well;
            // otherwise, the product should have the same sign as b and not overflow.
            if *a != 0.into() {
                cols.product_upper = Word::from(sign_bits * sign_b).transform(F::from_canonical_u8)
            };

            cols.sign_1 = F::from_canonical_u32(sign_b);
            cols.sign_2 = F::from_canonical_u32(sign_c);
            cols.sign_remainder = F::from_canonical_u32(sign_remainder);
            cols.same_sign = F::from_bool(cols.sign_1 == cols.sign_2);
        } else {
            cols.remainder = (*b - (*a * *c)).transform(F::from_canonical_u8);
            cols.product_upper = Word::from(F::zero());
            cols.same_sign = F::one();
        };
    }
}

pub trait MachineWithDiv32Chip<F: PrimeField32>:
    MachineWithCpuChip<F>
    + MachineWithLt32Chip<F>
    + MachineWithMul32Chip<F>
    + MachineWithAdd32Chip<F>
    + MachineWithRangeCheckeru8<F>
{
    fn div_u32(&self) -> &Div32Chip;
    fn div_u32_mut(&mut self) -> &mut Div32Chip;
}

instructions!(Div32Instruction, SDiv32Instruction);

impl<M, F> Instruction<M, F> for Div32Instruction
where
    F: PrimeField,
    M: MachineWithDiv32Chip<F>
        + MachineWithRangeCheckeru8<F>
        + MachineWithLt32Chip<F>
        + MachineWithAdd32Chip<F>
        + MachineWithMul32Chip<F>,
    F: PrimeField32,
{
    const OPCODE: u32 = DIV32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;

        let mut imm: Option<Word<u8>> = None;
        let read_addr_1 = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        let b = M::read(state, clk, read_addr_1);
        let c = if ops.is_imm() == 1 {
            let c = (ops.c() as u32).into();
            imm = Some(c);
            c
        } else {
            let read_addr_2 = (state.machine.cpu().fp as i32 + ops.c()) as u32;
            M::read(state, clk, read_addr_2)
        };

        let a = b / c;
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .div_u32_mut()
                .operations
                .push(Operation::Div32(a, b, c));
            // We have to record the sends to the other ALU chips
            let au32: u32 = a.into();
            let bu32: u32 = b.into();
            let cu32: u32 = c.into();
            let rem = bu32 % cu32;
            let prod_lower = cu32 * au32;
            let prod_upper = (((cu32 as u64) * (au32 as u64)) >> 32) as u32;
            debug_assert_eq!(prod_upper, 0);

            // Comparison: rem < c (unsigned)
            state
                .machine
                .lt_u32_mut()
                .operations
                .push(lt::Operation::Lt32(true, rem.into(), c));

            // Lower 32 bits of a * c
            state
                .machine
                .mul_32_mut()
                .operations
                .push(mul::Operation::Mul32(
                    prod_lower.into(),
                    au32.into(),
                    cu32.into(),
                ));
            // Upper two bits of a * c (unsigned)
            state
                .machine
                .mul_32_mut()
                .operations
                .push(mul::Operation::Mulhu32(
                    prod_upper.into(),
                    au32.into(),
                    cu32.into(),
                ));

            // Sum: a * c + rem == b
            state
                .machine
                .add_u32_mut()
                .operations
                .push(add::Operation::Add32(b, prod_lower.into(), rem.into()));

            state.machine.push_bus_op(imm, opcode, ops);

            // Range checks from the Div32 chip: output of division
            state.machine.range_check_word(a);

            // Range checks from the Mul32 chip
            // from the Mul32 operation
            state.machine.range_check_word(prod_lower);
            state.machine.range_check_word(prod_upper);
            // from the Mulhu32 operation
            state.machine.range_check_word(prod_lower);
            state.machine.range_check_word(prod_upper);

            // internal range checks in the Mul32 chip
            let carries = get_carries(&Long::from(au32 as u64), &Long::from(cu32 as u64));
            // from the MUL32 operation
            state.machine.range_check_mul_chip_carries(&carries);
            // from the MULHU32 operation
            state.machine.range_check_mul_chip_carries(&carries);

            // Range checks from the Add32 chip
            state.machine.range_check_word(b);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for SDiv32Instruction
where
    M: MachineWithDiv32Chip<F> + MachineWithRangeCheckeru8<F> + MachineWithBytesChip<F>,
    F: PrimeField32,
{
    const OPCODE: u32 = SDIV32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;

        let mut imm: Option<Word<u8>> = None;
        let read_addr_1 = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        let b = M::read(state, clk, read_addr_1);
        let c = if ops.is_imm() == 1 {
            let c = (ops.c() as u32).into();
            imm = Some(c);
            c
        } else {
            let read_addr_2 = (state.machine.cpu().fp as i32 + ops.c()) as u32;
            M::read(state, clk, read_addr_2)
        };

        let a = b.sdiv(c);
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .div_u32_mut()
                .operations
                .push(Operation::SDiv32(a, b, c));
            // We have to record the sends to the other ALU chips
            let ai32: i32 = a.into();
            let bi32: i32 = b.into();
            let sign_b = (bi32 < 0) as u8;
            let ci32: i32 = c.into();
            let sign_c = (ci32 < 0) as u8;
            let same_sign = sign_b == sign_c;
            let rem = if same_sign {
                bi32 % ci32
            } else {
                -(bi32 % ci32)
            };
            let sign_remainder = (rem < 0) as u8;
            let prod = ci32 * ai32;
            // upper word for product of zero-extended values (only used inside mul32 implementation)
            let prod_upper_mul = (((ci32 as u32 as u64) * (ai32 as u32 as u64)) >> 32) as u32;
            let prod_upper_mulhs = (((ci32 as i64) * (ai32 as i64)) >> 32) as u32;
            let sign_extension_upper = u32::MAX;
            // a * c has the same sign as b, unless a * c == 0.
            debug_assert!(
                prod_upper_mulhs == sign_extension_upper * (sign_b as u32)
                    || (prod == 0 && prod_upper_mulhs == 0)
            );

            // sign bit checks
            let sign_1_res = state
                .machine
                .check_byte_op(*b.index_be(0), ByteOperation::MostSignificantBit);
            debug_assert_eq!(sign_1_res.into_vec(), vec![sign_b]);
            let sign_2_res = state
                .machine
                .check_byte_op(*c.index_be(0), ByteOperation::MostSignificantBit);
            debug_assert_eq!(sign_2_res.into_vec(), vec![sign_c]);
            let sign_remainder_res = state.machine.check_byte_op(
                *(Word::from(rem as u32).index_be(0)),
                ByteOperation::MostSignificantBit,
            );
            debug_assert_eq!(sign_remainder_res.into_vec(), vec![sign_remainder]);

            if ci32 > 0 {
                // 0 <= rem < c
                state
                    .machine
                    .lt_u32_mut()
                    .operations
                    .push(lt::Operation::Slt32(true, (rem as u32).into(), c));
            } else {
                // c < rem <= 0
                state
                    .machine
                    .lt_u32_mut()
                    .operations
                    .push(lt::Operation::Slt32(true, c, (rem as u32).into()));
            }

            // Lower 32 bits of a * c
            state
                .machine
                .mul_32_mut()
                .operations
                .push(mul::Operation::Mul32((prod as u32).into(), a, c));
            // Upper two bits of a * c (signed)
            state
                .machine
                .mul_32_mut()
                .operations
                .push(mul::Operation::Mulhs32(prod_upper_mulhs.into(), a, c));

            if same_sign {
                // Sum: b == a * c + rem
                state
                    .machine
                    .add_u32_mut()
                    .operations
                    .push(add::Operation::Add32(
                        b,
                        (prod as u32).into(),
                        (rem as u32).into(),
                    ));
            } else {
                // Sum: a * c == b + rem
                state
                    .machine
                    .add_u32_mut()
                    .operations
                    .push(add::Operation::Add32(
                        (prod as u32).into(),
                        b,
                        (rem as u32).into(),
                    ));
            }

            state.machine.push_bus_op(imm, opcode, ops);

            // Range checks from the Div32 chip: output of division
            state.machine.range_check_word(a);

            // Range checks from the Mul32 chip
            // from the Mul32 operation
            state.machine.range_check_word(prod as u32);
            state.machine.range_check_word(prod_upper_mul);
            // from the Mulhs32 operation
            state.machine.range_check_word(prod as u32);
            state.machine.range_check_word(prod_upper_mulhs);

            // internal range checks in the Mul32 chip
            let carries_mul = get_carries(
                &Long::from(ai32 as u32 as u64),
                &Long::from(ci32 as u32 as u64),
            );
            let carries_mulhs = get_carries(
                &Long::from(ai32 as i64 as u64),
                &Long::from(ci32 as i64 as u64),
            );
            // from the MUL32 operation: uses unsigned multiplication for upper word
            state.machine.range_check_mul_chip_carries(&carries_mul);
            // from the MULHS32 operation
            state.machine.range_check_mul_chip_carries(&carries_mulhs);

            // Sign bit check from the MULHS32 operation
            let res = state
                .machine
                .check_byte_op(*a.index_be(0), ByteOperation::MostSignificantBit);
            debug_assert_eq!(res.len(), 1);
            let sign_a_expected = res[0];
            debug_assert_eq!(sign_a_expected, (ai32 < 0) as u8);
            let res = state
                .machine
                .check_byte_op(*c.index_be(0), ByteOperation::MostSignificantBit);
            debug_assert_eq!(res.len(), 1);
            let sign_c_expected = res[0];
            debug_assert_eq!(sign_c_expected, (ci32 < 0) as u8);

            // Range checks from the Add32 chip
            if same_sign {
                state.machine.range_check_word(b);
            } else {
                state.machine.range_check_word(prod as u32);
            }
        }

        state.machine.step_pc();
    }
}
