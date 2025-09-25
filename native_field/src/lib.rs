#![no_std]

extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use columns::{NativeFieldCols, COL_MAP, NUM_NATIVE_FIELD_COLS};
use core::{borrow::Borrow, mem::transmute};
use valida_bus::{MachineWithGeneralBus, MachineWithRangeBus8};
use valida_bytes::{range8_sends_word, MachineWithRangeCheckeru8};
use valida_cpu::MachineWithCpuChip;
use valida_machine::{
    instructions, Chip, ChipTraceHeight, ChipWithPersistence, Instruction, Interaction, Operands,
    PublicTrace, RunningMachine, Word,
};
use valida_opcodes::{ADD, MUL, SUB};
use valida_util::pad_to_power_of_two;

use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use valida_machine::StarkConfig;

pub mod columns;
pub mod stark;

#[derive(Clone)]
pub enum Operation {
    Add(Word<u8>, Word<u8>, Word<u8>), // dst, src1, src2
    Sub(Word<u8>, Word<u8>, Word<u8>), // dst, src1, src2
    Mul(Word<u8>, Word<u8>, Word<u8>), // dst, src1, src2
}

pub struct NativeFieldChip {
    operations: Vec<Operation>,
}

impl ChipTraceHeight for NativeFieldChip {
    fn trace_height(&self) -> u32 {
        self.operations.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for NativeFieldChip
where
    M: MachineWithGeneralBus<SC::Val> + MachineWithRangeBus8<SC::Val>,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "NativeField".to_string()
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
                let cols: &NativeFieldCols<SC::Val> = row[..].borrow();
                log_prints.push(format!("NativeField row {index}: {:?}", cols));
            }
            Some(log_prints)
        } else {
            None
        };

        let mut trace = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_NATIVE_FIELD_COLS,
        );

        pad_to_power_of_two::<NUM_NATIVE_FIELD_COLS, SC::Val>(&mut trace.values);

        (Some(trace), log)
    }

    fn global_sends(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let is_real =
            VirtualPairCol::sum_main(vec![COL_MAP.is_add, COL_MAP.is_sub, COL_MAP.is_mul]);

        let output = COL_MAP.output.transform(VirtualPairCol::single_main);
        range8_sends_word(machine, output, &is_real)
    }

    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let opcode = VirtualPairCol::new_main(
            vec![
                (COL_MAP.is_add, SC::Val::from_canonical_u32(ADD)),
                (COL_MAP.is_sub, SC::Val::from_canonical_u32(SUB)),
                (COL_MAP.is_mul, SC::Val::from_canonical_u32(MUL)),
            ],
            SC::Val::zero(),
        );
        let input_1 = COL_MAP.input_1.transform(VirtualPairCol::single_main);
        let input_2 = COL_MAP.input_2.transform(VirtualPairCol::single_main);
        let output = COL_MAP.output.transform(VirtualPairCol::single_main);

        let mut fields = vec![opcode];
        fields.extend(input_1.into_iter_le());
        fields.extend(input_2.into_iter_le());
        fields.extend(output.into_iter_le());

        let is_real =
            VirtualPairCol::sum_main(vec![COL_MAP.is_add, COL_MAP.is_sub, COL_MAP.is_mul]);

        let receive = Interaction {
            fields,
            count: is_real,
            argument_index: machine.general_bus(),
        };
        vec![receive]
    }
}

impl<M, SC> ChipWithPersistence<M, SC> for NativeFieldChip
where
    M: MachineWithGeneralBus<SC::Val> + MachineWithRangeBus8<SC::Val>,
    SC: StarkConfig,
{
}
impl NativeFieldChip {
    fn op_to_row<F>(&self, op: &Operation) -> [F; NUM_NATIVE_FIELD_COLS]
    where
        F: PrimeField,
    {
        let mut row = [F::zero(); NUM_NATIVE_FIELD_COLS];
        let cols: &mut NativeFieldCols<F> = unsafe { transmute(&mut row) };

        match op {
            Operation::Add(a, b, c) => {
                cols.is_add = F::one();
                cols.input_1 = b.transform(F::from_canonical_u8);
                cols.input_2 = c.transform(F::from_canonical_u8);
                cols.output = a.transform(F::from_canonical_u8);
            }
            Operation::Sub(a, b, c) => {
                cols.is_sub = F::one();
                cols.input_1 = b.transform(F::from_canonical_u8);
                cols.input_2 = c.transform(F::from_canonical_u8);
                cols.output = a.transform(F::from_canonical_u8);
            }
            Operation::Mul(a, b, c) => {
                cols.is_mul = F::one();
                cols.input_1 = b.transform(F::from_canonical_u8);
                cols.input_2 = c.transform(F::from_canonical_u8);
                cols.output = a.transform(F::from_canonical_u8);
            }
        }

        row
    }
}

pub trait MachineWithNativeFieldChip<F: PrimeField>: MachineWithCpuChip<F> {
    fn native_field(&self) -> NativeFieldChip;
    fn native_field_mut(&mut self) -> &mut NativeFieldChip;
}

instructions!(AddInstruction, SubInstruction, MulInstruction);

impl<M, F> Instruction<M, F> for AddInstruction
where
    M: MachineWithNativeFieldChip<F> + MachineWithRangeCheckeru8<F>,
    F: PrimeField32,
{
    const OPCODE: u32 = ADD;

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

        let a_native = F::from_canonical_u32(b.into()) + F::from_canonical_u32(c.into());
        let a = Word::from(a_native.as_canonical_u32());
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .native_field_mut()
                .operations
                .push(Operation::Add(a, b, c));

            state.machine.push_bus_op(imm, opcode, ops);
            state.machine.range_check_word(a);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for SubInstruction
where
    M: MachineWithNativeFieldChip<F> + MachineWithRangeCheckeru8<F>,
    F: PrimeField32,
{
    const OPCODE: u32 = SUB;

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

        let a_native = F::from_canonical_u32(b.into()) - F::from_canonical_u32(c.into());
        let a = Word::from(a_native.as_canonical_u32());
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .native_field_mut()
                .operations
                .push(Operation::Sub(a, b, c));

            state.machine.push_bus_op(imm, opcode, ops);
            state.machine.range_check_word(a);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for MulInstruction
where
    M: MachineWithNativeFieldChip<F> + MachineWithRangeCheckeru8<F>,
    F: PrimeField32,
{
    const OPCODE: u32 = MUL;

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

        let a_m31 = F::from_canonical_u32(b.into()) * F::from_canonical_u32(c.into());
        let a = Word::from(a_m31.as_canonical_u32());
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .native_field()
                .operations
                .push(Operation::Mul(a, b, c));

            state.machine.push_bus_op(imm, opcode, ops);
            state.machine.range_check_word(a);
        }

        state.machine.step_pc();
    }
}
