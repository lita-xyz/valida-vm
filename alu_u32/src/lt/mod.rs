extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use columns::{Lt32Cols, LT_COL_MAP, NUM_LT_COLS};
use core::{borrow::Borrow, mem::transmute};
use valida_bus::MachineWithGeneralBus;
use valida_cpu::MachineWithCpuChip;
use valida_machine::{
    instructions, Chip, ChipTraceHeight, ChipWithPersistence, Instruction, Interaction, Operands,
    PublicTrace, RunningMachine, Word, MEMORY_CELL_BYTES,
};
use valida_opcodes::{map_opcode, map_opcode_to_field_value, LT32, LTE32, SLE32, SLT32};

use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use valida_machine::StarkConfig;
use valida_util::pad_to_power_of_two;

use valida_memory_footprint::MemoryFootprint;
pub mod columns;
pub mod stark;

#[derive(Clone)]
pub enum Operation {
    Lt32(bool, Word<u8>, Word<u8>),  // (dst, src1, src2)
    Lte32(bool, Word<u8>, Word<u8>), // (dst, src1, src2)
    Slt32(bool, Word<u8>, Word<u8>), // (dst, src1, src2)
    Sle32(bool, Word<u8>, Word<u8>), // (dst, src1, src2)
}

impl MemoryFootprint for Operation {
    fn memory_footprint(&self) -> usize {
        match self {
            Operation::Lt32(a, b, c) => a.memory_footprint() + 2 * b.memory_footprint(),
            Operation::Lte32(a, b, c) => a.memory_footprint() + 2 * b.memory_footprint(),
            Operation::Slt32(a, b, c) => a.memory_footprint() + 2 * b.memory_footprint(),
            Operation::Sle32(a, b, c) => a.memory_footprint() + 2 * b.memory_footprint(),
        }
    }
}

#[derive(Default)]
pub struct Lt32Chip {
    pub operations: Vec<Operation>,
}

impl MemoryFootprint for Lt32Chip {
    fn memory_footprint(&self) -> usize {
        self.operations.memory_footprint()
    }
}

impl ChipTraceHeight for Lt32Chip {
    fn trace_height(&self) -> u32 {
        self.operations.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for Lt32Chip
where
    M: MachineWithGeneralBus<SC::Val>,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "Lt32".to_string()
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
                let cols: &Lt32Cols<SC::Val> = row[..].borrow();
                log_prints.push(format!("Lt32 row {index}: {:?}", cols));
            }
            Some(log_prints)
        } else {
            None
        };

        let mut trace =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_LT_COLS);

        pad_to_power_of_two::<NUM_LT_COLS, SC::Val>(&mut trace.values);

        (Some(trace), log)
    }

    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let opcode = VirtualPairCol::new_main(
            vec![
                (LT_COL_MAP.is_lt, map_opcode_to_field_value(LT32)),
                (LT_COL_MAP.is_lte, map_opcode_to_field_value(LTE32)),
                (LT_COL_MAP.is_slt, map_opcode_to_field_value(SLT32)),
                (LT_COL_MAP.is_sle, map_opcode_to_field_value(SLE32)),
            ],
            SC::Val::zero(),
        );
        let input_1 = LT_COL_MAP.input_1.transform(VirtualPairCol::single_main);
        let input_2 = LT_COL_MAP.input_2.transform(VirtualPairCol::single_main);
        let output = Word::from_components_le([
            VirtualPairCol::single_main(LT_COL_MAP.output),
            VirtualPairCol::constant(SC::Val::zero()),
            VirtualPairCol::constant(SC::Val::zero()),
            VirtualPairCol::constant(SC::Val::zero()),
        ]);

        let mut fields = vec![opcode];
        fields.extend(input_1.into_iter_le());
        fields.extend(input_2.into_iter_le());
        fields.extend(output.into_iter_le());

        let receive = Interaction {
            fields,
            count: VirtualPairCol::single_main(LT_COL_MAP.multiplicity),
            argument_index: machine.general_bus(),
        };
        vec![receive]
    }
}

impl<M, SC> ChipWithPersistence<M, SC> for Lt32Chip
where
    M: MachineWithGeneralBus<SC::Val>,
    SC: StarkConfig,
{
}
impl Lt32Chip {
    fn op_to_row<F>(&self, op: &Operation) -> [F; NUM_LT_COLS]
    where
        F: PrimeField,
    {
        let mut row = [F::zero(); NUM_LT_COLS];
        let cols: &mut Lt32Cols<F> = unsafe { transmute(&mut row) };

        match op {
            Operation::Lt32(a, b, c) => {
                cols.is_lt = F::one();
                self.set_cols(cols, false, a, b, c);
            }
            Operation::Lte32(a, b, c) => {
                cols.is_lte = F::one();
                self.set_cols(cols, false, a, b, c);
            }
            Operation::Slt32(a, b, c) => {
                cols.is_slt = F::one();
                self.set_cols(cols, true, a, b, c);
            }
            Operation::Sle32(a, b, c) => {
                cols.is_sle = F::one();
                self.set_cols(cols, true, a, b, c);
            }
        }
        row
    }

    fn set_cols<F>(
        &self,
        cols: &mut Lt32Cols<F>,
        is_signed: bool,
        a: &bool,
        b: &Word<u8>,
        c: &Word<u8>,
    ) where
        F: PrimeField,
    {
        // Set the input columns
        debug_assert_eq!(MEMORY_CELL_BYTES, 4);
        cols.input_1 = b.transform(F::from_canonical_u8);
        cols.input_2 = c.transform(F::from_canonical_u8);
        cols.output = F::from_bool(*a);

        if let Some((n, x, y)) = b
            .into_iter_be()
            .zip(c.into_iter_be())
            .enumerate()
            .find_map(|(n, (x, y))| if x == y { None } else { Some((n, x, y)) })
        {
            let z = 256u16 + x as u16 - y as u16;
            for i in 0..9 {
                cols.bits[i] = F::from_canonical_u16(z >> i & 1);
            }
            cols.byte_flag[n] = F::one();
            // b[n] != c[n] always here, so the difference is never zero.
            cols.diff_inv = (*cols.input_1.index_be(n) - *cols.input_2.index_be(n)).inverse();
        }
        // compute (little-endian) bit decomposition of the top bytes
        for i in 0..8 {
            cols.top_bits_1[i] = F::from_canonical_u8(*b.index_be(0) >> i & 1);
            cols.top_bits_2[i] = F::from_canonical_u8(*c.index_be(0) >> i & 1);
        }
        // check if sign bits agree and set different_signs accordingly
        cols.different_signs = if is_signed {
            if cols.top_bits_1[7] != cols.top_bits_2[7] {
                F::one()
            } else {
                F::zero()
            }
        } else {
            F::zero()
        };

        cols.multiplicity = F::one();
    }

    fn execute_with_closure<M, E, F>(
        state: &mut RunningMachine<'_, E, M>,
        ops: Operands<i32>,
        opcode: u32,
        comp: F,
    ) -> (bool, Word<u8>, Word<u8>)
    where
        M: MachineWithLt32Chip<E>,
        E: PrimeField,
        F: Fn(Word<u8>, Word<u8>) -> bool,
    {
        let clk = state.machine.cpu().clock;
        let mut imm: Option<Word<u8>> = None;
        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        let src1: Word<u8> = if ops.is_left_imm() == 1 {
            let b = (ops.b() as u32).into();
            imm = Some(b);
            b
        } else {
            let read_addr_1 = (state.machine.cpu().fp as i32 + ops.b()) as u32;
            M::read(state, clk, read_addr_1)
        };
        let src2: Word<u8> = if ops.is_imm() == 1 {
            let c = (ops.c() as u32).into();
            imm = Some(c);
            c
        } else {
            let read_addr_2 = (state.machine.cpu().fp as i32 + ops.c()) as u32;
            M::read(state, clk, read_addr_2)
        };

        let res = comp(src1, src2);

        M::write(state, clk, write_addr, (res as u32).into());

        if state.machine.log_enabled() {
            if ops.is_left_imm() == 1 {
                state.machine.push_left_imm_bus_op(imm, opcode, ops);
            } else {
                state.machine.push_bus_op(imm, opcode, ops);
            }
        }

        (res, src1, src2)
    }
}

pub trait MachineWithLt32Chip<F: PrimeField>: MachineWithCpuChip<F> {
    fn lt_u32(&self) -> &Lt32Chip;
    fn lt_u32_mut(&mut self) -> &mut Lt32Chip;
}

instructions!(
    Lt32Instruction,
    Lte32Instruction,
    Slt32Instruction,
    Sle32Instruction
);

impl<M, F> Instruction<M, F> for Lt32Instruction
where
    M: MachineWithLt32Chip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = LT32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let comp = |a, b| a < b;
        let (dst, src1, src2) = Lt32Chip::execute_with_closure(state, ops, opcode, comp);
        if state.machine.log_enabled() {
            state
                .machine
                .lt_u32_mut()
                .operations
                .push(Operation::Lt32(dst, src1, src2));
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for Lte32Instruction
where
    M: MachineWithLt32Chip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = LTE32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let comp = |a, b| a <= b;
        let (dst, src1, src2) = Lt32Chip::execute_with_closure(state, ops, opcode, comp);
        if state.machine.log_enabled() {
            state
                .machine
                .lt_u32_mut()
                .operations
                .push(Operation::Lte32(dst, src1, src2));
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for Slt32Instruction
where
    M: MachineWithLt32Chip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = SLT32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let comp = |a: Word<u8>, b: Word<u8>| {
            let a_i: i32 = a.into();
            let b_i: i32 = b.into();
            a_i < b_i
        };
        let (dst, src1, src2) = Lt32Chip::execute_with_closure(state, ops, opcode, comp);
        if state.machine.log_enabled() {
            state
                .machine
                .lt_u32_mut()
                .operations
                .push(Operation::Slt32(dst, src1, src2));
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for Sle32Instruction
where
    M: MachineWithLt32Chip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = SLE32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let comp = |a: Word<u8>, b: Word<u8>| {
            let a_i: i32 = a.into();
            let b_i: i32 = b.into();
            a_i <= b_i
        };
        let (dst, src1, src2) = Lt32Chip::execute_with_closure(state, ops, opcode, comp);
        if state.machine.log_enabled() {
            state
                .machine
                .lt_u32_mut()
                .operations
                .push(Operation::Sle32(dst, src1, src2));
        }

        state.machine.step_pc();
    }
}
