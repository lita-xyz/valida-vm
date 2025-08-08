extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use columns::{Bitwise32Cols, COL_MAP, NUM_BITWISE_COLS};
use core::{borrow::Borrow, mem::transmute};
use valida_bus::MachineWithGeneralBus;
use valida_cpu::MachineWithCpuChip;
use valida_machine::{
    instructions, Chip, ChipTraceHeight, ChipWithPersistence, Instruction, Interaction, Operands,
    PublicTrace, RunningMachine, Word,
};
use valida_opcodes::{convert_opcode, AND32, OR32, XOR32};

use valida_memory_footprint::MemoryFootprint;

use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use valida_machine::StarkConfig;
use valida_util::pad_to_power_of_two;

pub mod columns;
pub mod stark;

#[derive(Clone, Debug)]
pub enum Operation {
    And32(Word<u8>, Word<u8>, Word<u8>), // (dst, src1, src2)
    Or32(Word<u8>, Word<u8>, Word<u8>),  // ''
    Xor32(Word<u8>, Word<u8>, Word<u8>), // ''
}

impl MemoryFootprint for Operation {
    fn memory_footprint(&self) -> usize {
        match self {
            // Could just be one branch, but need a value so...
            Operation::And32(a, _b, _c) => 3 * a.memory_footprint(),
            Operation::Or32(a, _b, _c) => 3 * a.memory_footprint(),
            Operation::Xor32(a, _b, _c) => 3 * a.memory_footprint(),
        }
    }
}

#[derive(Default)]
pub struct Bitwise32Chip {
    pub operations: Vec<Operation>,
}

impl MemoryFootprint for Bitwise32Chip {
    fn memory_footprint(&self) -> usize {
        self.operations.memory_footprint()
    }
}

impl ChipTraceHeight for Bitwise32Chip {
    fn trace_height(&self) -> u32 {
        self.operations.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for Bitwise32Chip
where
    M: MachineWithGeneralBus<SC::Val>,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "Bitwise32".to_string()
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
                let cols: &Bitwise32Cols<SC::Val> = row[..].borrow();
                log_prints.push(format!("Bitwise32 row {index}: {:?}", cols));
            }
            Some(log_prints)
        } else {
            None
        };

        let mut trace = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_BITWISE_COLS,
        );

        pad_to_power_of_two::<NUM_BITWISE_COLS, SC::Val>(&mut trace.values);

        (Some(trace), log)
    }

    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let opcode = VirtualPairCol::new_main(
            vec![
                (
                    COL_MAP.is_and,
                    SC::Val::from_canonical_u32(convert_opcode(AND32)),
                ),
                (
                    COL_MAP.is_or,
                    SC::Val::from_canonical_u32(convert_opcode(OR32)),
                ),
                (
                    COL_MAP.is_xor,
                    SC::Val::from_canonical_u32(convert_opcode(XOR32)),
                ),
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

        let is_real = VirtualPairCol::sum_main(vec![COL_MAP.is_and, COL_MAP.is_or, COL_MAP.is_xor]);

        let receive = Interaction {
            fields,
            count: is_real,
            argument_index: machine.general_bus(),
        };
        vec![receive]
    }
}

impl<M, SC> ChipWithPersistence<M, SC> for Bitwise32Chip
where
    M: MachineWithGeneralBus<SC::Val>,
    SC: StarkConfig,
{
}

impl Bitwise32Chip {
    fn op_to_row<F>(&self, op: &Operation) -> [F; NUM_BITWISE_COLS]
    where
        F: PrimeField,
    {
        let mut row = [F::zero(); NUM_BITWISE_COLS];
        let cols: &mut Bitwise32Cols<F> = unsafe { transmute(&mut row) };

        match op {
            Operation::Xor32(a, b, c) => {
                cols.is_xor = F::one();
                self.set_cols(a, b, c, cols);
            }
            Operation::And32(a, b, c) => {
                cols.is_and = F::one();
                self.set_cols(a, b, c, cols);
            }
            Operation::Or32(a, b, c) => {
                cols.is_or = F::one();
                self.set_cols(a, b, c, cols);
            }
        }

        row
    }

    fn set_cols<F>(&self, a: &Word<u8>, b: &Word<u8>, c: &Word<u8>, cols: &mut Bitwise32Cols<F>)
    where
        F: PrimeField,
    {
        cols.input_1 = b.transform(F::from_canonical_u8);
        cols.input_2 = c.transform(F::from_canonical_u8);
        cols.output = a.transform(F::from_canonical_u8);

        let mut bits_1 = [[F::zero(); 8]; 4];
        let mut bits_2 = [[F::zero(); 8]; 4];
        for i in 0..4 {
            for j in 0..8 {
                bits_1[i][j] = F::from_canonical_u8(*b.index_be(i) >> j & 1);
                bits_2[i][j] = F::from_canonical_u8(*c.index_be(i) >> j & 1);
            }
        }
        cols.bits_1 = bits_1;
        cols.bits_2 = bits_2;
    }
}

pub trait MachineWithBitwise32Chip<F: PrimeField>: MachineWithCpuChip<F> {
    fn bitwise_u32(&self) -> &Bitwise32Chip;
    fn bitwise_u32_mut(&mut self) -> &mut Bitwise32Chip;
}

instructions!(And32Instruction, Or32Instruction, Xor32Instruction);

impl<M, F> Instruction<M, F> for Xor32Instruction
where
    M: MachineWithBitwise32Chip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = XOR32;

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

        let a = b ^ c;
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .bitwise_u32_mut()
                .operations
                .push(Operation::Xor32(a, b, c));
            state.machine.push_bus_op(imm, opcode, ops);
        }
        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for And32Instruction
where
    M: MachineWithBitwise32Chip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = AND32;

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

        let a = b & c;
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .bitwise_u32_mut()
                .operations
                .push(Operation::And32(a, b, c));
            state.machine.push_bus_op(imm, opcode, ops);
        }
        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for Or32Instruction
where
    M: MachineWithBitwise32Chip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = OR32;

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

        let a = b | c;
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .bitwise_u32_mut()
                .operations
                .push(Operation::Or32(a, b, c));
            state.machine.push_bus_op(imm, opcode, ops);
        }
        state.machine.step_pc();
    }
}
