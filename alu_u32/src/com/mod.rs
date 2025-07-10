extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use columns::{Com32Cols, COM_COL_MAP, NUM_COM_COLS};
use core::borrow::Borrow;
use core::mem::transmute;
use valida_bus::MachineWithGeneralBus;
use valida_cpu::MachineWithCpuChip;
use valida_machine::{
    instructions, Chip, ChipTraceHeight, Instruction, Interaction, Operands, PublicTrace, Word,
};
use valida_machine::{ChipWithPersistence, RunningMachine, StarkConfig};
use valida_opcodes::{EQ32, NE32};

use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use valida_util::pad_to_power_of_two;

use valida_memory_footprint::MemoryFootprint;

pub mod columns;
pub mod stark;

#[derive(Clone)]
pub enum Operation {
    Ne32(bool, Word<u8>, Word<u8>), // (dst, src1, src2)
    Eq32(bool, Word<u8>, Word<u8>), // (dst, src1, src2)
}

impl MemoryFootprint for Operation {
    fn memory_footprint(&self) -> usize {
        match self {
            Operation::Ne32(a, b, _c) => a.memory_footprint() + 2 * b.memory_footprint(),
            Operation::Eq32(a, b, _c) => a.memory_footprint() + 2 * b.memory_footprint(),
        }
    }
}

#[derive(Default)]
pub struct Com32Chip {
    pub operations: Vec<Operation>,
}

impl MemoryFootprint for Com32Chip {
    fn memory_footprint(&self) -> usize {
        self.operations.memory_footprint()
    }
}

impl ChipTraceHeight for Com32Chip {
    fn trace_height(&self) -> u32 {
        self.operations.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for Com32Chip
where
    M: MachineWithGeneralBus<SC::Val>,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "Com32".to_string()
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
                let cols: &Com32Cols<SC::Val> = row[..].borrow();
                log_prints.push(format!("Com32 row {index}: {:?}", cols));
            }
            Some(log_prints)
        } else {
            None
        };

        let mut trace =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_COM_COLS);

        pad_to_power_of_two::<NUM_COM_COLS, SC::Val>(&mut trace.values);

        (Some(trace), log)
    }
    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let opcode = VirtualPairCol::new_main(
            vec![
                (COM_COL_MAP.is_ne, SC::Val::from_canonical_u32(NE32)),
                (COM_COL_MAP.is_eq, SC::Val::from_canonical_u32(EQ32)),
            ],
            SC::Val::zero(),
        );
        let input_1 = COM_COL_MAP.input_1.transform(VirtualPairCol::single_main);
        let input_2 = COM_COL_MAP.input_2.transform(VirtualPairCol::single_main);
        let output = Word::from_components_le([
            VirtualPairCol::single_main(COM_COL_MAP.output),
            VirtualPairCol::constant(SC::Val::zero()),
            VirtualPairCol::constant(SC::Val::zero()),
            VirtualPairCol::constant(SC::Val::zero()),
        ]);

        let mut fields = vec![opcode];
        fields.extend(input_1.into_iter_le());
        fields.extend(input_2.into_iter_le());
        fields.extend(output.into_iter_le());

        let is_real = VirtualPairCol::sum_main(vec![COM_COL_MAP.is_ne, COM_COL_MAP.is_eq]);

        let receive = Interaction {
            fields,
            count: is_real,
            argument_index: machine.general_bus(),
        };
        vec![receive]
    }
}

impl<M, SC> ChipWithPersistence<M, SC> for Com32Chip
where
    M: MachineWithGeneralBus<SC::Val>,
    SC: StarkConfig,
{
}
impl Com32Chip {
    fn op_to_row<F>(&self, op: &Operation) -> [F; NUM_COM_COLS]
    where
        F: PrimeField,
    {
        let mut row = [F::zero(); NUM_COM_COLS];
        let cols: &mut Com32Cols<F> = unsafe { transmute(&mut row) };

        let (a, b, c) = match op {
            Operation::Ne32(a, b, c) => {
                cols.is_ne = F::one();
                (a, b, c)
            }
            Operation::Eq32(a, b, c) => {
                cols.is_eq = F::one();
                (a, b, c)
            }
        };

        cols.input_1 = b.transform(F::from_canonical_u8);
        cols.input_2 = c.transform(F::from_canonical_u8);
        // This maps 'true' to 1 and 'false' to 0
        cols.output = F::from_bool(*a);

        cols.diff = F::from_canonical_u32(
            b.into_iter_le()
                .zip(c.into_iter_le())
                .map(|(b, c)| (((b as i32 - c as i32) * (b as i32 - c as i32)) as u32))
                .sum(),
        );
        if b != c {
            cols.diff_inv = cols.diff.inverse();
            cols.not_equal = F::one();
        } else {
            cols.not_equal = F::zero();
        }

        row
    }
}

pub trait MachineWithCom32Chip<F: PrimeField>: MachineWithCpuChip<F> {
    fn com_u32(&self) -> &Com32Chip;
    fn com_u32_mut(&mut self) -> &mut Com32Chip;
}

instructions!(Ne32Instruction, Eq32Instruction);

impl<M, F> Instruction<M, F> for Ne32Instruction
where
    M: MachineWithCom32Chip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = NE32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;

        let mut imm: Option<Word<u8>> = None;
        let read_addr_1 = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        let src1 = M::read(state, clk, read_addr_1);
        let src2 = if ops.is_imm() == 1 {
            let c = (ops.c() as u32).into();
            imm = Some(c);
            c
        } else {
            let read_addr_2 = (state.machine.cpu().fp as i32 + ops.c()) as u32;
            M::read(state, clk, read_addr_2)
        };

        let dst = if src1 != src2 {
            Word::from(1)
        } else {
            Word::from(0)
        };
        M::write(state, clk, write_addr, dst);

        if state.machine.log_enabled() {
            state
                .machine
                .com_u32_mut()
                .operations
                .push(Operation::Ne32(src1 != src2, src1, src2));
            state.machine.push_bus_op(imm, opcode, ops);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for Eq32Instruction
where
    M: MachineWithCom32Chip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = EQ32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;

        let mut imm: Option<Word<u8>> = None;
        let read_addr_1 = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        let src1 = M::read(state, clk, read_addr_1);
        let src2 = if ops.is_imm() == 1 {
            let c = (ops.c() as u32).into();
            imm = Some(c);
            c
        } else {
            let read_addr_2 = (state.machine.cpu().fp as i32 + ops.c()) as u32;
            M::read(state, clk, read_addr_2)
        };

        let dst = if src1 == src2 {
            Word::from(1)
        } else {
            Word::from(0)
        };
        M::write(state, clk, write_addr, dst);

        if state.machine.log_enabled() {
            state
                .machine
                .com_u32_mut()
                .operations
                .push(Operation::Eq32(src1 == src2, src1, src2));
            state.machine.push_bus_op(imm, opcode, ops);
        }

        state.machine.step_pc();
    }
}
