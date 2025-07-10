extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use columns::{Sub32Cols, NUM_SUB_COLS, SUB_COL_MAP};
use core::borrow::Borrow;
use core::mem::transmute;
use valida_bus::{MachineWithGeneralBus, MachineWithRangeBus8};
use valida_cpu::MachineWithCpuChip;
use valida_machine::{
    instructions, Chip, ChipTraceHeight, ChipWithPersistence, Instruction, Interaction, Operands,
    PublicTrace, RunningMachine, Word,
};
use valida_opcodes::SUB32;

use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use valida_machine::StarkConfig;
use valida_util::pad_to_power_of_two;

use valida_bytes::{range8_sends_word, MachineWithRangeCheckeru8};

use valida_memory_footprint::MemoryFootprint;

pub mod columns;
pub mod stark;

#[derive(Clone)]
pub enum Operation {
    Sub32(Word<u8>, Word<u8>, Word<u8>),
}

impl MemoryFootprint for Operation {
    fn memory_footprint(&self) -> usize {
        match self {
            Operation::Sub32(a, b, c) => 3 * b.memory_footprint(),
        }
    }
}

#[derive(Default)]
pub struct Sub32Chip {
    pub operations: Vec<Operation>,
}

impl MemoryFootprint for Sub32Chip {
    fn memory_footprint(&self) -> usize {
        self.operations.memory_footprint()
    }
}

impl ChipTraceHeight for Sub32Chip {
    fn trace_height(&self) -> u32 {
        self.operations.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for Sub32Chip
where
    M: MachineWithGeneralBus<SC::Val> + MachineWithRangeBus8<SC::Val>,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "Sub32".to_string()
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
                let cols: &Sub32Cols<SC::Val> = row[..].borrow();
                log_prints.push(format!("Sub32 row {index}: {:?}", cols));
            }
            Some(log_prints)
        } else {
            None
        };

        let mut trace =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_SUB_COLS);

        pad_to_power_of_two::<NUM_SUB_COLS, SC::Val>(&mut trace.values);

        (Some(trace), log)
    }

    fn global_sends(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let is_real = VirtualPairCol::single_main(SUB_COL_MAP.is_real);
        let output = SUB_COL_MAP.output.transform(VirtualPairCol::single_main);
        range8_sends_word(machine, output, &is_real)
    }

    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let opcode = VirtualPairCol::constant(SC::Val::from_canonical_u32(SUB32));
        let input_1 = SUB_COL_MAP.input_1.transform(VirtualPairCol::single_main);
        let input_2 = SUB_COL_MAP.input_2.transform(VirtualPairCol::single_main);
        let output = SUB_COL_MAP.output.transform(VirtualPairCol::single_main);

        let mut fields = vec![opcode];
        fields.extend(input_1.into_iter_le());
        fields.extend(input_2.into_iter_le());
        fields.extend(output.into_iter_le());

        let receive = Interaction {
            fields,
            count: VirtualPairCol::single_main(SUB_COL_MAP.is_real),
            argument_index: machine.general_bus(),
        };
        vec![receive]
    }
}
impl<M, SC> ChipWithPersistence<M, SC> for Sub32Chip
where
    M: MachineWithGeneralBus<SC::Val> + MachineWithRangeBus8<SC::Val>,
    SC: StarkConfig,
{
}
impl Sub32Chip {
    fn op_to_row<F>(&self, op: &Operation) -> [F; NUM_SUB_COLS]
    where
        F: PrimeField32,
    {
        let mut row = [F::zero(); NUM_SUB_COLS];
        let cols: &mut Sub32Cols<F> = unsafe { transmute(&mut row) };

        match op {
            Operation::Sub32(a, b, c) => {
                cols.input_1 = b.transform(F::from_canonical_u8);
                cols.input_2 = c.transform(F::from_canonical_u8);
                cols.output = a.transform(F::from_canonical_u8);

                if *b.index_be(3) < *c.index_be(3) {
                    *cols.borrow.index_mut_be(3) = F::one();
                }
                if (*b.index_be(2) as u32)
                    < (*c.index_be(2) as u32) + cols.borrow.index_be(3).as_canonical_u32()
                {
                    *cols.borrow.index_mut_be(2) = F::one();
                }
                if (*b.index_be(1) as u32)
                    < (*c.index_be(1) as u32) + cols.borrow.index_mut_be(2).as_canonical_u32()
                {
                    *cols.borrow.index_mut_be(1) = F::one();
                }
                if (*b.index_be(0) as u32)
                    < (*c.index_be(0) as u32) + cols.borrow.index_be(1).as_canonical_u32()
                {
                    *cols.borrow.index_mut_be(0) = F::one();
                }
                cols.is_real = F::one();
            }
        }

        row
    }
}

pub trait MachineWithSub32Chip<F: PrimeField>: MachineWithCpuChip<F> {
    fn sub_u32(&self) -> &Sub32Chip;
    fn sub_u32_mut(&mut self) -> &mut Sub32Chip;
}

instructions!(Sub32Instruction);

impl<M, F> Instruction<M, F> for Sub32Instruction
where
    M: MachineWithSub32Chip<F> + MachineWithRangeCheckeru8<F>,
    F: PrimeField,
{
    const OPCODE: u32 = SUB32;

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

        let a = b - c;
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .sub_u32_mut()
                .operations
                .push(Operation::Sub32(a, b, c));

            state.machine.push_bus_op(imm, opcode, ops);
        }

        state.machine.step_pc();
        state.machine.range_check_word(a);
    }
}
