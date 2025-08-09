extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use columns::{Add32Cols, ADD_COL_MAP, NUM_ADD_COLS};
use core::{borrow::Borrow, mem::transmute};
use valida_bus::{MachineWithGeneralBus, MachineWithRangeBus8};
use valida_cpu::MachineWithCpuChip;
use valida_machine::{
    instructions, Chip, ChipTraceHeight, ChipWithPersistence, Instruction, Interaction, Operands,
    PublicTrace, RunningMachine, Word,
};
use valida_opcodes::{map_opcode_to_field_value, ADD32};

use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use valida_machine::StarkConfig;
use valida_memory_footprint::MemoryFootprint;
use valida_util::pad_to_power_of_two;

use valida_bytes::{range8_sends_word, MachineWithRangeCheckeru8};

pub mod columns;
pub mod stark;

#[derive(Clone)]
pub enum Operation {
    Add32(Word<u8>, Word<u8>, Word<u8>),
}

impl MemoryFootprint for Operation {
    fn memory_footprint(&self) -> usize {
        match self {
            Operation::Add32(a, _b, _c) => 3 * a.memory_footprint(),
        }
    }
}

#[derive(Default)]
pub struct Add32Chip {
    pub operations: Vec<Operation>,
}

impl MemoryFootprint for Add32Chip {
    fn memory_footprint(&self) -> usize {
        self.operations.memory_footprint()
    }
}

impl ChipTraceHeight for Add32Chip {
    fn trace_height(&self) -> u32 {
        self.operations.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for Add32Chip
where
    M: MachineWithGeneralBus<SC::Val>
        + MachineWithRangeCheckeru8<SC::Val>
        + MachineWithRangeBus8<SC::Val>,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "Add32".to_string()
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
                let cols: &Add32Cols<SC::Val> = row[..].borrow();
                log_prints.push(format!("Add32 row {index}: {:?}", cols));
            }
            Some(log_prints)
        } else {
            None
        };

        let mut trace =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_ADD_COLS);

        pad_to_power_of_two::<NUM_ADD_COLS, SC::Val>(&mut trace.values);

        (Some(trace), log)
    }

    fn global_sends(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let is_real = VirtualPairCol::single_main(ADD_COL_MAP.is_real);
        let output = ADD_COL_MAP.output.transform(VirtualPairCol::single_main);
        range8_sends_word(machine, output, &is_real)
    }

    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let opcode = VirtualPairCol::constant(map_opcode_to_field_value(ADD32));
        let input_1 = ADD_COL_MAP.input_1.transform(VirtualPairCol::single_main);
        let input_2 = ADD_COL_MAP.input_2.transform(VirtualPairCol::single_main);
        let output = ADD_COL_MAP.output.transform(VirtualPairCol::single_main);

        let mut fields = vec![opcode];
        fields.extend(input_1.into_iter_le());
        fields.extend(input_2.into_iter_le());
        fields.extend(output.into_iter_le());

        let receive = Interaction {
            fields,
            count: VirtualPairCol::single_main(ADD_COL_MAP.is_real),
            argument_index: machine.general_bus(),
        };
        vec![receive]
    }
}

impl<M, SC> ChipWithPersistence<M, SC> for Add32Chip
where
    M: MachineWithGeneralBus<SC::Val>
        + MachineWithRangeCheckeru8<SC::Val>
        + MachineWithRangeBus8<SC::Val>,
    SC: StarkConfig,
{
}

impl Add32Chip {
    fn op_to_row<F>(&self, op: &Operation) -> [F; NUM_ADD_COLS]
    where
        F: PrimeField,
    {
        let mut row = [F::zero(); NUM_ADD_COLS];
        let cols: &mut Add32Cols<F> = unsafe { transmute(&mut row) };

        match op {
            Operation::Add32(a, b, c) => {
                cols.input_1 = b.transform(F::from_canonical_u8);
                cols.input_2 = c.transform(F::from_canonical_u8);
                cols.output = a.transform(F::from_canonical_u8);

                let mut carry_1 = 0;
                let mut carry_2 = 0;
                if *b.index_be(3) as u32 + *c.index_be(3) as u32 > 255 {
                    carry_1 = 1;
                    cols.carry[0] = F::one();
                }
                if *b.index_be(2) as u32 + *c.index_be(2) as u32 + carry_1 > 255 {
                    carry_2 = 1;
                    cols.carry[1] = F::one();
                }
                if *b.index_be(1) as u32 + *c.index_be(1) as u32 + carry_2 > 255 {
                    cols.carry[2] = F::one();
                }
                cols.is_real = F::one();
            }
        }
        row
    }
}

pub trait MachineWithAdd32Chip<F: PrimeField>: MachineWithCpuChip<F> {
    fn add_u32(&self) -> &Add32Chip;
    fn add_u32_mut(&mut self) -> &mut Add32Chip;
}

instructions!(Add32Instruction);

impl<M, F> Instruction<M, F> for Add32Instruction
where
    M: MachineWithAdd32Chip<F> + MachineWithRangeCheckeru8<F>,
    F: PrimeField,
{
    const OPCODE: u32 = ADD32;

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

        let a = b + c;
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .add_u32_mut()
                .operations
                .push(Operation::Add32(a, b, c));
            state.machine.push_bus_op(imm, opcode, ops);
        }
        state.machine.step_pc();

        state.machine.range_check_word(a);
    }
}
