#![no_std]

#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
use std::io::Write;

extern crate alloc;
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use crate::columns::{OutputCols, NUM_OUTPUT_COLS, OUTPUT_COL_MAP, PUBLIC_OUTPUT_COL_MAP};
use columns::{PublicOutputCols, NUM_PUBLIC_OUTPUT_COLS};
use core::{borrow::Borrow, mem::transmute};
use valida_bus::{
    MachineWithBytesBus, MachineWithGeneralBus, MachineWithOutputBus, MachineWithRangeBus8,
};
use valida_bytes::{half_baby_bear_range_sends, MachineWithBytesChip};
use valida_cpu::{MachineWithCpuChip, Operation};
use valida_machine::{
    instructions, Chip, ChipTraceHeight, ChipWithPersistence, Instruction, Interaction,
    MachineRuntime, Operands, PublicTrace, RunningMachine, StarkConfig, Word,
};
use valida_memory_footprint::MemoryFootprint;
use valida_opcodes::WRITE;
use valida_util::pad_to_power_of_two;

use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

pub mod columns;
pub mod stark;

#[derive(Default)]
pub struct OutputChip {
    pub tape: Vec<u8>,
    // clks_log[i] is the clock cycle at which tape[i] was written.
    pub clks_log: Vec<u32>,
}

impl MemoryFootprint for OutputChip {
    fn memory_footprint(&self) -> usize {
        self.tape.memory_footprint() + self.clks_log.memory_footprint()
    }
}

impl OutputChip {
    pub fn bytes(&self) -> &[u8] {
        &self.tape
    }
}

impl ChipTraceHeight for OutputChip {
    fn trace_height(&self) -> u32 {
        self.tape.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for OutputChip
where
    M: MachineWithGeneralBus<SC::Val>
        + MachineWithOutputBus<SC::Val>
        + MachineWithRangeBus8<SC::Val>
        + MachineWithBytesBus<SC::Val>,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "Output".to_string()
    }

    fn generate_main_trace(
        &self,
        _machine: &M,
        verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>) {
        // Only assert equality when we have clock logs (i.e., when logging was enabled)
        // In multi-segment mode, the first pass runs with logging disabled but may still
        // call generate_main_trace, causing tape and clks_log to have different lengths
        debug_assert!(self.clks_log.is_empty() || self.tape.len() == self.clks_log.len());

        let mut rows = self
            .clks_log
            .as_slice()
            .par_windows(2)
            .map(|window| {
                let (clk_1, clk_2) = (window[0], window[1]);
                let clk_diff: Word<u8> = (clk_2 - clk_1).into();
                let mut row = [SC::Val::zero(); NUM_OUTPUT_COLS];
                let cols: &mut OutputCols<SC::Val> = unsafe { transmute(&mut row) };
                cols.clk = SC::Val::from_canonical_u32(clk_1);
                cols.diff = clk_diff.transform(SC::Val::from_canonical_u8);
                cols.is_real = SC::Val::one();
                row
            })
            .collect::<Vec<_>>();

        // Add final row
        if let Some(last_clk) = self.clks_log.last() {
            let mut row = [SC::Val::zero(); NUM_OUTPUT_COLS];
            let cols: &mut OutputCols<SC::Val> = unsafe { transmute(&mut row) };
            cols.clk = SC::Val::from_canonical_u32(*last_clk);
            cols.is_real = SC::Val::one();
            rows.push(row);
        }

        let log = if verbose {
            let mut log_prints = Vec::with_capacity(rows.len());
            for (i, row) in rows.iter().enumerate() {
                let cols: &OutputCols<SC::Val> = row[..].borrow();
                log_prints.push(format!("Output row {}: {:?}", i, cols));
            }
            Some(log_prints)
        } else {
            None
        };

        let mut trace = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_OUTPUT_COLS,
        );

        pad_to_power_of_two::<NUM_OUTPUT_COLS, SC::Val>(&mut trace.values);

        (Some(trace), log)
    }

    /// To ensure that the output is correctly sorted, we check that the difference between consecutive
    /// clock values is in the range 0..2^31.
    fn global_sends(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let is_real = VirtualPairCol::single_main(OUTPUT_COL_MAP.is_real);
        let diff = OUTPUT_COL_MAP.diff.transform(VirtualPairCol::single_main);
        half_baby_bear_range_sends(machine, &diff, is_real)
    }

    fn generate_public_values(&self, verbose: bool) -> (Option<Self::Public>, Option<Vec<String>>) {
        let rows = self
            .tape
            .par_iter()
            .map(|b| {
                let mut row = [SC::Val::zero(); NUM_PUBLIC_OUTPUT_COLS];
                let cols: &mut PublicOutputCols<SC::Val> = unsafe { transmute(&mut row) };
                cols.value = SC::Val::from_canonical_u8(*b);
                row
            })
            .collect::<Vec<_>>();

        let log = if verbose {
            let mut log_prints = Vec::with_capacity(rows.len());
            for (i, row) in rows.iter().enumerate() {
                let cols: &PublicOutputCols<SC::Val> = row[..].borrow();
                log_prints.push(format!("Output row {}: {:?}", i, cols));
            }
            Some(log_prints)
        } else {
            None
        };

        let mut output_matrix = RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_PUBLIC_OUTPUT_COLS,
        );

        pad_to_power_of_two::<NUM_PUBLIC_OUTPUT_COLS, SC::Val>(&mut output_matrix.values);

        (Some(PublicTrace::from_matrix(output_matrix)), log)
    }

    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let clk = VirtualPairCol::single_main(OUTPUT_COL_MAP.clk);
        let value = VirtualPairCol::single_public(PUBLIC_OUTPUT_COL_MAP.value);

        let fields = vec![clk, value];

        let receive = Interaction {
            fields,
            count: VirtualPairCol::single_main(OUTPUT_COL_MAP.is_real),
            argument_index: machine.output_bus(),
        };
        vec![receive]
    }
}

impl<M, SC> ChipWithPersistence<M, SC> for OutputChip
where
    M: MachineWithGeneralBus<SC::Val>
        + MachineWithOutputBus<SC::Val>
        + MachineWithRangeBus8<SC::Val>
        + MachineWithBytesBus<SC::Val>,
    SC: StarkConfig,
{
}

pub trait MachineWithOutputChip<F: PrimeField>: MachineWithCpuChip<F> {
    fn output(&self) -> &OutputChip;
    fn output_mut(&mut self) -> &mut OutputChip;

    fn output_tape(&self) -> &[u8] {
        self.output().bytes()
    }

    fn output_byte(state: &mut RunningMachine<F, Self>, clk: u32, byte: u8) {
        #[cfg(feature = "std")]
        {
            std::io::stdout().write_all(&[byte]).unwrap();
        }
        state.runtime.write_to_file(byte);
        state.machine.output_mut().tape.push(byte);
        if state.machine.log_enabled() {
            state.machine.output_mut().clks_log.push(clk);
        }
    }
}

instructions!(WriteInstruction);

impl<M, F> Instruction<M, F> for WriteInstruction
where
    M: MachineWithOutputChip<F> + MachineWithBytesChip<F> + MachineWithBytesBus<F>,
    F: PrimeField,
{
    const OPCODE: u32 = WRITE;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;
        let read_addr_1 = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        let b = M::read(state, clk, read_addr_1);
        let last_clk = *(state.machine.output().clks_log.last().unwrap_or(&clk));

        M::output_byte(state, clk, b.trunc_to_u8());

        if state.machine.log_enabled() {
            // The range check counter should be updated.
            state
                .machine
                .check_half_baby_bear_range(&(clk - last_clk).into());
        }

        state.machine.push_op(Operation::Write, opcode, ops);

        // The immediate value flag should be set, and the immediate operand value should
        // equal zero. We only write one byte of one word at a time to output.
        assert_eq!(ops.is_imm(), 1);
        assert_eq!(ops.c(), 0);

        state.machine.step_pc();
    }
}
