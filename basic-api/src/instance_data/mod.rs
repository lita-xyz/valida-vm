use std::{borrow::BorrowMut, collections::BTreeMap, mem::transmute};

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use valida_machine::{MachineInstanceData, ProgramROM, PublicRow, PublicTrace, Word, NUM_CHIPS};
use valida_memory::PersistentMemoryRecord;
use valida_memory_footprint::MemoryFootprint;
use valida_output::columns::{PublicOutputCols, NUM_PUBLIC_OUTPUT_COLS};
use valida_program::rom_to_table;
use valida_static_data::StaticDataChip;
use valida_util::pad_to_power_of_two;

#[derive(Debug, Clone)]
pub struct ValidaInstanceData {
    // The ROM is part of the instance data when the machine is run in universal setup mode, but
    // not when run with circuit-specific setup. The static data is part of the first-segment instance
    // data when the machine is run in universal setup mode.
    pub rom: Option<ProgramROM<i32>>,

    pub pc_init: u32,
    pub fp_init: u32,

    pub output: Vec<u8>,

    pub did_fail: bool, // TODO: we can safely remove this because we can't prove executions that fail

    pub segments: Vec<ValidaSegmentInstanceData>,
}

impl<F: PrimeField32> MachineInstanceData<F> for ValidaInstanceData {
    fn public_traces(&self, verbose: Vec<bool>) -> Vec<Vec<Option<PublicTrace<F>>>> {
        // +1 for the static data chip, which belongs to the multi-segment machine.
        let mut public_traces: Vec<Vec<Option<PublicTrace<F>>>> =
            Vec::with_capacity(self.segments.len());
        self.segments
            .iter()
            .map(|segment| {
                let mut segment_public_traces = segment.public_traces(verbose.clone()).remove(0);
                debug_assert!(segment_public_traces.is_empty());
                // program ROM chip
                segment_public_traces[1] = self.rom.as_ref().map(|rom| {
                    let (trace, log) = rom_to_table(rom, verbose[1]);
                    if let Some(log) = log {
                        println!("Public trace for ROM chip:");
                        for line in log {
                            println!("{}", line);
                        }
                    }
                    PublicTrace::from_matrix(trace)
                });

                // output chip
                segment_public_traces[11] = {
                    let mut output_tape = self
                        .output
                        .iter()
                        .enumerate()
                        .flat_map(|(index, &byte)| {
                            let mut row = vec![F::zero(); NUM_PUBLIC_OUTPUT_COLS];
                            let cols: &mut PublicOutputCols<F> = row[..].borrow_mut();
                            cols.value = F::from_canonical_u8(byte);
                            if verbose[11] {
                                println!("Output public row {}: {:?}", index, cols);
                            }
                            row
                        })
                        .collect::<Vec<_>>();

                    pad_to_power_of_two::<NUM_PUBLIC_OUTPUT_COLS, F>(&mut output_tape);

                    Some(PublicTrace::from_matrix(RowMajorMatrix::new(
                        output_tape,
                        NUM_PUBLIC_OUTPUT_COLS,
                    )))
                };
                segment_public_traces
            })
            .collect()
    }
}

// TODO: define the type of a commitment to a memory state.
#[derive(Debug)]
pub struct MemoryCommitment {}

#[derive(Clone, Debug)]
pub struct ValidaSegmentInstanceData {
    // In the context of the `run` (program execution) to generate the instance data
    // we set `rom` and `static_data` to `None` for all segments, as this would be duplicated
    // data. In the context of prover and verifier we then assign the program rom
    // so that the public traces can be correctly generated.
    pub rom: Option<ProgramROM<i32>>,
    // The output of *this segment*
    pub output: Vec<u8>,
    // Static data will be empty (but not None) for all segments but the first
    pub static_data: Option<BTreeMap<u32, Word<u8>>>,

    // Initial program counter for this segment
    pub pc_init: u32,
    // Final program counter for this segment
    pub pc_final: u32,
    // Initial frame pointer for this segment
    pub fp_init: u32,
    // Final frame pointer for this segment
    pub fp_final: u32,

    // Whether this segment halted
    pub did_stop: bool,

    pub did_fail: bool, // TODO: we can safely remove this because we can't prove executions that fail

    pub segment_number: u32,
    pub is_last_segment: bool,
}

impl MemoryFootprint for ValidaSegmentInstanceData {
    fn memory_footprint(&self) -> usize {
        let mut result = 0;
        result += self.rom.memory_footprint();

        result += self.output.memory_footprint();

        result += self.static_data.memory_footprint();

        result += self.pc_init.memory_footprint();
        result += self.pc_final.memory_footprint();
        result += self.fp_init.memory_footprint();
        result += self.fp_final.memory_footprint();

        result += self.did_stop.memory_footprint();

        result += self.did_fail.memory_footprint();

        result += self.segment_number.memory_footprint();
        result += self.is_last_segment.memory_footprint();
        result
    }
}

impl<F: PrimeField32> MachineInstanceData<F> for ValidaSegmentInstanceData {
    fn public_traces(&self, verbose: Vec<bool>) -> Vec<Vec<Option<PublicTrace<F>>>> {
        vec![(0..NUM_CHIPS)
            .map(|i| {
                match i {
                    // CPU chip
                    0 => {
                        let public_vector =
                            vec![self.pc_init, self.fp_init, self.is_last_segment as u32]
                                .into_iter()
                                .map(F::from_canonical_u32)
                                .collect::<Vec<_>>();
                        if verbose[0] {
                            println!("CPU public vector: {:?}", public_vector);
                        }
                        Some(PublicTrace::from_vec(public_vector))
                    }
                    // program chip
                    1 => self.rom.as_ref().map(|rom| {
                        let (trace, log) = rom_to_table(rom, verbose[1]);
                        if let Some(log) = log {
                            println!("Public trace for ROM chip:");
                            for line in log {
                                println!("{}", line);
                            }
                        }
                        PublicTrace::from_matrix(trace)
                    }),
                    // memory chip
                    2 => {
                        let segment_number = self.segment_number;
                        if verbose[2] {
                            println!("Memory public segment number: {}", segment_number);
                        }
                        Some(PublicTrace::from_vec(vec![F::from_canonical_u32(
                            segment_number,
                        )]))
                    }
                    // output chip
                    11 => {
                        let mut output_tape = self
                            .output
                            .iter()
                            .enumerate()
                            .flat_map(|(index, &byte)| {
                                let mut row = vec![F::zero(); NUM_PUBLIC_OUTPUT_COLS];
                                let cols: &mut PublicOutputCols<F> = row[..].borrow_mut();
                                cols.value = F::from_canonical_u8(byte);
                                if verbose[11] {
                                    println!("Output public row {}: {:?}", index, cols);
                                }
                                row
                            })
                            .collect::<Vec<_>>();

                        pad_to_power_of_two::<NUM_PUBLIC_OUTPUT_COLS, F>(&mut output_tape);

                        Some(PublicTrace::from_matrix(RowMajorMatrix::new(
                            output_tape,
                            NUM_PUBLIC_OUTPUT_COLS,
                        )))
                    }
                    13 => self.static_data.as_ref().map(|cells| {
                        let (trace, log) = StaticDataChip::cells_to_table(cells, verbose[13]);
                        if let Some(log) = log {
                            println!("Public trace for StaticData chip:");
                            for line in log {
                                println!("{}", line);
                            }
                        }
                        PublicTrace::from_matrix(trace)
                    }),
                    _ => None,
                }
            })
            .collect::<Vec<_>>()]
    }
}
