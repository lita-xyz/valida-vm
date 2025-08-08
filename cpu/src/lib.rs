//#![no_std]

extern crate alloc;

use crate::columns::{CpuCols, CpuPublicVector, CPU_COL_MAP, NUM_CPU_COLS, NUM_CPU_PUBLIC_VALUES};
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::borrow::Borrow;
use core::iter;
use core::marker::Sync;
use core::mem::transmute;
use valida_bus::{
    MachineWithBytesBus, MachineWithGeneralBus, MachineWithMemBus, MachineWithOutputBus,
    MachineWithPointerBus, MachineWithProgramBus, MachineWithRangeBus8,
};
use valida_bytes::{byte_send_simple, ByteOperation, MachineWithBytesChip};
use valida_machine::{
    addr_of_word, index_le_of_byte, instructions, Chip, ChipTraceHeight, ChipWithPersistence,
    Instruction, InstructionWord, Interaction, Machine, Operands, PublicRow, PublicTrace,
    RunningMachine, Word, CPU_MEMORY_WRITE_CHANNELS, MEMORY_CELL_BYTES,
};
use valida_machine::{is_mul_4, MachineRuntime, MemoryRecord, CPU_MEMORY_READ_CHANNELS};
use valida_memory::{MachineWithMemoryChip, Operation as MemoryOperation};
use valida_opcodes::{
    BEQ, BNE, BYTES_PER_INSTR, FAIL, IMM32, JAL, JALV, LOAD32, LOADFP, LOADS8, LOADU8, MEMCPY,
    READ_ADVICE, STOP, STORE32, STOREU8,
};
use valida_program::columns::NUM_PROGRAM_COLS;
use valida_program::columns::PROGRAM_COL_MAP;

use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use valida_machine::StarkConfig;
use valida_memory_footprint::MemoryFootprint;
use valida_util::batch_multiplicative_inverse_allowing_zero;

pub mod columns;
pub mod stark;

#[derive(Clone)]
pub enum Operation {
    Store32,
    StoreU8,
    Load32,
    LoadU8,
    LoadS8,
    Jal,
    Jalv,
    Beq(Option<Word<u8>> /*imm*/),
    Bne(Option<Word<u8>> /*imm*/),
    Imm32,
    Bus(Option<Word<u8>> /*imm*/),
    Pointer,
    BusLeftImm(Option<Word<u8>> /*imm*/),
    ReadAdvice,
    Stop,
    Fail,
    LoadFp,
    Write,
    Memcpy,
}

impl MemoryFootprint for Operation {
    fn memory_footprint(&self) -> usize {
        core::mem::size_of::<Self>()
    }
}

#[derive(Default)]
pub struct CpuChip {
    pub clock: u32,
    pub pc: u32,
    pub fp: u32,
    pub registers: Vec<Registers>,
    pub operations: Vec<Operation>,
    pub instructions: Vec<InstructionWord<i32>>,
    pub pc_init: u32,
    pub fp_init: u32,
    pub is_last_segment: u32,
}

impl MemoryFootprint for CpuChip {
    fn memory_footprint(&self) -> usize {
        self.clock.memory_footprint()
            + self.pc.memory_footprint()
            + self.fp.memory_footprint()
            + self.registers.memory_footprint()
            + self.operations.memory_footprint()
            + self.instructions.memory_footprint()
            + self.pc_init.memory_footprint()
            + self.fp_init.memory_footprint()
            + self.is_last_segment.memory_footprint()
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Registers {
    pub pc: u32,
    pub fp: u32,
}

impl MemoryFootprint for Registers {
    fn memory_footprint(&self) -> usize {
        self.pc.memory_footprint() + self.fp.memory_footprint()
    }
}

impl ChipTraceHeight for CpuChip {
    fn trace_height(&self) -> u32 {
        self.operations.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for CpuChip
where
    M: MachineWithProgramBus<SC::Val>
        + MachineWithMemoryChip<SC::Val>
        + MachineWithGeneralBus<SC::Val>
        + MachineWithMemBus<SC::Val>
        + MachineWithOutputBus<SC::Val>
        + MachineWithBytesBus<SC::Val>
        + MachineWithRangeBus8<SC::Val>
        + MachineWithPointerBus<SC::Val>
        + Sync,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "CPU".to_string()
    }

    fn generate_main_trace(
        &self,
        machine: &M,
        verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>) {
        let mut rows = self
            .operations
            .as_slice()
            .into_par_iter()
            .enumerate()
            .map(|(n, op)| {
                self.op_to_row::<M, SC>(
                    n.try_into()
                        .expect("number of memory operations is not larger than 2^32"),
                    op,
                    machine,
                )
            })
            .collect::<Vec<_>>();

        // Set diff, diff_inv, and not_equal
        Self::compute_word_diffs(&mut rows);

        let log = if verbose {
            let mut log_prints = Vec::with_capacity(rows.len());
            for (i, row) in rows.iter().enumerate() {
                let cols: &CpuCols<SC::Val> = row[..].borrow();
                log_prints.push(format!("CPU row {i}: {:?}", cols));
            }
            Some(log_prints)
        } else {
            None
        };
        let mut trace =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_CPU_COLS);

        Self::pad_to_power_of_two(&mut trace.values);

        (Some(trace), log)
    }

    fn generate_public_values(
        &self,
        verbose: bool,
    ) -> (Option<PublicTrace<SC::Val>>, Option<Vec<String>>) {
        let (pc_init, fp_init, is_last_segment) =
            (self.pc_init, self.fp_init, self.is_last_segment);

        let mut row = [SC::Val::zero(); NUM_CPU_PUBLIC_VALUES];
        let public_vector: &mut CpuPublicVector<SC::Val> = unsafe { transmute(&mut row) };
        public_vector.pc_init = SC::Val::from_canonical_u32(pc_init);
        public_vector.fp_init = SC::Val::from_canonical_u32(fp_init);
        public_vector.is_last_segment = SC::Val::from_canonical_u32(is_last_segment);

        let log = if verbose {
            Some(vec![format!("CPU public vector: {:?}", public_vector)])
        } else {
            None
        };

        (
            Some(PublicTrace::PublicVector(PublicRow(row.to_vec()))),
            log,
        )
    }

    fn global_sends(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        // Memory bus channels
        let mem_read_sends = (0..CPU_MEMORY_READ_CHANNELS).map(|i| {
            let channel = &CPU_COL_MAP.mem_read_channels[i];
            let is_read = VirtualPairCol::constant(SC::Val::one());
            let clk = VirtualPairCol::single_main(CPU_COL_MAP.clk);
            let addr = VirtualPairCol::single_main(channel.addr);
            let value = channel.value.transform(VirtualPairCol::single_main);

            let mut fields = vec![is_read, clk, addr];
            fields.extend(value.into_iter_le());

            Interaction {
                fields,
                count: VirtualPairCol::single_main(channel.used),
                argument_index: machine.mem_bus(),
            }
        });
        let mem_write_sends = (0..CPU_MEMORY_WRITE_CHANNELS).map(|i| {
            let channel = &CPU_COL_MAP.mem_write_channels[i];
            let is_read = VirtualPairCol::constant(SC::Val::zero());
            let clk = VirtualPairCol::single_main(CPU_COL_MAP.clk);
            let addr = VirtualPairCol::single_main(channel.addr);
            let value = channel.value.transform(VirtualPairCol::single_main);

            let mut fields = vec![is_read, clk, addr];
            fields.extend(value.into_iter_le());

            Interaction {
                fields,
                count: VirtualPairCol::single_main(channel.used),
                argument_index: machine.mem_bus(),
            }
        });
        let mem_read_send_single_byte = {
            let channel = &CPU_COL_MAP.mem_write_channels[0];
            let is_read = VirtualPairCol::constant(SC::Val::one());
            let clk = VirtualPairCol::single_main(CPU_COL_MAP.clk);
            let addr = VirtualPairCol::single_main(channel.addr);
            let old_value = channel.old_value.transform(VirtualPairCol::single_main);

            let mut fields = vec![is_read, clk, addr];
            fields.extend(old_value.into_iter_le());

            Interaction {
                fields,
                count: VirtualPairCol::single_main(CPU_COL_MAP.opcode_flags.is_store_u8),
                argument_index: machine.mem_bus(),
            }
        };
        // Check the sign bit for the single-byte load instruction
        let sign_bit = VirtualPairCol::single_main(CPU_COL_MAP.sign_bit);
        let is_load_s8 = VirtualPairCol::single_main(CPU_COL_MAP.opcode_flags.is_load_s8);
        let single_loaded_byte =
            VirtualPairCol::single_main(*CPU_COL_MAP.mem_write_channels[0].value.index_le(0));
        let load_s8_sign_bit_send = byte_send_simple(
            machine,
            single_loaded_byte,
            Some(sign_bit.clone()),
            is_load_s8,
            ByteOperation::MostSignificantBit,
        );
        // Check the sign bit for the jalv instruction
        let is_jalv = VirtualPairCol::single_main(CPU_COL_MAP.opcode_flags.is_jalv);
        let top_byte_of_second_read =
            VirtualPairCol::single_main(*CPU_COL_MAP.mem_read_channels[1].value.index_be(0));
        let jalv_sign_bit_send = byte_send_simple(
            machine,
            top_byte_of_second_read,
            Some(sign_bit),
            is_jalv,
            ByteOperation::MostSignificantBit,
        );

        // General bus channel
        let mut fields = vec![VirtualPairCol::single_main(CPU_COL_MAP.instruction.opcode)];
        fields.extend(
            CPU_COL_MAP
                .mem_read_channels
                .iter()
                .flat_map(|c| c.value.into_iter_le().map(VirtualPairCol::single_main))
                .collect::<Vec<_>>(),
        );
        fields.extend(
            CPU_COL_MAP
                .mem_write_channels
                .iter()
                .flat_map(|c| c.value.into_iter_le())
                .map(VirtualPairCol::single_main)
                .collect::<Vec<_>>(),
        );
        let send_general = Interaction {
            fields,
            count: VirtualPairCol::single_main(CPU_COL_MAP.opcode_flags.is_bus_op),
            argument_index: machine.general_bus(),
        };

        // Pointer bus channel
        let mut fields = vec![VirtualPairCol::single_main(CPU_COL_MAP.instruction.opcode)];
        fields.extend(
            CPU_COL_MAP.mem_read_channels[0..1]
                .iter()
                .flat_map(|c| c.value.into_iter_le().map(VirtualPairCol::single_main))
                .collect::<Vec<_>>(),
        );
        let send_pointer = Interaction {
            fields,
            count: VirtualPairCol::single_main(CPU_COL_MAP.opcode_flags.is_pointer_op),
            argument_index: machine.pointer_bus(),
        };

        // Output bus channel
        let fields_output = vec![
            VirtualPairCol::single_main(CPU_COL_MAP.clk),
            VirtualPairCol::single_main(
                CPU_COL_MAP.mem_read_channels[0]
                    .value
                    .into_iter_le()
                    .next()
                    .unwrap(),
            ),
        ];
        let send_output = Interaction {
            fields: fields_output,
            count: VirtualPairCol::single_main(CPU_COL_MAP.opcode_flags.is_write),
            argument_index: machine.output_bus(),
        };

        // Program ROM bus channel
        let mut program_sends: Vec<Interaction<SC::Val>> = vec![];
        let pc: VirtualPairCol<SC::Val> = VirtualPairCol::single_main(CPU_COL_MAP.pc);
        let opcode = VirtualPairCol::single_main(CPU_COL_MAP.instruction.opcode);
        let is_imm_op = VirtualPairCol::single_main(CPU_COL_MAP.opcode_flags.is_imm_op);
        let is_left_imm_op = VirtualPairCol::single_main(CPU_COL_MAP.opcode_flags.is_left_imm_op);
        // 1 - is_imm_op - is_left_imm_op (note that is_imm_op and is_left_imm_op are mutually exclusive, so this is always 0 or 1).
        // Send if _not_ an immediate op _and_ if it is a real row (because padded rows have both zero too)
        let is_not_imm_op = VirtualPairCol::new_main(
            vec![
                (CPU_COL_MAP.is_real, SC::Val::one()),
                (CPU_COL_MAP.opcode_flags.is_imm_op, -SC::Val::one()),
                (CPU_COL_MAP.opcode_flags.is_left_imm_op, -SC::Val::one()),
            ],
            SC::Val::zero(),
        );

        let mut fields_common = vec![VirtualPairCol::constant(SC::Val::zero()); NUM_PROGRAM_COLS];
        // the send to the program chip should be exactly a row of the program table, in the same order.

        // set the program columns that are common to all instructions
        fields_common[PROGRAM_COL_MAP.pc] = pc.clone();
        fields_common[PROGRAM_COL_MAP.opcode] = opcode.clone();

        CPU_COL_MAP
            .instruction
            .operands
            .into_iter()
            .zip(PROGRAM_COL_MAP.operands)
            .for_each(|(op_col_cpu, op_col_program)| {
                fields_common[op_col_program] = VirtualPairCol::single_main(op_col_cpu);
            });
        let fields_no_imm = fields_common.clone();

        let send_program_no_imm = Interaction {
            fields: fields_no_imm,
            count: is_not_imm_op,
            argument_index: machine.program_bus(),
        };
        program_sends.push(send_program_no_imm);

        let mut fields_left_imm = fields_common.clone();

        PROGRAM_COL_MAP
            .imm
            .into_iter_le()
            .zip(CPU_COL_MAP.mem_read_channels[0].value.into_iter_le())
            .for_each(|(imm_col_program, imm_col_cpu)| {
                fields_left_imm[imm_col_program] = VirtualPairCol::single_main(imm_col_cpu);
            });

        let send_program_left_imm = Interaction {
            fields: fields_left_imm,
            count: is_left_imm_op,
            argument_index: machine.program_bus(),
        };
        program_sends.push(send_program_left_imm);

        let mut fields_imm = fields_common;
        PROGRAM_COL_MAP
            .imm
            .into_iter_le()
            .zip(CPU_COL_MAP.mem_read_channels[1].value.into_iter_le())
            .for_each(|(imm_col_program, imm_col_cpu)| {
                fields_imm[imm_col_program] = VirtualPairCol::single_main(imm_col_cpu);
            });

        let send_program_imm = Interaction {
            fields: fields_imm,
            count: is_imm_op,
            argument_index: machine.program_bus(),
        };
        program_sends.push(send_program_imm);

        mem_read_sends
            .chain(mem_write_sends)
            .chain(iter::once(mem_read_send_single_byte))
            .chain(iter::once(load_s8_sign_bit_send))
            .chain(iter::once(jalv_sign_bit_send))
            .chain(iter::once(send_general))
            .chain(iter::once(send_pointer))
            .chain(iter::once(send_output))
            .chain(program_sends)
            .collect()
    }
}

impl<M, SC> ChipWithPersistence<M, SC> for CpuChip
where
    M: MachineWithProgramBus<SC::Val>
        + MachineWithMemoryChip<SC::Val>
        + MachineWithGeneralBus<SC::Val>
        + MachineWithMemBus<SC::Val>
        + MachineWithOutputBus<SC::Val>
        + MachineWithBytesBus<SC::Val>
        + MachineWithRangeBus8<SC::Val>
        + MachineWithPointerBus<SC::Val>
        + Sync,
    SC: StarkConfig,
{
}
impl CpuChip {
    fn op_to_row<M, SC>(&self, clk: u32, op: &Operation, machine: &M) -> [SC::Val; NUM_CPU_COLS]
    where
        M: MachineWithMemoryChip<SC::Val>,
        SC: StarkConfig,
    {
        let mut row = [SC::Val::zero(); NUM_CPU_COLS];
        let cols: &mut CpuCols<SC::Val> = unsafe { transmute(&mut row) };

        cols.pc = SC::Val::from_canonical_u32(self.registers[clk as usize].pc);
        cols.fp = SC::Val::from_canonical_u32(self.registers[clk as usize].fp);
        cols.clk = SC::Val::from_canonical_u32(clk);
        cols.is_last_segment = SC::Val::from_canonical_u32(self.is_last_segment);
        cols.is_real = SC::Val::one();
        self.set_instruction_values(clk, cols);

        match op {
            Operation::Store32 => {
                cols.opcode_flags.is_store = SC::Val::one();
            }
            Operation::Load32 => {
                cols.opcode_flags.is_load = SC::Val::one();
            }
            Operation::StoreU8 => {
                cols.opcode_flags.is_store_u8 = SC::Val::one();
            }
            Operation::LoadU8 => {
                cols.opcode_flags.is_load_u8 = SC::Val::one();
            }
            Operation::LoadS8 => {
                cols.opcode_flags.is_load_s8 = SC::Val::one();
            }
            Operation::Jal => {
                cols.opcode_flags.is_jal = SC::Val::one();
            }
            Operation::Jalv => {
                cols.opcode_flags.is_jalv = SC::Val::one();
            }
            Operation::Beq(imm) => {
                cols.opcode_flags.is_beq = SC::Val::one();
                self.set_imm_value(cols, *imm);
            }
            Operation::Bne(imm) => {
                cols.opcode_flags.is_bne = SC::Val::one();
                self.set_imm_value(cols, *imm);
            }
            Operation::Imm32 => {
                cols.opcode_flags.is_imm32 = SC::Val::one();
            }
            Operation::Bus(imm) => {
                cols.opcode_flags.is_bus_op = SC::Val::one();
                self.set_imm_value(cols, *imm);
            }
            Operation::Pointer => {
                cols.opcode_flags.is_pointer_op = SC::Val::one();
            }
            Operation::BusLeftImm(imm) => {
                cols.opcode_flags.is_bus_op = SC::Val::one();
                self.set_left_imm_value(cols, *imm);
            }
            Operation::ReadAdvice => {
                cols.opcode_flags.is_advice = SC::Val::one();
            }
            Operation::Stop => {
                cols.opcode_flags.is_stop = SC::Val::one();
            }
            Operation::Fail => {}
            Operation::LoadFp => {
                cols.opcode_flags.is_loadfp = SC::Val::one();
            }
            Operation::Write => {
                cols.opcode_flags.is_write = SC::Val::one();
            }
            Operation::Memcpy => {
                todo!("memcpy not implemented");
            }
        }

        self.set_memory_channel_values::<M, SC>(clk, cols, machine);

        row
    }
    fn set_offset_flags<F: PrimeField32>(&self, cols: &mut CpuCols<F>) {
        let offset = if cols.opcode_flags.is_store_u8 == F::one() {
            let addr_bottom_byte = cols.mem_read_channels[1]
                .value
                .index_le(0)
                .as_canonical_u32();
            addr_bottom_byte % (MEMORY_CELL_BYTES as u32)
        } else if cols.opcode_flags.is_load_u8 == F::one()
            || cols.opcode_flags.is_load_s8 == F::one()
        {
            let addr_bottom_byte = cols.mem_read_channels[0]
                .value
                .index_le(0)
                .as_canonical_u32();
            addr_bottom_byte % (MEMORY_CELL_BYTES as u32)
        } else {
            0
        };
        if cols.opcode_flags.is_load_s8 == F::one()
            || cols.opcode_flags.is_load_u8 == F::one()
            || cols.opcode_flags.is_store_u8 == F::one()
        {
            *cols.addr_offset_flags.index_mut_le(offset as usize) = F::one();
        }
    }

    fn set_instruction_values<F: PrimeField>(&self, clk: u32, cols: &mut CpuCols<F>) {
        cols.instruction.opcode = F::from_canonical_u32(self.instructions[clk as usize].opcode);
        cols.instruction.opcode_lo16 =
            F::from_canonical_u32(self.instructions[clk as usize].opcode & 0x0F);
        cols.instruction.opcode_hi16 =
            F::from_canonical_u32((self.instructions[clk as usize].opcode >> 4) & 0x0F);
        cols.instruction.operands =
            Operands::<F>::from_i32_slice(&self.instructions[clk as usize].operands.0);
    }

    fn set_memory_channel_values<M: MachineWithMemoryChip<SC::Val>, SC: StarkConfig>(
        &self,
        clk: u32,
        cols: &mut CpuCols<SC::Val>,
        machine: &M,
    ) {
        let is_left_imm_op = cols.opcode_flags.is_left_imm_op == SC::Val::one();
        let is_pointer_op = cols.opcode_flags.is_pointer_op == SC::Val::one();
        let memory = machine.mem();
        for ops in memory.operations.get(&clk).iter() {
            let mut read_index = 0;
            for op in ops.iter() {
                match op {
                    MemoryOperation::DummyRead(..) => {}
                    MemoryOperation::Read(addr, MemoryRecord { value, .. }) => {
                        // The first read sets the left memory channel unless the operation is left-immediate
                        if read_index == 0 && !is_left_imm_op {
                            cols.mem_read_channels[0].used = SC::Val::one();
                            cols.mem_read_channels[0].addr = SC::Val::from_canonical_u32(*addr);
                            cols.mem_read_channels[0].value =
                                value.transform(SC::Val::from_canonical_u8);
                            read_index += 1;
                            // The second read, or the first read for a left-immediate instruction, sets the right memory channel
                        } else if read_index < 2 && !is_pointer_op {
                            cols.mem_read_channels[1].used = SC::Val::one();
                            cols.mem_read_channels[1].addr = SC::Val::from_canonical_u32(*addr);
                            cols.mem_read_channels[1].value =
                                value.transform(SC::Val::from_canonical_u8);
                            read_index += 1;
                            if cols.opcode_flags.is_jalv == SC::Val::one() {
                                cols.sign_bit = SC::Val::from_canonical_u8(*value.index_be(0) >> 7);
                            }
                            // The only circumstance in which there is a third read is in the case of store_u8,
                            // which reads the old value of the memory cell
                        } else if !is_pointer_op {
                            cols.mem_write_channels[0].old_value =
                                value.transform(SC::Val::from_canonical_u8);
                        }
                    }
                    MemoryOperation::Write(addr, value) => {
                        if !is_pointer_op {
                            cols.mem_write_channels[0].used = SC::Val::one();
                            cols.mem_write_channels[0].addr = SC::Val::from_canonical_u32(*addr);
                            cols.mem_write_channels[0].value =
                                value.transform(SC::Val::from_canonical_u8);
                            if cols.opcode_flags.is_load_s8 == SC::Val::one() {
                                cols.sign_bit = SC::Val::from_canonical_u8(*value.index_le(0) >> 7);
                            }
                        }
                    }
                }
            }
        }
        self.set_offset_flags(cols);
    }

    fn compute_word_diffs<F: PrimeField>(rows: &mut [[F; NUM_CPU_COLS]]) {
        // Compute `diff`
        let mut diff = vec![F::zero(); rows.len()];
        for i in 0..rows.len() {
            let word_1 = CPU_COL_MAP.mem_read_channels[0]
                .value
                .into_iter_le()
                .map(|j| rows[i][j])
                .collect::<Vec<_>>();
            let word_2 = CPU_COL_MAP.mem_read_channels[1]
                .value
                .into_iter_le()
                .map(|j| rows[i][j])
                .collect::<Vec<_>>();
            for (a, b) in word_1.into_iter().zip(word_2) {
                diff[i] += (a - b).square();
            }
        }

        // Compute `diff_inv`
        let diff_inv = batch_multiplicative_inverse_allowing_zero(diff.clone());

        // Set trace values
        for i in 0..rows.len() {
            rows[i][CPU_COL_MAP.diff] = diff[i];
            rows[i][CPU_COL_MAP.diff_inv] = diff_inv[i];
            if diff[i] != F::zero() {
                rows[i][CPU_COL_MAP.not_equal] = F::one();
            }
        }
    }

    /// We simply pad the CPU rows in the same way as other chips nowadays. That means
    /// we extend by lots of zeros. This works now, because we added an `is_real` column
    /// which we explicitly check for and set to true in `op_to_row`.
    /// This is important now in the context of continuations. Padding with STOP instructions
    /// as was done previously is not a valid solution for all but the last segment. But as
    /// we require a working solution for segments other than the last, there is also no reason
    /// to pad with STOP in the last segment either.
    fn pad_to_power_of_two<F: PrimeField>(values: &mut Vec<F>) {
        let n_real_rows = values.len() / NUM_CPU_COLS;
        values.resize(n_real_rows.next_power_of_two() * NUM_CPU_COLS, F::zero());
    }

    fn set_imm_value<F: PrimeField>(&self, cols: &mut CpuCols<F>, imm: Option<Word<u8>>) {
        if let Some(imm) = imm {
            cols.opcode_flags.is_imm_op = F::one();
            let imm = imm.transform(F::from_canonical_u8);
            cols.mem_read_channels[1].value = imm;
            cols.instruction.operands.0[2] = imm.reduce();
        }
    }

    fn set_left_imm_value<F: PrimeField>(&self, cols: &mut CpuCols<F>, imm: Option<Word<u8>>) {
        if let Some(imm) = imm {
            cols.opcode_flags.is_left_imm_op = F::one();
            let imm = imm.transform(F::from_canonical_u8);
            cols.mem_read_channels[0].value = imm;
            cols.instruction.operands.0[1] = imm.reduce();
        }
    }
}
pub trait MachineWithRegisters<F: PrimeField>: Machine<F> {
    fn set_initial_register_values(&mut self, reg: Registers);
    fn initial_register_values(&self) -> Registers;
}

pub trait MachineWithCpuChip<F: PrimeField>:
    MachineWithMemoryChip<F> + MachineWithRegisters<F>
{
    fn cpu(&self) -> &CpuChip;
    fn cpu_mut(&mut self) -> &mut CpuChip;

    fn set_pc(&mut self, new_pc: u32);
    fn step_pc(&mut self);

    fn set_fp(&mut self, new_fp: u32);
    fn inc_fp(&mut self, offset: i32);

    fn push_bus_op(&mut self, imm: Option<Word<u8>>, opcode: u32, operands: Operands<i32>);
    fn push_pointer_op(&mut self, opcode: u32, operands: Operands<i32>);
    fn push_left_imm_bus_op(&mut self, imm: Option<Word<u8>>, opcode: u32, operands: Operands<i32>);
    fn push_op(&mut self, op: Operation, opcode: u32, operands: Operands<i32>);
}

instructions!(
    Load32Instruction,
    LoadU8Instruction,
    LoadS8Instruction,
    Store32Instruction,
    StoreU8Instruction,
    JalInstruction,
    JalvInstruction,
    BeqInstruction,
    BneInstruction,
    Imm32Instruction,
    ReadAdviceInstruction,
    StopInstruction,
    FailInstruction,
    LoadFpInstruction,
    MemcpyInstruction
);

/// Non-deterministic instructions
impl<M, F> Instruction<M, F> for ReadAdviceInstruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = READ_ADVICE;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>)
    where
        M: MachineWithCpuChip<F>,
    {
        let clk = state.machine.cpu().clock;
        let fp = state.machine.cpu().fp as i32;
        let mem_addr = fp + ops.a();

        // Read from the advice tape into memory
        let advice_opt = state.runtime.read_from_file();
        let advice_byte = match advice_opt {
            Some(advice) => Word::from_u8(advice),
            // eof
            None => Word::from(u32::MAX),
        };
        M::write(state, clk, mem_addr as u32, advice_byte);

        if state.machine.log_enabled() {
            state.machine.push_op(
                Operation::ReadAdvice,
                <Self as Instruction<M, F>>::OPCODE,
                ops,
            );
        }

        state.machine.step_pc();
    }
}

/// Deterministic instructions
impl<M, F> Instruction<M, F> for Load32Instruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = LOAD32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;
        let fp = state.machine.cpu().fp;

        let read_addr_1 = (fp as i32 + ops.c()) as u32;
        assert!(
            is_mul_4(read_addr_1),
            "LOAD32: Read address location is not a multiple of 4!"
        );

        let read_addr_2 = M::read(state, clk, read_addr_1);
        assert!(
            is_mul_4(read_addr_2.into()),
            "LOAD32: Read address is not a multiple of 4!"
        );

        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        assert!(
            is_mul_4(write_addr),
            "LOAD32: Write address location is not a multiple of 4!"
        );

        let cell = M::read(state, clk, read_addr_2.into());
        M::write(state, clk, write_addr, cell);
        if state.machine.log_enabled() {
            state.machine.push_op(Operation::Load32, opcode, ops);
        }

        state.machine.step_pc();
    }
}

fn read_byte<M, F>(read_addr: u32, state: &mut RunningMachine<'_, F, M>, clk: u32) -> u8
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    let read_addr_index = addr_of_word(read_addr);

    // The word from the read address.
    let cell = M::read(state, clk, read_addr_index);

    // The (little-endian) array index of the word for the byte to read from
    let index_of_read = index_le_of_byte(read_addr);

    // The byte from the read cell.
    *cell.index_le(index_of_read)
}

fn write_byte<M, F>(write_addr: u32, to_write: u8, state: &mut RunningMachine<'_, F, M>, clk: u32)
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    // The array index of the word for the byte to write to
    let index_of_write = index_le_of_byte(write_addr);

    // The key to the memory map, converted to a multiple of 4.
    let write_addr_index = addr_of_word(write_addr);

    // The original content of the cell to write to. If the cell is empty, initiate it with a default value.
    let cell_write = M::read(state, clk, write_addr_index);

    // The Word to write, with one byte overwritten to the read byte
    let cell_to_write = cell_write.update_byte(to_write, index_of_write);

    M::write(state, clk, write_addr_index, cell_to_write);
}

impl<M, F> Instruction<M, F> for LoadU8Instruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = LOADU8;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;
        let fp = state.machine.cpu().fp;

        let read_addr_loc = (fp as i32 + ops.c()) as u32;

        assert!(
            is_mul_4(read_addr_loc),
            "LOADU8: Read address location is not a multiple of 4!"
        );

        let read_addr = M::read(state, clk, read_addr_loc);
        let cell_byte = read_byte(read_addr.into(), state, clk);

        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;

        assert!(
            is_mul_4(write_addr),
            "LOADU8: Write address is not a multiple of 4!"
        );

        // The Word to write, with one byte overwritten to the read byte
        M::write(state, clk, write_addr, Word::from_u8(cell_byte));
        if state.machine.log_enabled() {
            state.machine.push_op(Operation::LoadU8, opcode, ops);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for LoadS8Instruction
where
    M: MachineWithCpuChip<F> + MachineWithBytesChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = LOADS8;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;
        let fp = state.machine.cpu().fp;

        let read_addr_loc = (fp as i32 + ops.c()) as u32;

        assert!(
            is_mul_4(read_addr_loc),
            "LOADS8: Read address location is not a multiple of 4!"
        );

        let read_addr = M::read(state, clk, read_addr_loc);

        let cell_byte = read_byte(read_addr.into(), state, clk);

        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;

        assert!(
            is_mul_4(write_addr),
            "LOADS8: Write address is not a multiple of 4!"
        );

        // The Word to write, with one byte overwritten to the read byte
        let cell_to_write = Word::sign_extend_byte(cell_byte);
        M::write(state, clk, write_addr, cell_to_write);

        // Record the byte operation lookup to extract the sign bit from celL_byte
        if state.machine.log_enabled() {
            let _ = state
                .machine
                .check_byte_op(cell_byte, ByteOperation::MostSignificantBit);

            state.machine.push_op(Operation::LoadS8, opcode, ops);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for Store32Instruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = STORE32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;

        let read_addr = (state.machine.cpu().fp as i32 + ops.c()) as u32;
        assert!(
            is_mul_4(read_addr),
            "STORE32: Read address is not a multiple of 4!"
        );
        let cell = M::read(state, clk, read_addr);

        let write_addr_loc = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        assert!(
            is_mul_4(write_addr_loc),
            "STORE32: Write address location is not a multiple of 4!"
        );
        let write_addr = M::read(state, clk, write_addr_loc);
        assert!(
            is_mul_4(write_addr.into()),
            "STORE32: Write address is not a multiple of 4!"
        );

        M::write(state, clk, write_addr.into(), cell);
        if state.machine.log_enabled() {
            state.machine.push_op(Operation::Store32, opcode, ops);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for StoreU8Instruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = STOREU8;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;

        let read_addr = (state.machine.cpu().fp as i32 + ops.c()) as u32;
        assert!(
            is_mul_4(read_addr),
            "STOREU8: Read address is not a multiple of 4!"
        );
        // Read the cell from the read address.
        let cell = M::read(state, clk, read_addr);

        // Make sure we get to the correct and non empty map for the byte.
        let write_addr_loc = (state.machine.cpu().fp as i32 + ops.b()) as u32;

        assert!(
            is_mul_4(write_addr_loc),
            "STOREU8: Write address location is not a multiple of 4!"
        );
        let write_addr = M::read(state, clk, write_addr_loc);

        write_byte(write_addr.into(), cell.trunc_to_u8(), state, clk);
        if state.machine.log_enabled() {
            state.machine.push_op(Operation::StoreU8, opcode, ops);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for JalInstruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = JAL;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let clk = state.machine.cpu().clock;
        // Store 24 * (pc + 1) to local stack variable at offset a
        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        if state.machine.log_enabled() {
            state
                .machine
                .push_op(Operation::Jal, <Self as Instruction<M, F>>::OPCODE, ops);
        }
        let next_pc = state.machine.cpu().pc + 1;
        M::write(state, clk, write_addr, (BYTES_PER_INSTR * next_pc).into());
        // Set pc to the field element b / 24
        state.machine.set_pc((ops.b() as u32) / BYTES_PER_INSTR);
        // Set fp to fp + c
        state.machine.inc_fp(ops.c());
    }
}

impl<M, F> Instruction<M, F> for JalvInstruction
where
    M: MachineWithCpuChip<F> + MachineWithBytesChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = JALV;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;
        if state.machine.log_enabled() {
            state.machine.push_op(Operation::Jalv, opcode, ops);
        }
        // Store pc + 1 to local stack variable at offset a
        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        let next_pc = state.machine.cpu().pc + 1;
        M::write(state, clk, write_addr, (BYTES_PER_INSTR * next_pc).into());
        // Set pc to the field element [b]
        let read_addr = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        let read_value: u32 = M::read(state, clk, read_addr).into();
        state.machine.set_pc(read_value / BYTES_PER_INSTR);
        // Set fp to [c]
        let read_addr = (state.machine.cpu().fp as i32 + ops.c()) as u32;
        let cell: u32 = M::read(state, clk, read_addr).into();
        let offset: i32 = cell as i32;
        state.machine.inc_fp(offset);

        if state.machine.log_enabled() {
            let _ = state
                .machine
                .check_byte_op(cell.to_be_bytes()[0], ByteOperation::MostSignificantBit);
        }
    }
}

impl<M, F> Instruction<M, F> for BeqInstruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = BEQ;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;
        let mut imm: Option<Word<u8>> = None;
        let read_addr_1 = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        let cell_1 = M::read(state, clk, read_addr_1);
        let cell_2 = if ops.is_imm() == 1 {
            let c = (ops.c() as u32).into();
            imm = Some(c);
            c
        } else {
            let read_addr_2 = (state.machine.cpu().fp as i32 + ops.c()) as u32;
            M::read(state, clk, read_addr_2)
        };

        if state.machine.log_enabled() {
            state.machine.push_op(Operation::Beq(imm), opcode, ops);
        }

        if cell_1 == cell_2 {
            state.machine.set_pc((ops.a() as u32) / BYTES_PER_INSTR);
        } else {
            state.machine.step_pc();
        }
    }
}

impl<M, F> Instruction<M, F> for BneInstruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = BNE;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;
        let mut imm: Option<Word<u8>> = None;
        let read_addr_1 = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        let cell_1 = M::read(state, clk, read_addr_1);
        let cell_2 = if ops.is_imm() == 1 {
            let c = (ops.c() as u32).into();
            imm = Some(c);
            c
        } else {
            let read_addr_2 = (state.machine.cpu().fp as i32 + ops.c()) as u32;
            M::read(state, clk, read_addr_2)
        };
        if state.machine.log_enabled() {
            state.machine.push_op(Operation::Bne(imm), opcode, ops);
        }
        if cell_1 != cell_2 {
            state.machine.set_pc((ops.a() as u32) / BYTES_PER_INSTR);
        } else {
            state.machine.step_pc();
        }
    }
}

impl<M, F> Instruction<M, F> for Imm32Instruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = IMM32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let clk = state.machine.cpu().clock;
        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        let value =
            Word::from_components_le([ops.b(), ops.c(), ops.d(), ops.e()]).transform(|x| x as u8);
        M::write(state, clk, write_addr, value);
        if state.machine.log_enabled() {
            state
                .machine
                .push_op(Operation::Imm32, <Self as Instruction<M, F>>::OPCODE, ops);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for StopInstruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = STOP;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        // don't change pc
        //state.machine.cpu_mut().pc = state.machine.cpu().pc;
        if state.machine.log_enabled() {
            state
                .machine
                .push_op(Operation::Stop, <Self as Instruction<M, F>>::OPCODE, ops);
        }
    }
}

impl<M, F> Instruction<M, F> for FailInstruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = FAIL;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        if state.machine.log_enabled() {
            state
                .machine
                .push_op(Operation::Fail, <Self as Instruction<M, F>>::OPCODE, ops);
        }
    }
}

impl<M, F> Instruction<M, F> for MemcpyInstruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = MEMCPY;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;
        let fp = state.machine.cpu().fp;

        let size_addr = (fp as i32 + ops.c()) as u32;
        assert!(
            is_mul_4(size_addr),
            "MEMCPY: Size address location is not a multiple of 4!"
        );
        let mut size: u32 = M::read(state, clk, size_addr).into();

        let dst_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        assert!(
            is_mul_4(dst_addr),
            "MEMCPY: Dest address location is not a multiple of 4!"
        );
        let dst: u32 = M::read(state, clk, dst_addr).into();

        let src_addr = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        assert!(
            is_mul_4(src_addr),
            "MEMCPY: Src address location is not a multiple of 4!"
        );
        let src: u32 = M::read(state, clk, src_addr).into();

        let mut d8 = dst;
        let mut s8 = src;

        // The following implementation for mempcy for nonoverlapping memory regions
        // is equivalent to:
        //
        // ```
        // while size != 0 {
        //     write_byte(d8, read_byte(s8, state, clk), state, clk);
        //     d8 = d8 + 1;
        //     s8 = s8 + 1;
        //     size = size - 1;
        // }
        // ```
        //
        // The above succint implementation does not optimize for the number of memory accesses though.
        //
        // The actual implementation optimizes for the number of memory accesses by
        // reading and writing to a given memory cell only once. The only exception to this
        // is writing initial (at most 3) bytes and final (at most 3) bytes. In principle this
        // could be optimized as well to avoid accessing the memory cells at region boundaries more than once.

        if size < 4 {
            while size != 0 {
                write_byte(d8, read_byte(s8, state, clk), state, clk);
                d8 = d8 + 1;
                s8 = s8 + 1;
                size = size - 1;
            }
        } else {
            let mut align: u32 = dst & 3;
            if align != 0 {
                align = 4 - align;
                size = size - align;
                while align != 0 {
                    write_byte(d8, read_byte(s8, state, clk), state, clk);
                    d8 = d8 + 1;
                    s8 = s8 + 1;
                    align = align - 1;
                }
            }

            let mut d32: u32 = d8;

            let src_misalignment: u32 = s8 & 3;
            let mut s32: u32 = s8 - src_misalignment;

            let mut words: u32 = size >> 2;

            if src_misalignment == 0 {
                while words != 0 {
                    let val = M::read(state, clk, s32);
                    M::write(state, clk, d32, val);
                    s32 = s32 + 4;
                    d32 = d32 + 4;
                    words = words - 1;
                }
            } else {
                let mut prev_word: u32 = M::read(state, clk, s32).into();
                s32 = s32 + 4;
                let shift_right: u32 = src_misalignment * 8;
                let shift_left: u32 = 32 - shift_right;

                while words != 0 {
                    let next_word: u32 = M::read(state, clk, s32).into();
                    s32 = s32 + 4;

                    let word: u32 =
                        ((prev_word >> shift_right) as u32) | ((next_word << shift_left) as u32);

                    M::write(state, clk, d32, Word::from(word));
                    d32 = d32 + 4;

                    prev_word = next_word;

                    words = words - 1;
                }
            }

            d8 = d32;
            s8 = s32 - (4 - src_misalignment) % 4;

            let mut remaining = size & 3;
            while remaining != 0 {
                write_byte(d8, read_byte(s8, state, clk), state, clk);
                d8 = d8 + 1;
                s8 = s8 + 1;
                remaining = remaining - 1;
            }
        }

        if state.machine.log_enabled() {
            state.machine.push_op(Operation::Memcpy, opcode, ops);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for LoadFpInstruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = LOADFP;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let clk = state.machine.cpu().clock;
        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        let value = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        M::write(state, clk, write_addr, value.into());
        if state.machine.log_enabled() {
            state
                .machine
                .push_op(Operation::LoadFp, <Self as Instruction<M, F>>::OPCODE, ops);
        }

        state.machine.step_pc();
    }
}

impl CpuChip {
    pub fn push_bus_op(&mut self, imm: Option<Word<u8>>, opcode: u32, operands: Operands<i32>) {
        self.push_op(Operation::Bus(imm), opcode, operands);
    }

    pub fn push_left_imm_bus_op(
        &mut self,
        imm: Option<Word<u8>>,
        opcode: u32,
        operands: Operands<i32>,
    ) {
        self.push_op(Operation::BusLeftImm(imm), opcode, operands);
    }

    pub fn push_op(&mut self, op: Operation, opcode: u32, operands: Operands<i32>) {
        self.operations.push(op);
        self.instructions.push(InstructionWord { opcode, operands });
        self.save_register_state();
        self.clock += 1;
    }

    pub fn push_pointer_op(&mut self, opcode: u32, operands: Operands<i32>) {
        self.push_op(Operation::Pointer, opcode, operands);
    }

    pub fn set_initial_register_values(&mut self, reg: Registers) {
        let Registers {
            pc: pc_init,
            fp: fp_init,
        } = reg;
        self.pc_init = pc_init;
        self.fp_init = fp_init;
        self.pc = pc_init;
        self.fp = fp_init;
    }

    fn save_register_state(&mut self) {
        let registers = Registers {
            pc: self.pc,
            fp: self.fp,
        };
        self.registers.push(registers);
    }
}
