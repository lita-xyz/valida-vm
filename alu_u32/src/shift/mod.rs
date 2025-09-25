extern crate alloc;

use valida_bytes::{
    byte_send_shift_by_flags_and_remainder, byte_send_shifts, byte_send_simple, ByteOperation,
    BytesTable, MachineWithBytesChip,
};

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use columns::{Shift32Cols, COL_MAP, NUM_SHIFT_COLS};
use core::borrow::{Borrow, BorrowMut};
use core::iter;
use spin::Mutex;
use valida_bus::{MachineWithBytesBus, MachineWithGeneralBus, MachineWithRangeBus8};
use valida_cpu::MachineWithCpuChip;
use valida_machine::{
    instructions, Chip, ChipTraceHeight, ChipWithPersistence, Instruction, Interaction, Operands,
    PublicTrace, RunningMachine, Sra, Word, MEMORY_CELL_BYTES,
};
use valida_opcodes::{map_opcode_to_field_value, SHL32, SHR32, SRA32};

use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use valida_machine::StarkConfig;

use valida_memory_footprint::MemoryFootprint;

pub mod columns;
pub mod stark;

#[derive(Clone)]
pub enum Operation {
    Shl32(Word<u8>, Word<u8>, Word<u8>), // (dst, src, shift)
    Shr32(Word<u8>, Word<u8>, Word<u8>), // ''
    Sra32(Word<u8>, Word<u8>, Word<u8>), // ''
}

impl MemoryFootprint for Operation {
    fn memory_footprint(&self) -> usize {
        match self {
            Operation::Shl32(a, b, c) => 3 * b.memory_footprint(),
            Operation::Shr32(a, b, c) => 3 * b.memory_footprint(),
            Operation::Sra32(a, b, c) => 3 * b.memory_footprint(),
        }
    }
}

#[derive(Default)]
pub struct Shift32Chip {
    pub operations: Vec<Operation>,
}

impl MemoryFootprint for Shift32Chip {
    fn memory_footprint(&self) -> usize {
        self.operations.memory_footprint()
    }
}

impl ChipTraceHeight for Shift32Chip {
    fn trace_height(&self) -> u32 {
        self.operations.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for Shift32Chip
where
    M: MachineWithGeneralBus<SC::Val>
        + MachineWithBytesBus<SC::Val>
        + MachineWithRangeBus8<SC::Val>,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "Shift32".to_string()
    }

    fn generate_main_trace(
        &self,
        _machine: &M,
        verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>) {
        let num_ops = self.operations.len();
        let num_padded_ops = num_ops.next_power_of_two();
        let values = Mutex::new(vec![SC::Val::zero(); num_padded_ops * NUM_SHIFT_COLS]);

        // Encode the real operations
        self.operations.par_iter().enumerate().for_each(|(i, op)| {
            let mut values = values.lock();
            let row = &mut values[i * NUM_SHIFT_COLS..(i + 1) * NUM_SHIFT_COLS];
            let cols: &mut Shift32Cols<SC::Val> = row.borrow_mut();
            self.op_to_row(op, cols);
        });

        let log = if verbose {
            let mut log_prints = Vec::with_capacity(num_ops);
            for (i, row) in values
                .lock()
                .chunks(NUM_SHIFT_COLS)
                .take(num_ops)
                .enumerate()
            {
                let cols: &Shift32Cols<SC::Val> = row.borrow();
                log_prints.push(format!("Shift32 row {}: {:?}", i, cols));
            }
            Some(log_prints)
        } else {
            None
        };

        (
            Some(RowMajorMatrix::new(values.into_inner(), NUM_SHIFT_COLS)),
            log,
        )
    }

    fn global_sends(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let byte_shifted_word_cols = COL_MAP
            .byte_shifted_word
            .transform(VirtualPairCol::single_main);
        let left_shifted_byte_cols = COL_MAP
            .bit_shifted_bytes_left
            .transform(VirtualPairCol::single_main);
        let right_shifted_byte_cols = COL_MAP
            .bit_shifted_bytes_right
            .transform(VirtualPairCol::single_main);
        let is_real =
            VirtualPairCol::sum_main(vec![COL_MAP.is_shl, COL_MAP.is_shr, COL_MAP.is_sra]);

        // The `byte_send_shifts` lookup send here computes (byte << left_shift_by_mod_8, byte >> (8 - left_shift_by_mod_8)),
        // with a shift of 8 in either direction yielding zero.
        let bit_shifted_byte_sends = byte_shifted_word_cols
            .into_iter_le()
            .zip(left_shifted_byte_cols.into_iter_le())
            .zip(right_shifted_byte_cols.into_iter_le())
            .map(
                |((byte_col, bit_shifted_byte_col_left), bit_shifted_byte_col_right)| {
                    byte_send_shifts(
                        machine,
                        byte_col,
                        bit_shifted_byte_col_left,
                        bit_shifted_byte_col_right,
                        is_real.clone(),
                        COL_MAP.left_shift_by_mod_8,
                    )
                },
            )
            .collect::<Vec<_>>();

        let sign_extension_byte_col = VirtualPairCol::single_main(COL_MAP.sign_extension_byte);
        let left_shifted_sign_extension_byte_col =
            VirtualPairCol::single_main(COL_MAP.sign_extension_byte_overflow);
        let is_sra_col = VirtualPairCol::single_main(COL_MAP.is_sra);
        let right_shifted_sign_extension_byte_col = VirtualPairCol::new_main(
            vec![
                (COL_MAP.sign_extension_byte, SC::Val::one()),
                (COL_MAP.sign_extension_byte_overflow, -SC::Val::one()),
            ],
            SC::Val::zero(),
        );
        let sign_extension_byte_overflow_send = byte_send_shifts(
            machine,
            sign_extension_byte_col,
            left_shifted_sign_extension_byte_col,
            right_shifted_sign_extension_byte_col,
            is_sra_col.clone(),
            COL_MAP.left_shift_by_mod_8,
        );

        let input_2_col = VirtualPairCol::single_main(*COL_MAP.input_2.index_le(0));
        let result_flag_cols: [_; MEMORY_CELL_BYTES] = [
            COL_MAP.shift_by_zero_full_bytes,
            COL_MAP.shift_by_one_full_byte,
            COL_MAP.shift_by_two_full_bytes,
            COL_MAP.shift_by_three_full_bytes,
        ]
        .into_iter()
        .map(VirtualPairCol::single_main)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
        let is_shl = VirtualPairCol::single_main(COL_MAP.is_shl);
        let remainder_col_left = VirtualPairCol::single_main(COL_MAP.left_shift_by_mod_8);
        let shift_by_send_left = byte_send_shift_by_flags_and_remainder(
            machine,
            input_2_col.clone(),
            result_flag_cols.clone(),
            remainder_col_left,
            is_shl,
        );
        let is_right_shift = VirtualPairCol::sum_main(vec![COL_MAP.is_shr, COL_MAP.is_sra]);
        let remainder_col_right = VirtualPairCol::new_main(
            vec![(COL_MAP.left_shift_by_mod_8, -SC::Val::one())],
            SC::Val::from_canonical_u8(8),
        );
        let shift_by_send_right = byte_send_shift_by_flags_and_remainder(
            machine,
            input_2_col,
            result_flag_cols,
            remainder_col_right,
            is_right_shift,
        );

        let input_1_top_byte_col = VirtualPairCol::single_main(*COL_MAP.input_1.index_be(0));
        let sign_bit_col = VirtualPairCol::single_main(COL_MAP.sign_1);

        let sign_bit_send = byte_send_simple(
            machine,
            input_1_top_byte_col,
            Some(sign_bit_col),
            is_sra_col,
            ByteOperation::MostSignificantBit,
        );

        bit_shifted_byte_sends
            .into_iter()
            .chain(iter::once(sign_extension_byte_overflow_send))
            .chain(iter::once(shift_by_send_left))
            .chain(iter::once(shift_by_send_right))
            .chain(iter::once(sign_bit_send))
            .collect()
    }

    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let opcode = VirtualPairCol::new_main(
            vec![
                (COL_MAP.is_shl, map_opcode_to_field_value(SHL32)),
                (COL_MAP.is_shr, map_opcode_to_field_value(SHR32)),
                (COL_MAP.is_sra, map_opcode_to_field_value(SRA32)),
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
            VirtualPairCol::sum_main(vec![COL_MAP.is_shl, COL_MAP.is_shr, COL_MAP.is_sra]);

        let receive = Interaction {
            fields,
            count: is_real,
            argument_index: machine.general_bus(),
        };
        vec![receive]
    }
}

impl<M, SC> ChipWithPersistence<M, SC> for Shift32Chip
where
    M: MachineWithGeneralBus<SC::Val>
        + MachineWithBytesBus<SC::Val>
        + MachineWithRangeBus8<SC::Val>,
    SC: StarkConfig,
{
}
impl Shift32Chip {
    fn op_to_row<F>(&self, op: &Operation, cols: &mut Shift32Cols<F>)
    where
        F: PrimeField,
    {
        match op {
            Operation::Shr32(a, b, c) => {
                cols.is_shr = F::one();
                self.set_cols(cols, a, b, c);
            }
            Operation::Sra32(a, b, c) => {
                cols.is_sra = F::one();
                self.set_cols(cols, a, b, c);
            }
            Operation::Shl32(a, b, c) => {
                cols.is_shl = F::one();
                self.set_cols(cols, a, b, c);
            }
        }
    }

    fn set_cols<F>(&self, cols: &mut Shift32Cols<F>, a: &Word<u8>, b: &Word<u8>, c: &Word<u8>)
    where
        F: PrimeField,
    {
        // Set the input columns
        cols.input_1 = b.transform(F::from_canonical_u8);
        cols.input_2 = c.transform(F::from_canonical_u8);
        cols.output = a.transform(F::from_canonical_u8);

        let sign_b = (b.index_be(0) & 0x80) != 0;
        if cols.is_sra == F::one() {
            cols.sign_1 = F::from_bool(sign_b);
        }

        let sign_extension_byte = if cols.is_sra == F::one() {
            u8::MAX * sign_b as u8
        } else {
            0
        };

        cols.sign_extension_byte = F::from_canonical_u8(sign_extension_byte);

        let shift_by = c.index_le(0) & 0x1f;
        let shift_by_full_bytes = shift_by / 8;
        debug_assert!(shift_by_full_bytes < 4);

        let left_shift_by_mod_8 = if cols.is_shl == F::one() {
            shift_by % 8
        } else {
            8 - (shift_by % 8)
        };
        cols.left_shift_by_mod_8 = F::from_canonical_u8(left_shift_by_mod_8);

        match shift_by_full_bytes {
            0 => {
                cols.shift_by_zero_full_bytes = F::one();
            }
            1 => {
                cols.shift_by_one_full_byte = F::one();
            }
            2 => {
                cols.shift_by_two_full_bytes = F::one();
            }
            3 => {
                cols.shift_by_three_full_bytes = F::one();
            }
            _ => (),
        }

        let byte_shifted_word = if cols.is_shl == F::one() {
            Word::from_components_be(
                b.into_iter_be()
                    .skip(shift_by_full_bytes as usize)
                    .chain(iter::repeat(sign_extension_byte).take(shift_by_full_bytes as usize))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            )
        } else {
            Word::from_components_le(
                b.into_iter_le()
                    .skip(shift_by_full_bytes as usize)
                    .chain(iter::repeat(sign_extension_byte).take(shift_by_full_bytes as usize))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            )
        };
        cols.byte_shifted_word = byte_shifted_word.transform(F::from_canonical_u8);

        let (left_shifted_bytes, right_shifted_bytes) = byte_shifted_word
            .iter_le()
            .map(|&byte| {
                let res = BytesTable::byte_op(
                    byte,
                    ByteOperation::LeftShiftAndRightShift(left_shift_by_mod_8),
                );
                debug_assert_eq!(res.len(), 2);
                (res[0], res[1])
            })
            .collect::<(Vec<_>, Vec<_>)>();

        cols.bit_shifted_bytes_left =
            Word::from_components_le(left_shifted_bytes.try_into().unwrap())
                .transform(F::from_canonical_u8);
        cols.bit_shifted_bytes_right =
            Word::from_components_le(right_shifted_bytes.try_into().unwrap())
                .transform(F::from_canonical_u8);

        let (sign_extension_byte_overflow, _) = {
            let res = BytesTable::byte_op(
                sign_extension_byte,
                ByteOperation::LeftShiftAndRightShift(left_shift_by_mod_8),
            );
            debug_assert_eq!(res.len(), 2);
            (res[0], res[1])
        };

        cols.sign_extension_byte_overflow = F::from_canonical_u8(sign_extension_byte_overflow);
    }
}

pub trait MachineWithShift32Chip<F: PrimeField32>:
    MachineWithCpuChip<F> + MachineWithBytesChip<F>
{
    fn shift_u32(&self) -> &Shift32Chip;
    fn shift_u32_mut(&mut self) -> &mut Shift32Chip;
}

instructions!(Shl32Instruction, Shr32Instruction, Sra32Instruction);

impl<M, F> Instruction<M, F> for Shl32Instruction
where
    M: MachineWithShift32Chip<F> + MachineWithBytesChip<F>,
    F: PrimeField32,
{
    const OPCODE: u32 = SHL32;

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

        // Write the shifted value to memory
        let a = b << c;
        M::write(state, clk, write_addr, a);
        // note: we do NOT need to record the range check to
        // 'a' here, as we record this range check below,
        // as `prod_remainder`.

        if state.machine.log_enabled() {
            let res = state
                .machine
                .check_byte_op(*c.index_le(0), ByteOperation::ShiftByFlagsAndRemainder);
            debug_assert_eq!(res.len(), MEMORY_CELL_BYTES + 1);
            let shift_by = c.index_le(0) % (8 * MEMORY_CELL_BYTES as u8);
            let shift_by_mod_8 = c.index_le(0) % 8;
            debug_assert_eq!(res[MEMORY_CELL_BYTES], shift_by_mod_8);
            let shift_by_full_bytes = shift_by / 8;
            for (i, flag_column) in res.iter().take(MEMORY_CELL_BYTES).enumerate() {
                debug_assert_eq!(
                    *flag_column,
                    if i == shift_by_full_bytes as usize {
                        1
                    } else {
                        0
                    }
                );
            }
            for shifted_byte in b
                .iter_be()
                .skip(shift_by_full_bytes as usize)
                .chain(iter::repeat(&0).take(shift_by_full_bytes as usize))
            {
                let _ = state.machine.check_byte_op(
                    *shifted_byte,
                    ByteOperation::LeftShiftAndRightShift(shift_by_mod_8),
                );
            }
            state
                .machine
                .shift_u32_mut()
                .operations
                .push(Operation::Shl32(a, b, c));
            state.machine.push_bus_op(imm, opcode, ops);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for Shr32Instruction
where
    M: MachineWithShift32Chip<F> + MachineWithBytesChip<F>,
    F: PrimeField32,
{
    const OPCODE: u32 = SHR32;

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

        // Write the shifted value to memory
        let a = b >> c;
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            let res = state
                .machine
                .check_byte_op(*c.index_le(0), ByteOperation::ShiftByFlagsAndRemainder);
            debug_assert_eq!(res.len(), MEMORY_CELL_BYTES + 1);
            let shift_by = c.index_le(0) % (8 * MEMORY_CELL_BYTES as u8);
            let shift_by_mod_8 = c.index_le(0) % 8;
            let left_shift_by_mod_8 = 8 - shift_by_mod_8;
            debug_assert_eq!(res[MEMORY_CELL_BYTES], shift_by_mod_8);
            let shift_by_full_bytes = shift_by / 8;
            for (i, flag_column) in res.iter().take(MEMORY_CELL_BYTES).enumerate() {
                debug_assert_eq!(
                    *flag_column,
                    if i == shift_by_full_bytes as usize {
                        1
                    } else {
                        0
                    }
                );
            }
            for shifted_byte in b
                .iter_le()
                .skip(shift_by_full_bytes as usize)
                .chain(iter::repeat(&0).take(shift_by_full_bytes as usize))
            {
                let _ = state.machine.check_byte_op(
                    *shifted_byte,
                    ByteOperation::LeftShiftAndRightShift(left_shift_by_mod_8),
                );
            }
            state
                .machine
                .shift_u32_mut()
                .operations
                .push(Operation::Shr32(a, b, c));

            state.machine.push_bus_op(imm, opcode, ops);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for Sra32Instruction
where
    M: MachineWithShift32Chip<F> + MachineWithBytesChip<F>,
    F: PrimeField32,
{
    const OPCODE: u32 = SRA32;

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

        // Write the shifted value to memory
        let a = b.sra(c);
        M::write(state, clk, write_addr, a);
        if state.machine.log_enabled() {
            let res = state
                .machine
                .check_byte_op(*c.index_le(0), ByteOperation::ShiftByFlagsAndRemainder);
            debug_assert_eq!(res.len(), MEMORY_CELL_BYTES + 1);
            let shift_by = c.index_le(0) % (8 * MEMORY_CELL_BYTES as u8);
            let shift_by_mod_8 = c.index_le(0) % 8;
            let left_shift_by_mod_8 = 8 - shift_by_mod_8;
            debug_assert_eq!(res[MEMORY_CELL_BYTES], shift_by_mod_8);
            let shift_by_full_bytes = shift_by / 8;
            for (i, flag_column) in res.iter().take(MEMORY_CELL_BYTES).enumerate() {
                debug_assert_eq!(
                    *flag_column,
                    if i == shift_by_full_bytes as usize {
                        1
                    } else {
                        0
                    }
                );
            }

            let res = state
                .machine
                .check_byte_op(*b.index_be(0), ByteOperation::MostSignificantBit);
            debug_assert_eq!(res.len(), 1);
            let sign_bit = res[0];
            let sign_extension_byte = sign_bit * 0xff;

            for shifted_byte in b
                .iter_le()
                .skip(shift_by_full_bytes as usize)
                .chain(iter::repeat(&sign_extension_byte).take(shift_by_full_bytes as usize))
            {
                let _ = state.machine.check_byte_op(
                    *shifted_byte,
                    ByteOperation::LeftShiftAndRightShift(left_shift_by_mod_8),
                );
            }
            let _ = state.machine.check_byte_op(
                sign_extension_byte,
                ByteOperation::LeftShiftAndRightShift(left_shift_by_mod_8),
            );

            state
                .machine
                .shift_u32_mut()
                .operations
                .push(Operation::Sra32(a, b, c));

            state.machine.push_bus_op(imm, opcode, ops);
        }

        state.machine.step_pc();
    }
}
