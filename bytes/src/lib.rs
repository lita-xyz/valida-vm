#![no_std]

extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::iter;
use p3_air::VirtualPairCol;

use valida_bus::{MachineWithBytesBus, MachineWithRangeBus8};
use valida_lookups::{LookupChip, MachineWithMultiLookupChip, MultiLookupTable};

use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use smallvec::SmallVec;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount, EnumIter};
use valida_machine::{Interaction, Word, MEMORY_CELL_BYTES, SMALLVEC_SIZE};
use valida_memory_footprint::MemoryFootprint;

#[derive(EnumIter, EnumCount, Copy, Clone, PartialEq, Debug)]
pub enum ByteOperation {
    Range,
    MostSignificantBit,
    LeastSignificantBit,
    SixBits,
    FiveBits,
    FourBits,
    ThreeBits,
    TwoBits,
    Negation,
    // Used in comparing 32-bit values to half of the baby bear field size.
    Under0x3C,
    // Computes `byte` mod 32, returns `byte` mod `8` and flags specifying (byte % 32) / 8.
    ShiftByFlagsAndRemainder,
    // The u8 provides the 'shift by' amount, and must be in [0, 8]
    LeftShiftAndRightShift(u8),
}

fn num_bytes_cols() -> usize {
    1 + // The byte itself
    ByteOperation::iter()
    .take(ByteOperation::COUNT - 1)
    .map(output_length_of_byte_op)
    .sum::<usize>()
    // There are 9 variants of the LeftShiftAndRightShift operation
    + 9 * output_length_of_byte_op(ByteOperation::LeftShiftAndRightShift(0))
}

pub fn output_length_of_byte_op(op: ByteOperation) -> usize {
    match op {
        ByteOperation::Range => 0,
        ByteOperation::LeftShiftAndRightShift(_) => 2,
        ByteOperation::ShiftByFlagsAndRemainder => MEMORY_CELL_BYTES + 1,
        _ => 1,
    }
}

pub fn receive_index_of_byte_op(op: ByteOperation) -> usize {
    match op {
        ByteOperation::LeftShiftAndRightShift(shift_by) => {
            assert!(shift_by <= 8);
            ByteOperation::COUNT + (shift_by as usize) - 1
        }
        _ => ByteOperation::iter().position(|x| x == op).unwrap(),
    }
}

fn is_simple_op(op: ByteOperation) -> bool {
    !matches!(
        op,
        ByteOperation::ShiftByFlagsAndRemainder | ByteOperation::LeftShiftAndRightShift(_)
    )
}

pub fn starting_column_index_of_byte_op(op: ByteOperation) -> usize {
    if is_simple_op(op) {
        receive_index_of_byte_op(op)
    } else {
        let num_simple_ops = ByteOperation::iter().filter(|op| is_simple_op(*op)).count();
        match op {
            ByteOperation::ShiftByFlagsAndRemainder => num_simple_ops,
            ByteOperation::LeftShiftAndRightShift(shift_by) => {
                assert!(shift_by <= 8);
                num_simple_ops
                    + output_length_of_byte_op(ByteOperation::ShiftByFlagsAndRemainder)
                    + ((shift_by as usize) * output_length_of_byte_op(op))
            }
            _ => unreachable!(),
        }
    }
}

pub fn byte_send_simple<F: PrimeField, M: MachineWithBytesBus<F> + MachineWithRangeBus8<F>>(
    machine: &M,
    byte: VirtualPairCol<F>,
    result: Option<VirtualPairCol<F>>,
    is_real: VirtualPairCol<F>,
    op: ByteOperation,
) -> Interaction<F> {
    let opcode = VirtualPairCol::constant(F::from_canonical_usize(receive_index_of_byte_op(op)));
    match op {
        ByteOperation::Range => Interaction {
            fields: vec![opcode, byte],
            count: is_real,
            argument_index: machine.range_bus_8(),
        },
        ByteOperation::LeftShiftAndRightShift(_shift_by) => {
            panic!("LeftShiftAndRightShift is not a simple byte_send operation");
        }
        _ => Interaction {
            fields: vec![
                opcode,
                byte,
                result.expect("result must be Some for non-Range ops"),
            ],
            count: is_real,
            argument_index: machine.bytes_bus(),
        },
    }
}

pub fn byte_send_shifts<F: PrimeField, M: MachineWithBytesBus<F>>(
    machine: &M,
    byte: VirtualPairCol<F>,
    result_1: VirtualPairCol<F>,
    result_2: VirtualPairCol<F>,
    is_real: VirtualPairCol<F>,
    shift_by_col: usize,
) -> Interaction<F> {
    let opcode = VirtualPairCol::new_main(
        vec![(shift_by_col, F::one())],
        F::from_canonical_usize(receive_index_of_byte_op(
            ByteOperation::LeftShiftAndRightShift(0),
        )),
    );

    Interaction {
        fields: vec![opcode, byte, result_1, result_2],
        count: is_real,
        argument_index: machine.bytes_bus(),
    }
}

pub fn byte_send_shift_by_flags_and_remainder<F: PrimeField, M: MachineWithBytesBus<F>>(
    machine: &M,
    byte: VirtualPairCol<F>,
    result_flag_cols: [VirtualPairCol<F>; MEMORY_CELL_BYTES],
    remainder_col: VirtualPairCol<F>,
    is_real: VirtualPairCol<F>,
) -> Interaction<F> {
    let opcode = VirtualPairCol::constant(F::from_canonical_usize(receive_index_of_byte_op(
        ByteOperation::ShiftByFlagsAndRemainder,
    )));
    let fields = (vec![opcode, byte])
        .into_iter()
        .chain(result_flag_cols)
        .chain(iter::once(remainder_col))
        .collect::<Vec<_>>();
    Interaction {
        fields,
        count: is_real,
        argument_index: machine.bytes_bus(),
    }
}

pub fn range8_send<F: PrimeField, M: MachineWithRangeBus8<F>>(
    machine: &M,
    byte: VirtualPairCol<F>,
    is_real: VirtualPairCol<F>,
) -> Interaction<F> {
    let opcode = VirtualPairCol::constant(F::from_canonical_usize(receive_index_of_byte_op(
        ByteOperation::Range,
    )));
    Interaction {
        fields: vec![opcode, byte],
        count: is_real,
        argument_index: machine.range_bus_8(),
    }
}

pub fn range8_sends_word<F: PrimeField, M: MachineWithRangeBus8<F>>(
    machine: &M,
    word: Word<VirtualPairCol<F>>,
    is_real: &VirtualPairCol<F>,
) -> Vec<Interaction<F>> {
    word.into_iter_le()
        .map(|byte| range8_send(machine, byte, is_real.clone()))
        .collect()
}

// Checks that a word is at most 0x3C000000, which is 1/2(P - 1)
// where P = 0x78000001 is the modulus of the "baby bear" prime field.
pub fn half_baby_bear_range_sends<
    F: PrimeField,
    M: MachineWithRangeBus8<F> + MachineWithBytesBus<F>,
>(
    machine: &M,
    word: &Word<VirtualPairCol<F>>,
    is_real: VirtualPairCol<F>,
) -> Vec<Interaction<F>> {
    let bottom_sends = word
        .iter_le()
        .take(MEMORY_CELL_BYTES - 1)
        .map(|byte| range8_send(machine, byte.clone(), is_real.clone()))
        .collect::<Vec<_>>();
    let top_send = byte_send_simple(
        machine,
        word.index_be(0).clone(),
        Some(VirtualPairCol::constant(F::one())),
        is_real,
        ByteOperation::Under0x3C,
    );
    bottom_sends
        .into_iter()
        .chain(iter::once(top_send))
        .collect()
}

#[derive(Default)]
pub struct BytesTable;

impl MemoryFootprint for BytesTable {
    fn memory_footprint(&self) -> usize {
        core::mem::size_of::<Self>()
    }
}

impl BytesTable {
    /// This returns a `SmallVec` to avoid heap allocations. With a regular `Vec`, this function
    /// otherwise is the number one hotspot for heap allocations.
    pub fn byte_op(byte: u8, op: ByteOperation) -> SmallVec<[u8; 5]> {
        let mut res = SmallVec::with_capacity(5);
        match op {
            ByteOperation::Range => (),
            ByteOperation::MostSignificantBit => res.push((byte & 0x80) >> 7),
            ByteOperation::LeastSignificantBit => res.push(byte & 0x01),
            ByteOperation::SixBits => res.push(((byte >> 6) == 0) as u8),
            ByteOperation::FiveBits => res.push(((byte >> 5) == 0) as u8),
            ByteOperation::FourBits => res.push(((byte >> 4) == 0) as u8),
            ByteOperation::ThreeBits => res.push(((byte >> 3) == 0) as u8),
            ByteOperation::TwoBits => res.push(((byte >> 2) == 0) as u8),
            ByteOperation::Negation => res.push(!byte),
            ByteOperation::Under0x3C => res.push((byte < 0x3C) as u8),
            ByteOperation::ShiftByFlagsAndRemainder => {
                let shift_by = byte % (MEMORY_CELL_BYTES as u8 * 8);
                let remainder = shift_by % 8;
                res.resize(5, 0);
                res[shift_by as usize / 8] = 1;
                res[4] = remainder;
            }
            ByteOperation::LeftShiftAndRightShift(shift_by) => {
                assert!(shift_by <= 8);
                res.resize(2, 0);
                if shift_by == 0 {
                    res[0] = byte;
                } else if shift_by == 8 {
                    res[1] = byte;
                } else {
                    res[0] = byte << shift_by;
                    res[1] = byte >> (8 - shift_by);
                }
            }
        };
        debug_assert_eq!(res.len(), output_length_of_byte_op(op));
        res
    }
    fn byte_to_row<F: PrimeField>(byte: u8, verbose: bool) -> (Vec<F>, Option<String>) {
        let mut row = vec![F::zero(); num_bytes_cols()];

        for op in ByteOperation::iter() {
            let start = starting_column_index_of_byte_op(op);
            let res = Self::byte_op(byte, op);
            match op {
                ByteOperation::Range => {
                    row[start] = F::from_canonical_u8(byte);
                }
                ByteOperation::LeftShiftAndRightShift(_) => {
                    for shift_by in 0..9 {
                        let res =
                            Self::byte_op(byte, ByteOperation::LeftShiftAndRightShift(shift_by));
                        let output_len = output_length_of_byte_op(op);
                        for (i, output_byte) in res.into_iter().enumerate() {
                            row[start + output_len * (shift_by as usize) + i] =
                                F::from_canonical_u8(output_byte);
                        }
                    }
                }
                _ => {
                    for (i, &x) in res.iter().enumerate() {
                        row[start + i] = F::from_canonical_u8(x);
                    }
                }
            }
        }

        let printed_row = if verbose {
            let mut _printed_row = String::new();
            for op in ByteOperation::iter().filter(|op| is_simple_op(*op)) {
                match op {
                    ByteOperation::Range => {
                        _printed_row.push_str(&format!("byte: {:b} | ", byte));
                    }
                    _ => {
                        let res = Self::byte_op(byte, op);
                        debug_assert_eq!(res.len(), 1);
                        _printed_row.push_str(&format!("{op:?}: {:b} | ", res[0]));
                    }
                };
            }
            _printed_row.push('\n');
            let shift_by_res = Self::byte_op(byte, ByteOperation::ShiftByFlagsAndRemainder);
            debug_assert_eq!(shift_by_res.len(), MEMORY_CELL_BYTES + 1);
            _printed_row.push_str(&format!(
                "{:?}: zero full bytes: {}, one full byte: {}, two full bytes: {}, three full bytes: {}, remainder: {} | ",
                ByteOperation::ShiftByFlagsAndRemainder,
                shift_by_res[0],
                shift_by_res[1],
                shift_by_res[2],
                shift_by_res[3],
                shift_by_res[4],
            ));
            _printed_row.push('\n');
            _printed_row.push_str("LeftShiftAndRightShift: ");
            for shift_by in 0..9 {
                let op = ByteOperation::LeftShiftAndRightShift(shift_by);
                let res = Self::byte_op(byte, op);
                debug_assert_eq!(res.len(), 2);
                _printed_row.push_str(&format!(
                    "shift_by {} bits: (left: {:b}, right: {:b}) | ",
                    shift_by, res[0], res[1]
                ));
            }
            Some(_printed_row)
        } else {
            None
        };

        (row, printed_row)
    }
}

impl<F: PrimeField> MultiLookupTable<F> for BytesTable {
    type M<'a> = RowMajorMatrix<F>;

    fn num_receives(&self) -> usize {
        ByteOperation::iter().filter(|op| is_simple_op(*op)).count()
            + 1 // ShiftByFlagsAndRemainder
            + 9 // LeftShiftAndRightShift
    }

    fn num_preprocessed_columns(&self) -> usize {
        num_bytes_cols()
    }
    fn preprocessed_columns(&self, verbose: bool) -> (Option<Self::M<'_>>, Option<Vec<String>>) {
        let rows = (0..=u8::MAX)
            .into_par_iter()
            .map(|byte| Self::byte_to_row(byte, verbose))
            .collect::<Vec<_>>();
        let log = if verbose {
            let mut log_prints = Vec::with_capacity(rows.len());
            for (_row, printed_row) in rows.iter() {
                log_prints.push(
                    printed_row
                        .clone()
                        .expect("if verbose is true, the printed row must be Some"),
                );
            }
            Some(log_prints)
        } else {
            None
        };
        // no need for padding, since the height is always 256 exactly
        let values = rows
            .into_iter()
            .flat_map(|(row, _printed_row)| row)
            .collect::<Vec<_>>();
        (Some(RowMajorMatrix::new(values, num_bytes_cols())), log)
    }

    fn fields_for_receive(&self, i: usize) -> SmallVec<[VirtualPairCol<F>; SMALLVEC_SIZE]> {
        let range_index = receive_index_of_byte_op(ByteOperation::Range);
        let byte_column = VirtualPairCol::single_preprocessed(range_index);
        let opcode = VirtualPairCol::constant(F::from_canonical_usize(i));
        let op =
            if let Some(op) = ByteOperation::iter().find(|op| receive_index_of_byte_op(*op) == i) {
                op
            } else {
                let shift_by = i - ByteOperation::COUNT + 1;
                ByteOperation::LeftShiftAndRightShift(shift_by as u8)
            };
        let res_cols = (starting_column_index_of_byte_op(op)
            ..starting_column_index_of_byte_op(op) + output_length_of_byte_op(op))
            .map(|col| VirtualPairCol::<F>::single_preprocessed(col))
            .collect::<Vec<_>>();

        vec![opcode, byte_column]
            .into_iter()
            .chain(res_cols)
            .collect()
    }

    fn height(&self) -> usize {
        (u8::MAX as usize + 1).next_power_of_two()
    }

    fn name(&self) -> String {
        "Single Byte Operation Table".to_string()
    }
}

pub type BytesChip<F> = LookupChip<BytesTable, F>;

pub trait MachineWithBytesChip<F: PrimeField> {
    fn bytes(&self) -> &BytesChip<F>;
    fn bytes_mut(&mut self) -> &mut BytesChip<F>;

    /// Lookup the result of a single-byte operation
    fn check_byte_op(&mut self, byte: u8, op: ByteOperation) -> SmallVec<[u8; 5]>;

    /// Check that a word is less than 0x3C000000, which is 1/2(P - 1)
    /// where P = 0x78000001 is the modulus of the "baby bear" prime field.
    fn check_half_baby_bear_range(&mut self, word: &Word<u8>) {
        // bottom bytes are arbitrary
        for byte in word.iter_le().take(MEMORY_CELL_BYTES - 1) {
            self.check_byte_op(*byte, ByteOperation::Range);
        }
        // Most significant byte must be under 0x3C
        let res = self.check_byte_op(*word.index_be(0), ByteOperation::Under0x3C);
        debug_assert_eq!(res.len(), 1);
        if res[0] != 1 {
            panic!(
                "res[0]: {}, byte: {:?}, word: {:?}",
                res[0],
                *word.index_be(0),
                word
            );
        }
    }
}

impl<F, M> MachineWithBytesChip<F> for M
where
    F: PrimeField,
    M: MachineWithMultiLookupChip<F, BytesTable>,
{
    fn bytes(&self) -> &BytesChip<F> {
        self.lookup_chip() as &BytesChip<F>
    }
    fn bytes_mut(&mut self) -> &mut BytesChip<F> {
        self.lookup_chip_mut() as &mut BytesChip<F>
    }

    fn check_byte_op(&mut self, byte: u8, op: ByteOperation) -> SmallVec<[u8; 5]> {
        let receive_index = receive_index_of_byte_op(op);
        let opcode = F::from_canonical_usize(receive_index);
        let log = self.log_enabled();
        let res = BytesTable::byte_op(byte, op);
        debug_assert_eq!(res.len(), output_length_of_byte_op(op));

        // Instead of creating intermediate vectors, build the lookup vector directly
        let mut lookup_fields = SmallVec::with_capacity(2 + res.len());
        lookup_fields.push(opcode);
        lookup_fields.push(F::from_canonical_u8(byte));
        lookup_fields.extend(res.iter().map(|&b| F::from_canonical_u8(b)));

        self.vector_multi_lookup(lookup_fields, log, receive_index);
        res
    }
}

pub trait MachineWithRangeCheckeru8<F: PrimeField> {
    /// Record a single 32-bit or smaller unsigned integer in the range check counter
    fn range_check_byte(&mut self, value: u8);

    /// Record the components of the word in the range check counter
    fn range_check_word<I: Into<Word<u8>>>(&mut self, value: I) {
        for v in value.into().into_iter_le() {
            self.range_check_byte(v);
        }
    }
}

impl<F, M> MachineWithRangeCheckeru8<F> for M
where
    F: PrimeField,
    M: MachineWithBytesChip<F>,
{
    fn range_check_byte(&mut self, byte: u8) {
        self.check_byte_op(byte, ByteOperation::Range);
    }
}
