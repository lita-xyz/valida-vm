//#![no_std]

extern crate alloc;

use alloc::{format, string::String, vec, vec::Vec};
use columns::{RangeLookupCols, NUM_RANGE_LOOKUP_COLS};
use core::borrow::{Borrow, BorrowMut};
use valida_lookups::{
    LookupChip, LookupTable, LookupType, MachineWithLookupChip, MultiLookupTableWrapper,
};
use valida_machine::{Machine, Word};
use valida_util::pad_to_power_of_two;

use p3_field::{Field, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

pub mod columns;

#[derive(Default)]
pub struct RangeTable<const MAX: u32>;

impl<F: Field, const MAX: u32> LookupTable<F> for RangeTable<MAX> {
    type M<'a> = RowMajorMatrix<F>;

    fn name(&self) -> String {
        format!("RangeTable<{}>", MAX)
    }

    fn lookup_type(&self) -> LookupType {
        LookupType::Preprocessed
    }

    fn lookup_matrix(&self, verbose: bool) -> (RowMajorMatrix<F>, Option<Vec<String>>) {
        let rows = (0..=MAX)
            .into_par_iter()
            .map(|n| {
                let mut row = [F::zero(); NUM_RANGE_LOOKUP_COLS];
                let cols: &mut RangeLookupCols<F> = row[..].borrow_mut();
                cols.counter = F::from_canonical_u32(n);
                row
            })
            .collect::<Vec<_>>();

        let log = if verbose {
            let mut log_prints = vec![String::new(); MAX as usize];
            rows.iter().enumerate().for_each(|(index, row)| {
                let cols: &RangeLookupCols<F> = (row[..]).borrow();
                log_prints[index].push_str(&format!("Range Table {}: {:?}", index, cols));
            });
            Some(log_prints)
        } else {
            None
        };

        let mut values = rows.into_iter().flatten().collect::<Vec<_>>();

        pad_to_power_of_two::<NUM_RANGE_LOOKUP_COLS, F>(&mut values);
        (RowMajorMatrix::new(values, NUM_RANGE_LOOKUP_COLS), log)
    }

    fn width(&self) -> usize {
        NUM_RANGE_LOOKUP_COLS
    }

    fn height(&self) -> usize {
        (MAX + 1).next_power_of_two() as usize
    }
}

pub type RangeCheckerChip<F, const MAX: u32> =
    LookupChip<MultiLookupTableWrapper<RangeTable<MAX>>, F>;

pub trait MachineWithRangeChip<F: PrimeField, const MAX: u32>: Machine<F> {
    fn range(&self) -> &RangeCheckerChip<F, MAX>;
    fn range_mut(&mut self) -> &mut RangeCheckerChip<F, MAX>;

    /// Record a single field element in the range check counter
    fn range_check_scalar(&mut self, value: F);

    /// Record a single 32-bit or smaller unsigned integer in the range check counter
    fn range_check<I: Into<u32>>(&mut self, value: I) {
        if self.log_enabled() {
            self.range_check_scalar(F::from_canonical_u32(value.into()))
        };
    }

    /// Record the components of the word in the range check counter
    fn range_check_word<I: Into<u32> + std::fmt::Debug>(&mut self, value: Word<I>) {
        if self.log_enabled() {
            for v in value.into_iter_le() {
                self.range_check(v);
            }
        }
    }
}

impl<F, M, const MAX: u32> MachineWithRangeChip<F, MAX> for M
where
    F: PrimeField,
    M: MachineWithLookupChip<F, RangeTable<MAX>>,
{
    fn range(&self) -> &RangeCheckerChip<F, MAX> {
        self.lookup_chip() as &RangeCheckerChip<F, MAX>
    }
    fn range_mut(&mut self) -> &mut RangeCheckerChip<F, MAX> {
        self.lookup_chip_mut() as &mut RangeCheckerChip<F, MAX>
    }

    fn range_check_scalar(&mut self, value: F) {
        let log = self.log_enabled();
        self.scalar_lookup(value, log);
    }
}
