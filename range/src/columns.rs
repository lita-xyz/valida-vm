use core::borrow::{Borrow, BorrowMut};
use core::mem::{size_of, transmute};
use valida_derive::AlignedBorrow;
use valida_util::indices_arr;

#[derive(Default, Debug, AlignedBorrow)]
pub struct RangeLookupCols<T> {
    pub counter: T, // The numbers 0, 1,..., MAX-1
}

pub const NUM_RANGE_LOOKUP_COLS: usize = size_of::<RangeLookupCols<u8>>();
pub const RANGE_LOOKUP_COL_MAP: RangeLookupCols<usize> = make_lookup_col_map();

const fn make_lookup_col_map() -> RangeLookupCols<usize> {
    let indices_arr = indices_arr::<NUM_RANGE_LOOKUP_COLS>();
    unsafe { transmute::<[usize; NUM_RANGE_LOOKUP_COLS], RangeLookupCols<usize>>(indices_arr) }
}
