use core::borrow::{Borrow, BorrowMut};
use core::marker::PhantomData;
use core::mem::{size_of, transmute};
use valida_derive::AlignedBorrow;
use valida_machine::Word;
use valida_util::indices_arr;

#[derive(AlignedBorrow, Default, Debug)]
pub struct StaticDataLookupCols<T> {
    /// Memory address
    pub addr: T,

    /// Memory cell
    pub value: Word<T>,

    /// Whether this row represents a real (address, value) pair
    pub is_real: T,
}

pub struct StaticDataPrivateCols<T> {
    pub _phantom_data: PhantomData<T>,
}

pub const NUM_STATIC_DATA_LOOKUP_COLS: usize = size_of::<StaticDataLookupCols<u8>>();
pub const STATIC_DATA_LOOKUP_COL_MAP: StaticDataLookupCols<usize> = make_lookup_col_map();

pub const NUM_STATIC_DATA_PRIVATE_COLS: usize = size_of::<StaticDataPrivateCols<u8>>();
pub const STATIC_DATA_PRIVATE_COL_MAP: StaticDataPrivateCols<usize> = make_private_col_map();

const fn make_lookup_col_map() -> StaticDataLookupCols<usize> {
    let indices_arr = indices_arr::<NUM_STATIC_DATA_LOOKUP_COLS>();
    unsafe {
        transmute::<[usize; NUM_STATIC_DATA_LOOKUP_COLS], StaticDataLookupCols<usize>>(indices_arr)
    }
}

const fn make_private_col_map() -> StaticDataPrivateCols<usize> {
    let indices_arr = indices_arr::<NUM_STATIC_DATA_PRIVATE_COLS>();
    unsafe {
        transmute::<[usize; NUM_STATIC_DATA_PRIVATE_COLS], StaticDataPrivateCols<usize>>(
            indices_arr,
        )
    }
}
