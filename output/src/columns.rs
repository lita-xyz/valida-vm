use core::borrow::{Borrow, BorrowMut};
use core::mem::{size_of, transmute};
use valida_derive::AlignedBorrow;
use valida_machine::Word;
use valida_util::indices_arr;

#[derive(AlignedBorrow, Default, Debug)]
pub struct OutputCols<T> {
    /// CPU clock
    pub clk: T,

    /// clk' - clk, in bytes
    pub diff: Word<T>,

    pub is_real: T,
}

#[derive(AlignedBorrow, Default, Debug)]
pub struct PublicOutputCols<T> {
    /// Output byte value
    pub value: T,
}

pub const NUM_OUTPUT_COLS: usize = size_of::<OutputCols<u8>>();
pub const OUTPUT_COL_MAP: OutputCols<usize> = make_col_map();
pub const NUM_PUBLIC_OUTPUT_COLS: usize = size_of::<PublicOutputCols<u8>>();
pub const PUBLIC_OUTPUT_COL_MAP: PublicOutputCols<usize> = make_public_col_map();

const fn make_col_map() -> OutputCols<usize> {
    let indices_arr = indices_arr::<NUM_OUTPUT_COLS>();
    unsafe { transmute::<[usize; NUM_OUTPUT_COLS], OutputCols<usize>>(indices_arr) }
}
const fn make_public_col_map() -> PublicOutputCols<usize> {
    let indices_arr = indices_arr::<NUM_PUBLIC_OUTPUT_COLS>();
    unsafe { transmute::<[usize; NUM_PUBLIC_OUTPUT_COLS], PublicOutputCols<usize>>(indices_arr) }
}
