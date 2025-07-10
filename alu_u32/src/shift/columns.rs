use core::borrow::{Borrow, BorrowMut};
use core::mem::{size_of, transmute};
use valida_derive::AlignedBorrow;
use valida_machine::Word;
use valida_util::indices_arr;

// this is log_2(8 * MEMORY_CELL_BYTES)
pub const SHIFT_BY_BITS: usize = 5;

#[derive(AlignedBorrow, Default, Debug)]
pub struct Shift32Cols<T> {
    pub input_1: Word<T>,
    pub input_2: Word<T>,
    pub output: Word<T>,

    /// The number of bits to shift by rounded down to a multiple of 8, which is always in the range [0, 3]
    pub shift_by_zero_full_bytes: T,
    pub shift_by_one_full_byte: T,
    pub shift_by_two_full_bytes: T,
    pub shift_by_three_full_bytes: T,

    /// The number of bits to shift *left* by modulo 8, which is always in the range [0, 8]
    /// For a right shift, this is 8 - (number of bits to shift right by modulo 8).
    /// A value of 8 (possible only for right shifts) means that the left shift yields 0
    /// and the right shift returns the original byte.
    pub left_shift_by_mod_8: T,

    /// The byte-wise shifted input_1
    pub byte_shifted_word: Word<T>,

    /// The byte to sign-extend input_1 with: 0 if shl or shr, 0xff * MSB(input_1) if sra
    pub sign_extension_byte: T,

    /// The result of shifting each byte of byte_shifted_word left by left_shift_by_mod_8
    pub bit_shifted_bytes_left: Word<T>,
    /// The result of shifting each byte of byte_shifted_word right by (8 - left_shift_by_mod_8)
    pub bit_shifted_bytes_right: Word<T>,
    pub sign_extension_byte_overflow: T,

    /// Sign bit of input_1
    pub sign_1: T,

    pub is_shl: T,
    pub is_shr: T,
    pub is_sra: T,
}

pub const NUM_SHIFT_COLS: usize = size_of::<Shift32Cols<u8>>();
pub const COL_MAP: Shift32Cols<usize> = make_col_map();

const fn make_col_map() -> Shift32Cols<usize> {
    let indices_arr = indices_arr::<NUM_SHIFT_COLS>();
    unsafe { transmute::<[usize; NUM_SHIFT_COLS], Shift32Cols<usize>>(indices_arr) }
}
