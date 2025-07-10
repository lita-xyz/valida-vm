use core::borrow::{Borrow, BorrowMut};
use core::mem::{size_of, transmute};
use valida_derive::AlignedBorrow;
use valida_machine::Word;
use valida_util::indices_arr;

#[derive(AlignedBorrow, Default, Debug)]
pub struct Div32Cols<T> {
    pub input_1: Word<T>,
    pub input_2: Word<T>,

    /// Sign bit of input_1
    pub sign_1: T,
    /// Sign bit of input_2
    pub sign_2: T,
    /// 1 if input_1 and input_2 have the same sign, 0 otherwise
    pub same_sign: T,

    /// Witnessed output
    pub output: Word<T>,

    /// Witnessed product input_2 * output
    pub product_lower: Word<T>,
    pub product_upper: Word<T>,

    /// Witnessed remainder: input_1 % input_2
    /// |remainder| < |input_2| and remainder has the same sign as input_2
    pub remainder: Word<T>,
    /// sign bit of the remainder
    pub sign_remainder: T,

    pub is_div: T,
    pub is_sdiv: T,
}

pub const NUM_DIV_COLS: usize = size_of::<Div32Cols<u8>>();
pub const DIV_COL_MAP: Div32Cols<usize> = make_col_map();

const fn make_col_map() -> Div32Cols<usize> {
    let indices_arr = indices_arr::<NUM_DIV_COLS>();
    unsafe { transmute::<[usize; NUM_DIV_COLS], Div32Cols<usize>>(indices_arr) }
}
