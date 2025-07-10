use core::borrow::{Borrow, BorrowMut};
use core::mem::{size_of, transmute};
use valida_derive::AlignedBorrow;
use valida_machine::{Word, MEMORY_CELL_BYTES};
use valida_util::indices_arr;

pub const PRODUCT_LENGTH: usize = MEMORY_CELL_BYTES * 2;

pub const LIMB_SIZE: usize = 2;

// We compute the multiplication modulo successive powers of 2^16, i.e. two-byte limbs
pub const PRODUCT_LIMBS: usize = (PRODUCT_LENGTH - 1) / LIMB_SIZE + 1;
pub const CARRY_LENGTH: usize = PRODUCT_LIMBS;

// The maximum size of a partially reduced product `pi`. This is the maximum
// base power 2^(8 * (LIMB_SIZE - 1)) occuring in the expansion of a product of limbs,
// times 2*16, the maximum size of a product of two bytes.
pub const PI_MAX: usize = MEMORY_CELL_BYTES * (1 << (8 * (LIMB_SIZE + 1) + 1));
// This is PI_MAX >> 8 * LIMB_SIZE, the maximum size of a carry element.
pub const CARRY_MAX: usize = PRODUCT_LENGTH * (1 << 8);
#[derive(AlignedBorrow, Default, Debug)]
pub struct Mul32Cols<T> {
    pub input_1: Word<T>,
    pub input_2: Word<T>,

    // 1 if the input is negative and the opcode is MULHS, 0 otherwise
    pub sign_1: T,
    pub sign_2: T,

    /// Witnessed output
    pub lower_word: Word<T>,
    pub upper_word: Word<T>,

    // Carry elements used to compute the product.
    pub carry: Word<T>,

    pub is_mul: T,
    pub is_mulhs: T,
    pub is_mulhu: T,

    // For the range check on the elements of `carry`: `counter` lists
    // indices from 0 to CARRY_MAX, and `counter_mult` records the number of times
    // this counter value is encountered.
    pub counter: T,
    pub counter_mult: T,
}

pub const NUM_MUL_COLS: usize = size_of::<Mul32Cols<u8>>();
pub const MUL_COL_MAP: Mul32Cols<usize> = make_col_map();

const fn make_col_map() -> Mul32Cols<usize> {
    let indices_arr = indices_arr::<NUM_MUL_COLS>();
    unsafe { transmute::<[usize; NUM_MUL_COLS], Mul32Cols<usize>>(indices_arr) }
}
