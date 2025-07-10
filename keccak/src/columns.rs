use core::borrow::{Borrow, BorrowMut};
use core::fmt::Debug;
use core::mem::{size_of, transmute};
use valida_derive::AlignedBorrow;
use valida_machine::Word;
use valida_util::indices_arr;

use crate::constants::R;
use crate::{NUM_ROUNDS, U64_LIMBS};

#[derive(AlignedBorrow, Debug)]
pub struct WrapKeccakCols<T> {
    /// The `i`th value is set to 1 if we are in the `i`th round, otherwise 0.
    pub step_flags: [T; NUM_ROUNDS],

    pub a: [[[T; U64_LIMBS]; 5]; 5],

    /// ```ignore
    /// C[x] = xor(A[x, 0], A[x, 1], A[x, 2], A[x, 3], A[x, 4])
    /// ```
    pub c: [[T; 64]; 5],

    /// ```ignore
    /// C'[x, z] = xor(C[x, z], C[x - 1, z], C[x + 1, z - 1])
    /// ```
    pub c_prime: [[T; 64]; 5],

    // Note: D is inlined, not stored in the witness.
    /// ```ignore
    /// A'[x, y] = xor(A[x, y], D[x])
    ///          = xor(A[x, y], C[x - 1], ROT(C[x + 1], 1))
    /// ```
    pub a_prime: [[[T; 64]; 5]; 5],

    /// ```ignore
    /// A''[x, y] = xor(B[x, y], andn(B[x + 1, y], B[x + 2, y])).
    /// ```
    pub a_prime_prime: [[[T; U64_LIMBS]; 5]; 5],

    /// The bits of `A''[0, 0]`.
    pub a_prime_prime_0_0_bits: [T; 64],

    /// ```ignore
    /// A'''[0, 0, z] = A''[0, 0, z] ^ RC[k, z]
    /// ```
    pub a_prime_prime_prime_0_0_limbs: [T; U64_LIMBS],
    pub export_input: T,
    pub export_output: T,
    pub clk: T,
    pub is_real: T,
    pub base_address: Word<T>,
}

pub const NUM_WRAP_KECCAK_COLS: usize = size_of::<WrapKeccakCols<u8>>();

pub const WRAPKECCAK_COL_MAP: WrapKeccakCols<usize> = make_wrapkeccak_col_map();

const fn make_wrapkeccak_col_map() -> WrapKeccakCols<usize> {
    let indices_arr = indices_arr::<NUM_WRAP_KECCAK_COLS>();
    unsafe { transmute::<[usize; NUM_WRAP_KECCAK_COLS], WrapKeccakCols<usize>>(indices_arr) }
}

impl<T: Copy> WrapKeccakCols<T> {
    pub fn b(&self, x: usize, y: usize, z: usize) -> T {
        debug_assert!(x < 5);
        debug_assert!(y < 5);
        debug_assert!(z < 64);

        // B is just a rotation of A', so these are aliases for A' registers.
        // From the spec,
        //     B[y, (2x + 3y) % 5] = ROT(A'[x, y], r[x, y])
        // So,
        //     B[x, y] = f((x + 3y) % 5, x)
        // where f(a, b) = ROT(A'[a, b], r[a, b])
        let a = (x + 3 * y) % 5;
        let b = x;
        let rot = R[a][b] as usize;
        self.a_prime[b][a][(z + 64 - rot) % 64]
    }

    pub fn a_prime_prime_prime(&self, x: usize, y: usize, limb: usize) -> T {
        debug_assert!(x < 5);
        debug_assert!(y < 5);
        debug_assert!(limb < U64_LIMBS);

        if x == 0 && y == 0 {
            self.a_prime_prime_prime_0_0_limbs[limb]
        } else {
            self.a_prime_prime[y][x][limb]
        }
    }
}

pub fn indexes(i: usize) -> (usize, usize, usize) {
    let i_u64 = i / U64_LIMBS;
    let limb_index = i % U64_LIMBS;

    // The 5x5 state is treated as y-major, as per the Keccak spec.
    let y = i_u64 / 5;
    let x = i_u64 % 5;
    (x, y, limb_index)
}

pub fn indexes_x(i: usize) -> (usize, usize, usize) {
    let i_u64 = i / U64_LIMBS;
    let limb_index = i % U64_LIMBS;

    // Swap to x-major order: x = i_u64 / 5, y = i_u64 % 5
    let x = i_u64 / 5;
    let y = i_u64 % 5;
    (x, y, limb_index)
}

impl<T: Default + Copy> Default for WrapKeccakCols<T> {
    fn default() -> Self {
        Self {
            step_flags: [T::default(); NUM_ROUNDS],
            a: [[[T::default(); U64_LIMBS]; 5]; 5],
            c: [[T::default(); 64]; 5],
            c_prime: [[T::default(); 64]; 5],
            a_prime: [[[T::default(); 64]; 5]; 5],
            a_prime_prime: [[[T::default(); U64_LIMBS]; 5]; 5],
            a_prime_prime_0_0_bits: [T::default(); 64],
            a_prime_prime_prime_0_0_limbs: [T::default(); U64_LIMBS],
            export_input: T::default(),
            export_output: T::default(),
            clk: T::default(),
            is_real: T::default(),
            base_address: Word::<T>::default(),
        }
    }
}
