use super::MEMORY_CELL_BYTES;
use core::cmp::Ordering;
use core::num::Wrapping;
use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Shl, Shr, Sub};
use p3_air::AirBuilder;
use p3_field::{Field, PrimeField, PrimeField32};

use valida_memory_footprint::MemoryFootprint;

// Currently stored in big-endian form.
#[derive(Copy, Clone, Debug, Default, Hash)]
pub struct Word<F>(pub [F; MEMORY_CELL_BYTES]);

use core::mem;
impl<T: Sized> MemoryFootprint for Word<T> {
    fn memory_footprint(&self) -> usize {
        mem::size_of::<T>() * MEMORY_CELL_BYTES
    }
}

// Functions for byte manipulations
/// Get the little-endian index of a byte in a memory cell.
pub fn index_le_of_byte(addr: u32) -> usize {
    (addr & 3) as usize
}

/// Get the address of the memory cells which is not empty (a multiple of 4).
pub fn addr_of_word(addr: u32) -> u32 {
    addr & !3
}

pub fn is_mul_4(addr: u32) -> bool {
    addr.rem_euclid(4) == 0
}
//----------------------------------

impl Word<u8> {
    pub fn from_u8(byte: u8) -> Self {
        Self::from(byte as u32)
    }

    pub fn sign_extend_byte(byte: u8) -> Self {
        Self::from(byte as i8 as i32 as u32)
    }

    pub fn update_byte(self, byte: u8, loc: usize) -> Self {
        let mut result = self;
        result.0[loc] = byte;
        result
    }

    pub fn trunc_to_u8(self) -> u8 {
        u32::from_le_bytes(self.0) as u8
    }
}

impl<F: Copy> Word<F> {
    pub fn transform<T, G>(self, f: G) -> Word<T>
    where
        G: FnMut(F) -> T,
    {
        Word(self.0.map(f))
    }

    /// This is slower than `update_from_slice_le`, which should be preferred.
    pub fn update_from_slice_be(&mut self, slice: &[F]) {
        self.update_from_slice_le(slice);
        self.0.reverse();
    }

    pub fn update_from_slice_le(&mut self, slice: &[F]) {
        self.0.copy_from_slice(slice)
    }
}

impl<F> Word<F> {
    pub fn from_components_be(x: [F; MEMORY_CELL_BYTES]) -> Self {
        let [a, b, c, d] = x;
        Self([d, c, b, a])
    }

    pub fn from_components_le(x: [F; MEMORY_CELL_BYTES]) -> Self {
        Self(x)
    }

    pub fn index_be(&self, index: usize) -> &F {
        &self.0[MEMORY_CELL_BYTES - index - 1]
    }

    pub fn index_mut_be(&mut self, index: usize) -> &mut F {
        &mut self.0[MEMORY_CELL_BYTES - index - 1]
    }

    pub fn index_le(&self, index: usize) -> &F {
        &self.0[index]
    }

    pub fn index_mut_le(&mut self, index: usize) -> &mut F {
        &mut self.0[index]
    }

    pub fn iter_be(&self) -> impl Iterator<Item = &F> {
        self.0.iter().rev()
    }

    pub fn iter_mut_be(&mut self) -> impl Iterator<Item = &mut F> {
        self.0.iter_mut().rev()
    }

    pub fn into_iter_be(self) -> impl Iterator<Item = F> {
        self.0.into_iter().rev()
    }

    pub fn iter_le(&self) -> impl Iterator<Item = &F> {
        self.0.iter()
    }

    pub fn iter_mut_le(&mut self) -> impl Iterator<Item = &mut F> {
        self.0.iter_mut()
    }

    pub fn into_iter_le(self) -> impl Iterator<Item = F> {
        self.0.into_iter()
    }
}

impl<F: PrimeField> Word<F> {
    pub fn reduce(self) -> F {
        let mut result = F::zero();
        for (n, item) in self.0.into_iter().enumerate() {
            result += item * F::from_canonical_u32(1 << (8 * n));
        }
        result
    }
}

pub fn reduce_word<AB: AirBuilder>(base: &Word<AB::Expr>, input: Word<AB::Var>) -> AB::Expr {
    input
        .into_iter_le()
        .zip(base.iter_le())
        .map(|(i, b)| i * b.clone())
        .sum()
}

impl From<Word<u8>> for u32 {
    fn from(val: Word<u8>) -> Self {
        u32::from_le_bytes(val.0)
    }
}

impl From<Word<u8>> for i32 {
    fn from(val: Word<u8>) -> Self {
        i32::from_le_bytes(val.0)
    }
}

impl From<u32> for Word<u8> {
    fn from(value: u32) -> Self {
        Self(value.to_le_bytes())
    }
}

impl Add for Word<u8> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let b: u32 = self.into();
        let c: u32 = other.into();
        let res = (Wrapping(b) + Wrapping(c)).0;
        res.into()
    }
}

impl Sub for Word<u8> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let b: u32 = self.into();
        let c: u32 = other.into();
        let res = (Wrapping(b) - Wrapping(c)).0;
        res.into()
    }
}

impl Mul for Word<u8> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let b: u32 = self.into();
        let c: u32 = other.into();
        let res = (Wrapping(b) * Wrapping(c)).0;
        res.into()
    }
}

pub trait Mulhs<Rhs = Self> {
    /// The resulting type after applying the `/` operator.
    type Output;

    fn mulhs(self, rhs: Rhs) -> Self::Output;
}

impl Mulhs for Word<u8> {
    type Output = Self;
    fn mulhs(self, other: Self) -> Self {
        let bu32: u32 = self.into();
        let bi64 = (bu32 as i32) as i64;
        let cu32: u32 = other.into();
        let ci64 = (cu32 as i32) as i64;
        // The result of regular multiplication represented in i64
        let mul_res = bi64 * ci64;
        let res = (mul_res >> 32) as i32 as u32;
        res.into()
    }
}

pub trait Mulhu<Rhs = Self> {
    /// The resulting type after applying the `/` operator.
    type Output;

    fn mulhu(self, rhs: Rhs) -> Self::Output;
}

impl Mulhu for Word<u8> {
    type Output = Self;
    fn mulhu(self, other: Self) -> Self {
        let bu32: u32 = self.into();
        let bu64 = bu32 as u64;
        let cu32: u32 = other.into();
        let cu64 = cu32 as u64;
        // The result of regular multiplication represented in u64
        let mul_res = bu64 * cu64;
        let res = (mul_res >> 32) as u32;
        res.into()
    }
}

impl Div for Word<u8> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let b: u32 = self.into();
        let c: u32 = other.into();
        let res = b / c;
        res.into()
    }
}

pub trait SDiv<Rhs = Self> {
    /// The resulting type after applying the `/` operator.
    type Output;

    fn sdiv(self, rhs: Rhs) -> Self::Output;
}

impl SDiv for Word<u8> {
    type Output = Self;
    fn sdiv(self, other: Self) -> Self {
        let bu: u32 = self.into();
        let b = bu as i32;
        let cu: u32 = other.into();
        let c = cu as i32;
        // perform the division in i32 first, then convert it to u32
        let res = (b / c) as u32;
        res.into()
    }
}

impl Shl for Word<u8> {
    type Output = Self;
    fn shl(self, other: Self) -> Self {
        let b: u32 = self.into();
        let c: u32 = other.into();
        let res = b << (c & 0x1f);
        res.into()
    }
}

impl Shr for Word<u8> {
    type Output = Self;
    fn shr(self, other: Self) -> Self {
        let b: u32 = self.into();
        let c: u32 = other.into();
        let res = b >> (c & 0x1f);
        res.into()
    }
}

pub trait Sra<Rhs = Self> {
    /// The resulting type after applying the `/` operator.
    type Output;

    fn sra(self, rhs: Rhs) -> Self::Output;
}

impl Sra for Word<u8> {
    type Output = Self;
    fn sra(self, other: Self) -> Self {
        let bu: u32 = self.into();
        let b = bu as i32;
        let c: u32 = other.into();
        let res = (b >> (c & 0x1f)) as u32;
        res.into()
    }
}

impl BitXor for Word<u8> {
    type Output = Self;
    fn bitxor(self, other: Self) -> Self {
        let mut res = self;
        for i in 0..MEMORY_CELL_BYTES {
            res.0[i] ^= other.0[i];
        }
        res
    }
}

impl BitAnd for Word<u8> {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        let mut res = self;
        for i in 0..MEMORY_CELL_BYTES {
            res.0[i] &= other.0[i];
        }
        res
    }
}

impl BitOr for Word<u8> {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        let mut res = self;
        for i in 0..MEMORY_CELL_BYTES {
            res.0[i] |= other.0[i];
        }
        res
    }
}

impl<F: Field> From<F> for Word<F> {
    fn from(bytes: F) -> Self {
        Self::from_components_be([F::zero(), F::zero(), F::zero(), bytes])
    }
}

impl<F: Ord> Eq for Word<F> {}

impl<F: Ord> PartialEq for Word<F> {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
    }
}

impl<F: Ord> PartialOrd for Word<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Ord> Ord for Word<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .iter()
            .zip(other.0.iter())
            .rev()
            .map(|(a, b)| a.cmp(b))
            .find(|&ord| ord != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    }
}

impl<F: PrimeField32> Word<F> {
    /// Convert a Word<F> to Word<u8>
    pub fn to_u8_word(&self) -> Word<u8> {
        // Convert each field element to a u8
        let bytes = self.0.map(|x| x.as_canonical_u32() as u8);
        Word(bytes)
    }

    /// Convert a Word<u8> to Word<F>
    pub fn from_u8_word(word: &Word<u8>) -> Self {
        // Convert each u8 to a field element
        let field_elements = word.0.map(|x| F::from_canonical_u8(x));
        Word(field_elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;

    #[test]
    fn test_word_helper_methods() {
        // Create a Word<u8> with test values
        let word_u8 = Word([1u8, 2u8, 3u8, 4u8]);

        // Convert to Word<F> and back to Word<u8>
        let word_f = Word::<BabyBear>::from_u8_word(&word_u8);
        let word_u8_back = word_f.to_u8_word();

        // Round trip consistency check
        assert_eq!(word_u8.0, word_u8_back.0);

        // Test with max u8 values
        let word_u8_max = Word([255u8, 255u8, 255u8, 255u8]);
        let word_f_max = Word::<BabyBear>::from_u8_word(&word_u8_max);
        let word_u8_max_back = word_f_max.to_u8_word();

        // Round trip consistency check
        assert_eq!(word_u8_max.0, word_u8_max_back.0);
    }
}
