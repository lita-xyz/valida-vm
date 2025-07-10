use core::borrow::{Borrow, BorrowMut};
use core::mem::{size_of, transmute};
use valida_derive::AlignedBorrow;
use valida_machine::{Operands, Word, CPU_MEMORY_READ_CHANNELS, CPU_MEMORY_WRITE_CHANNELS};
use valida_util::indices_arr;

#[derive(AlignedBorrow, Default, Debug)]
pub struct CpuPublicVector<T> {
    /// Initial program counter
    pub pc_init: T,

    /// Initial frame pointer
    pub fp_init: T,

    pub is_last_segment: T,
}

#[derive(AlignedBorrow, Default, Debug)]
pub struct CpuCols<T> {
    /// Clock cycle
    pub clk: T,

    /// Program counter.
    pub pc: T,

    /// Frame pointer.
    pub fp: T,

    /// The instruction that was read, i.e. `program_code[pc]`.
    pub instruction: InstructionCols<T>,

    /// Flags indicating what type of operation is being performed this cycle.
    pub opcode_flags: OpcodeFlagCols<T>,

    /// When doing an equality test between two words, `x` and `y`, this holds the sum of
    /// `(x_i - y_i)^2`, which is zero if and only if `x = y`.
    pub diff: T,
    /// The inverse of `diff`, or undefined if `diff = 0`.
    pub diff_inv: T,
    /// A boolean flag indicating whether `diff != 0`.
    pub not_equal: T,

    /// Channels to the memory bus.
    pub mem_read_channels: [ReadChannelCols<T>; CPU_MEMORY_READ_CHANNELS],
    pub mem_write_channels: [WriteChannelCols<T>; CPU_MEMORY_WRITE_CHANNELS],
    /// Boolean flags indicating whether the given byte index is the offset of the address
    /// for single-byte operations
    pub addr_offset_flags: Word<T>,

    /// For loads8 instruction, holds the sign bit of the byte to be stored
    /// For jalv instruction, holds the sign bit of the fp offset
    pub sign_bit: T,

    pub is_last_segment: T,

    /// Indicates whether this is a real trace row or a padding row
    pub is_real: T,
}

#[derive(Default, Debug)]
pub struct InstructionCols<T> {
    pub opcode: T,
    pub operands: Operands<T>,
}

#[derive(Default, Debug)]
pub struct OpcodeFlagCols<T> {
    pub is_bus_op: T,
    pub is_pointer_op: T,
    pub is_imm_op: T,
    pub is_left_imm_op: T,
    pub is_load: T,
    pub is_load_u8: T,
    pub is_load_s8: T,
    pub is_store: T,
    pub is_store_u8: T,
    pub is_beq: T,
    pub is_bne: T,
    pub is_jal: T,
    pub is_jalv: T,
    pub is_imm32: T,
    pub is_advice: T,
    pub is_stop: T,
    pub is_loadfp: T,
    pub is_write: T,
}

#[derive(Debug)]
pub enum MemoryChannelCols<T> {
    ReadCols(ReadChannelCols<T>),
    WriteCols(WriteChannelCols<T>),
}
#[derive(Default, Debug)]
pub struct ReadChannelCols<T> {
    pub used: T,
    pub addr: T,
    pub value: Word<T>,
}

#[derive(Default, Debug)]
pub struct WriteChannelCols<T> {
    pub used: T,
    pub addr: T,
    pub value: Word<T>,
    pub old_value: Word<T>,
}

impl<T: Copy> CpuCols<T> {
    pub fn read_addr_1(&self) -> T {
        self.mem_read_channels[0].addr
    }
    pub fn read_addr_2(&self) -> T {
        self.mem_read_channels[1].addr
    }
    pub fn write_addr(&self) -> T {
        self.mem_write_channels[0].addr
    }

    pub fn read_value_1(&self) -> Word<T> {
        self.mem_read_channels[0].value
    }
    pub fn read_value_2(&self) -> Word<T> {
        self.mem_read_channels[1].value
    }
    pub fn write_value(&self) -> Word<T> {
        self.mem_write_channels[0].value
    }
    pub fn old_value_for_single_byte_write(&self) -> Word<T> {
        self.mem_write_channels[0].old_value
    }

    pub fn read_1_used(&self) -> T {
        self.mem_read_channels[0].used
    }
    pub fn read_2_used(&self) -> T {
        self.mem_read_channels[1].used
    }
    pub fn write_used(&self) -> T {
        self.mem_write_channels[0].used
    }
}

// `u8` is guaranteed to have a `size_of` of 1.
pub const NUM_CPU_COLS: usize = size_of::<CpuCols<u8>>();
pub const CPU_COL_MAP: CpuCols<usize> = make_col_map();

pub const NUM_CPU_PUBLIC_VALUES: usize = size_of::<CpuPublicVector<u8>>();
pub const CPU_PUBLIC_VECTOR_MAP: CpuPublicVector<usize> = make_public_vector_map();

const fn make_col_map() -> CpuCols<usize> {
    let indices_arr = indices_arr::<NUM_CPU_COLS>();
    unsafe { transmute::<[usize; NUM_CPU_COLS], CpuCols<usize>>(indices_arr) }
}

const fn make_public_vector_map() -> CpuPublicVector<usize> {
    let indices_arr = indices_arr::<NUM_CPU_PUBLIC_VALUES>();
    unsafe { transmute::<[usize; NUM_CPU_PUBLIC_VALUES], CpuPublicVector<usize>>(indices_arr) }
}

#[cfg(test)]
mod tests {
    type F = p3_baby_bear::BabyBear;

    #[test]
    fn aligned_borrow() {
        use super::*;
        use p3_field::AbstractField;

        let mut row = [F::zero(); NUM_CPU_COLS];
        let cols: &mut CpuCols<F> = unsafe { transmute(&mut row) };

        cols.mem_read_channels[0].used = F::one();

        let local: &CpuCols<F> = row[..].borrow();
        assert_eq!(local.mem_read_channels[0].used, F::one());
    }
}
