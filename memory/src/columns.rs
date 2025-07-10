use core::borrow::{Borrow, BorrowMut};
use core::mem::{size_of, transmute};
use valida_derive::AlignedBorrow;
use valida_machine::Word;
use valida_util::indices_arr;

#[derive(AlignedBorrow, Debug)]
pub struct MemoryCols<T> {
    /// Memory address
    pub addr: T,

    /// address bytes
    pub addr_bytes: Word<T>,

    /// Memory cell
    pub value: Word<T>,

    /// `clk` value in the current segment for the operation
    pub clk: T,

    pub diff_bytes: Word<T>,

    /// Flag indicating if this row encodes a dummy read
    pub is_dummy_read: T,
    /// Whether memory operation is a read
    pub is_read: T,
    /// Whether memory operation is a write
    pub is_write: T,

    /// Either addr' - addr (if address is changed), or clk' - clk (if address is not changed)
    pub diff: T,
    /// The inverse of `diff`, or 0 if `diff = 0`.
    pub diff_inv: T,

    /// A boolean flag, 1 iff `addr' == addr`
    pub addr_equal: T,

    /// is_initial is true iff this is a dummy read or a read to an address that was not previously accessed in this segment
    /// and the address does *NOT* contain data from the static data chip (i.e. previous row was `is_static_write == 1`
    pub is_initial: T,

    /// For initial reads, the timestamp of the last access to the address
    pub prior_timestamp: T,

    /// A boolean flag, 1 iff this is an operation involving a zero-initialized memory cell
    /// This is true iff it's the first dummy read or read to that address across _all_ segments
    pub is_zero_initialized: T,
    //
    pub is_final: T,
    /// Only true for first writes of static data into memory chip. Segment 0
    pub is_static_write: T,
    /// Whether we perform a persistent send for this row. If 0 we send, if 1 we skip the send.
    pub skip_persistent_send: T,
    /// Whether we perform a persistent receive for this row. If 0 we send, if 1 we skip the send.
    /// This is for static data access that happens for the first time in the program in any segment
    /// other than the first.
    pub skip_persistent_receive: T,
}

#[derive(AlignedBorrow, Default, Debug)]
pub struct MemoryPublicVector<T> {
    pub segment_number: T,
}

pub const NUM_MEM_COLS: usize = size_of::<MemoryCols<u8>>();
pub const MEM_COL_MAP: MemoryCols<usize> = make_col_map();

pub const NUM_MEM_PUBLIC_VALUES: usize = size_of::<MemoryPublicVector<u8>>();
pub const MEM_PUBLIC_VECTOR_MAP: MemoryPublicVector<usize> = make_public_vector_map();

// Used as an offset for persistent sends to match persistent receive segment numbers.
// `as_scalar` for `MemoryAccessTimestamp` starts at 3 for segment index 0 and then increases
// from there.
pub const AS_SCALAR_SEGMENT_OFFSET: u32 = 3;

const fn make_col_map() -> MemoryCols<usize> {
    let indices_arr = indices_arr::<NUM_MEM_COLS>();
    unsafe { transmute::<[usize; NUM_MEM_COLS], MemoryCols<usize>>(indices_arr) }
}

const fn make_public_vector_map() -> MemoryPublicVector<usize> {
    let indices_arr = indices_arr::<NUM_MEM_PUBLIC_VALUES>();
    unsafe { transmute::<[usize; NUM_MEM_PUBLIC_VALUES], MemoryPublicVector<usize>>(indices_arr) }
}
