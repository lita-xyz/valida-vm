use alloc::fmt::Formatter;
use core::borrow::{Borrow, BorrowMut};
use core::fmt::{Debug, Display, Result};
use core::mem::{size_of, transmute};
use p3_field::PrimeField32;
use valida_derive::AlignedBorrow;
use valida_machine::{Operands, Word};
use valida_opcodes::{unmap_field_value_to_opcode, Opcode};
use valida_util::indices_arr;

#[derive(AlignedBorrow, Default, Debug)]
pub struct ProgramCols<T> {
    pub pc: T,
    pub opcode: T,
    pub operands: Operands<T>,
    pub imm: Word<T>,
}

impl<F: PrimeField32> Display for ProgramCols<F> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let opcode_enum =
            Opcode::try_from(unmap_field_value_to_opcode(self.opcode)).expect("Invalid opcode");
        write!(
            f,
            "ProgramCols {{ pc: {}, opcode: {:?}, operands: {:?}, imm: {:?} }}",
            self.pc, opcode_enum, self.operands, self.imm
        )
    }
}

pub const NUM_PROGRAM_COLS: usize = size_of::<ProgramCols<u8>>();
pub const PROGRAM_COL_MAP: ProgramCols<usize> = make_col_map();

const fn make_col_map() -> ProgramCols<usize> {
    let indices_arr = indices_arr::<NUM_PROGRAM_COLS>();
    unsafe { transmute::<[usize; NUM_PROGRAM_COLS], ProgramCols<usize>>(indices_arr) }
}
