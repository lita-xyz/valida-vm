#![no_std]

extern crate alloc;

use crate::columns::NUM_PROGRAM_COLS;
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use columns::ProgramCols;
use core::{
    borrow::{Borrow, BorrowMut},
    mem::{transmute, MaybeUninit},
};
use valida_machine::{InstructionWord, Machine, Operands, OperandsIndex as OpIdx, ProgramROM};

use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use smallvec::SmallVec;
use valida_lookups::{
    LookupChip, LookupTable, LookupType, MachineWithLookupChip, MultiLookupTableWrapper,
};
use valida_machine::SMALLVEC_SIZE;
use valida_opcodes::*;

use valida_memory_footprint::MemoryFootprint;

pub mod columns;

#[derive(Clone)]
pub enum CpuOperation {
    Pointer,
    Load32,
    LoadU8,
    LoadS8,
    Store32,
    StoreU8,
    Beq,
    Bne,
    Jal,
    Jalv,
    Imm32,
    ReadAdvice,
    Stop,
    LoadFp,
    Write,
}

pub fn opcode_to_cpuoperation_code(opcode: u32) -> u32 {
    if opcode == 0 {
        return 0;
    }

    match Opcode::try_from(opcode).unwrap() {
        Opcode::LOAD32 => CpuOperation::Load32 as u32 + 1,
        Opcode::LOADU8 => CpuOperation::LoadU8 as u32 + 1,
        Opcode::LOADS8 => CpuOperation::LoadS8 as u32 + 1,
        Opcode::STORE32 => CpuOperation::Store32 as u32 + 1,
        Opcode::STOREU8 => CpuOperation::StoreU8 as u32 + 1,
        Opcode::BEQ => CpuOperation::Beq as u32 + 1,
        Opcode::BNE => CpuOperation::Bne as u32 + 1,
        Opcode::JAL => CpuOperation::Jal as u32 + 1,
        Opcode::JALV => CpuOperation::Jalv as u32 + 1,
        Opcode::IMM32 => CpuOperation::Imm32 as u32 + 1,
        Opcode::READ_ADVICE => CpuOperation::ReadAdvice as u32 + 1,
        Opcode::STOP => CpuOperation::Stop as u32 + 1,
        Opcode::LOADFP => CpuOperation::LoadFp as u32 + 1,
        Opcode::WRITE => CpuOperation::Write as u32 + 1,
        _ => 0,
    }
}

fn instruction_to_row<F: PrimeField32>(
    (pc, word): (usize, &InstructionWord<i32>),
) -> SmallVec<[F; SMALLVEC_SIZE]> {
    // In these operations, operands a,b,c are always signed memory offsets or ignored.
    let ops_with_three_offset_operands = [
        JALV,
        LOADFP,
        LOAD32,
        LOADU8,
        LOADS8,
        STORE32,
        STOREU8,
        READ_ADVICE,
        WRITE,
        STOP,
        FAIL,
        MEMCPY,
        COMBSECP256K1,
        SMULSECP256K1,
        MULSSECP256K1,
    ];

    // These operations have operands which may come in immediate or non-immediate forms,
    // with the last two operations boolean flags indicating which form is used.
    // These two flags are mutually exclusive.
    let bus_ops = [
        ADD32,
        SUB32,
        MUL32,
        MULHS32,
        MULHU32,
        DIV32,
        SDIV32,
        SHL32,
        SHR32,
        SRA32,
        LT32,
        LTE32,
        SLT32,
        SLE32,
        AND32,
        OR32,
        XOR32,
        NE32,
        EQ32,
        ADD,
        SUB,
        MUL,
        KECCAKF,
        SINVSECP256K1,
    ];
    let branch_ops = [BEQ, BNE];

    // A bunch of static sanity checks for the operands: these may all be enforced at compile-time,
    // so there is no need to check them inside the STARK.
    if ops_with_three_offset_operands.contains(&word.opcode) {
        // a,b,c are signed address offsets, and should be in the range (-p/2, p/2)
        assert!(word.operands.a().unsigned_abs() * 2 < F::ORDER_U32);
        assert!(word.operands.b().unsigned_abs() * 2 < F::ORDER_U32);
        assert!(word.operands.c().unsigned_abs() * 2 < F::ORDER_U32);
    } else if branch_ops.contains(&word.opcode) {
        // a is a program address and should be in the range [0, p)
        assert!(word.operands.a() >= 0);
        assert!((word.operands.a() as u32) < F::ORDER_U32);
        if !word.operands.is_imm() == 1 {
            // b is a signed address offset, and should be in the range (-p/2, p/2)
            assert!(word.operands.b().unsigned_abs() * 2 < F::ORDER_U32);
        }
        if !word.operands.is_left_imm() == 1 {
            // c is a signed address offset, and should be in the range (-p/2, p/2)
            assert!(word.operands.c().unsigned_abs() * 2 < F::ORDER_U32);
        }
    } else if bus_ops.contains(&word.opcode) {
        // a is a signed address offset, and should be in the range (-p/2, p/2)
        assert!(word.operands.a().unsigned_abs() * 2 < F::ORDER_U32);
        if !word.operands.is_left_imm() == 1 {
            // b is a signed address offset, and should be in the range (-p/2, p/2)
            assert!(word.operands.b().unsigned_abs() * 2 < F::ORDER_U32);
        }
        if !word.operands.is_imm() == 1 {
            // c is a signed address offset, and should be in the range (-p/2, p/2)
            assert!(word.operands.c().unsigned_abs() * 2 < F::ORDER_U32);
        }
    } else if word.opcode == JAL {
        // b is a program address and should be in the range [0, p)
        assert!(word.operands.b() >= 0);
        assert!((word.operands.b() as u32) < F::ORDER_U32);

        // a, c are signed address offsets, and should be in the range (-p/2, p/2)
        assert!(word.operands.a().unsigned_abs() * 2 < F::ORDER_U32);
        assert!(word.operands.c().unsigned_abs() * 2 < F::ORDER_U32);
    } else if word.opcode == IMM32 {
        assert!((word.operands.b() as u32) < 1 << 8);
        assert!((word.operands.c() as u32) < 1 << 8);
        assert!((word.operands.d() as u32) < 1 << 8);
        assert!((word.operands.e() as u32) < 1 << 8);
    } else {
        unimplemented!("unknown opcode: {}", word.opcode);
    }

    // SAFETY: initializing uninits is a no-op
    let mut row: [MaybeUninit<F>; NUM_PROGRAM_COLS] =
        unsafe { MaybeUninit::uninit().assume_init() };
    let cols: &mut ProgramCols<MaybeUninit<F>> = { unsafe { transmute(&mut row) } };
    cols.pc.write(F::from_canonical_usize(pc));
    cols.opcode.write(F::from_canonical_u32(word.opcode));
    cols.operation_code
        .write(F::from_canonical_u32(opcode_to_cpuoperation_code(
            word.opcode,
        )));

    let operands = Operands::<F>::from_operands_i32(&word.operands);

    // SAFETY: sources are different from row (dst)
    unsafe {
        core::ptr::copy_nonoverlapping(
            operands.0.map(MaybeUninit::new).as_ptr(),
            cols.operands.0.as_mut_ptr(),
            5,
        );
        cols.imm
            .update_from_slice_le(&[0; 4].map(F::from_canonical_u8).map(MaybeUninit::new));
    }

    // SAFETY: at this point, every entry of row is initialized
    let mut row = unsafe { row.map(|x| x.assume_init()) };
    let cols: &mut ProgramCols<F> = row[..].borrow_mut();

    // Load the immediate values bytewise into the four extra columns in the lookup, and reduce the
    // corresponding operand modulo the order of F.
    if bus_ops.contains(&word.opcode) || branch_ops.contains(&word.opcode) {
        // The immediate values are loaded into the memory channel for the unused read,
        // so these should be included in the lookup.
        if word.operands.is_left_imm() == 1 {
            let b = word.operands.b() as u32;
            cols.operands[OpIdx::B] = F::from_wrapped_u32(b);
            cols.imm
                .update_from_slice_le(&b.to_le_bytes().map(F::from_canonical_u8))
        } else if word.operands.is_imm() == 1 {
            let c = word.operands.c() as u32;
            cols.operands[OpIdx::C] = F::from_wrapped_u32(c);
            cols.imm
                .update_from_slice_le(&c.to_le_bytes().map(F::from_canonical_u8));
        }
    }

    SmallVec::from_buf(row)
}

pub fn rom_to_table<F: PrimeField32>(
    rom: &ProgramROM<i32>,
    verbose: bool,
) -> (RowMajorMatrix<F>, Option<Vec<String>>) {
    let n = rom.0.len();

    let mut values = rom
        .0
        .par_iter()
        .enumerate()
        .map(instruction_to_row)
        .flat_map(|row| row.to_vec())
        .collect::<Vec<_>>();

    let log = if verbose {
        let mut log_prints = Vec::with_capacity(n);
        for (i, row) in values.chunks(NUM_PROGRAM_COLS).enumerate() {
            let cols: &ProgramCols<F> = row[..].borrow();
            log_prints.push(format!("Program row {i}: {}", cols));
        }
        Some(log_prints)
    } else {
        None
    };

    // Pad the ROM to a power of two.
    values.resize(n.next_power_of_two() * NUM_PROGRAM_COLS, F::zero());
    (RowMajorMatrix::new(values, NUM_PROGRAM_COLS), log)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProgramTableType {
    #[default]
    Public,
    Preprocessed,
}

impl MemoryFootprint for ProgramTableType {
    fn memory_footprint(&self) -> usize {
        core::mem::size_of::<Self>()
    }
}

#[derive(Default, Clone)]
pub struct ProgramTable {
    pub table_type: ProgramTableType,
    pub rom: ProgramROM<i32>,
}

impl MemoryFootprint for ProgramTable {
    fn memory_footprint(&self) -> usize {
        self.table_type.memory_footprint() + self.rom.memory_footprint()
    }
}

impl<F: PrimeField32> LookupTable<F> for ProgramTable {
    type M<'a> = RowMajorMatrix<F>;

    fn name(&self) -> String {
        match self.table_type {
            ProgramTableType::Public => "Public Program Table".to_string(),
            ProgramTableType::Preprocessed => "Preprocessed Program Table".to_string(),
        }
    }

    fn lookup_type(&self) -> LookupType {
        match self.table_type {
            ProgramTableType::Public => LookupType::Public,
            ProgramTableType::Preprocessed => LookupType::Preprocessed,
        }
    }

    fn lookup_matrix(&self, verbose: bool) -> (RowMajorMatrix<F>, Option<Vec<String>>) {
        rom_to_table(&self.rom, verbose)
    }

    fn width(&self) -> usize {
        NUM_PROGRAM_COLS
    }

    fn height(&self) -> usize {
        self.rom.0.len().next_power_of_two()
    }
}

pub type ProgramChip<F> = LookupChip<MultiLookupTableWrapper<ProgramTable>, F>;

pub trait MachineWithProgramROM<F: Field>: Machine<F> {
    fn program_rom(&self) -> &ProgramROM<i32>;
    fn set_program_rom(&mut self, rom: ProgramROM<i32>, table_type: ProgramTableType);

    fn program_table_type(&self) -> ProgramTableType;
}

pub trait MachineWithProgramChip<F: PrimeField32>: Machine<F> + MachineWithProgramROM<F> {
    fn program(&self) -> &ProgramChip<F>;

    fn program_mut(&mut self) -> &mut ProgramChip<F>;

    fn read_word(&mut self, index: u32, log: bool);
}

impl<F, M> MachineWithProgramChip<F> for M
where
    F: PrimeField32,
    M: MachineWithLookupChip<F, ProgramTable> + MachineWithProgramROM<F>,
{
    fn program(&self) -> &ProgramChip<F> {
        self.lookup_chip()
    }
    fn program_mut(&mut self) -> &mut ProgramChip<F> {
        self.lookup_chip_mut()
    }

    fn read_word(&mut self, index: u32, log: bool) {
        if log {
            let instruction = self.program().table.0.rom.get_instruction(index);
            self.vector_lookup(instruction_to_row((index as usize, instruction)), log);
        }
    }
}
