use core::array::IntoIter;
use core::fmt::{Display, Formatter, Result};
use core::ops::{Index, IndexMut};
use core::slice::{Iter, IterMut};

use crate::{Machine, RunningMachine, Word, INSTRUCTION_ELEMENTS, OPERAND_ELEMENTS};
use byteorder::{ByteOrder, LittleEndian};
use p3_field::Field;

use valida_opcodes::{Opcode, IMM32};

use valida_memory_footprint::MemoryFootprint;

pub trait Instruction<M: Machine<F>, F: Field> {
    const OPCODE: u32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>);
}

#[derive(Copy, Clone, Default, Debug)]
pub struct InstructionWord<F> {
    pub opcode: u32,
    pub operands: Operands<F>,
}

impl<F> MemoryFootprint for InstructionWord<F> {
    fn memory_footprint(&self) -> usize {
        self.opcode.memory_footprint() + self.operands.memory_footprint()
    }
}

impl Display for InstructionWord<i32> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let opcode = match Opcode::try_from(self.opcode) {
            Ok(opcode_name) => {
                format!("{:?}", opcode_name)
            }
            Err(_) => {
                format!("UNKNOWN_OP:{}", self.opcode)
            }
        };
        write!(f, "{} {}", opcode, self.print_operands())
    }
}

impl InstructionWord<i32> {
    pub fn flatten<F: Field>(&self) -> [F; INSTRUCTION_ELEMENTS] {
        let mut result = [F::default(); INSTRUCTION_ELEMENTS];
        result[0] = F::from_canonical_u32(self.opcode);
        result[1..].copy_from_slice(&Operands::<F>::from_i32_slice(&self.operands.0).0);
        result
    }

    pub fn print_imm32(&self) -> String {
        assert!(self.opcode == IMM32, "Instruction is not immediate");

        //extract the immediate value
        let imm0 = self.operands.0[1];
        let imm1 = self.operands.0[2];
        let imm2 = self.operands.0[3];
        let imm3 = self.operands.0[4];
        format!(
            "{}(fp), {}",
            self.operands.0[0],
            imm3 << 24 | imm2 << 16 | imm1 << 8 | imm0
        )
    }

    pub fn print_first_operand(&self) -> String {
        format!("{}(fp)", self.operands.0[1])
    }

    pub fn print_second_operand(&self) -> String {
        let second_opnd_is_imm = self.operands.0[4] != 0;
        if second_opnd_is_imm {
            format!("{}", self.operands.0[2])
        } else {
            format!("{}(fp)", self.operands.0[2])
        }
    }

    pub fn print_address(&self, index: usize) -> String {
        format!("{}", self.operands.0[index] / 24)
    }

    pub fn print_operands(&self) -> String {
        match self.opcode {
            valida_opcodes::IMM32 => self.print_imm32(),
            valida_opcodes::JAL => format!(
                "{}(fp), PC: {}, {}",
                self.operands.0[0],
                self.print_address(1),
                self.operands.0[2]
            ),
            valida_opcodes::JALV => format!(
                "{}(fp), {}(fp), {}(fp)",
                self.operands.0[0], self.operands.0[1], self.operands.0[2]
            ),
            valida_opcodes::LOADFP => format!("{}(fp), {}", self.operands.0[0], self.operands.0[1]),
            valida_opcodes::BEQ | valida_opcodes::BNE => format!(
                "{}, {}, {}",
                self.print_address(0),
                self.print_first_operand(),
                self.print_second_operand()
            ),
            valida_opcodes::STOP => "".to_string(),
            valida_opcodes::FAIL => "terminated with error".to_string(),
            valida_opcodes::LOAD32 => {
                format!("{}(fp), {}(fp)", self.operands.0[0], self.operands.0[2])
            }
            valida_opcodes::LOADU8 => {
                format!("{}(fp), {}(fp)", self.operands.0[0], self.operands.0[2])
            }
            valida_opcodes::LOADS8 => {
                format!("{}(fp), {}(fp)", self.operands.0[0], self.operands.0[2])
            }
            valida_opcodes::STORE32 => {
                format!("{}(fp), {}(fp)", self.operands.0[1], self.operands.0[2])
            }
            valida_opcodes::STOREU8 => {
                format!("{}(fp), {}(fp)", self.operands.0[1], self.operands.0[2])
            }
            valida_opcodes::MEMCPY => {
                format!(
                    "{}(fp), {}(fp), {}(fp)",
                    self.operands.0[0], self.operands.0[1], self.operands.0[2]
                )
            }
            _ => {
                format!(
                    "{}(fp), {}, {}",
                    self.operands.0[0],
                    self.print_first_operand(),
                    self.print_second_operand()
                )
            }
        }
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Operands<F>(pub [F; OPERAND_ELEMENTS]);

impl<F> MemoryFootprint for Operands<F> {
    fn memory_footprint(&self) -> usize {
        core::mem::size_of::<F>() * OPERAND_ELEMENTS
    }
}

impl<F: Copy> Operands<F> {
    pub fn a(&self) -> F {
        self.0[0]
    }
    pub fn b(&self) -> F {
        self.0[1]
    }
    pub fn c(&self) -> F {
        self.0[2]
    }
    pub fn d(&self) -> F {
        self.0[3]
    }
    pub fn e(&self) -> F {
        self.0[4]
    }
    pub fn is_left_imm(&self) -> F {
        self.0[3]
    }
    pub fn is_imm(&self) -> F {
        self.0[4]
    }
    pub fn imm32(&self) -> Word<F> {
        Word::from_components_le([self.0[1], self.0[2], self.0[3], self.0[4]])
    }
}
pub enum OperandsIndex {
    A,
    B,
    C,
    D,
    E,
}

impl<F> Index<OperandsIndex> for Operands<F> {
    type Output = F;

    fn index(&self, index: OperandsIndex) -> &F {
        match index {
            OperandsIndex::A => &self.0[0],
            OperandsIndex::B => &self.0[1],
            OperandsIndex::C => &self.0[2],
            OperandsIndex::D => &self.0[3],
            OperandsIndex::E => &self.0[4],
        }
    }
}

impl<F> IndexMut<OperandsIndex> for Operands<F> {
    fn index_mut(&mut self, index: OperandsIndex) -> &mut F {
        match index {
            OperandsIndex::A => &mut self.0[0],
            OperandsIndex::B => &mut self.0[1],
            OperandsIndex::C => &mut self.0[2],
            OperandsIndex::D => &mut self.0[3],
            OperandsIndex::E => &mut self.0[4],
        }
    }
}

impl<T> Operands<T> {
    pub fn iter(&self) -> Iter<T> {
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<T> {
        self.0.iter_mut()
    }
}

impl<T> IntoIterator for Operands<T> {
    type Item = T;
    type IntoIter = IntoIter<T, OPERAND_ELEMENTS>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<F: Field> Operands<F> {
    pub fn from_i32_slice(slice: &[i32]) -> Self {
        let mut operands = [F::zero(); OPERAND_ELEMENTS];
        for (i, &operand) in slice.iter().enumerate() {
            let abs = F::from_canonical_u32(operand.unsigned_abs());
            operands[i] = if operand < 0 { -abs } else { abs };
        }
        Self(operands)
    }

    pub fn from_operands_i32(other: &Operands<i32>) -> Self {
        Self(other.0.map(|x| {
            let abs = F::from_canonical_u32(x.unsigned_abs());
            if x < 0 {
                -abs
            } else {
                abs
            }
        }))
    }
}

pub fn convert_machine_code_to_opcode_value(original_opcode: u32) -> u32 {
    if original_opcode == 1 {
        1 // LOAD32
    } else if original_opcode == 2 {
        17 // STORE32
    } else if original_opcode == 3 {
        33 // JAL
    } else if original_opcode == 4 {
        49 // JALV
    } else if original_opcode == 5 {
        65 // BEQ
    } else if original_opcode == 6 {
        81 // BNE
    } else if original_opcode == 7 {
        97 // IMM32
    } else if original_opcode == 8 {
        113 // STOP
    } else if original_opcode == 9 {
        129 // READ_ADVICE
    } else if original_opcode == 10 {
        145 // LOADFP
    } else if original_opcode == 11 {
        161 // LOADU8
    } else if original_opcode == 12 {
        177 // LOADS8
    } else if original_opcode == 13 {
        193 // STOREU8
    } else if original_opcode == 16 {
        16 // FAIL
    } else if original_opcode == 20 {
        209 // MEMCPY
    } else if original_opcode == 100 {
        100 // ADD32
    } else if original_opcode == 101 {
        101 // SUB32
    } else if original_opcode == 102 {
        102 // MUL32
    } else if original_opcode == 109 {
        109 // XOR32
    } else if original_opcode == 111 {
        111 // NE32
    } else if original_opcode == 112 {
        112 // MULHU32
    } else if original_opcode == 113 {
        114 // SRA32
    } else if original_opcode == 114 {
        115 // MULH32
    } else if original_opcode == 115 {
        116 // LTE32
    } else if original_opcode == 116 {
        117 // EQ32
    } else if original_opcode == 117 {
        118 // SLT32
    } else if original_opcode == 118 {
        119 // SLE32
    } else if original_opcode == 200 {
        200 // ADD
    } else if original_opcode == 201 {
        201 // SUB
    } else if original_opcode == 202 {
        202 // MUL
    } else if original_opcode == 300 {
        225 // WRITE
    } else if original_opcode == 120 {
        120 // KECCAKF
    } else if original_opcode == 134 {
        134 // COMBSECP256K1
    } else {
        original_opcode
    }
}

#[derive(Default, Clone, Debug)]
pub struct ProgramROM<F>(pub Vec<InstructionWord<F>>);

impl<F> MemoryFootprint for ProgramROM<F> {
    fn memory_footprint(&self) -> usize {
        self.0.memory_footprint()
    }
}

impl<F> ProgramROM<F> {
    pub fn new(instructions: Vec<InstructionWord<F>>) -> Self {
        Self(instructions)
    }

    pub fn get_instruction(&self, pc: u32) -> &InstructionWord<F> {
        debug_assert!(pc < self.0.len() as u32, "PC out of bounds");
        &self.0[pc as usize]
    }
}

impl ProgramROM<i32> {
    pub fn from_machine_code(mc: &[u8], should_convert_opcode: bool) -> Self {
        let mut instructions = Vec::new();
        for chunk in mc.chunks_exact(INSTRUCTION_ELEMENTS * 4) {
            instructions.push(InstructionWord {
                opcode: if should_convert_opcode {
                    convert_machine_code_to_opcode_value(LittleEndian::read_u32(&chunk[0..4]))
                } else {
                    LittleEndian::read_u32(&chunk[0..4])
                },
                operands: Operands([
                    LittleEndian::read_i32(&chunk[4..8]),
                    LittleEndian::read_i32(&chunk[8..12]),
                    LittleEndian::read_i32(&chunk[12..16]),
                    LittleEndian::read_i32(&chunk[16..20]),
                    LittleEndian::read_i32(&chunk[20..24]),
                ]),
            });
        }
        Self(instructions)
    }

    #[cfg(feature = "std")]
    pub fn from_file(filename: &str) -> std::io::Result<Self> {
        use byteorder::ReadBytesExt;
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);
        let mut instructions = Vec::new();

        while let Ok(opcode) = reader.read_u32::<LittleEndian>() {
            let mut operands_arr = [0i32; OPERAND_ELEMENTS];
            for operand in operands_arr.iter_mut() {
                *operand = reader.read_i32::<LittleEndian>()?;
            }
            let operands = Operands(operands_arr);
            instructions.push(InstructionWord {
                opcode: opcode,
                operands,
            });
        }

        Ok(ProgramROM::new(instructions))
    }
}
