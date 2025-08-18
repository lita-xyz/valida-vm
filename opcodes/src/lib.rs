use num_enum::TryFromPrimitive;

use p3_field::PrimeField32;

pub const BYTES_PER_INSTR: u32 = 24; // 4 bytes per word * 6 words per instruction

#[repr(u32)]
#[derive(Debug, TryFromPrimitive, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Opcode {
    LOAD32 = 1,
    STORE32 = 2,
    JAL = 3,
    JALV = 4,
    BEQ = 5,
    BNE = 6,
    IMM32 = 7,
    STOP = 8,
    #[allow(non_camel_case_types)]
    READ_ADVICE = 9,
    LOADFP = 10,
    LOADU8 = 11,
    LOADS8 = 12,
    STOREU8 = 13,
    FAIL = 16,
    MEMCPY = 20,

    ADD32 = 100,
    SUB32 = 101,
    MUL32 = 102,
    DIV32 = 103,
    SDIV32 = 110,
    LT32 = 104,
    SHL32 = 105,
    SHR32 = 106,
    AND32 = 107,
    OR32 = 108,
    XOR32 = 109,
    NE32 = 111,
    MULHU32 = 112,
    SRA32 = 113,
    MULHS32 = 114,
    LTE32 = 115,
    EQ32 = 116,
    SLT32 = 117,
    SLE32 = 118,
    ADD = 200,
    SUB = 201,
    MUL = 202,
    WRITE = 300,

    // Note that atm we know that opcodes over 256 are a problem and so we have to use smaller numbers than 256
    // Crypto

    // KECCAK
    KECCAKF = 120,

    COMBSECP256K1 = 134,
    SMULSECP256K1,
    SINVSECP256K1,
    MULSSECP256K1,
}

pub fn map_opcode(opcode: u32) -> u32 {
    match opcode {
        1 => 1,     // LOAD32
        2 => 17,    // STORE32
        3 => 33,    // JAL
        4 => 49,    // JALV
        5 => 65,    // BEQ
        6 => 81,    // BNE
        7 => 97,    // IMM32
        8 => 113,   // STOP
        9 => 129,   // READ_ADVICE
        10 => 145,  // LOADFP
        11 => 161,  // LOADU8
        12 => 177,  // LOADS8
        13 => 193,  // STOREU8
        16 => 16,   // FAIL
        20 => 209,  // MEMCPY
        100 => 100, // ADD32
        101 => 101, // SUB32
        102 => 102, // MUL32
        109 => 109, // XOR32
        111 => 111, // NE32
        112 => 112, // MULHU32
        113 => 114, // SRA32
        114 => 115, // MULH32
        115 => 116, // LTE32
        116 => 117, // EQ32
        117 => 118, // SLT32
        118 => 119, // SLE32
        200 => 200, // ADD
        201 => 201, // SUB
        202 => 202, // MUL
        300 => 225, // WRITE
        120 => 241, // KECCAKF
        134 => 134, // COMBSECP256K1
        135 => 135, // SMULSECP256K1
        136 => 136, // SINVSECP256K1
        137 => 137, // MULSSECP256K1
        _ => 0,
    }
}

pub fn map_opcode_to_field_value<F: PrimeField32>(opcode: u32) -> F {
    F::from_canonical_u32(map_opcode(opcode))
}

pub fn unmap_opcode(opcode_value: u32) -> u32 {
    match opcode_value {
        1 => 1,     // LOAD32
        17 => 2,    // STORE32
        33 => 3,    // JAL
        49 => 4,    // JALV
        65 => 5,    // BEQ
        81 => 6,    // BNE
        97 => 7,    // IMM32
        113 => 8,   // STOP
        129 => 9,   // READ_ADVICE
        145 => 10,  // LOADFP
        161 => 11,  // LOADU8
        177 => 12,  // LOADS8
        193 => 13,  // STOREU8
        16 => 16,   // FAIL
        209 => 20,  // MEMCPY
        100 => 100, // ADD32
        101 => 101, // SUB32
        102 => 102, // MUL32
        109 => 109, // XOR32
        111 => 111, // NE32
        112 => 112, // MULHU32
        114 => 113, // SRA32
        115 => 114, // MULH32
        116 => 115, // LTE32
        117 => 116, // EQ32
        118 => 117, // SLT32
        119 => 118, // SLE32
        200 => 200, // ADD
        201 => 201, // SUB
        202 => 202, // MUL
        225 => 300, // WRITE
        241 => 120, // KECCAKF
        134 => 134, // COMBSECP256K1
        135 => 135, // SMULSECP256K1
        136 => 136, // SINVSECP256K1
        137 => 137, // MULSSECP256K1
        _ => 0,
    }
}

pub fn unmap_field_value_to_opcode<F: PrimeField32>(value: F) -> u32 {
    unmap_opcode(value.as_canonical_u32())
}

macro_rules! declare_opcode {
    ($opcode : ident) => {
        pub const $opcode: u32 = Opcode::$opcode as u32;
    };
}

// TODO: should combine enum together

// CORE
declare_opcode!(LOAD32);
declare_opcode!(STORE32);
declare_opcode!(JAL);
declare_opcode!(JALV);
declare_opcode!(BEQ);
declare_opcode!(BNE);
declare_opcode!(IMM32);
declare_opcode!(STOP);
declare_opcode!(FAIL);
declare_opcode!(MEMCPY);
declare_opcode!(LOADFP);
declare_opcode!(LOADU8);
declare_opcode!(LOADS8);
declare_opcode!(STOREU8);

// NONDETERMINISTIC
declare_opcode!(READ_ADVICE);

// U32 ALU
declare_opcode!(ADD32);
declare_opcode!(SUB32);
declare_opcode!(MUL32);
declare_opcode!(DIV32);
declare_opcode!(SDIV32);
declare_opcode!(LT32);
declare_opcode!(SHL32);
declare_opcode!(SHR32);
declare_opcode!(AND32);
declare_opcode!(OR32);
declare_opcode!(XOR32);
declare_opcode!(NE32);
declare_opcode!(MULHU32);
declare_opcode!(SRA32);
declare_opcode!(MULHS32);
declare_opcode!(LTE32);
declare_opcode!(EQ32);
declare_opcode!(SLT32);
declare_opcode!(SLE32);

// NATIVE FIELD
declare_opcode!(ADD);
declare_opcode!(SUB);
declare_opcode!(MUL);

// OUTPUT
declare_opcode!(WRITE);

// KECCAK
declare_opcode!(KECCAKF);

declare_opcode!(COMBSECP256K1);
declare_opcode!(SMULSECP256K1);
declare_opcode!(SINVSECP256K1);
declare_opcode!(MULSSECP256K1);
