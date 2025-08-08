use num_enum::TryFromPrimitive;

pub const BYTES_PER_INSTR: u32 = 24; // 4 bytes per word * 6 words per instruction

#[repr(u32)]
#[derive(Debug, TryFromPrimitive, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Opcode {
    // Opcodes that set any opcode_flags other than is_bus_op of the cpu chip
    // to 1 have their first digit in hexadecimal equal to 1
    LOAD32 = 1,
    STORE32 = 17,
    JAL = 33,
    JALV = 49,
    BEQ = 65,
    BNE = 81,
    IMM32 = 97,
    STOP = 113,
    #[allow(non_camel_case_types)]
    READ_ADVICE = 129,
    LOADFP = 145,
    LOADU8 = 161,
    LOADS8 = 177,
    STOREU8 = 193,
    FAIL = 16,
    MEMCPY = 209,

    WRITE = 225,

    // is_bus_op of the cpu chip is set to 1 if and only if the first digit in the base-16
    // representation of the opcode is NOT 1.
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
    SRA32 = 114,
    MULHS32 = 115,
    LTE32 = 116,
    EQ32 = 117,
    SLT32 = 118,
    SLE32 = 119,
    ADD = 200,
    SUB = 201,
    MUL = 202,

    // Note that atm we know that opcodes over 256 are a problem and so we have to use smaller numbers than 256
    // Crypto

    // KECCAK
    KECCAKF = 241,

    COMBSECP256K1 = 134,
    SMULSECP256K1,
    SINVSECP256K1,
    MULSSECP256K1,
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
