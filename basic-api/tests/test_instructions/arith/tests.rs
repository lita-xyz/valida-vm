use crate::common::{test_mem, test_rimm};

use valida_alu_u32::{
    add::Add32Instruction,
    div::{Div32Instruction, SDiv32Instruction},
    mul::{Mul32Instruction, Mulhs32Instruction, Mulhu32Instruction},
    sub::Sub32Instruction,
};

#[test]
fn test_add() {
    test_mem::<Add32Instruction>(0x0011, 0x101, 0x0112);
}

#[test]
fn test_add_imm() {
    test_rimm::<Add32Instruction>(0x0011, 0x101, 0x0112);
}

#[test]
fn test_add_overflow() {
    test_mem::<Add32Instruction>(u32::MAX, 1, 0);
}

#[test]
fn test_sub() {
    test_mem::<Sub32Instruction>(0x1110, 0x0101, 0x100f);
}

#[test]
fn test_sub_imm() {
    test_rimm::<Sub32Instruction>(0x1110, 0x0101, 0x100f);
}

#[test]
fn test_sub_underflow() {
    test_mem::<Sub32Instruction>(0, 1, u32::MAX);
}

#[test]
fn test_mul() {
    test_mem::<Mul32Instruction>(17, 257, 4369);
}

#[test]
fn test_mul_imm() {
    test_rimm::<Mul32Instruction>(17, 257, 4369);
}

#[test]
fn test_mul_overflow1() {
    test_mem::<Mul32Instruction>(u32::MAX, 2, u32::MAX - 1);
}

#[test]
fn test_mul_overflow2() {
    test_mem::<Mul32Instruction>(u32::MAX, u32::MAX, 1);
}

#[test]
fn test_mulhs_small_positive() {
    test_mem::<Mulhs32Instruction>(0xffff, 0xffff, 0);
}

#[test]
fn test_mulhs_small_negative() {
    test_mem::<Mulhs32Instruction>(0xfffffff0, 0xfffffff0, 0);
}

#[test]
fn test_mulhs_small_minusone() {
    test_mem::<Mulhs32Instruction>(u32::MAX, 1, u32::MAX);
}

#[test]
fn test_mulhs_large_positive() {
    test_mem::<Mulhs32Instruction>(0x7fffffff, 0x7fffffff, 0x3fffffff);
}

#[test]
fn test_mulhs_large_negative() {
    test_mem::<Mulhs32Instruction>(0x80000000, 0x80000000, 0x40000000);
}

#[test]
fn test_mulhs_imm() {
    test_rimm::<Mulhs32Instruction>(0xffff, 0xffff, 0);
}

#[test]
fn test_mulhu_small() {
    test_mem::<Mulhu32Instruction>(0xffff, 0xffff, 0);
}

#[test]
fn test_mulhu_large() {
    test_mem::<Mulhu32Instruction>(u32::MAX, u32::MAX, u32::MAX - 1);
}

#[test]
fn test_mulhu_imm() {
    test_rimm::<Mulhu32Instruction>(0xffff, 0xffff, 0);
}

#[test]
fn test_div() {
    test_mem::<Div32Instruction>(0x1010, 0x0011, 0x1010 / 0x0011);
}

#[test]
fn test_div_imm() {
    test_rimm::<Div32Instruction>(0x1010, 0x0011, 0x1010 / 0x0011);
}

#[test]
#[should_panic(expected = "attempt to divide by zero")]
fn test_div_zero() {
    test_mem::<Div32Instruction>(1, 0, 0);
}

#[test]
fn test_sdiv1() {
    test_mem::<SDiv32Instruction>(-1234567i32 as u32, -51i32 as u32, 1234567 / 51);
}

#[test]
fn test_sdiv2() {
    test_mem::<SDiv32Instruction>(u32::MAX - 2, 2, u32::MAX);
}

#[test]
#[should_panic(expected = "attempt to divide by zero")]
fn test_sdiv_zero() {
    test_mem::<SDiv32Instruction>(1, 0, 0);
}
