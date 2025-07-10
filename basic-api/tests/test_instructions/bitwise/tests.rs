use crate::common::{test_mem, test_rimm};

use valida_alu_u32::{
    bitwise::{And32Instruction, Or32Instruction, Xor32Instruction},
    shift::{Shl32Instruction, Shr32Instruction, Sra32Instruction},
};

#[test]
fn test_and() {
    test_mem::<And32Instruction>(0x0011, 0x0101, 0x0001);
}

#[test]
fn test_and_imm() {
    test_rimm::<And32Instruction>(0x0011, 0x0101, 0x0001);
}

#[test]
fn test_or() {
    test_mem::<Or32Instruction>(0x0011, 0x0101, 0x0111);
}

#[test]
fn test_or_imm() {
    test_rimm::<Or32Instruction>(0x0011, 0x0101, 0x0111);
}

#[test]
fn test_xor() {
    test_mem::<Xor32Instruction>(0x0011, 0x0101, 0x0110);
}

#[test]
fn test_xor_imm() {
    test_rimm::<Xor32Instruction>(0x0011, 0x0101, 0x0110);
}

#[test]
fn test_shl() {
    test_mem::<Shl32Instruction>(0xaaaaaaaa, 33, 0x55555554);
}

#[test]
fn test_shl_imm() {
    test_rimm::<Shl32Instruction>(0xaaaaaaaa, 33, 0x55555554);
}

#[test]
fn test_shr() {
    test_mem::<Shr32Instruction>(0x55555555, 33, 0x2aaaaaaa);
}

#[test]
fn test_shr_imm() {
    test_rimm::<Shr32Instruction>(0x55555555, 33, 0x2aaaaaaa);
}

#[test]
fn test_sra_positive() {
    test_mem::<Sra32Instruction>(0x55555555, 33, 0x2aaaaaaa);
}

#[test]
fn test_sra_negative() {
    test_mem::<Sra32Instruction>(i32::MIN as u32, 31, u32::MAX);
}

#[test]
fn test_sra_imm() {
    test_rimm::<Sra32Instruction>(0x55555555, 33, 0x2aaaaaaa);
}
