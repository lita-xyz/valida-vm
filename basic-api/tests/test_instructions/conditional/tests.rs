use crate::common::{test_limm, test_mem, test_rimm};

use valida_alu_u32::{
    com::{Eq32Instruction, Ne32Instruction},
    lt::{Lt32Instruction, Lte32Instruction, Sle32Instruction, Slt32Instruction},
};

#[test]
fn test_lte_lt() {
    test_mem::<Lte32Instruction>(1, 0x80000000, 1);
}

#[test]
fn test_lte_eq() {
    test_mem::<Lte32Instruction>(5, 5, 1);
}

#[test]
fn test_lte_gt() {
    test_mem::<Lte32Instruction>(0x80000000, 1, 0);
}

#[test]
fn test_lte_imm() {
    test_limm::<Lte32Instruction>(0, i32::min as u32, 1);
    test_rimm::<Lte32Instruction>(0, i32::min as u32, 1);
}

#[test]
fn test_lt_lt() {
    test_mem::<Lt32Instruction>(1, 0x80000000, 1);
}

#[test]
fn test_lt_eq() {
    test_mem::<Lt32Instruction>(5, 5, 0);
}

#[test]
fn test_lt_gt() {
    test_mem::<Lt32Instruction>(0x80000000, 1, 0);
}

#[test]
fn test_lt_imm() {
    test_limm::<Lt32Instruction>(0, i32::min as u32, 1);
    test_rimm::<Lt32Instruction>(0, i32::min as u32, 1);
}

#[test]
fn test_sle_lt() {
    test_mem::<Sle32Instruction>(0x80000000, 1, 1);
}

#[test]
fn test_sle_eq() {
    test_mem::<Sle32Instruction>(5, 5, 1);
}

#[test]
fn test_sle_gt() {
    test_mem::<Sle32Instruction>(0x7fffffff, 0x80000000, 0);
}

#[test]
fn test_sle_imm() {
    test_limm::<Sle32Instruction>(0x80000000, 1, 1);
    test_rimm::<Sle32Instruction>(0x80000000, 1, 1);
}

#[test]
fn test_slt_lt() {
    test_mem::<Slt32Instruction>(0x80000000, 1, 1);
}

#[test]
fn test_slt_eq() {
    test_mem::<Slt32Instruction>(5, 5, 0);
}

#[test]
fn test_slt_gt() {
    test_mem::<Slt32Instruction>(0x7fffffff, 0x80000000, 0);
}

#[test]
fn test_slt_imm() {
    test_limm::<Slt32Instruction>(0x80000000, 1, 1);
    test_rimm::<Slt32Instruction>(0x80000000, 1, 1);
}

#[test]
fn test_eq_eq() {
    test_mem::<Eq32Instruction>(0xaaaa, 0xaaaa, 1);
}

#[test]
fn test_eq_ne() {
    test_mem::<Eq32Instruction>(0xaaaa, 0xaa55, 0);
}

#[test]
fn test_eq_rimm() {
    test_rimm::<Eq32Instruction>(0x5555, 0x5555, 1);
}

#[test]
fn test_ne_eq() {
    test_mem::<Ne32Instruction>(0xaaaa, 0xaaaa, 0);
}

#[test]
fn test_ne_ne() {
    test_mem::<Ne32Instruction>(0xaaaa, 0xaa55, 1);
}

#[test]
fn test_ne_rimm() {
    test_rimm::<Ne32Instruction>(0x5555, 0x5555, 0);
}
