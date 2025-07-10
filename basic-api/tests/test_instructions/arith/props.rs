use crate::common::{bb_state_strategy, test_arith, BBState, DataStrategy, MemorySize};

use std::num::Wrapping;
use valida_alu_u32::{
    add::Add32Instruction,
    div::{Div32Instruction, SDiv32Instruction},
    mul::{Mul32Instruction, Mulhs32Instruction, Mulhu32Instruction},
    sub::Sub32Instruction,
};

use proptest::prelude::*;

fn arith_strat() -> BoxedStrategy<BBState> {
    bb_state_strategy(DataStrategy::Numerical, MemorySize::default()).boxed()
}

proptest! {
    #[test]
    fn add_property(state in arith_strat()) {
        test_arith::<Add32Instruction, _>(
            |a, b| (Wrapping::<u32>(a) + Wrapping(b)).0,
            state,
            false)?
    }

    #[test]
    fn sub_property(state in arith_strat()) {
        test_arith::<Sub32Instruction, _>(
            |a, b| (Wrapping::<u32>(a) - Wrapping(b)).0,
            state,
            false)?
    }

    #[test]
    fn mul_property(state in arith_strat()) {
        test_arith::<Mul32Instruction, _>(
            |a, b| (Wrapping::<u32>(a) * Wrapping(b)).0,
            state,
            false)?
    }

    #[test]
    fn mulhs_property(state in arith_strat()) {
        test_arith::<Mulhs32Instruction, _>(
            |a, b| {
                let a: u32 = a;
                let b: u32 = b;
                let a = Wrapping((a as i32) as i64);
                let b = Wrapping((b as i32) as i64);
                (((a * b).0 >> 32) as i32) as u32
            },
            state,
            false)?
    }

    #[test]
    fn mulhu_property(state in arith_strat()) {
        test_arith::<Mulhu32Instruction, _>(
            |a, b| {
                let a: u32 = a;
                let b: u32 = b;
                let a = Wrapping(a as u64);
                let b = Wrapping(b as u64);
                ((a * b).0 >> 32) as u32
            },
            state,
            false)?
    }

    #[test]
    fn div_property(state in arith_strat()) {
        test_arith::<Div32Instruction, _>(
            |a, b| (Wrapping::<u32>(a) / Wrapping(b)).0,
            state,
            true)?
    }

    #[test]
    fn sdiv_property(state in arith_strat()) {
        test_arith::<SDiv32Instruction, _>(
            |a, b| {
                let a: u32 = a;
                let b: u32 = b;
                (a as i32 / b as i32) as u32
            },
            state,
            true)?
    }
}
