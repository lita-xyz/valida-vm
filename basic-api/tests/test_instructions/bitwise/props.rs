use crate::common::{bb_state_strategy, test_arith, BBState, DataStrategy, MemorySize};
use proptest::prelude::*;

use valida_alu_u32::{
    bitwise::{And32Instruction, Or32Instruction, Xor32Instruction},
    shift::{Shl32Instruction, Shr32Instruction, Sra32Instruction},
};

fn bit_strat() -> BoxedStrategy<BBState> {
    bb_state_strategy(DataStrategy::Bitwise, MemorySize::default()).boxed()
}

proptest! {
    #[test]
    fn and_property(state in bit_strat()) {
        test_arith::<And32Instruction, _>
            (|a, b| a & b, state, false)?
    }

    #[test]
    fn xor_property(state in bit_strat()) {
        test_arith::<Xor32Instruction, _>
            (|a, b| a ^ b, state, false)?
    }

    #[test]
    fn or_property(state in bit_strat()) {
        test_arith::<Or32Instruction, _>
            (|a, b| a | b, state, false)?
    }

    #[test]
    fn shl_property(state in bit_strat()) {
        test_arith::<Shl32Instruction, _>
            (|a, b| a << (b & 0x1f), state, false)?
    }

    #[test]
    fn shr_property(state in bit_strat()) {
        test_arith::<Shr32Instruction, _>
            (|a, b| a >> (b & 0x1f), state, false)?
    }

    #[test]
    fn sra_property(state in bit_strat()) {
        test_arith::<Sra32Instruction, _>
            (|a, b| ((a as i32) >> (b & 0x1f)) as u32, state, false)?
    }
}
