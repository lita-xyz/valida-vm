use crate::common::{bb_state_strategy, test_arith, BBState, DataStrategy, MemorySize};

use valida_alu_u32::{
    com::{Eq32Instruction, Ne32Instruction},
    lt::{Lt32Instruction, Lte32Instruction, Sle32Instruction, Slt32Instruction},
};

use proptest::prelude::*;

fn cond_strat() -> BoxedStrategy<BBState> {
    bb_state_strategy(DataStrategy::Numerical, MemorySize::default()).boxed()
}

proptest! {
    #[test]
    fn lte_property(state in cond_strat()) {
        test_arith::<Lte32Instruction, _>
            (|a, b| (a <= b) as u32, state, false)?
    }

    #[test]
    fn lt_property(state in cond_strat()) {
        test_arith::<Lt32Instruction, _>
            (|a, b| (a < b) as u32, state, false)?
    }

    #[test]
    fn sle_property(state in cond_strat()) {
        test_arith::<Sle32Instruction, _>
            (|a, b| ((a as i32) <= (b as i32)) as u32, state, false)?
    }

    #[test]
    fn slt_property(state in cond_strat()) {
        test_arith::<Slt32Instruction, _>
            (|a, b| ((a as i32) < (b as i32)) as u32, state, false)?
    }

    #[test]
    fn eq_property(state in cond_strat()) {
        test_arith::<Eq32Instruction, _>
            (|a, b| (a == b) as u32, state, false)?
    }

    #[test]
    fn ne_property(state in cond_strat()) {
        test_arith::<Ne32Instruction, _>
            (|a, b| (a != b) as u32, state, false)?
    }
}
