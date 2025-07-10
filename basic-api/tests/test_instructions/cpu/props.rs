use crate::common::{
    bb_state_strategy, calc_offset, new_addr_strat, with_unique, BBMachine, BBState, MemorySize,
};
use crate::testmachine::TestMachine;

use p3_baby_bear::BabyBear;
use std::ops::RangeInclusive;
use valida_basic_api::machine::basic::pc_strategy;
use valida_basic_api::ValidaRuntime;
use valida_cpu::{
    BeqInstruction, BneInstruction, JalInstruction, JalvInstruction, StopInstruction,
};
use valida_machine::{Instruction, Machine, Operands};
use valida_machine::{MemoryAccessTimestamp, MemoryRecord, StorageBackendTrait};
use valida_memory::Operation::{DummyRead, Read, Write};

use proptest::prelude::*;
use proptest::sample::Selector;

use crate::common::{fp_strategy, DataStrategy};

static FP_STRAT: fn() -> RangeInclusive<u32> = fp_strategy::<BabyBear>;
static PC_STRAT: fn() -> RangeInclusive<u32> = || pc_strategy::<BabyBear>();

fn cpu_strat() -> BoxedStrategy<BBState> {
    bb_state_strategy(DataStrategy::Program, MemorySize::default()).boxed()
}

proptest! {
    #[test]
    // [fp + a] = 24 * (pc + 1)
    // pc = b / 24
    // fp = fp + c
    fn jal_property(
        (mut state, dst) in new_addr_strat::<BabyBear>(DataStrategy::Program, MemorySize(0)),
        expected_fp in FP_STRAT(),
        expected_pc in PC_STRAT(),
    ) {
        let (old_pc, old_fp) = state.registers();
        let clk = state.clk();

        let dst_offset = calc_offset(old_fp, dst);
        let fp_offset = calc_offset(old_fp, expected_fp);

        let expected_ops = match state.memory_backend.get(&dst) {
            None => vec![
            DummyRead(dst, Default::default()),
            Write(dst, (24 * (old_pc + 1)).into()),
        ],
        _ => vec![Write(dst, (24 * (old_pc + 1)).into())],
        };

        let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
        runtime.memory_backend = state.memory_backend.clone();
        let mut running_machine = state.machine.start(&mut runtime);
        JalInstruction::execute(&mut running_machine, Operands([dst_offset, (24 * expected_pc) as i32, fp_offset, 0, 0]));
        state.machine = BBMachine::stop(running_machine);
        state.memory_backend = runtime.memory_backend.clone();
        let (new_pc, new_fp) =state.registers();
        let result = state.get(dst).expect("nonempty");

        prop_assert_eq!(expected_pc, new_pc);
        prop_assert_eq!(expected_fp, new_fp);
        let result_u32: u32 = result.into();
        prop_assert_eq!(24 * (old_pc + 1), result_u32);
        prop_assert_eq!(state.memory_log_size(), 1);
        prop_assert_eq!(state.memory_log(clk), expected_ops);
    }

    #[test]
    // [fp + a] = 24 * (pc + 1)
    // pc = [fp + b]
    // fp += [fp + c]
    fn jalv_property(
        (state, dst) in new_addr_strat::<BabyBear>(DataStrategy::Program, MemorySize(2)),
        sel: Selector,
        expected_fp in FP_STRAT(),
        expected_pc in PC_STRAT(),
    ) {
        with_unique(sel, state, |mut state, old_pc, old_fp, clk, v| {
            let [b, c] = v;
            let [dst_offset, b_offset, c_offset, fp_offset] =
              [dst, b, c, expected_fp].map(|x| calc_offset(old_fp, x));

           state.set(b, expected_pc * 24);
           state.set(c, fp_offset as u32);

            let expected_ops = match state.memory_backend.get(&dst) {

            None => vec![
                DummyRead(dst, Default::default()),
                Write(dst, (24 * (old_pc + 1)).into()),
                Read(b, MemoryRecord{value: (expected_pc * 24).into(), last_accessed: MemoryAccessTimestamp::ThisSegment}),
                Read(c, MemoryRecord{value: (fp_offset as u32).into(), last_accessed: MemoryAccessTimestamp::ThisSegment}),
            ], _ => vec![
                Write(dst, (24 * (old_pc + 1)).into()),
                Read(b, MemoryRecord{value: (expected_pc * 24).into(), last_accessed: MemoryAccessTimestamp::ThisSegment}),
                Read(c, MemoryRecord{value: (fp_offset as u32).into(), last_accessed: MemoryAccessTimestamp::ThisSegment}),
            ],
            };

            let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
            runtime.memory_backend = state.memory_backend.clone();
            let mut running_machine = state.machine.start(&mut runtime);
            JalvInstruction::execute(&mut running_machine, Operands([dst_offset, b_offset, c_offset, 0, 0]));
            let state = BBState {machine: BBMachine::stop(running_machine), memory_backend: runtime.memory_backend};
            let (new_pc, new_fp) = state.registers();
            let result = state.get(dst).expect("nonempty");

            prop_assert_eq!(expected_pc, new_pc);
            prop_assert_eq!(expected_fp, new_fp);
            let result_u32: u32 = result.into();
            prop_assert_eq!(24 * (old_pc + 1), result_u32);
            prop_assert_eq!(state.memory_log_size(), 1);
            prop_assert_eq!(state.memory_log(clk), expected_ops);
            Ok(())
          })?
    }

    #[test]
    // if [fp + b] == [fp + c] => pc = a / 24
    // else => pc += 1
    fn beq_property(
        state in cpu_strat(),
        sel: Selector,
        dst in PC_STRAT()
    ) {
        let (old_pc, old_fp) = state.registers();
        let clk = state.clk();

        let (b, bv) = sel.select(state.assigned_cells());
        let (c, cv) = sel.select(state.assigned_cells());
        let expected_pc = if bv == cv { dst / 24 } else { old_pc + 1 };

        let [b_offset, c_offset] = [b, c].map(|x| calc_offset(old_fp, x));

        let expected_ops = vec![
            Read(b, MemoryRecord{value: bv, last_accessed: MemoryAccessTimestamp::ThisSegment}),
            Read(c, MemoryRecord{value: cv, last_accessed: MemoryAccessTimestamp::ThisSegment}),
        ];

        let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
        runtime.memory_backend = state.memory_backend.clone();
        let mut running_machine = state.machine.start(&mut runtime);
        BeqInstruction::execute(&mut running_machine, Operands([dst as i32, b_offset, c_offset, 0, 0]));
        let state = BBState {machine: BBMachine::stop(running_machine), memory_backend: runtime.memory_backend};
        let (new_pc, new_fp) = state.registers();

        prop_assert_eq!(expected_pc, new_pc);
        prop_assert_eq!(old_fp, new_fp);
        prop_assert_eq!(state.memory_log_size(), 1);
        prop_assert_eq!(state.memory_log(clk), expected_ops);
    }

    #[test]
    // if [fp + b] != [fp + c] => pc = a / 24
    // else => pc += 1
    fn bne_property(
        state in cpu_strat(),
        sel: Selector,
        dst in PC_STRAT()
    ) {
        let (old_pc, old_fp) = state.registers();
        let clk = state.clk();

        let (b, bv) = sel.select(state.assigned_cells());
        let (c, cv) = sel.select(state.assigned_cells());
        let expected_pc = if bv != cv { dst / 24 } else { old_pc + 1 };

        let [b_offset, c_offset] = [b, c].map(|x| calc_offset(old_fp, x));

        let expected_ops = vec![
            Read(b, MemoryRecord{value: bv, last_accessed: MemoryAccessTimestamp::ThisSegment}),
            Read(c, MemoryRecord{value: cv, last_accessed: MemoryAccessTimestamp::ThisSegment}),
        ];

        let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
        runtime.memory_backend = state.memory_backend.clone();
        let mut running_machine = state.machine.start(&mut runtime);
        BneInstruction::execute(&mut running_machine, Operands([dst as i32, b_offset, c_offset, 0, 0]));
        let state = BBState {machine: BBMachine::stop(running_machine), memory_backend: runtime.memory_backend};
        let (new_pc, new_fp) = state.registers();

        prop_assert_eq!(expected_pc, new_pc);
        prop_assert_eq!(old_fp, new_fp);
        prop_assert_eq!(state.memory_log_size(), 1);
        prop_assert_eq!(state.memory_log(clk), expected_ops);
    }

    #[test]
    fn stop_property(state in cpu_strat()) {
        let (old_pc, old_fp) = state.registers();
        let clk = state.clk();
        let expected_ops = vec![];

        let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
        runtime.memory_backend = state.memory_backend.clone();
        let mut running_machine = state.machine.start( &mut runtime);
        StopInstruction::execute(&mut running_machine, Operands([0, 0, 0, 0, 0]));
        let state = BBState {machine: BBMachine::stop(running_machine), memory_backend: runtime.memory_backend};
        let (new_pc, new_fp) = state.registers();

        prop_assert_eq!(old_pc, new_pc);
        prop_assert_eq!(old_fp, new_fp);
        prop_assert_eq!(state.memory_log_size(), 0);
        prop_assert_eq!(state.memory_log(clk), expected_ops);
    }
}
