use crate::common::{
    bb_state_strategy, calc_offset, new_addr, with_unique, BBMachine, BBState, DataStrategy,
    MemorySize, MAX_ADDR,
};
use crate::testmachine::TestMachine;

use p3_baby_bear::BabyBear;
use valida_basic_api::{BasicMachine, ValidaRuntime};
use valida_cpu::{
    Imm32Instruction, Load32Instruction, LoadFpInstruction, LoadS8Instruction, LoadU8Instruction,
    Store32Instruction, StoreU8Instruction,
};
use valida_machine::{Instruction, Machine, Operands, StorageBackendTrait};
use valida_memory::Operation::{DummyRead, Read, Write};

use proptest::prelude::*;
use proptest::sample::Selector;

use rand::Rng;

fn mem_strat() -> BoxedStrategy<BBState> {
    bb_state_strategy(DataStrategy::Numerical, MemorySize::default()).boxed()
}

proptest! {
    #[test]
    // [fp + a] = [[fp + c]]
    fn load32_property(state in mem_strat(), sel: Selector) {
        with_unique(sel, state, |mut state, old_pc, old_fp, clk, [j, k]| {
            let mut rng = rand::thread_rng();
            let expected = *state.memory_backend.get(&k).expect("nonempty");
           state.set(j, k);


            // ensure that dst is different from j and k
            let (dst, dst_offset) = { let (mut dst, mut dst_offset) = new_addr(old_fp, *MAX_ADDR, &mut rng);
                while dst == j || dst == k { (dst, dst_offset) = new_addr(old_fp, *MAX_ADDR, &mut rng); }
                (dst, dst_offset)
            };
            let offset = calc_offset(old_fp, j);

            let expected_ops = match state.memory_backend.get(&dst) {
                None => vec![
                Read(j, k.into()),
                Read(k, expected),
                DummyRead(dst, Default::default()),
                Write(dst, expected.value),
                ],
                _ => vec![Read(j, k.into()), Read(k, expected), Write(dst, expected.value)],
            };

            let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
            runtime.memory_backend = state.memory_backend;
            let mut running_machine = state.machine.start(&mut runtime);
            Load32Instruction::execute(&mut running_machine, Operands([dst_offset, 0, offset, 0, 0]));
            state = BBState {machine: BBMachine::stop(running_machine), memory_backend: runtime.memory_backend};
            let (new_pc, new_fp) =state.registers();
            let result =state.get(dst).expect("nonempty");

            prop_assert_eq!(old_pc + 1, new_pc);
            prop_assert_eq!(old_fp, new_fp);
            prop_assert_eq!(expected.value, result);
            prop_assert_eq!(state.memory_log_size(), 1);
            prop_assert_eq!(state.memory_log(clk), expected_ops);
            Ok(())
        })?
    }

    #[test]
    fn loads8_property(state in mem_strat(), sel: Selector) {
        let byte = sel.select([0, 1, 2, 3]);
        with_unique(sel, state, |mut state, old_pc, old_fp, clk, [j, k]| {
            let mut rng = rand::thread_rng();
            let read = *state.memory_backend.get(&k).expect("nonempty");
            let expected = *read.value.index_le(byte as usize) as i8 as i32 as u32;
           state.set(j, k + byte);

            // ensure that dst is different from j and k
            let (dst, dst_offset) = {
                let (mut dst, mut dst_offset) = new_addr(old_fp, *MAX_ADDR, &mut rng);
                while dst == j || dst == k { (dst, dst_offset) = new_addr(old_fp, *MAX_ADDR, &mut rng); }
                (dst, dst_offset)
            };
            let offset = calc_offset(old_fp, j);

            let expected_ops = match state.memory_backend.get(&dst) {
                None => vec![
                Read(j, (k + byte).into()),
                Read(k,
                    read),
                DummyRead(dst, Default::default()),
                Write(dst, expected.into()),
                ],
                _ => vec![Read(j, (k + byte).into()), Read(k, read), Write(dst, expected.into())],
            };

            let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
            runtime.memory_backend = state.memory_backend;
            let mut running_machine = state.machine.start(&mut runtime);
            LoadS8Instruction::execute(&mut running_machine, Operands([dst_offset, 0, offset, 0, 0]));
            let state = BBState {machine: BBMachine::stop(running_machine), memory_backend: runtime.memory_backend};
            let (new_pc, new_fp) =state.registers();
            let result =state.get(dst).expect("nonempty");

            prop_assert_eq!(old_pc + 1, new_pc);
            prop_assert_eq!(old_fp, new_fp);
            let result_u32: u32 = result.into();
            prop_assert_eq!(expected, result_u32);
            prop_assert_eq!(state.memory_log_size(), 1);
            prop_assert_eq!(state.memory_log(clk), expected_ops);
            Ok(())
        })?
    }

    #[test]
    fn loadu8_property(state in mem_strat(), sel: Selector) {
        let byte = sel.select([0, 1, 2, 3]);
        with_unique(sel, state, |mut state, old_pc, old_fp, clk, [j, k]| {
            let mut rng = rand::thread_rng();
            let read = *state.memory_backend.get(&k).expect("nonempty");
            let expected = *read.value.index_le(byte as usize) as u32;
           state.set(j, k + byte);
            // ensure that dst is different from j and k
            let (dst, dst_offset) = {
                let (mut dst, mut dst_offset) = new_addr(old_fp, *MAX_ADDR, &mut rng);
                while dst == j || dst== k {
                    (dst, dst_offset) = new_addr(old_fp, *MAX_ADDR, &mut rng);
                }
            (dst, dst_offset)
            };
            let offset = calc_offset(old_fp, j);

            let expected_ops = match state.memory_backend.get(&dst) {
                None => vec![
                Read(j, (k + byte).into()),
                Read(k, read),
                DummyRead(dst, Default::default()),
                Write(dst, expected.into()),
                ],
                _ => vec![Read(j, (k + byte).into()), Read(k, read), Write(dst, expected.into())],
            };

            let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
            runtime.memory_backend = state.memory_backend;
            let mut running_machine = state.machine.start(&mut runtime);
            LoadU8Instruction::execute(&mut running_machine, Operands([dst_offset, 0, offset, 0, 0]));
            let state = BBState {machine: BBMachine::stop(running_machine), memory_backend: runtime.memory_backend};
            let (new_pc, new_fp) =state.registers();
            let result =state.get(dst).expect("nonempty");

            prop_assert_eq!(old_pc + 1, new_pc);
            prop_assert_eq!(old_fp, new_fp);
            let result_u32: u32 = result.into();
            prop_assert_eq!(expected, result_u32);
            prop_assert_eq!(state.memory_log_size(), 1);
            prop_assert_eq!(state.memory_log(clk), expected_ops);
            Ok(())
        })?
    }

    #[test]
    // [[fp + b]] = [fp + c]
    fn store32_property(state in mem_strat(), sel: Selector) {
        with_unique(sel, state, |mut state, old_pc, old_fp, clk, [j, k]| {
            let mut rng = rand::thread_rng();
            let expected = *state.memory_backend.get(&k).expect("nonempty");
            let (dst, _) = {
                let (mut dst, _) = new_addr(old_fp, *MAX_ADDR, &mut rng);
                while dst == j || dst == k { (dst, _) = new_addr(old_fp, *MAX_ADDR, &mut rng); }
                (dst, 0)
            };

            let expected_ops = match state.memory_backend.get(&dst) {
                None => vec![
                    Read(k, expected),
                    Read(j, dst.into()),
                    DummyRead(dst, Default::default()),
                    Write(dst, expected.value),
                    ],
                    _ => vec![Read(k, expected), Read(j, dst.into()), Write(dst, expected.value)]
                };

            // [j] = dst
            state.set(j, dst);

            let j_offset = calc_offset(old_fp, j);
            let k_offset = calc_offset(old_fp, k);

            let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
            runtime.memory_backend = state.memory_backend;
            let mut running_machine = state.machine.start(&mut runtime);
            Store32Instruction::execute(&mut running_machine, Operands([0, j_offset, k_offset, 0, 0]));
            state = BBState{machine: BasicMachine::stop(running_machine), memory_backend: runtime.memory_backend};
            let (new_pc, new_fp) =state.registers();
            let result =state.get(dst).expect("nonempty");

            prop_assert_eq!(old_pc + 1, new_pc);
            prop_assert_eq!(old_fp, new_fp);
            prop_assert_eq!(expected.value, result);
            prop_assert_eq!(state.memory_log_size(), 1);
            prop_assert_eq!(state.memory_log(clk), expected_ops);
            Ok(())
        })?
    }

    #[test]
    fn storeu8_property(state in mem_strat(), sel: Selector) {
        let byte = sel.select([0, 1, 2, 3]);
        with_unique(sel, state, |mut state, old_pc, old_fp, clk, [j, k]| {
            let mut rng = rand::thread_rng();
            let read: u32 =state.get(k).expect("nonempty").into();
            let old_value: u32 = rng.gen();
            let expected = (old_value & !(0xff << (8 * byte)))
                | ((read & 0xff) << (8 * byte));
            let (dst, _) = new_addr(old_fp, *MAX_ADDR, &mut rng);

            let expected_ops = match state.memory_backend.get(&dst) {
                None => vec![
                Read(k, read.into()),
                Read(j, (dst + byte).into()),
                Read(dst, old_value.into()),
                Write(dst, expected.into()),
            ], _ => vec![
                Read(k, read.into()),
                Read(j, (dst + byte).into()),
                Write(dst, expected.into()),
            ],
            };

            // [j] = dst + byte
           state.set(j, dst + byte);
           state.set(dst, old_value);

            let j_offset = calc_offset(old_fp, j);
            let k_offset = calc_offset(old_fp, k);

            let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
            runtime.memory_backend = state.memory_backend;
            let mut running_machine = state.machine.start( &mut runtime);

            StoreU8Instruction::execute(&mut running_machine, Operands([0, j_offset, k_offset, 0, 0]));
                state = BBState{machine: BasicMachine::stop(running_machine), memory_backend: runtime.memory_backend};
            let (new_pc, new_fp) =state.registers();
            let result =state.get(dst).expect("nonempty");

            prop_assert_eq!(old_pc + 1, new_pc);
            prop_assert_eq!(old_fp, new_fp);
            let result_u32: u32 = result.into();
            prop_assert_eq!(expected, result_u32);
            prop_assert_eq!(state.memory_log_size(), 1);
            prop_assert_eq!(state.memory_log(clk), expected_ops);
            Ok(())
        })?
    }

    #[test]
    // [fp + a] = fp + b
    fn loadfp_property(mut state in mem_strat()) {
        let mut rng = rand::thread_rng();
        let (old_pc, old_fp) =state.registers();
        let clk =state.clk();

        let (expected, offset) = new_addr(old_fp, *MAX_ADDR, &mut rng);
        let (dst, dst_offset) = new_addr(old_fp, *MAX_ADDR, &mut rng);

        let expected_ops = match state.memory_backend.get(&dst) {
            None => vec![
            DummyRead(dst, Default::default()),
            Write(dst, expected.into()),
            ], _ => vec![Write(dst, expected.into())]};

        let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
        runtime.memory_backend = state.memory_backend;
        let mut running_machine = state.machine.start(&mut runtime);
        LoadFpInstruction::execute(&mut running_machine, Operands([dst_offset, offset, 0, 0, 0]));
        state = BBState{machine: BasicMachine::stop(running_machine), memory_backend: runtime.memory_backend};
        let (new_pc, new_fp) =state.registers();
        let result =state.get(dst).expect("nonempty");

        prop_assert_eq!(old_pc + 1, new_pc);
        prop_assert_eq!(old_fp, new_fp);
        let result_u32: u32 = result.into();
        prop_assert_eq!(expected, result_u32);
        prop_assert_eq!(state.memory_log_size(), 1);
        prop_assert_eq!(state.memory_log(clk), expected_ops);
    }

    #[test]
    fn imm32_property(state in mem_strat(), expected: u32) {
        let mut rng = rand::thread_rng();
        let (old_pc, old_fp) =state.registers();
        let clk =state.clk();

        let (dst, dst_offset) = new_addr(old_fp, *MAX_ADDR, &mut rng);

        let expected_ops = match state.memory_backend.get(&dst) {
            None => vec![
            DummyRead(dst, Default::default()),
            Write(dst, expected.into()),
            ], _ => vec![Write(dst, expected.into())]};

        let imm = expected.to_le_bytes().map(|x| x as i32);
        let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
        runtime.memory_backend = state.memory_backend;
        let mut running_machine = state.machine.start(&mut runtime);

        Imm32Instruction::execute(&mut running_machine, Operands([dst_offset, imm[0], imm[1], imm[2], imm[3]]));
        let state = BBState {machine: BBMachine::stop(running_machine), memory_backend: runtime.memory_backend};
        let (new_pc, new_fp) =state.registers();
        let result =state.get(dst).expect("nonempty");

        prop_assert_eq!(old_pc + 1, new_pc);
        prop_assert_eq!(old_fp, new_fp);
        let result_u32: u32 = result.into();
        prop_assert_eq!(expected, result_u32);
        prop_assert_eq!(state.memory_log_size(), 1);
        prop_assert_eq!(state.memory_log(clk), expected_ops);
    }
}
