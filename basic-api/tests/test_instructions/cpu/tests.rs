use crate::common::{BBMachine, BBState};
use crate::testmachine::TestMachine;

use p3_baby_bear::BabyBear;
use valida_basic_api::ValidaRuntime;
use valida_cpu::{
    BeqInstruction, BneInstruction, JalInstruction, JalvInstruction, MachineWithCpuChip,
    StopInstruction,
};
use valida_machine::{Instruction, Machine, Operands, StorageBackendTrait};
use valida_memory::Operation::{DummyRead, Read, Write};

#[test]
// [fp + a] = 24 * (pc + 1)
// pc = b / 24
// fp = fp + c
fn test_jal() {
    let mut state = BBState::default();

    let (old_pc, _) = state.registers();
    let clk = state.clk();

    state.machine_mut().cpu_mut().fp = 64;

    let expected_ops = match state.memory_backend.get(&72) {
        None => vec![
            DummyRead(72, Default::default()),
            Write(72, (24 * (old_pc + 1)).into()),
        ],
        _ => vec![Write(72, (24 * (old_pc + 1)).into())],
    };

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    JalInstruction::execute(&mut running_machine, Operands([8, 24 * 44, 12, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();
    let result = state.get(72).expect("nonempty");

    assert_eq!(44, new_pc);
    assert_eq!(76, new_fp);
    let result_u32: u32 = result.into();
    assert_eq!(24 * (old_pc + 1), result_u32);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
// [fp + a] = 24 * (pc + 1)
// pc = [fp + b]
// fp += [fp + c]
fn test_jalv() {
    let mut state = BBState::default();

    let (old_pc, _) = state.registers();
    let clk = state.clk();

    state.machine_mut().cpu_mut().fp = 64;
    state.set(72, 44 * 24);
    state.set(76, 16);

    let expected_ops = match state.memory_backend.get(&68) {
        None => vec![
            DummyRead(68, Default::default()),
            Write(68, (24 * (old_pc + 1)).into()),
            Read(72, (44 * 24).into()),
            Read(76, 16.into()),
        ],
        _ => vec![
            Write(68, (24 * (old_pc + 1)).into()),
            Read(72, (44 * 24).into()),
            Read(76, 16.into()),
        ],
    };

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend.clone();
    let mut running_machine = state.machine.start(&mut runtime);
    JalvInstruction::execute(&mut running_machine, Operands([4, 8, 12, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();
    let result = state.get(68).expect("nonempty");

    assert_eq!(44, new_pc);
    assert_eq!(80, new_fp);
    let result_u32: u32 = result.into();
    assert_eq!(24 * (old_pc + 1), result_u32);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
// if [fp + b] != [fp + c] => pc = a / 24
// else => pc += 1
fn test_beq_eq() {
    let mut state = BBState::default();

    let (_, old_fp) = state.registers();
    let clk = state.clk();

    state.set(24, 51);
    state.set(28, 51);

    let expected_ops = vec![Read(24, 51.into()), Read(28, 51.into())];

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    BeqInstruction::execute(&mut running_machine, Operands([24 * 77, 24, 28, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();

    assert_eq!(77, new_pc);
    assert_eq!(old_fp, new_fp);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
// if [fp + b] != [fp + c] => pc = a / 24
// else => pc += 1
fn test_beq_ne() {
    let mut state = BBState::default();

    let (_, old_fp) = state.registers();
    let clk = state.clk();

    state.set(24, 51);
    state.set(28, 52);

    let expected_ops = vec![Read(24, 51.into()), Read(28, 52.into())];

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    BeqInstruction::execute(&mut running_machine, Operands([24 * 77, 24, 28, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();

    assert_eq!(1, new_pc);
    assert_eq!(old_fp, new_fp);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
// if [fp + b] != [fp + c] => pc = a / 24
// else => pc += 1
fn test_beq_imm() {
    let mut state = BBState::default();

    let (_, old_fp) = state.registers();
    let clk = state.clk();

    state.set(24, 51);

    let expected_ops = vec![Read(24, 51.into())];

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    BeqInstruction::execute(&mut running_machine, Operands([24 * 77, 24, 51, 0, 1]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();

    assert_eq!(77, new_pc);
    assert_eq!(old_fp, new_fp);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
// if [fp + b] != [fp + c] => pc = a / 24
// else => pc += 1
fn test_bne_eq() {
    let mut state = BBState::default();

    let (_, old_fp) = state.registers();
    let clk = state.clk();

    state.set(24, 51);
    state.set(28, 51);

    let expected_ops = vec![Read(24, 51.into()), Read(28, 51.into())];

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    BneInstruction::execute(&mut running_machine, Operands([24 * 77, 24, 28, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();

    assert_eq!(1, new_pc);
    assert_eq!(old_fp, new_fp);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
// if [fp + b] != [fp + c] => pc = a / 24
// else => pc += 1
fn test_bne_ne() {
    let mut state = BBState::default();

    let (_, old_fp) = state.registers();
    let clk = state.clk();

    state.set(24, 51);
    state.set(28, 52);

    let expected_ops = vec![Read(24, 51.into()), Read(28, 52.into())];

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    BneInstruction::execute(&mut running_machine, Operands([24 * 77, 24, 28, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();

    assert_eq!(77, new_pc);
    assert_eq!(old_fp, new_fp);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
// if [fp + b] != [fp + c] => pc = a / 24
// else => pc += 1
fn test_bne_imm() {
    let mut state = BBState::default();

    let (_, old_fp) = state.registers();
    let clk = state.clk();

    state.set(24, 51);

    let expected_ops = vec![Read(24, 51.into())];

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    BneInstruction::execute(&mut running_machine, Operands([24 * 77, 24, 51, 0, 1]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();

    assert_eq!(1, new_pc);
    assert_eq!(old_fp, new_fp);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
fn test_stop() {
    let state = BBState::default();

    let (old_pc, old_fp) = state.registers();
    let clk = state.clk();
    let expected_ops = vec![];

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    let mut running_machine = state.machine.start(&mut runtime);
    StopInstruction::execute(&mut running_machine, Operands([0, 0, 0, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();

    assert_eq!(old_pc, new_pc);
    assert_eq!(old_fp, new_fp);
    assert_eq!(state.memory_log_size(), 0);
    assert_eq!(state.memory_log(clk), expected_ops);
}
