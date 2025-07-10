use crate::common::{BBMachine, BBState};
use crate::testmachine::TestMachine;

use p3_baby_bear::BabyBear;
use valida_basic_api::ValidaRuntime;
use valida_cpu::{
    Imm32Instruction, Load32Instruction, LoadFpInstruction, LoadS8Instruction, LoadU8Instruction,
    MachineWithCpuChip, Store32Instruction, StoreU8Instruction,
};
use valida_machine::{Instruction, Machine, Operands, StorageBackendTrait};
use valida_memory::Operation::{DummyRead, Read, Write};

#[test]
// [fp + a] = [[fp + c]]
fn test_load32() {
    let mut state = BBState::default();
    let (old_pc, old_fp) = state.registers();
    let clk = state.clk();

    state.set(32, 64);
    state.set(64, 0xdeadbeef);

    let expected_ops = match state.memory_backend.get(&0) {
        None => vec![
            Read(32, 64.into()),
            Read(64, 0xdeadbeef.into()),
            DummyRead(0, Default::default()),
            Write(0, 0xdeadbeef.into()),
        ],
        _ => vec![Write(0, 0xdeadbeef.into())],
    };

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    Load32Instruction::execute(&mut running_machine, Operands([0, 0, 32, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();
    let result = state.get(0).expect("nonempty");

    assert_eq!(old_pc + 1, new_pc);
    assert_eq!(old_fp, new_fp);
    let result_u32: u32 = result.into();
    assert_eq!(0xdeadbeef_u32, result_u32);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
fn test_loads8() {
    let mut state = BBState::default();

    let (old_pc, old_fp) = state.registers();
    let clk = state.clk();

    state.set(32, 66);
    state.set(64, 0xdeadbeef);
    // In little endian, at loc 66, the byte is 0xad.
    // Address:    64    65    66    67
    // Byte value: 0xef  0xbe  0xad  0xde
    // Binary:     11101111 10111110 10101101 11011110
    // sign extended to [0xAD, 0xFF, 0xFF, 0xFF]
    let expected_ops = match state.memory_backend.get(&0) {
        None => vec![
            Read(32, 66.into()),
            Read(64, 0xdeadbeef.into()),
            DummyRead(0, Default::default()),
            Write(0, 0xffffffad.into()),
        ],
        _ => vec![
            Read(32, 66.into()),
            Read(64, 0xdeadbeef.into()),
            Write(0, 0xffffffad.into()),
        ],
    };

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    LoadS8Instruction::execute(&mut running_machine, Operands([0, 0, 32, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();
    let result = state.get(0).expect("nonempty");

    assert_eq!(old_pc + 1, new_pc);
    assert_eq!(old_fp, new_fp);
    let result_u32: u32 = result.into();
    assert_eq!(0xffffffad_u32, result_u32);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
fn test_loadu8() {
    let mut state = BBState::default();

    let (old_pc, old_fp) = state.registers();
    let clk = state.clk();

    state.set(32, 66);
    state.set(64, 0xdeadbeef);

    // In little endian, at loc 66, the byte is 0xad.
    // Address:    64    65    66    67
    // Byte value: 0xef  0xbe  0xad  0xde
    // Binary:     11101111 10111110 10101101 11011110
    let expected_ops = match state.memory_backend.get(&0) {
        None => vec![
            Read(32, 66.into()),
            Read(64, 0xdeadbeef.into()),
            DummyRead(0, Default::default()),
            Write(0, 0xad.into()),
        ],
        _ => vec![
            Read(32, 66.into()),
            Read(64, 0xdeadbeef.into()),
            Write(0, 0xad.into()),
        ],
    };

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    LoadU8Instruction::execute(&mut running_machine, Operands([0, 0, 32, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();
    let result = state.get(0).expect("nonempty");

    assert_eq!(old_pc + 1, new_pc);
    assert_eq!(old_fp, new_fp);
    let result_u32: u32 = result.into();
    assert_eq!(0xad, result_u32);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
// [[fp + b]] = [fp + c]
fn test_store32() {
    let mut state = BBState::default();
    let (old_pc, old_fp) = state.registers();
    let clk = state.clk();

    state.set(32, 0);
    state.set(64, 0xdeadbeef);

    let expected_ops = match state.memory_backend.get(&0) {
        None => vec![
            Read(64, 0xdeadbeef.into()),
            Read(32, 0.into()),
            DummyRead(0, Default::default()),
            Write(0, 0xdeadbeef.into()),
        ],
        _ => vec![
            Read(64, 0xdeadbeef.into()),
            Read(32, 0.into()),
            Write(0, 0xdeadbeef.into()),
        ],
    };

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    Store32Instruction::execute(&mut running_machine, Operands([77, 32, 64, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();
    let result = state.get(0).expect("nonempty");

    assert_eq!(old_pc + 1, new_pc);
    assert_eq!(old_fp, new_fp);
    let result_u32: u32 = result.into();
    assert_eq!(0xdeadbeef_u32, result_u32);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
fn test_storeu8() {
    let mut state = BBState::default();
    let (old_pc, old_fp) = state.registers();
    let clk = state.clk();

    state.set(32, 1);
    state.set(64, 0xdeadbeef);
    state.set(0, 0xcafebabe);

    // In little endian, at loc 64, the byte is 0xef
    // Address:    64    65    66    67
    // Byte value: 0xef  0xbe  0xad  0xde
    // Binary:     11101111 10111110 10101101 11011110
    // or [239, 190, 173, 222]
    let expected_ops = vec![
        Read(64, 0xdeadbeef.into()),
        Read(32, 1.into()),
        // Read(0, Word([190, 186, 254, 202])) is the cell to store the byte. At location 1,
        // i.e., 186 should be overwritten by 239.
        Read(0, 0xcafebabe.into()),
        // Write(0, Word([190, 239, 254, 202])) is the result
        Write(0, 0xCAFEEFBE.into()),
    ];

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    StoreU8Instruction::execute(&mut running_machine, Operands([77, 32, 64, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();
    let result = state.get(0).expect("nonempty");

    assert_eq!(old_pc + 1, new_pc);
    assert_eq!(old_fp, new_fp);
    let result_u32: u32 = result.into();
    assert_eq!(0xCAFEEFBE_u32, result_u32);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
// [fp + a] = fp + b
fn test_loadfp() {
    let mut state = BBState::default();
    state.machine.cpu_mut().fp = 0x40000;
    let (old_pc, old_fp) = state.registers();
    let clk = state.clk();

    let expected_ops = match state.memory_backend.get(&0x3ffe0) {
        None => vec![
            DummyRead(0x3ffe0, Default::default()),
            Write(0x3ffe0, 0x40040.into()),
        ],
        _ => vec![Write(0x3ffe0, 0x40040.into())],
    };

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    LoadFpInstruction::execute(&mut running_machine, Operands([-0x20, 0x40, 0, 0, 0]));
    let state = BBState {
        machine: BBMachine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = state.registers();
    let result = state.get(0x3ffe0).expect("nonempty");

    assert_eq!(old_pc + 1, new_pc);
    assert_eq!(old_fp, new_fp);
    let result_u32: u32 = result.into();
    assert_eq!(0x40040, result_u32);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

#[test]
fn test_imm32() {
    let mut state = BBState::default();
    let (old_pc, old_fp) = state.registers();
    let clk = state.clk();

    let expected_ops = match state.memory_backend.get(&0x40000) {
        None => vec![
            DummyRead(0x40000, Default::default()),
            Write(0x40000, 0xdeadbeef.into()),
        ],
        _ => vec![Write(0x40000, 0xdeadbeef.into())],
    };

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);

    Imm32Instruction::execute(
        &mut running_machine,
        Operands([0x40000, 0xef, 0xbe, 0xad, 0xde]),
    );
    state = BBState {
        machine: Machine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let (new_pc, new_fp) = (state).registers();
    let result = state.get(0x40000).expect("nonempty");

    assert_eq!(old_pc + 1, new_pc);
    assert_eq!(old_fp, new_fp);
    let result_u32: u32 = result.into();
    assert_eq!(0xdeadbeef_u32, result_u32);
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}
