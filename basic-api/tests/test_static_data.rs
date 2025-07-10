extern crate core;

use p3_baby_bear::BabyBear;
use p3_fri::{TwoAdicFriPcs, TwoAdicFriPcsConfig};
use std::collections::BTreeMap;
use valida_basic_api::{
    BasicMachine, BasicMachineMetrics, MultiSegmentBasicMachine, ValidaBootData, ValidaRuntime,
    ValidaSegmentBootData,
};

use valida_cpu::{
    BneInstruction, Imm32Instruction, Load32Instruction, MachineWithRegisters, Registers,
    StopInstruction,
};
use valida_machine::{
    Instruction, InstructionWord, Machine, MachineMetrics, Operands, ProgramROM, ProverOptions,
    SegmentMachine, Word,
};

use valida_program::{MachineWithProgramROM, ProgramTableType};
use valida_static_data::{MachineWithStaticDataChip, StaticDataChipType};

use p3_challenger::DuplexChallenger;
use p3_dft::Radix2Bowers;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::FriConfig;
use p3_keccak::Keccak256Hash;
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon::Poseidon;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
use rand::thread_rng;
use valida_machine::StarkConfigImpl;
use valida_machine::__internal::p3_commit::ExtensionMmcs;

mod common;
use common::*;

/// This program checks three different aspects of the static data chip:
/// 1. Whether static data can be correctly loaded, (single segment & multi segment machines)
///    in initial imm32 + load + bnei
/// 2. Whether a static data cell read for the first time in program execution in a segmente
///    later than the first works correctly. Requires `skip_persistent_receive` in the ephemeral
///    memory chip to behave correctly to avoid duplicate persistent receives.
/// 3. Whether the existence of static data cells that are _never_ used causes no issues.
///    Requires `skip_persistent_send` & `is_static_write` to work correctly in the ephemeral memory
///    chip so that we don't create a persistent send for static cells that are never used.
fn prove_static_data_program() -> Vec<InstructionWord<i32>> {
    // _start:
    //  imm32 0(fp), 0, 0x10, 0, 0            // write 0x10 to fp+0
    //  load32 -4(fp), 0(fp), 0, 0, 0         // load fp+0 to fp-4. If static data written, now contains 0x25
    //  bnei _start, 0(fp), 0x25, 0, 1        // infinite loop unless static value is loaded
    //  imm32 -12(fp), 0, 7, 0, 0             // write 7 to  fp-12
    //  imm32 -16(fp), 0, 9, 0, 0             // write 9 to  fp-16
    //  imm32 -20(fp), 0, 11, 0, 0            // write 11 to fp-20
    //  imm32 -24(fp), 0, 13, 0, 0            // write 13 to fp-24
    //  imm32 -28(fp), 0, 15, 0, 0            // write 15 to fp-28
    //  imm32 -32(fp), 0, 17, 0, 0            // write 17 to fp-32
    //  stop
    vec![
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([0, 0x10, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Load32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 0, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <BneInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([0, -4, 0x25, 0, 1]),
        },
        // we now perform a few dummy operations that just writes values to frame pointer offset locations
        // This is to stretch the program length, so that we can have multiple segments.
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-12, 7, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-16, 9, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-20, 11, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-24, 13, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-28, 15, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-32, 17, 0, 0, 0]),
        },
        // Now we perform another static data load. This is to check the case where we have static data
        // that we do *not* load in the first segment, but _later_.
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([0, 0x18, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Load32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 0, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <BneInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([0, -4, 0x46, 0, 1]),
        },
        // Assuming `fp-4 == 0x46` as expected from static data written to `0x18`, we exit here
        InstructionWord {
            opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands::default(),
        },
    ]
}

fn prepare_boot_data(max_trace_height: u32) -> ValidaBootData {
    let program = prove_static_data_program();
    let rom = ProgramROM::new(program);

    let static_data = BTreeMap::from([
        (0x10, Word::from_components_be([0, 0, 0, 0x25])),
        (0x14, Word::from_components_be([0, 0, 0, 0x32])),
        (0x18, Word::from_components_be([0, 0, 0, 0x46])),
    ]);
    let initial_register_values = Registers { pc: 0, fp: 0x1000 };
    let boot_data = ValidaBootData {
        program_rom: rom,
        program_table_type: ProgramTableType::Public,
        static_data,
        static_data_chip_type: StaticDataChipType::Public,
        initial_register_values,
        max_trace_height,
        program_file: vec![], // don't have a real program ELF
    };
    boot_data
}

fn to_segment_boot_data(bd: ValidaBootData) -> ValidaSegmentBootData {
    ValidaSegmentBootData {
        program_rom: bd.program_rom,
        program_table_type: bd.program_table_type,
        segment_number: 0,
        max_trace_height: bd.max_trace_height,
        program_file: bd.program_file,
        initial_register_values: bd.initial_register_values,
        static_data: Some(bd.static_data),
        static_data_chip_type: Some(bd.static_data_chip_type),
        log_enabled: true,
    }
}

#[test]
/// WARNING: If for some reason the static data chip does not work as expected
/// resulting in e.g. the data not being copied into memory, this test case will
/// never terminate!
/// TODO: Change the test to instead return a boolean value of whether the
/// static data was read or not!
fn prove_static_data() {
    let boot_data = prepare_boot_data(65536);
    let mut machine = BasicMachine::<Val>::default();
    machine.init(to_segment_boot_data(boot_data));

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    let mut state = machine.start(&mut runtime);
    let mut metrics = BasicMachineMetrics::initialize();

    let (instance_data, _output) = BasicMachine::run(&mut state, &mut metrics);

    let (prover_opts, show_preprocessed, show_preprocessed_dims, show_public_verifier) =
        prover_options();

    let config = get_machine_config();

    let (pk, vk) = state
        .machine
        .pre_process(&config, show_preprocessed, show_preprocessed_dims);
    let proof = state
        .machine
        .prove(&config, &pk, prover_opts, &instance_data);
    state
        .machine
        .verify(&config, &proof, &vk, &instance_data, show_public_verifier)
        .expect("verification failed");
}

#[test]
fn prove_multi_segment_static_data() {
    let boot_data = prepare_boot_data(4);
    let mut machine = MultiSegmentBasicMachine::<Val>::default();
    machine.init(boot_data);

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    let mut state = machine.start(&mut runtime);
    let mut metrics = BasicMachineMetrics::initialize();

    let (instance_data, _output) = MultiSegmentBasicMachine::run(&mut state, &mut metrics);

    let (prover_opts, show_preprocessed, show_preprocessed_dims, show_public_verifier) =
        prover_options();

    let config = get_machine_config();

    let (pk, vk) = state
        .machine
        .pre_process(&config, show_preprocessed, show_preprocessed_dims);
    let proof = state
        .machine
        .prove(&config, &pk, prover_opts, &instance_data);
    state
        .machine
        .verify(&config, &proof, &vk, &instance_data, show_public_verifier)
        .expect("verification failed");
}
