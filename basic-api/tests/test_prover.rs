extern crate core;

use p3_baby_bear::BabyBear;
use p3_fri::{TwoAdicFriPcs, TwoAdicFriPcsConfig};
use valida_alu_u32::add::{Add32Instruction, MachineWithAdd32Chip};
use valida_alu_u32::div::{Div32Instruction, SDiv32Instruction};
use valida_alu_u32::lt::{Lt32Instruction, Lte32Instruction, Sle32Instruction, Slt32Instruction};
use valida_alu_u32::sub::Sub32Instruction;
use valida_basic_api::instance_data::ValidaInstanceData;
use valida_basic_api::{
    BasicMachine, BasicMachineMetrics, MultiSegmentBasicMachine, ValidaRuntime,
};
use valida_cpu::{
    BeqInstruction, BneInstruction, Imm32Instruction, JalInstruction, JalvInstruction,
    LoadFpInstruction, LoadS8Instruction, LoadU8Instruction, MachineWithCpuChip,
    MachineWithRegisters, StopInstruction, StoreU8Instruction,
};
use valida_machine::{
    Instruction, InstructionWord, Machine, MachineProof, MachineRuntime, MemoryBackendTrait,
    MultiSegmentMachineProof, Operands, ProgramROM, ProverOptions, SegmentMachine, StarkField,
    ValidaMemoryBackend, Word,
};

use valida_memory::MachineWithMemoryChip;
use valida_opcodes::BYTES_PER_INSTR;
use valida_output::{MachineWithOutputChip, WriteInstruction};
use valida_program::{MachineWithProgramROM, ProgramTableType};

use p3_challenger::DuplexChallenger;
use p3_dft::Radix2Bowers;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeField32, TwoAdicField};
use p3_fri::FriConfig;
use p3_keccak::Keccak256Hash;
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon::Poseidon;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
use rand::thread_rng;
use tiny_keccak::keccakf;
use valida_keccak::KeccakFInstruction;
use valida_machine::StarkConfigImpl;
use valida_machine::__internal::p3_commit::ExtensionMmcs;

/// We import everything from common as it's just a helper to not define the same types and helper functions
/// in this and other test files
mod common;
use common::*;

fn div_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];

    // imm32 -4(fp), 0, 1, 0, 0
    // imm32 -8(fp), 3, 0, 0, 0
    // imm32 -12(fp), 0, 0, 0, 0
    // div32 8(fp), -4(fp), 3, 0, 1
    // div32 12(fp), -8(fp), -4(fp), 0, 0
    // div32 16(fp), -12(fp), -8(fp), 0, 0
    // stop
    program.extend(vec![
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 0, 1, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, 3, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-12, 0, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Div32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([8, -4, 3, 0, 1]),
        },
        InstructionWord {
            opcode: <Div32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([12, -8, -4, 0, 0]),
        },
        InstructionWord {
            opcode: <Div32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([16, -12, -8, 0, 0]),
        },
        InstructionWord {
            opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands::default(),
        },
    ]);

    program
}

fn sdiv_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];

    // imm32 -4(fp), 0, 1, 0, 0
    // imm32 -8(fp), 3, 0, 0, 0
    // imm32 -12(fp), 0, 0, 0, 0
    // imm32 -16(fp), 0, 255, 255, 255 // -256
    // imm32 -20(fp), 253, 255, 255, 255 // -3

    program.extend(vec![
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 0, 1, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, 3, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-12, 0, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-16, 0, 255, 255, 255]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-20, 253, 255, 255, 255]),
        },
    ]);

    // sdiv32 4(fp), -4(fp), 3, 0, 1
    // sdiv32 8(fp), -8(fp), -4(fp), 0, 0
    // sdiv32 12(fp), -4(fp), -3, 0, 1
    // sdiv32 16(fp), -8(fp), -16(fp), 0, 0
    // sdiv32 20(fp), -16(fp), 3, 0, 1
    // sdiv32 24(fp), -20(fp), -4(fp), 0, 0
    // sdiv32 28(fp), -16(fp), -3, 0, 1
    // sdiv32 32(fp), -20(fp), -16(fp), 0, 0
    // sdiv32 36(fp), -12(fp), 3, 0, 1
    // sdiv32 40(fp), -12(fp), -3, 0, 1
    // stop
    program.extend(vec![
        InstructionWord {
            opcode: <SDiv32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([4, -4, 3, 0, 1]),
        },
        InstructionWord {
            opcode: <SDiv32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([8, -8, -4, 0, 0]),
        },
        InstructionWord {
            opcode: <SDiv32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([12, -4, -3, 0, 1]),
        },
        InstructionWord {
            opcode: <SDiv32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([16, -8, -16, 0, 0]),
        },
        InstructionWord {
            opcode: <SDiv32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([20, -16, 3, 0, 1]),
        },
        InstructionWord {
            opcode: <SDiv32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([24, -20, -4, 0, 0]),
        },
        InstructionWord {
            opcode: <SDiv32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([28, -16, -3, 0, 1]),
        },
        InstructionWord {
            opcode: <SDiv32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([32, -20, -16, 0, 0]),
        },
        InstructionWord {
            opcode: <SDiv32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([36, -12, 3, 0, 1]),
        },
        InstructionWord {
            opcode: <SDiv32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([40, -12, -3, 0, 1]),
        },
        InstructionWord {
            opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands::default(),
        },
    ]);

    program
}

fn single_byte_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];

    // imm32 -4(fp), 1, 255, 3, 4
    // loadfp -8(fp) -3(fp) 0 0 0
    program.extend(vec![
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 1, 255, 3, 4]),
        },
        InstructionWord {
            opcode: <LoadFpInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, -3, 0, 0, 0]),
        },
    ]);

    // LOADU8 4(fp), 0, -8(fp), 0, 0
    // LOADS8 8(fp), 0, -8(fp), 0, 0
    // STOREU8 0, -8(fp), -4(fp), 0, 0
    program.extend(vec![
        InstructionWord {
            opcode: <LoadU8Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([4, 0, -8, 0, 0]),
        },
        InstructionWord {
            opcode: <LoadS8Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([8, 0, -8, 0, 0]),
        },
        InstructionWord {
            opcode: <StoreU8Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([0, -8, -4, 0, 0]),
        },
    ]);
    // stop
    program.push(InstructionWord {
        opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
        operands: Operands::default(),
    });
    program
}

fn sub_with_overflow_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];

    // imm32 -4(fp), 3, 0, 0, 0
    // imm32 -8(fp), 2, 0, 0, 0
    program.extend(vec![
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 3, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, 2, 0, 0, 0]),
        },
    ]);
    // sub32 12(fp), -8(fp), -4(fp), 0, 0
    program.push(InstructionWord {
        opcode: <Sub32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
        operands: Operands([12, -8, -4, 0, 0]),
    });
    // stop
    program.push(InstructionWord {
        opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
        operands: Operands::default(),
    });

    program
}

fn sub_with_borrow_propagation_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];

    // imm32 -4(fp), 1, 0, 0, 0
    // imm32 -8(fp), 0, 0, 0, 1
    program.extend(vec![
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 1, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, 0, 0, 0, 1]),
        },
    ]);
    // sub32 12(fp), -8(fp), -4(fp), 0, 0
    program.push(InstructionWord {
        opcode: <Sub32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
        operands: Operands([12, -8, -4, 0, 0]),
    });
    // stop
    program.push(InstructionWord {
        opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
        operands: Operands::default(),
    });

    program
}

fn add_with_overflow_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];

    // imm32 -4(fp), 255, 255, 255, 255 // 0xffffffff
    // imm32 -8(fp), 2, 0, 0, 0
    program.extend(vec![
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 255, 255, 255, 255]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, 2, 0, 0, 0]),
        },
    ]);
    // add32 12(fp), -8(fp), -4(fp), 0, 0
    program.push(InstructionWord {
        opcode: <Add32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
        operands: Operands([12, -8, -4, 0, 0]),
    });
    // stop
    program.push(InstructionWord {
        opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
        operands: Operands::default(),
    });

    program
}

fn fib_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];

    // Label locations
    let bytes_per_instr = BYTES_PER_INSTR as i32;
    let fib_bb0 = 8 * bytes_per_instr;
    let fib_bb0_1 = 13 * bytes_per_instr;
    let fib_bb0_2 = 15 * bytes_per_instr;
    let fib_bb0_3 = 19 * bytes_per_instr;
    let fib_bb0_4 = 21 * bytes_per_instr;

    //main:                                   ; @main
    //; %bb.0:
    //	imm32	-4(fp), 0, 0, 0, 0
    //	imm32	-8(fp), 25, 0, 0, 0
    //	addi	-16(fp), -8(fp), 0
    //	imm32	-20(fp), 28, 0, 0, 0
    //	jal	-28(fp), fib, -28
    //	addi	-12(fp), -24(fp), 0
    //	addi	4(fp), -12(fp), 0
    //	exit
    //...
    program.extend([
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 0, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, 25, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Add32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-16, -8, 0, 0, 1]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-20, 28, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <JalInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-28, fib_bb0, -28, 0, 0]),
        },
        InstructionWord {
            opcode: <Add32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-12, -24, 0, 0, 1]),
        },
        InstructionWord {
            opcode: <Add32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([4, -12, 0, 0, 1]),
        },
        InstructionWord {
            opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands::default(),
        },
    ]);

    //fib:                                    ; @fib
    //; %bb.0:
    //	addi	-4(fp), 12(fp), 0
    //	imm32	-8(fp), 0, 0, 0, 0
    //	imm32	-12(fp), 1, 0, 0, 0
    //	imm32	-16(fp), 0, 0, 0, 0
    //	beq	.LBB0_1, 0(fp), 0(fp)
    program.extend([
        InstructionWord {
            opcode: <Add32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 12, 0, 0, 1]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, 0, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-12, 1, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-16, 0, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <BeqInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([fib_bb0_1, 0, 0, 0, 0]),
        },
    ]);

    //.LBB0_1:
    //	bne	.LBB0_2, -16(fp), -4(fp)
    //	beq	.LBB0_4, 0(fp), 0(fp)
    program.extend([
        InstructionWord {
            opcode: <BneInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([fib_bb0_2, -16, -4, 0, 0]),
        },
        InstructionWord {
            opcode: <BeqInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([fib_bb0_4, 0, 0, 0, 0]),
        },
    ]);

    //; %bb.2:
    //	add	-20(fp), -8(fp), -12(fp)
    //	addi	-8(fp), -12(fp), 0
    //	addi	-12(fp), -20(fp), 0
    //	beq	.LBB0_3, 0(fp), 0(fp)
    program.extend([
        InstructionWord {
            opcode: <Add32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-20, -8, -12, 0, 0]),
        },
        InstructionWord {
            opcode: <Add32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, -12, 0, 0, 1]),
        },
        InstructionWord {
            opcode: <Add32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-12, -20, 0, 0, 1]),
        },
        InstructionWord {
            opcode: <BeqInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([fib_bb0_3, 0, 0, 0, 0]),
        },
    ]);

    //; %bb.3:
    //	addi	-16(fp), -16(fp), 1
    //	beq	.LBB0_1, 0(fp), 0(fp)
    program.extend([
        InstructionWord {
            opcode: <Add32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-16, -16, 1, 0, 1]),
        },
        InstructionWord {
            opcode: <BeqInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([fib_bb0_1, 0, 0, 0, 0]),
        },
    ]);

    //.LBB0_4:
    //	addi	4(fp), -8(fp), 0
    //	jalv	-4(fp), 0(fp), 8(fp)
    program.extend([
        InstructionWord {
            opcode: <Add32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([4, -8, 0, 0, 1]),
        },
        InstructionWord {
            opcode: <JalvInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 0, 8, 0, 0]),
        },
    ]);
    program
}

fn left_imm_ops_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];

    program.extend([
        // imm32	-4(fp), 3, 0, 0, 0
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 3, 0, 0, 0]),
        },
        // ;(0, 1, 0, 0) == 256
        // imm32   -8(fp), 0, 1, 0, 0
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, 0, 1, 0, 0]),
        },
        // lt32    4(fp), 3, -4(fp), 1, 0
        InstructionWord {
            opcode: <Lt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([4, 3, -4, 1, 0]),
        },
        // lte32    8(fp), 3, -4(fp), 1, 0
        InstructionWord {
            opcode: <Lte32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([8, 3, -4, 1, 0]),
        },
        // lt32    12(fp), 4, -4(fp), 1, 0
        InstructionWord {
            opcode: <Lt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([12, 4, -4, 1, 0]),
        },
        // lte32   16(fp), 4, -4(fp), 1, 0
        InstructionWord {
            opcode: <Lte32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([16, 4, -4, 1, 0]),
        },
        // lt32 20(fp), 2, -4(fp), 1, 0
        InstructionWord {
            opcode: <Lt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([20, 2, -4, 1, 0]),
        },
        // lte32 24(fp), 2, -4(fp), 1, 0
        InstructionWord {
            opcode: <Lte32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([24, 2, -4, 1, 0]),
        },
        // lt32 28(fp), 256, -4(fp), 1, 0
        InstructionWord {
            opcode: <Lt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([28, 256, -4, 1, 0]),
        },
        // lte32 32(fp), 256, -4(fp), 1, 0
        InstructionWord {
            opcode: <Lte32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([32, 256, -4, 1, 0]),
        },
        // lt32 36(fp), 3, -8(fp), 1, 0
        InstructionWord {
            opcode: <Lt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([36, 3, -8, 1, 0]),
        },
        // lte32 40(fp), 3, -8(fp), 1, 0
        InstructionWord {
            opcode: <Lte32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([40, 3, -8, 1, 0]),
        },
        // stop 0, 0, 0, 0, 0
        InstructionWord {
            opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands::default(),
        },
    ]);
    program
}

fn output_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];

    // imm32 -4(fp), 4, 0, 0, 0
    // imm32 -8(fp), 5, 1, 0, 0
    program.extend([
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 4, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, 5, 1, 0, 0]),
        },
    ]);
    // write 0(fp), -4(fp), 0, 0, 1
    // write 0(fp), -8(fp), 0, 0, 1
    program.extend([
        InstructionWord {
            opcode: <WriteInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([0, -4, 0, 0, 1]),
        },
        InstructionWord {
            opcode: <WriteInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([0, -8, 0, 0, 1]),
        },
    ]);
    // stop 0, 0, 0, 0, 0
    program.push(InstructionWord {
        opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
        operands: Operands::default(),
    });
    program
}

fn signed_inequality_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];

    // imm32 -4(fp), 1, 0, 0, 0
    // imm32 -8(fp), 255, 255, 255, 255
    // imm32 -12(fp), 254, 255, 255, 255
    program.extend([
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 1, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, 255, 255, 255, 255]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-12, 254, 255, 255, 255]),
        },
    ]);

    // slt32 4(fp), -12(fp), -8(fp), 0, 0
    // slt32 8(fp), -12(fp), -4(fp), 0, 0
    // slt32 12(fp), -4(fp), -1, 0, 1
    // slt32 16(fp), -1, -8(fp), 1, 0
    // sle32 20(fp), -1, -8(fp), 1, 0
    // slt32 24(fp), -1, -12(fp), 1, 0
    // slt32 28(fp), -8(fp), -12(fp), 0, 0
    // slt32 32(fp), -8(fp), -4(fp), 0, 0

    program.extend([
        InstructionWord {
            opcode: <Slt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([4, -12, -8, 0, 0]),
        },
        InstructionWord {
            opcode: <Slt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([8, -12, -4, 0, 0]),
        },
        InstructionWord {
            opcode: <Slt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([12, -4, -1, 0, 1]),
        },
        InstructionWord {
            opcode: <Slt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([16, -1, -8, 1, 0]),
        },
        InstructionWord {
            opcode: <Sle32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([20, -1, -8, 1, 0]),
        },
        InstructionWord {
            opcode: <Slt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([24, -1, -12, 1, 0]),
        },
        InstructionWord {
            opcode: <Slt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([28, -8, -12, 0, 0]),
        },
        InstructionWord {
            opcode: <Slt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([32, -8, -4, 0, 0]),
        },
    ]);

    // lt32 36(fp), -12(fp), -8(fp), 0, 0
    // lt32 40(fp), -12(fp), -4(fp), 0, 0
    // lt32 44(fp), -4(fp), -1, 0, 1
    // lt32 48(fp), -1, -8(fp), 1, 0
    // lte32 52(fp), -1, -8(fp), 1, 0
    // lt32 56(fp), -1, -12(fp), 1, 0
    // lt32 60(fp), -8(fp), -12(fp), 0, 0
    // lt32 64(fp), -8(fp), -4(fp), 0, 0
    // stop
    program.extend([
        InstructionWord {
            opcode: <Lt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([36, -12, -8, 0, 0]),
        },
        InstructionWord {
            opcode: <Lt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([40, -12, -4, 0, 0]),
        },
        InstructionWord {
            opcode: <Lt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([44, -4, -1, 0, 1]),
        },
        InstructionWord {
            opcode: <Lt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([48, -1, -8, 1, 0]),
        },
        InstructionWord {
            opcode: <Lte32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([52, -1, -8, 1, 0]),
        },
        InstructionWord {
            opcode: <Lt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([56, -1, -12, 1, 0]),
        },
        InstructionWord {
            opcode: <Lt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([60, -8, -12, 0, 0]),
        },
        InstructionWord {
            opcode: <Lt32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([64, -8, -4, 0, 0]),
        },
        // stop 0, 0, 0, 0, 0
        InstructionWord {
            opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([0, 0, 0, 0, 0]),
        },
    ]);

    program
}

fn loadfp_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];
    // loadfp 4(fp), 0, 0, 0, 0
    // loadfp 8(fp), 3, 0, 0, 0
    // stop
    program.extend([
        InstructionWord {
            opcode: <LoadFpInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([4, 0, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <LoadFpInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([8, 3, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands::default(),
        },
    ]);

    program
}

fn persistent_memory_example<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];
    // stop
    program.extend([
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-4, 1, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-8, 2, 0, 0, 0]),
        },
        // now calculate some stuff
        InstructionWord {
            opcode: <Add32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([4, -8, 0, 0, 1]),
        },
        // the above should be in the first segment
        // now we just define a bunch of instructions to end up in the second segment
        // (we'll use max trace height 4)
        // NOTE: Using distinct numbers for easier debugging of mismatched sends/receives
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
        // and now perform another add with the result from the Add32 in the first segment
        // 4(fp) == 2
        // -8(fp) == 2
        // Result: 4
        InstructionWord {
            opcode: <Add32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([8, 4, -8, 0, 0]),
        },
        // this should now be a persistent memory setup
        InstructionWord {
            opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands::default(),
        },
    ]);

    program
}

fn keccak_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];
    // chunck of data we want to hash
    for i in 0..25 {
        program.extend([InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-200 + i * 8, 1, 0, 0, 0]),
        }]);
        program.extend([InstructionWord {
            opcode: <Imm32Instruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-200 + i * 8 + 4, 0, 0, 0, 0]),
        }]);
    }

    // pointer to the base address
    program.extend([
        InstructionWord {
            opcode: <LoadFpInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([-400, -200, 0, 0, 0]),
        },
        //Keccak hash
        InstructionWord {
            opcode: <KeccakFInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([0, -400, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <KeccakFInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([0, -400, 0, 0, 0]),
        },
        InstructionWord {
            opcode: <KeccakFInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands([0, -400, 0, 0, 0]),
        },
        // Stop
        InstructionWord {
            opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands::default(),
        },
    ]);

    program
}

// TODO: Unless there's a good reason we should use `ValidaBootData` and `init` instead of manually
// calling `machine.set_*`.
fn multi_segment_prove_program(
    program: Vec<InstructionWord<i32>>,
    program_table_type: ProgramTableType,
    max_trace_height: u32,
) -> (
    MultiSegmentBasicMachine<BabyBear>,
    ValidaInstanceData,
    ValidaMemoryBackend,
    bool, // whether proof was verified correctly
) {
    let mut machine = MultiSegmentBasicMachine::<BabyBear>::default();

    // We need to set the trace height to be a fairly small number for testing purposes.
    // This ensures that we transition through a segment machine boundary. Otherwise,
    // this test is only exercising the first segment machine, not the logic
    // of transitioning between segment machines.
    //
    // Unfortunately, finding where this transition occurs depends on the specific
    // program, so this is customized for each program/test.
    machine.set_max_trace_height(max_trace_height);
    let rom = ProgramROM::new(program);
    machine.set_program_rom(0, rom, program_table_type);
    machine.set_initial_register_values(valida_cpu::Registers { pc: 0, fp: 0x1000 });

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    let mut state = machine.start(&mut runtime);
    let mut metrics = BasicMachineMetrics::initialize();

    let (instance_data, _output) = MultiSegmentBasicMachine::run(&mut state, &mut metrics);
    let memory_backend = state.runtime.memory_backend().clone();
    let finalized_machine = MultiSegmentBasicMachine::stop(state);

    let config = get_machine_config();
    let (prover_opts, show_preprocessed, show_preprocessed_dims, _) = prover_options();

    let (pk, vk) =
        finalized_machine.pre_process(&config, show_preprocessed, show_preprocessed_dims);

    let proof = finalized_machine.prove(&config, &pk, prover_opts, &instance_data);

    let show_traces = vec![false; BasicMachine::<Val>::NUM_CHIPS];
    let ver = finalized_machine.verify(&config, &proof, &vk, &instance_data, show_traces);

    // TODO(jen): DRY up with single segment prove_program

    (
        finalized_machine,
        instance_data,
        memory_backend,
        ver.is_ok(),
    )
}

fn prove_program(
    init_pc: u32,
    program: Vec<InstructionWord<i32>>,
    program_table_type: ProgramTableType,
) -> (BasicMachine<BabyBear>, ValidaMemoryBackend) {
    let mut machine = BasicMachine::<BabyBear>::default();
    machine.set_segment_number(0);
    machine.set_max_trace_height(65536);
    let rom = ProgramROM::new(program);
    machine.set_program_rom(init_pc, rom, program_table_type);
    machine.set_initial_register_values(valida_cpu::Registers {
        pc: init_pc,
        fp: 0x1000,
    });

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    let mut state = machine.start(&mut runtime);
    let mut metrics = BasicMachineMetrics::initialize();

    let (instance_data, _output) = BasicMachine::run(&mut state, &mut metrics);

    let config = get_machine_config();

    let (prover_opts, show_preprocessed, show_preprocessed_dims, show_public_verifier) =
        prover_options();

    let (pk, vk) = state
        .machine
        .pre_process(&config, show_preprocessed, show_preprocessed_dims);
    let proof = state
        .machine
        .prove(&config, &pk, prover_opts, &instance_data);

    let mut bytes = vec![];
    ciborium::into_writer(&proof, &mut bytes).expect("serialization failed");
    println!("Proof size: {} bytes", bytes.len());
    let deserialized_proof: MachineProof<MyConfig> =
        ciborium::from_reader(bytes.as_slice()).expect("deserialization failed");

    state
        .machine
        .verify(
            &config,
            &proof,
            &vk,
            &instance_data,
            show_public_verifier.clone(),
        )
        .expect("verification failed");
    state
        .machine
        .verify(
            &config,
            &deserialized_proof,
            &vk,
            &instance_data,
            show_public_verifier,
        )
        .expect("verification failed");

    (*state.machine, state.runtime.memory_backend().clone())
}

fn expected_div_memory_state(memory_backend: &ValidaMemoryBackend) {
    // div32: non-zero quotient
    // div32 8(fp), -4(fp), 3, 0, 1
    assert_eq!(
        memory_backend.get_value(0x1000 + 8),
        Word::from(85) // 256 / 3 = 85
    );
    // div32: zero quotient
    // div32 12(fp), -8(fp), 4(fp), 0, 1
    assert_eq!(
        memory_backend.get_value(0x1000 + 12),
        Word::from(0) // 3 / 256 = 0
    );
    // div32: zero dividend
    // div32 16(fp), 0, -4(fp), 0, 1
    assert_eq!(
        memory_backend.get_value(0x1000 + 16),
        Word::from(0) // 0 / 3 = 0
    );
}

#[test]
fn prove_div() {
    let program = div_program::<BabyBear>();
    let (_machine, memory_backend) = prove_program(0, program, ProgramTableType::Public);
    expected_div_memory_state(&memory_backend);
}

#[test]
fn prove_multi_segment_div() {
    let program = div_program::<BabyBear>();
    let (_finalized_machine, instance_data, memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 4);

    assert!(verified_ok);

    // Four machines should have been created to run the program.
    assert_eq!(instance_data.segments.len(), 3);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);

    expected_div_memory_state(&memory_backend);
}

fn expected_sdiv_memory_state(memory_backend: &ValidaMemoryBackend) {
    // sdiv32: positive / positive, non-zero quotient
    // sdiv32 4(fp), -4(fp), 3, 0, 1
    assert_eq!(
        memory_backend.get_value(0x1000 + 4),
        Word::from(85) // 256 / 3 = 85
    );
    // sdiv32: positive / positive, zero quotient
    // sdiv32 8(fp), -8(fp), -4(fp), 0, 0
    assert_eq!(
        memory_backend.get_value(0x1000 + 8),
        Word::from(0) // 3 / 256 = 0
    );
    // sdiv32: positive / negative, non-zero quotient
    // sdiv32 12(fp), -4(fp), -3, 0, 1
    assert_eq!(
        memory_backend.get_value(0x1000 + 12),
        Word::from(-85i32 as u32) // 256 / -3 = -85
    );
    // sdiv32: positive / negative, zero quotient
    // sdiv32 16(fp), -8(fp), -16(fp), 0, 0
    assert_eq!(
        memory_backend.get_value(0x1000 + 16),
        Word::from(0) // 3 / -256 = 0
    );
    // sdiv32: negative / positive, non-zero quotient
    // sdiv32 20(fp), -16(fp), 3, 0, 1
    assert_eq!(
        memory_backend.get_value(0x1000 + 20),
        Word::from(-85i32 as u32) // -256 / 3 = -85
    );
    // sdiv32: negative / positive, zero quotient
    // sdiv32 24(fp), -20(fp), -4(fp), 0, 0
    assert_eq!(
        memory_backend.get_value(0x1000 + 24),
        Word::from(0) // -3 / 256 = 0
    );
    // sdiv32: negative / negative, non-zero quotient
    // sdiv32 28(fp), -16(fp), -3, 0, 1
    assert_eq!(
        memory_backend.get_value(0x1000 + 28),
        Word::from(85) // -256 / -3 = 85
    );
    // sdiv32: negative / negative, zero quotient
    // sdiv32 32(fp), -20(fp), -16(fp), 0, 0
    assert_eq!(
        memory_backend.get_value(0x1000 + 32),
        Word::from(0) // -3 / -256 = 0
    );
    // sdiv32: zero / positive
    // sdiv32 36(fp), -12(fp), 3, 0, 1
    assert_eq!(
        memory_backend.get_value(0x1000 + 36),
        Word::from(0) // 0 / 3 = 0
    );
    // sdiv32: zero / negative
    // sdiv32 40(fp), -12(fp), -3, 0, 1
    assert_eq!(
        memory_backend.get_value(0x1000 + 40),
        Word::from(0) // 0 / -3 = 0
    );
}

#[test]
fn prove_sdiv() {
    let program = sdiv_program::<BabyBear>();
    let (_machine, memory_backend) = prove_program(0, program, ProgramTableType::Public);
    expected_sdiv_memory_state(&memory_backend);
}

#[test]
fn prove_sdiv_with_nonzero_initial_pc() {
    let program = sdiv_program::<BabyBear>();
    let (_machine, memory_backend) = prove_program(1234, program, ProgramTableType::Public);
    expected_sdiv_memory_state(&memory_backend);
}

#[test]
fn prove_multi_segment_sdiv() {
    let program = sdiv_program::<BabyBear>();
    let (_finalized_machine, instance_data, memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 8);

    assert!(verified_ok);
    // Four machines should have been created to run the program.
    assert_eq!(instance_data.segments.len(), 3);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);

    expected_sdiv_memory_state(&memory_backend);
}

fn expected_single_byte_memory_state(memory_backend: &ValidaMemoryBackend) {
    // LOADU8 4(fp), 0, -8(fp), 0, 0
    // should set 4(fp) to 255
    assert_eq!(memory_backend.get_value(0x1000 + 4), Word::from(255));
    // LOADS8 8(fp), 0, -8(fp), 0, 0
    // should set 8(fp) to -1
    assert_eq!(
        memory_backend.get_value(0x1000 + 8),
        Word::from_components_le([255, 255, 255, 255])
    );
    // STOREU8 0, -8(fp), -4(fp), 0, 0
    // should set the byte at -3(fp) to 1
    assert_eq!(
        memory_backend.get_value(0x1000 - 4),
        Word::from_components_le([1, 1, 3, 4])
    );
}

#[test]
fn prove_multi_segment_single_byte_instrs() {
    let program = single_byte_program::<BabyBear>();

    let (_finalized_machine, instance_data, memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 4);

    assert!(verified_ok);
    // Two machines should have been created to run the program.
    assert_eq!(instance_data.segments.len(), 2);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);

    expected_single_byte_memory_state(&memory_backend);
}

#[test]
fn prove_single_byte_instrs() {
    let program = single_byte_program::<BabyBear>();
    let (_machine, memory_backend) = prove_program(0, program, ProgramTableType::Public);
    expected_single_byte_memory_state(&memory_backend);
}

fn expected_fibonacci_memory_state(memory_backend: &ValidaMemoryBackend) {
    assert_eq!(
        memory_backend.get_value(0x1000 + 4), // Return value
        Word::from(75025)                     // 25th fibonacci number (75025)
    );
}

#[test]
fn prove_fibonacci() {
    let program = fib_program::<BabyBear>();

    let (machine, memory_backend) = prove_program(0, program, ProgramTableType::Public);

    assert_eq!(machine.cpu().clock, 192);
    assert_eq!(machine.cpu().operations.len(), 192);
    assert_eq!(machine.mem().operations.values().flatten().count(), 414);
    assert_eq!(machine.add_u32().operations.len(), 105);
    expected_fibonacci_memory_state(&memory_backend);
}

#[test]
fn prove_fibonacci_circuit_specific() {
    let program = fib_program::<BabyBear>();

    let (machine, memory_backend) = prove_program(0, program, ProgramTableType::Preprocessed);

    assert_eq!(machine.cpu().clock, 192);
    assert_eq!(machine.cpu().operations.len(), 192);
    assert_eq!(machine.mem().operations.values().flatten().count(), 414);
    assert_eq!(machine.add_u32().operations.len(), 105);
    expected_fibonacci_memory_state(&memory_backend);
}

#[test]
fn prove_multi_segment_fibonacci() {
    let program = fib_program::<BabyBear>();

    let (_finalized_machine, instance_data, memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 128);

    assert!(verified_ok);
    // Two machines should have been created to run the program.
    assert_eq!(instance_data.segments.len(), 2);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);
    let fp_final = instance_data.segments.last().unwrap().fp_final;

    //Todo()!
    assert_eq!(
        memory_backend.get_value(fp_final + 4), // Return value
        Word::from(75025)                       // 25th fibonacci number (75025)
    );
}

fn expected_add_with_overflow_memory_state(memory_backend: &ValidaMemoryBackend) {
    assert_eq!(
        memory_backend.get_value(0x1000 + 12),
        Word::from_components_be([0, 0, 0, 1]) // 0xFFFFFFFF + 2 = 1 (overflow)
    );
}

#[test]
fn prove_add_with_overflow() {
    let program = add_with_overflow_program::<BabyBear>();

    let (_machine, memory_backend) = prove_program(0, program, ProgramTableType::Public);

    expected_add_with_overflow_memory_state(&memory_backend);
}

#[test]
fn prove_multi_segment_add_with_overflow() {
    let program = add_with_overflow_program::<BabyBear>();

    let (_finalized_machine, instance_data, memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 2);

    assert!(verified_ok);
    // Two machines should have been created to run the program.
    assert_eq!(instance_data.segments.len(), 2);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);

    expected_add_with_overflow_memory_state(&memory_backend);
}

fn expected_sub_with_overflow_memory_state(memory_backend: &ValidaMemoryBackend) {
    assert_eq!(
        memory_backend.get_value(0x1000 + 12),
        Word::from_components_be([255, 255, 255, 255]) // 2 - 3 = 2^32 - 1 = 0xffffffff
    );
}

#[test]
fn prove_sub_with_overflow() {
    let program = sub_with_overflow_program::<BabyBear>();

    let (_machine, memory_backend) = prove_program(0, program, ProgramTableType::Public);

    expected_sub_with_overflow_memory_state(&memory_backend);
}

#[test]
fn prove_multi_segment_sub_with_overflow() {
    let program = sub_with_overflow_program::<BabyBear>();

    let (_finalized_machine, instance_data, memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 2);

    assert!(verified_ok);
    // Two machines should have been created to run the program.
    assert_eq!(instance_data.segments.len(), 2);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);

    expected_sub_with_overflow_memory_state(&memory_backend);
}

fn expected_sub_with_borrow_propagation_memory_state(memory_backend: &ValidaMemoryBackend) {
    assert_eq!(
        memory_backend.get_value(0x1000 + 12),
        Word::from_components_be([0, 255, 255, 255]) //0x0100000000 - 0x0000000001 = 0x00ffffffff
    );
}

#[test]
fn prove_sub_with_borrow_propagation() {
    let program = sub_with_borrow_propagation_program::<BabyBear>();

    let (_machine, memory_backend) = prove_program(0, program, ProgramTableType::Public);

    expected_sub_with_borrow_propagation_memory_state(&memory_backend);
}

#[test]
fn prove_multi_segment_sub_with_borrow_propagation() {
    let program = sub_with_borrow_propagation_program::<BabyBear>();

    let (_finalized_machine, instance_data, memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 2);

    assert!(verified_ok);
    // Two machines should have been created to run the program.
    assert_eq!(instance_data.segments.len(), 2);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);

    expected_sub_with_borrow_propagation_memory_state(&memory_backend);
}

#[test]
fn prove_output() {
    let program = output_program::<BabyBear>();

    let (machine, _memory_backend) = prove_program(0, program, ProgramTableType::Public);
    let clks = &machine.output().clks_log;
    let tape = machine.output_tape();
    assert_eq!(
        clks.len(),
        tape.len(),
        "output tape has length {} but log of clk values has length {}",
        tape.len(),
        clks.len()
    );
    assert_eq!((tape[0], clks[0]), (4u8, 2));
    assert_eq!((tape[1], clks[1]), (5u8, 3));
}

#[test]
fn prove_multi_segment_output() {
    let program = output_program::<BabyBear>();

    let (_finalized_machine, instance_data, _memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 4);

    assert!(verified_ok);
    // Two machines should have been created to run the program.
    assert_eq!(instance_data.segments.len(), 2);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);
}

/// Expected memory state after the left imm ops program.
fn left_imm_ops_expected_memory_state(memory_backend: &ValidaMemoryBackend) {
    assert_eq!(
        memory_backend.get_value(0x1000 + 4),
        Word::from(0) // 3 < 3 (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 8),
        Word::from(1) // 3 <= 3 (true)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 12),
        Word::from(0) // 4 < 3 (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 16),
        Word::from(0) // 4 <= 3 (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 20),
        Word::from(1) // 2 < 3 (true)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 24),
        Word::from(1) // 2 <= 3 (true)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 28),
        Word::from(0) // 256 < 3 (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 32),
        Word::from(0) // 256 <= 3 (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 36),
        Word::from(1) // 3 < 256 (true)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 40),
        Word::from(1) // 3 <= 256 (false)
    );
}

#[test]
fn prove_left_imm_ops() {
    let program = left_imm_ops_program::<BabyBear>();

    let (_machine, memory_backend) = prove_program(0, program, ProgramTableType::Public);

    left_imm_ops_expected_memory_state(&memory_backend);
}

#[test]
fn prove_multi_segment_left_imm_ops() {
    let program = left_imm_ops_program::<BabyBear>();

    let (_finalized_machine, instance_data, memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 8);

    assert!(verified_ok);
    // Two machines should have been created to run the left imm ops program.
    assert_eq!(instance_data.segments.len(), 2);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);

    left_imm_ops_expected_memory_state(&memory_backend);
}

/// Expected memory state after the signed inequality program.
fn signed_inequality_expected_memory_state(memory_backend: &ValidaMemoryBackend) {
    // signed inequalities
    assert_eq!(
        memory_backend.get_value(0x1000 + 4),
        Word::from(1) // -2 < -1 (true)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 8),
        Word::from(1) // -2 < 1 (true)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 12),
        Word::from(0) // 1 < -1 (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 16),
        Word::from(0) // -1 < -1 (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 20),
        Word::from(1) // -1 <= -1 (true)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 24),
        Word::from(0) // -1 < -2 (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 28),
        Word::from(0) // -1 < -2 (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 32),
        Word::from(1) // -1 < 1 (true)
    );

    // unsigned inequalities
    assert_eq!(
        memory_backend.get_value(0x1000 + 36),
        Word::from(1) // 0xFFFFFFFE < 0xFFFFFFFF (true)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 40),
        Word::from(0) // 0xFFFFFFFE < 1 (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 44),
        Word::from(1) // 1 < 0xFFFFFFFF (true)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 48),
        Word::from(0) // 0xFFFFFFFF < 0xFFFFFFFFFF (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 52),
        Word::from(1) // 0xFFFFFFFF <= 0xFFFFFFFF (true)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 56),
        Word::from(0) // 0xFFFFFFFF < 0xFFFFFFFE (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 60),
        Word::from(0) // 0xFFFFFFFF < 0xFFFFFFFE (false)
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 64),
        Word::from(0) // 0xFFFFFFFF < 1 (false)
    );
}

#[test]
fn prove_signed_inequality() {
    let program = signed_inequality_program::<BabyBear>();

    let (_machine, memory_backend) = prove_program(0, program, ProgramTableType::Public);

    signed_inequality_expected_memory_state(&memory_backend);
}

#[test]
fn prove_multi_segment_signed_inequality() {
    let program = signed_inequality_program::<BabyBear>();

    let (_finalized_machine, instance_data, memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 8);

    assert!(verified_ok);
    // Three machines should have been created to run the signed inequality program.
    assert_eq!(instance_data.segments.len(), 3);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);

    signed_inequality_expected_memory_state(&memory_backend);
}

/// Expected memory state after the LOADFP program.
fn loadfp_expected_memory_state(memory_backend: &ValidaMemoryBackend) {
    assert_eq!(
        memory_backend.get_value(0x1000 + 4),
        Word::from_components_le([0, 16, 0, 0]) // fp = 0x1000
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 8),
        Word::from_components_le([3, 16, 0, 0]) // fp(3) = 0x1003
    );
}

#[test]
fn prove_loadfp() {
    let program = loadfp_program::<BabyBear>();

    let (_machine, memory_backend) = prove_program(0, program, ProgramTableType::Public);

    loadfp_expected_memory_state(&memory_backend);
}

#[test]
fn prove_multi_segment_loadfp() {
    let program = loadfp_program::<BabyBear>();

    let (_finalized_machine, instance_data, memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 2);

    assert!(verified_ok);
    // Two machines should have been created to run the Keccak program.
    assert_eq!(instance_data.segments.len(), 2);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);

    loadfp_expected_memory_state(&memory_backend);
}

fn persistent_expected_memory_state(memory_backend: &ValidaMemoryBackend) {
    assert_eq!(
        memory_backend.get_value(0x1000 + 4),
        Word::from_components_le([2, 0, 0, 0])
    );
    assert_eq!(
        memory_backend.get_value(0x1000 + 8),
        Word::from_components_le([4, 0, 0, 0])
    );
}

#[test]
fn prove_multi_segment_persistent_example() {
    let program = persistent_memory_example::<BabyBear>();

    let (_finalized_machine, instance_data, memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 4);

    assert!(verified_ok);

    // Two machines should have been created to run the Keccak program.
    assert_eq!(instance_data.segments.len(), 3);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);

    persistent_expected_memory_state(&memory_backend);
}

/// Test the Keccak permutation across a single segment machine.
#[test]
fn prove_keccak() {
    let mut state = [1u64; 25];
    for _ in 0..3 {
        keccakf(&mut state);
    }

    let postimage = convert_array(state);

    let program = keccak_program::<BabyBear>();
    let (_machine, memory_backend) = prove_program(0, program, ProgramTableType::Public);

    // CHECK POSTIMAGE
    for i in 0..50 {
        assert_eq!(
            memory_backend.get_value(0x1000 - 0xc8 + i * 4),
            Word::from_components_le(postimage[i as usize])
        );
    }
}

/// Test the Keccak permutation across a multi-segment machine.
#[test]
fn prove_multi_segment_keccak() {
    let mut state = [1u64; 25];
    for _ in 0..3 {
        keccakf(&mut state);
    }

    let postimage = convert_array(state);

    let program = keccak_program::<BabyBear>();
    let (_finalized_machine, instance_data, memory_backend, verified_ok) =
        multi_segment_prove_program(program, ProgramTableType::Public, 32);

    assert!(verified_ok);

    // Two machines should have been created to run the Keccak program.
    assert_eq!(instance_data.segments.len(), 2);
    // The program should have ran successfully.
    assert!(!instance_data.did_fail);

    // CHECK POSTIMAGE
    for i in 0..50 {
        assert_eq!(
            memory_backend.get_value(0x1000 - 0xc8 + i * 4),
            Word::from_components_le(postimage[i as usize])
        );
    }
}

fn convert_array(input: [u64; 25]) -> [[u8; 4]; 50] {
    let mut output = [[0u8; 4]; 50];

    // Each u64 will be split into two [u8; 4] arrays
    // The lower 32 bits go to the first array
    // The upper 32 bits go to the second array
    for (i, &num) in input.iter().enumerate() {
        // Calculate indices for output array
        let out_idx1 = i * 2;
        let out_idx2 = i * 2 + 1;

        // Split into lower and upper 32 bits
        let lower = num as u32;
        let upper = (num >> 32) as u32;

        // Convert lower 32 bits to [u8; 4]
        output[out_idx1] = [
            (lower & 0xFF) as u8,
            ((lower >> 8) & 0xFF) as u8,
            ((lower >> 16) & 0xFF) as u8,
            ((lower >> 24) & 0xFF) as u8,
        ];

        // Convert upper 32 bits to [u8; 4]
        output[out_idx2] = [
            (upper & 0xFF) as u8,
            ((upper >> 8) & 0xFF) as u8,
            ((upper >> 16) & 0xFF) as u8,
            ((upper >> 24) & 0xFF) as u8,
        ];
    }

    output
}
