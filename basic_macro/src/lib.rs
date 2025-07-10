#![allow(unused)]

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_air::Air;
use p3_commit::{Pcs, UnivariatePcs, UnivariatePcsWithLde};
use p3_field::PrimeField32;
use p3_field::{extension::BinomialExtensionField, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Dimensions, Matrix};
use p3_maybe_rayon::*;
use p3_util::log2_ceil_usize;
use valida_alu_u32::{
    add::{Add32Chip, Add32Instruction, MachineWithAdd32Chip},
    bitwise::{
        And32Instruction, Bitwise32Chip, MachineWithBitwise32Chip, Or32Instruction,
        Xor32Instruction,
    },
    com::{Com32Chip, Eq32Instruction, MachineWithCom32Chip, Ne32Instruction},
    div::{Div32Chip, Div32Instruction, MachineWithDiv32Chip, SDiv32Instruction},
    lt::{Lt32Chip, Lt32Instruction, Lte32Instruction, MachineWithLt32Chip},
    mul::{
        MachineWithMul32Chip, Mul32Chip, Mul32Instruction, Mulhs32Instruction, Mulhu32Instruction,
    },
    shift::{
        MachineWithShift32Chip, Shift32Chip, Shl32Instruction, Shr32Instruction, Sra32Instruction,
    },
    sub::{MachineWithSub32Chip, Sub32Chip, Sub32Instruction},
};
use valida_bus::{
    MachineWithBytesBus, MachineWithGeneralBus, MachineWithMemBus, MachineWithOutputBus,
    MachineWithProgramBus, MachineWithRangeBus8,
};
use valida_bytes::{BytesChip, BytesTable, MachineWithBytesChip, MachineWithRangeCheckeru8};
use valida_cpu::{
    BeqInstruction, BneInstruction, Imm32Instruction, JalInstruction, JalvInstruction,
    Load32Instruction, LoadFpInstruction, ReadAdviceInstruction, Registers, StopInstruction,
    Store32Instruction,
};
use valida_cpu::{CpuChip, MachineWithCpuChip, Operation};
use valida_derive::Machine;
use valida_lookups::{MachineWithLookupChip, MachineWithMultiLookupChip};
use valida_machine::{
    AdviceProvider, BusArgument, Chip, ChipProof, Instruction, Machine, MachineInstanceData,
    MachineProof, MachineProverKey, MachineVerifierKey, Operands, PcsError, ProgramROM,
    PublicTrace, StoppingFlag, ValidaAirBuilder, VerificationError, Word,
};
use valida_memory::{MachineWithMemoryChip, MemoryChip};
use valida_output::{MachineWithOutputChip, OutputChip, WriteInstruction};
use valida_program::{MachineWithProgramChip, ProgramChip, ProgramTable, ProgramTableType};
use valida_range::{MachineWithRangeChip, RangeCheckerChip, RangeTable};
use valida_static_data::{MachineWithStaticDataChip, StaticDataChip, StaticDataChipType};

use p3_maybe_rayon::prelude::*;
use valida_machine::{ProverOptions, StarkConfig};

use valida_basic::instance_data::ValidaInstanceData;
#[derive(Machine, Default)]
#[machine_fields(F)]
pub struct BasicMachine<F: StarkField> {
    // Core instructions
    #[instruction]
    load32: Load32Instruction,

    #[instruction]
    store32: Store32Instruction,

    #[instruction]
    jal: JalInstruction,

    #[instruction]
    jalv: JalvInstruction,

    #[instruction]
    beq: BeqInstruction,

    #[instruction]
    bne: BneInstruction,

    #[instruction]
    imm32: Imm32Instruction,

    #[instruction]
    stop: StopInstruction,

    #[instruction]
    fail: FailInstruction,

    #[instruction]
    loadfp: LoadFpInstruction,

    // ALU instructions
    #[instruction(add_u32)]
    add32: Add32Instruction,

    #[instruction(sub_u32)]
    sub32: Sub32Instruction,

    #[instruction(mul_u32)]
    mul32: Mul32Instruction,

    #[instruction(mul_u32)]
    mulhs32: Mulhs32Instruction,

    #[instruction(mul_u32)]
    mulhu32: Mulhu32Instruction,

    #[instruction(div_u32)]
    div32: Div32Instruction,

    #[instruction(div_u32)]
    sdiv32: SDiv32Instruction,

    #[instruction(shift_u32)]
    shl32: Shl32Instruction,

    #[instruction(shift_u32)]
    shr32: Shr32Instruction,

    #[instruction(shift_u32)]
    sra32: Sra32Instruction,

    #[instruction(lt_u32)]
    lt32: Lt32Instruction,

    #[instruction(lt_u32)]
    lte32: Lte32Instruction,

    #[instruction(bitwise_u32)]
    and32: And32Instruction,

    #[instruction(bitwise_u32)]
    or32: Or32Instruction,

    #[instruction(bitwise_u32)]
    xor32: Xor32Instruction,

    #[instruction(com_u32)]
    ne32: Ne32Instruction,

    #[instruction(com_u32)]
    eq32: Eq32Instruction,

    // Input/output instructions
    #[instruction]
    read: ReadAdviceInstruction,

    #[instruction(output)]
    write: WriteInstruction,

    #[chip]
    cpu: CpuChip,

    #[chip]
    program: ProgramChip<F>,

    #[chip]
    mem: MemoryChip,

    #[chip]
    add_u32: Add32Chip,

    #[chip]
    sub_u32: Sub32Chip,

    #[chip]
    mul_32: Mul32Chip,

    #[chip]
    div_u32: Div32Chip,

    #[chip]
    shift_u32: Shift32Chip,

    #[chip]
    lt_u32: Lt32Chip,

    #[chip]
    com_u32: Com32Chip,

    #[chip]
    bitwise_u32: Bitwise32Chip,

    #[chip]
    output: OutputChip,

    #[chip]
    bytes: BytesChip<F>,

    #[chip]
    #[static_data_chip]
    static_data: StaticDataChip,

    _phantom_sc: PhantomData<fn() -> F>,
}

impl<F: StarkField> MachineWithGeneralBus<F> for BasicMachine<F> {
    fn general_bus(&self) -> BusArgument {
        BusArgument::Global(0)
    }
}

impl<F: StarkField> MachineWithProgramBus<F> for BasicMachine<F> {
    fn program_bus(&self) -> BusArgument {
        BusArgument::Global(1)
    }
}

impl<F: StarkField> MachineWithMemBus<F> for BasicMachine<F> {
    fn mem_bus(&self) -> BusArgument {
        BusArgument::Global(2)
    }
}

impl<F: StarkField> MachineWithBytesBus<F> for BasicMachine<F> {
    fn bytes_bus(&self) -> BusArgument {
        BusArgument::Global(3)
    }
}

impl<F: StarkField> MachineWithOutputBus<F> for BasicMachine<F> {
    fn output_bus(&self) -> BusArgument {
        BusArgument::Global(4)
    }
}

impl<F: StarkField> MachineWithCpuChip<F> for BasicMachine<F> {
    fn cpu(&self) -> &CpuChip {
        &self.cpu
    }

    fn cpu_mut(&mut self) -> &mut CpuChip {
        &mut self.cpu
    }

    fn set_pc(&mut self, new_pc: u32) {
        self.cpu.pc = new_pc;
    }
    fn step_pc(&mut self) {
        self.cpu.pc += 1;
    }

    fn set_fp(&mut self, new_fp: u32) {
        self.cpu.fp = new_fp;
    }
    fn inc_fp(&mut self, offset: i32) {
        self.cpu.fp = (self.cpu.fp as i32 + offset) as u32;
    }

    fn push_bus_op(&mut self, imm: Option<Word<u8>>, opcode: u32, operands: Operands<i32>) {
        self.cpu.push_bus_op(imm, opcode, operands);
    }
    fn push_left_imm_bus_op(
        &mut self,
        imm: Option<Word<u8>>,
        opcode: u32,
        operands: Operands<i32>,
    ) {
        self.cpu.push_left_imm_bus_op(imm, opcode, operands);
    }
    fn push_op(&mut self, op: Operation, opcode: u32, operands: Operands<i32>) {
        self.cpu.push_op(op, opcode, operands);
    }
    fn set_initial_register_values(&mut self, reg: Registers) {
        self.cpu.set_initial_register_values(reg);
    }
}

impl<F: StarkField> MachineWithLookupChip<F, ProgramTable> for BasicMachine<F> {
    fn lookup_chip(&self) -> &ProgramChip<F> {
        &self.program
    }

    fn lookup_chip_mut(&mut self) -> &mut ProgramChip<F> {
        &mut self.program
    }

    fn lookup_chip_bus(&self) -> BusArgument {
        self.program_bus()
    }
}

impl<F: StarkField> MachineWithMemoryChip<F> for BasicMachine<F> {
    fn mem(&self) -> &MemoryChip {
        &self.mem
    }

    fn mem_mut(&mut self) -> &mut MemoryChip {
        &mut self.mem
    }

    fn read(&mut self, clk: u32, address: u32) -> Word<u8> {
        self.mem.read(clk, address, self.log_enabled())
    }
    fn write(&mut self, clk: u32, address: u32, value: Word<u8>) {
        self.mem.write(clk, address, value, self.log_enabled())
    }
    fn write_static(&mut self, address: u32, value: Word<u8>) {
        self.mem.write_static(address, value, self.log_enabled())
    }
}

impl<F: StarkField> MachineWithAdd32Chip<F> for BasicMachine<F> {
    fn add_u32(&self) -> &Add32Chip {
        &self.add_u32
    }

    fn add_u32_mut(&mut self) -> &mut Add32Chip {
        &mut self.add_u32
    }
}

impl<F: StarkField> MachineWithSub32Chip<F> for BasicMachine<F> {
    fn sub_u32(&self) -> &Sub32Chip {
        &self.sub_u32
    }

    fn sub_u32_mut(&mut self) -> &mut Sub32Chip {
        &mut self.sub_u32
    }
}

impl<F: StarkField> MachineWithMul32Chip<F> for BasicMachine<F> {
    fn mul_32(&self) -> &Mul32Chip {
        &self.mul_32
    }

    fn mul_32_mut(&mut self) -> &mut Mul32Chip {
        &mut self.mul_32
    }
}

impl<F: StarkField> MachineWithDiv32Chip<F> for BasicMachine<F> {
    fn div_u32(&self) -> &Div32Chip {
        &self.div_u32
    }

    fn div_u32_mut(&mut self) -> &mut Div32Chip {
        &mut self.div_u32
    }
}

impl<F: StarkField> MachineWithBitwise32Chip<F> for BasicMachine<F> {
    fn bitwise_u32(&self) -> &Bitwise32Chip {
        &self.bitwise_u32
    }

    fn bitwise_u32_mut(&mut self) -> &mut Bitwise32Chip {
        &mut self.bitwise_u32
    }
}

impl<F: StarkField> MachineWithLt32Chip<F> for BasicMachine<F> {
    fn lt_u32(&self) -> &Lt32Chip {
        &self.lt_u32
    }

    fn lt_u32_mut(&mut self) -> &mut Lt32Chip {
        &mut self.lt_u32
    }
}
impl<F: StarkField> MachineWithCom32Chip<F> for BasicMachine<F> {
    fn com_u32(&self) -> &Com32Chip {
        &self.com_u32
    }

    fn com_u32_mut(&mut self) -> &mut Com32Chip {
        &mut self.com_u32
    }
}

impl<F: StarkField> MachineWithShift32Chip<F> for BasicMachine<F> {
    fn shift_u32(&self) -> &Shift32Chip {
        &self.shift_u32
    }

    fn shift_u32_mut(&mut self) -> &mut Shift32Chip {
        &mut self.shift_u32
    }
}

impl<F: StarkField> MachineWithOutputChip<F> for BasicMachine<F> {
    fn output(&self) -> &OutputChip {
        &self.output
    }

    fn output_mut(&mut self) -> &mut OutputChip {
        &mut self.output
    }
}

impl<F: StarkField> MachineWithMultiLookupChip<F, BytesTable> for BasicMachine<F> {
    fn lookup_chip(&self) -> &BytesChip<F> {
        &self.bytes
    }
    fn lookup_chip_mut(&mut self) -> &mut BytesChip<F> {
        &mut self.bytes
    }
    fn lookup_chip_bus(&self, receive_index: usize) -> BusArgument {
        if receive_index == 0 {
            self.range_bus_8()
        } else {
            self.bytes_bus()
        }
    }
}

impl<F: StarkField> MachineWithStaticDataChip<F> for BasicMachine<F> {
    fn static_data(&self) -> &StaticDataChip {
        &self.static_data
    }

    fn static_data_mut(&mut self) -> &mut StaticDataChip {
        &mut self.static_data
    }
}
