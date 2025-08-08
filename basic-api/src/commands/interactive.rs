use std::fs::{self, File};

use p3_baby_bear::BabyBear;

use crate::commands::common::MyMachine;
use crate::machine::multi_segment::MultiSegmentBasicRunningMachine;
use crate::{
    BasicMachine, BasicRunningMachine, MultiSegmentBasicMachine, ValidaBootData, ValidaRuntime,
    ValidaSegmentBootData,
};

use valida_cpu::{MachineWithCpuChip, Registers};
use valida_machine::{
    AdviceProvider, AdviceProviderWithDefault, Machine, MemoryBackendTrait, RunningMachine,
    StoppingFlag, StorageBackendTrait, StorageBackendType, ValidaMemoryBackend,
    WriteCallbackWithDefault,
};
use valida_memory::{add_diff_bytes_receives, MachineWithMemoryChip};

use valida_elf::{load_elf_object_file, Program};
use valida_program::{MachineWithProgramChip, MachineWithProgramROM, ProgramTableType};
use valida_static_data::{MachineWithStaticDataChip, StaticDataChipType};

pub struct Context<'a> {
    breakpoints_: Vec<u32>,
    stopped_: StoppingFlag,
    last_fp_: u32,
    recorded_current_fp_: u32,
    last_fp_size_: i32,
    advice: AdviceProvider,
    state_: MultiSegmentBasicRunningMachine<'a, BabyBear>,
}

impl<'a> Context<'a> {
    pub fn new(
        runtime: &'a mut ValidaRuntime,
        initial_fp: u32,
        //option_stdin: Option<String>, // TODO: This parameter is also unused on the main branch
        program_bytes: Vec<u8>,
        advice_provider: AdviceProviderWithDefault,
    ) -> Context<'a> {
        let mut machine = MultiSegmentBasicMachine::<BabyBear>::default();

        let Program {
            code,
            data,
            initial_program_counter,
        } = load_elf_object_file(&program_bytes);

        let initial_register_values = Registers {
            pc: initial_program_counter,
            fp: initial_fp,
        };
        let boot_data = ValidaBootData {
            program_rom: code,
            static_data: data,
            initial_register_values,
            program_table_type: ProgramTableType::Public, // TODO
            static_data_chip_type: StaticDataChipType::Public,
            max_trace_height: 1 << 20, // TODO: use input!
            program_file: program_bytes,
        };

        // now initialize the machine with the boot data
        machine.init(boot_data);
        // start the machine to get the `RunningMachine`
        let mut state = machine.start(runtime);

        let mut context = Context {
            breakpoints_: Vec::new(),
            stopped_: StoppingFlag::DidNotStop,
            last_fp_: initial_fp,
            recorded_current_fp_: initial_fp,
            last_fp_size_: 0,
            advice: advice_provider.0,
            state_: state,
        };

        context
    }

    pub fn step(&mut self) -> (StoppingFlag, u32) {
        // do not execute if already stopped
        if self.stopped_ == StoppingFlag::DidStop
            || self.stopped_ == StoppingFlag::DidFail
            || self.stopped_ == StoppingFlag::SizeLimitReached
        {
            return (self.stopped_, 0);
        }
        let did_stop = MultiSegmentBasicMachine::step(&mut self.state_);

        // Get pc, fp from the current segment
        let reg = self.state_.machine.get_registers().expect("Unexpectedly found no segments in MultiSegmentBasicMachine during interactive execution.");
        let (pc, fp) = (reg.pc, reg.fp);

        let instruction = self.state_.machine.program_rom().get_instruction(pc);
        println!("{:4} : {:?}", pc, instruction.to_string());

        // check if fp is changed
        if fp != self.recorded_current_fp_ {
            self.last_fp_size_ = self.recorded_current_fp_ as i32 - fp as i32;
            self.last_fp_ = self.recorded_current_fp_;
        } else if fp == self.last_fp_ {
            self.last_fp_size_ = 0;
        }
        self.recorded_current_fp_ = fp;

        (did_stop, pc)
    }
}

pub fn init_context(_context: &mut Context) -> String {
    String::from("created machine")
}

pub fn status(context: &mut Context) -> String {
    // construct machine status
    let mut status = String::new();
    status.push_str("FP: ");
    let reg = context.state_.machine.get_registers().expect(
        "Unexpectedly found no segments in MultiSegmentBasicMachine during interactive execution.",
    );
    let (pc, fp) = (reg.pc, reg.fp);
    status.push_str(&fp.to_string());
    status.push_str(", PC: ");
    status.push_str(&pc.to_string());
    status.push_str(match context.stopped_ {
        StoppingFlag::DidStop => ", Stopped",
        StoppingFlag::DidNotStop => ", Running",
        StoppingFlag::DidFail => ", Failed",
        StoppingFlag::SizeLimitReached => ", Size Limit Reached",
    });
    status
}

pub fn show_frame(size: i32, context: &mut Context) -> String {
    let mut frame = String::new();
    let reg = context.state_.machine.get_registers().expect(
        "Unexpectedly found no segments in MultiSegmentBasicMachine during interactive execution.",
    );
    let fp = reg.fp as i32;
    frame.push_str(format!("FP: {:x}\n", fp).as_str());
    for i in 0..size {
        let offset = i * -4;
        let read_addr = (fp + offset) as u32;
        let string_val = context.state_.runtime.memory_backend.examine(read_addr);
        let frameslot_addr = format!("0x{:8} | {:3}(fp)", read_addr, offset);
        let frameslot = format!("{:>7}", frameslot_addr);
        let frame_str = format!("\n{} : {}", frameslot, string_val);
        frame += &frame_str;
    }

    frame
}

pub fn last_frame(context: &mut Context) -> String {
    let mut frame = String::new();

    let lfp = context.last_fp_;
    let reg = context.state_.machine.get_registers().expect(
        "Unexpectedly found no segments in MultiSegmentBasicMachine during interactive execution.",
    );
    let fp = reg.fp as i32;
    let last_size = context.last_fp_size_ as i32;
    frame += format!("Last FP   : 0x{:x}, Frame size: {}\n", lfp, last_size).as_str();
    frame += format!("Current FP: 0x{:x}\n", fp).as_str();

    // print last frame
    for i in (-10..(last_size / 4) + 1).rev() {
        let offset: i32 = i * 4;
        let read_addr = (fp + offset) as u32;
        let string_val = context.state_.runtime.memory_backend.examine(read_addr);
        let frameslot_addr = format!("{}(fp)", offset);
        let frameslot = format!("0x{:<7x} | {:>7}", read_addr, frameslot_addr);
        let frame_str = format!("\n{} : {}", frameslot, string_val);
        frame += &frame_str;
    }
    frame
}

pub fn list_instrs(print_size_arg: Option<&String>, context: &mut Context) -> String {
    let reg = context.state_.machine.get_registers().expect(
        "Unexpectedly found no segments in MultiSegmentBasicMachine during interactive execution.",
    );
    let pc = reg.pc;

    let program_rom = context.state_.machine.program_rom();
    let total_size = program_rom.0.len();

    let print_size = match print_size_arg {
        Some(size) => size.parse::<u32>().unwrap(),
        None => 10,
    };

    let mut formatted = String::new();
    for i in 0..print_size {
        let cur_pc = pc + i;
        if cur_pc >= total_size as u32 {
            break;
        }
        let instruction = program_rom.get_instruction(cur_pc);
        formatted.push_str(format!("{:4} : {:?}\n", cur_pc, instruction.to_string()).as_str());
    }
    formatted
}

pub fn set_bp(pc: u32, context: &mut Context) -> String {
    context.breakpoints_.push(pc);
    let message = format!("Breakpoint set at pc: {}", pc);
    message
}
pub fn show_memory(addr: u32, context: &mut Context) -> String {
    // show memory at address, by default show 20 cells
    let mut memory = String::new();
    for i in 0..8 {
        let read_addr = addr + i * 4;
        let string_val = context.state_.runtime.memory_backend.examine(read_addr);
        let memory_str = format!("0x{:<8x} : {}\n", read_addr, string_val);
        memory += &memory_str;
    }

    memory
}

pub fn run_until(context: &mut Context) -> String {
    let mut message = String::new();
    loop {
        let (stop, pc) = context.step();
        if stop == StoppingFlag::DidStop {
            message.push_str("Execution stopped");
            break;
        }
        if stop == StoppingFlag::DidFail {
            message.push_str("Execution failed");
            break;
        }
        if stop == StoppingFlag::SizeLimitReached {
            message.push_str("Size limit reached");
            break;
        }
        if context.breakpoints_.contains(&pc) {
            let bp_index = context.breakpoints_.iter().position(|&x| x == pc).unwrap();
            message = format!("Execution stopped at breakpoint {}, PC: {}", bp_index, pc);
            break;
        }
    }
    message
}

pub fn step(context: &mut Context) -> Option<String> {
    let (stop, _) = context.step();
    if stop == StoppingFlag::DidStop {
        context.stopped_ = StoppingFlag::DidStop;
        post_process(context);
        Some(String::from("Execution stopped"))
    } else if stop == StoppingFlag::SizeLimitReached {
        context.stopped_ = StoppingFlag::SizeLimitReached;
        post_process(context);
        Some(String::from("Size limit reached"))
    } else if stop == StoppingFlag::DidFail {
        context.stopped_ = StoppingFlag::DidFail;
        post_process(context);
        Some(String::from("Execution failed"))
    } else {
        None
    }
}

/// Method for doing post-processing of the trace after execution has stopped.
///
/// Currently, this is only adding receives for `diff_bytes` range checks
/// to the trace, which is required for soundness of the memory argument.
pub fn post_process(context: &mut Context) {
    /// As the multi segment machine contains a vector of `BasicMachine`s and the last of
    /// those is being executed, we call `add_diff_bytes_receives` on that last machine.
    add_diff_bytes_receives(context.state_.machine.get_current_machine_mut());
}
