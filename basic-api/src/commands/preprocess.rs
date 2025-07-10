use crate::BasicMachine;

use ark_std::{end_timer, start_timer};
use p3_baby_bear::BabyBear;
use postcard::to_allocvec;

use valida_cpu::Registers;
use valida_elf::Program;
use valida_machine::Machine;

use super::common::{prepare_basic_machine, prepare_machine, MyConfig, MyPK, MyVK};

/// A preprocessing function for the basic machine (i.e. for a single segment)
pub fn preprocess_basic_machine(
    program: Program,
    config: MyConfig,
    show_preprocessed: bool,
    show_dims: bool,
    initial_fp: u32,
    specific: bool,
    max_trace_height: u32,
    program_file: Vec<u8>,
) -> (Vec<u8>, Vec<u8>) {
    let t_prep = start_timer!(|| "valida | Prepare machine");
    let mut machine = prepare_basic_machine(
        program.code,
        program.data,
        Registers {
            pc: program.initial_program_counter,
            fp: initial_fp,
        },
        specific,
        max_trace_height,
        program_file,
    );
    end_timer!(t_prep);

    let show_preprocessed = vec![show_preprocessed; BasicMachine::<BabyBear>::NUM_CHIPS];
    let (pk, vk) = machine.pre_process(&config, show_preprocessed, show_dims);
    let pk_bytes = to_allocvec(&pk).expect("good serialize");
    let vk_bytes = to_allocvec(&vk).expect("good serialize");

    (pk_bytes, vk_bytes)
}

/// The future default preprocessing function for the `MultiSegmentBasicMachine`
pub fn preprocess(
    program: Program,
    config: MyConfig,
    show_preprocessed: bool,
    show_dims: bool,
    initial_fp: u32,
    specific: bool,
    max_trace_height: u32,
    program_file: Vec<u8>,
) -> (Vec<u8>, Vec<u8>) {
    let t_prep = start_timer!(|| "valida | Prepare machine");
    let mut machine = prepare_machine(
        program.code,
        program.data,
        Registers {
            pc: program.initial_program_counter,
            fp: initial_fp,
        },
        specific,
        max_trace_height,
        program_file,
    );
    end_timer!(t_prep);

    let show_preprocessed = vec![show_preprocessed; BasicMachine::<BabyBear>::NUM_CHIPS];
    let (pk, vk) = machine.pre_process(&config, show_preprocessed, show_dims);
    let pk_bytes = to_allocvec(&pk).expect("good serialize");
    let vk_bytes = to_allocvec(&vk).expect("good serialize");

    (pk_bytes, vk_bytes)
}
