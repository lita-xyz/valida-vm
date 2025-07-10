use std::fs::File;
use std::io::{stdout, Write};

use p3_baby_bear::BabyBear;

use crate::{BasicMachineMetrics, MultiSegmentBasicMachine};
use valida_cpu::{MachineWithCpuChip, Registers};
use valida_machine::{
    AdviceProviderWithDefault, Machine, ReplayAdviceProvider, WriteCallbackWithDefault,
};

use valida_elf::Program;
use valida_static_data::{MachineWithStaticDataChip, StaticDataChipType};

use valida_output::MachineWithOutputChip;

use ark_std::{end_timer, start_timer};

use crate::commands::common::{
    prepare_basic_machine, prepare_machine, prepare_runtime, prepare_runtime_default,
};
use crate::BasicMachine;

/// A run function, which uses a single `BasicMachine` to execute the program
pub fn run_basic_machine(
    program: Program,
    fast: bool,
    write_callback: WriteCallbackWithDefault,
    advice_provider: AdviceProviderWithDefault,
    initial_fp: u32,
    specific: bool,
    max_trace_height: u32,
    program_file: Vec<u8>,
) -> (bool, Vec<u8>) {
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
        program_file.clone(),
    );
    end_timer!(t_prep);

    machine.enable_logging(!fast);
    machine.program_file = program_file.clone();

    // Assign the read and write callbacks to the `ValidaRuntime`
    let mut runtime = prepare_runtime(advice_provider, write_callback)
        .unwrap_or_else(|err| panic!("Failed to prepare runtime: {:?}", err));

    let mut metrics = BasicMachineMetrics::initialize();

    let t_run = start_timer!(|| "valida > run | machine.run(..)");
    let mut state = machine.start(&mut runtime);
    let (instance_data, output) = BasicMachine::run(&mut state, &mut metrics);
    end_timer!(t_run);

    metrics.finalize(&program_file);

    (!instance_data.did_fail, output)
}

/// The future default `run` function, which uses the `MultiSegmentBasicMachine` to execute
/// the program.
pub fn run(
    program: Program,
    fast: bool,
    write_callback: WriteCallbackWithDefault,
    advice_provider: AdviceProviderWithDefault,
    initial_fp: u32,
    specific: bool,
    max_trace_height: u32,
    program_file: Vec<u8>,
) -> (bool, Vec<u8>) {
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
        program_file.clone(),
    );
    end_timer!(t_prep);

    machine.enable_logging(!fast);
    machine.program_file = program_file.clone();

    // Assign the read and write callbacks to the `ValidaRuntime`
    let mut runtime = prepare_runtime(advice_provider, write_callback)
        .unwrap_or_else(|err| panic!("Failed to prepare runtime: {:?}", err));

    let t_run = start_timer!(|| "valida > run | machine.run(..)");

    let mut state = machine.start(&mut runtime);
    let mut metrics = BasicMachineMetrics::initialize();

    type M = MultiSegmentBasicMachine<BabyBear>;
    let (instance_data, output) = M::run(&mut state, &mut metrics);
    end_timer!(t_run);

    metrics.finalize(&program_file);

    (!instance_data.did_fail, output)
}
