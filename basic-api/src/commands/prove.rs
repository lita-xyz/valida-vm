use std::fs::{read, File};
use std::io::{stdout, Write};

use p3_baby_bear::BabyBear;
use postcard::from_bytes;

use valida_cpu::{MachineWithCpuChip, Registers};
use valida_machine::{
    AdviceProviderWithDefault, Machine, ProverOptions, ReplayAdviceProvider,
    WriteCallbackWithDefault,
};

use valida_elf::Program;
use valida_static_data::MachineWithStaticDataChip;

use ark_std::{end_timer, start_timer};

use crate::commands::common::{prepare_basic_machine, prepare_machine, prepare_runtime};
use crate::{BasicMachine, BasicMachineMetrics, MultiSegmentBasicMachine, ValidaRuntime};

use super::common::{MyConfig, MyPK, MyVK};

pub struct ProveDebugOptions {
    pub show_public: bool,
    pub show_preprocessed: bool,
    pub show_main: bool,
    pub show_interactions: bool,
    pub show_dims: bool,
}

impl Default for ProveDebugOptions {
    fn default() -> Self {
        Self {
            show_public: false,
            show_preprocessed: false,
            show_main: false,
            show_interactions: false,
            show_dims: false,
        }
    }
}

/// A prove function for the basic machine (i.e. for a single segment)
pub fn prove_basic_machine(
    program: Program,
    opts: ProveDebugOptions,
    advice_provider: AdviceProviderWithDefault,
    option_pk: Option<Vec<u8>>,
    config: MyConfig,
    initial_fp: u32,
    specific: bool,
    max_trace_height: u32,
    program_file: Vec<u8>,
) -> Result<Vec<u8>, String> {
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

    // Assign the read and write callbacks to the `ValidaRuntime`
    let mut runtime = prepare_runtime(advice_provider, WriteCallbackWithDefault::default())
        .unwrap_or_else(|err| panic!("Failed to prepare runtime: {:?}", err));
    let mut state = machine.start(&mut runtime);
    let mut metrics = BasicMachineMetrics::initialize();

    end_timer!(t_prep);

    let num_chips = BasicMachine::<BabyBear>::NUM_CHIPS;
    let show_public_prover = vec![opts.show_public; num_chips];
    let show_preprocessed_prover = vec![opts.show_preprocessed; num_chips];
    let show_main = vec![opts.show_main; num_chips];
    let show_interactions = vec![opts.show_interactions; num_chips];

    // These just default to false for now, as they are redundant with the above here.
    let show_preprocessed_verifier = vec![false; num_chips];
    let show_public_verifier = vec![false; num_chips];
    let prover_opts = ProverOptions {
        show_main,
        show_public: show_public_prover,
        show_interactions,
        show_public_dims: opts.show_dims,
        show_main_dims: opts.show_dims,
        show_permutation_dims: opts.show_dims,
    };
    let t_run = start_timer!(|| "valida > prove | machine.run(..)");

    let (instance_data, _output) = BasicMachine::run(&mut state, &mut metrics);
    end_timer!(t_run);

    metrics.finalize(&program_file);

    let pk = option_pk
        .map_or_else(
            || {
                Ok(state
                    .machine
                    .pre_process(&config, show_preprocessed_prover, opts.show_dims)
                    .0)
            },
            |pk: Vec<u8>| from_bytes(&pk),
        )
        .map_err(|err| format! {"Incorrect pk: {:?}", err})?;

    let t_prove = start_timer!(|| "valida > prove | machine.prove(..)");
    let proof = state
        .machine
        .prove(&config, &pk, prover_opts, &instance_data);
    end_timer!(t_prove);

    let t_verify = start_timer!(|| "valida > prove | machine.verify(..)");
    let vk = state
        .machine
        .pre_process(&config, show_preprocessed_verifier, false)
        .1;
    debug_assert!(state
        .machine
        .verify(&config, &proof, &vk, &instance_data, show_public_verifier)
        .is_ok());
    end_timer!(t_verify);

    let t_write = start_timer!(|| "valida > prove | proof serialization");
    let mut bytes = vec![];
    ciborium::into_writer(&proof, &mut bytes)
        .map_err(|err| format!("Proof serialization failed: {:?}", err))?;
    end_timer!(t_write);
    Ok(bytes)
}

/// The default prove function for the `MultiSegmentBasicMachine` with parallelization support
pub fn prove(
    program: Program,
    opts: ProveDebugOptions,
    advice_provider: AdviceProviderWithDefault,
    option_pk: Option<Vec<u8>>,
    config: MyConfig,
    initial_fp: u32,
    specific: bool,
    max_trace_height: u32,
    program_file: Vec<u8>,
    max_parallel_segments: usize,
) -> Result<Vec<u8>, String> {
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

    // Set the max_parallel_segments on the machine
    machine.set_max_parallel_segments(max_parallel_segments);

    // Assign the read and write callbacks to the `ValidaRuntime`
    let mut runtime = prepare_runtime(advice_provider, WriteCallbackWithDefault::default())
        .unwrap_or_else(|err| panic!("Failed to prepare runtime: {:?}", err));
    let mut state = machine.start(&mut runtime);
    let mut metrics = BasicMachineMetrics::initialize();

    end_timer!(t_prep);

    let num_chips = BasicMachine::<BabyBear>::NUM_CHIPS;
    let show_public_prover = vec![opts.show_public; num_chips];
    let show_preprocessed_prover = vec![opts.show_preprocessed; num_chips];
    let show_main = vec![opts.show_main; num_chips];
    let show_interactions = vec![opts.show_interactions; num_chips];

    // These just default to false for now, as they are redundant with the above here.
    let show_preprocessed_verifier = vec![false; num_chips];
    let show_public_verifier = vec![false; num_chips];
    let prover_opts = ProverOptions {
        show_main,
        show_public: show_public_prover,
        show_interactions,
        show_public_dims: opts.show_dims,
        show_main_dims: opts.show_dims,
        show_permutation_dims: opts.show_dims,
    };
    let t_run = start_timer!(|| "valida > prove | machine.run(..)");

    let (instance_data, _output) = MultiSegmentBasicMachine::run(&mut state, &mut metrics);
    end_timer!(t_run);

    metrics.finalize(&program_file);

    let pk = option_pk
        .map_or_else(
            || {
                Ok(state
                    .machine
                    .pre_process(&config, show_preprocessed_prover, opts.show_dims)
                    .0)
            },
            |pk: Vec<u8>| from_bytes(&pk),
        )
        .map_err(|err| format! {"Incorrect pk: {:?}", err})?;

    let t_prove = start_timer!(|| "valida > prove | machine.prove(..)");
    let proof = state
        .machine
        .prove(&config, &pk, prover_opts, &instance_data);
    end_timer!(t_prove);

    let t_verify = start_timer!(|| "valida > prove | machine.verify(..)");
    let vk = state
        .machine
        .pre_process(&config, show_preprocessed_verifier, false)
        .1;
    debug_assert!(state
        .machine
        .verify(&config, &proof, &vk, &instance_data, show_public_verifier)
        .is_ok());
    end_timer!(t_verify);

    let t_write = start_timer!(|| "valida > prove | proof serialization");
    let mut bytes = vec![];
    ciborium::into_writer(&proof, &mut bytes)
        .map_err(|err| format!("Proof serialization failed: {:?}", err))?;
    end_timer!(t_write);
    Ok(bytes)
}
