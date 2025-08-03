use std::collections::BTreeMap;
use std::fs::File;
use std::sync::{Arc, Mutex};

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, UnivariatePcs, UnivariatePcsWithLde};
use p3_field::{AbstractField, PrimeField32, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

use p3_matrix::Dimensions;
use valida_bus::{MachineWithProgramBus, MachineWithRangeBus8};
use valida_cpu::{MachineWithCpuChip, MachineWithRegisters, Registers};

use valida_machine::{
    check_interactions, AdviceProviderWithDefault, BusArgument, Chip, ChipProof,
    ChipWithPersistence, Commitments, ConstraintError, InteractionMap, InteractionVec, Machine,
    MachineInstanceData, MachineProof, MachineRuntime, MemoryAccessTimestamp,
    MultiSegmentMachineProof, OpenedValues, PcsError, ProgramROM, ProofSegmentInstanceData,
    ProverOptions, PublicTrace, ReplayAdviceProvider, RunningMachine, SegmentMachine, SegmentProof,
    StarkConfig, StarkField, StoppingFlag, StorageBackendTrait, ValidaMemoryBackend,
    VerificationError, Word, WriteCallbackWithDefault, NUM_CHIPS,
    {MachineProverKey, MachineVerifierKey},
};
use valida_memory::{MachineWithMemoryChip, PersistentMemoryRecord, PersistentMemoryTimestamp};
use valida_program::{MachineWithProgramROM, ProgramTable, ProgramTableType};

use ark_std::{end_timer, start_timer};

use crate::{
    instance_data::{ValidaInstanceData, ValidaSegmentInstanceData},
    BasicMachine, BasicMachineMetrics, ValidaBootData, ValidaRuntime, ValidaSegmentBootData,
};

use crate::machine::basic::{
    calc_zeta, observe_final_state, observe_instance_data, verify_chip_constraints,
    VerificationChallenges,
};

use valida_bytes::{BytesChip, BytesTable, MachineWithBytesChip};
use valida_lookups::{
    LookupType, MachineWithLookupChip, MachineWithMultiLookupChip, MultiLookupTableWrapper,
};

use valida_machine::__internal::{check_constraints, get_log_quotient_degree, quotient};

use valida_program::{MachineWithProgramChip, ProgramChip};

use p3_field::AbstractExtensionField;
use valida_keccak::MachineWithKeccakFChip;
use valida_output::{MachineWithOutputChip, OutputChip, WriteInstruction};
use valida_static_data::{MachineWithStaticDataChip, StaticDataChip, StaticDataChipType};
pub type MultiSegmentBasicRunningMachine<'a, F> =
    RunningMachine<'a, F, MultiSegmentBasicMachine<F>>;
use std::collections::HashSet;

use crate::commands::common::prepare_runtime_from_replay;

use valida_memory_footprint::MemoryFootprint;

#[derive(Default)]
pub struct MultiSegmentBasicMachine<F: StarkField> {
    // The *LAST* executed segment machine
    pub segment_machine: BasicMachine<F>,

    program_table: ProgramTable,

    // The static data for this program. The multi segment machine needs to know about it
    // so that it can pass it into the first segment's boot data.
    static_data: Option<BTreeMap<u32, Word<u8>>>,

    initial_register_values: Registers,

    // store the inverse so the derived default is to log
    no_log: bool,

    // Maximum trace height for each segment
    max_trace_height: u32,

    pub file_to_save_stdout: Option<File>,

    // a binary representation of a loaded ELF
    pub program_file: Vec<u8>,

    // The final memory state is used to deduce when persistent sends/receives between
    // segments are necessary. The very last update to a memory location never needs to
    // be sent to a next segment.
    // It needs to be stored in the MultiSegment machine so that we can assign it to each
    // basic machine during proof generation (as we do multiple execution passes as part of the prover)
    pub final_memory_state: HashSet<(u32, Word<u8>, u32)>,

    /// The advice (i.e. input) that was read during the initial program execution. Stored here at the end
    /// of the `run` function and used in the `prove` function.
    pub replay_advice: ReplayAdviceProvider,
}

impl<F: StarkField> MemoryFootprint for MultiSegmentBasicMachine<F> {
    fn memory_footprint(&self) -> usize {
        let mut result = 0;
        result += self.segment_machine.memory_footprint();
        result += self.program_table.memory_footprint();

        result += self.static_data.memory_footprint();
        result += self.initial_register_values.memory_footprint();
        result += self.no_log.memory_footprint();
        result += self.max_trace_height.memory_footprint();
        result += self.program_file.memory_footprint();
        result += self.final_memory_state.memory_footprint();
        result += self.replay_advice.memory_footprint();

        result
    }
}

// TODO: better define custom copy function that is not standard clone?
impl<F: StarkField> Clone for MultiSegmentBasicMachine<F> {
    /// Be careful using `clone` for the multi segment machine. Only use it if you are fine
    /// dropping the information in the segment machine!
    fn clone(&self) -> Self {
        MultiSegmentBasicMachine {
            segment_machine: BasicMachine::default(), // We don't need the segment machine after the clone!
            program_table: self.program_table.clone(),
            static_data: self.static_data.clone(),
            initial_register_values: self.initial_register_values,
            no_log: self.no_log,
            max_trace_height: self.max_trace_height,
            file_to_save_stdout: None, // Can't be cloned
            program_file: self.program_file.clone(),
            final_memory_state: self.final_memory_state.clone(),
            replay_advice: self.replay_advice.clone(),
        }
    }
}

type Val<SC> = <SC as StarkConfig>::Val;
type ValMat<SC> = RowMajorMatrix<Val<SC>>;
type Com<SC> = <<SC as StarkConfig>::Pcs as Pcs<Val<SC>, ValMat<SC>>>::Commitment;
type ProverData<SC> = <<SC as StarkConfig>::Pcs as Pcs<Val<SC>, ValMat<SC>>>::ProverData;

impl<F: StarkField> MultiSegmentBasicMachine<F> {
    fn memory_backend<'a>(state: &'a RunningMachine<F, Self>) -> &'a ValidaMemoryBackend {
        state.runtime.memory_backend()
    }

    fn memory_backend_mut<'a>(
        state: &'a mut RunningMachine<F, Self>,
    ) -> &'a mut ValidaMemoryBackend {
        state.runtime.memory_backend_mut()
    }

    /// Return the program counter and frame pointer at the current state of execution,
    /// i.e. of the last segment.
    pub fn get_registers<'a>(&self) -> Option<Registers> {
        let cpu = self.segment_machine.cpu();
        let (pc, fp) = (cpu.pc, cpu.fp);
        Some(Registers { pc, fp })
    }

    /// Returns the `BasicMachine` from the last segment, i.e. the "currently executing"
    /// basic machine.
    pub fn get_current_machine<'a>(&self) -> &BasicMachine<F> {
        &self.segment_machine
    }

    pub fn get_current_machine_mut<'a>(&mut self) -> &mut BasicMachine<F> {
        &mut self.segment_machine
    }

    /// Prepares the boot data of the first segment. Then updated with newo segment number.
    pub fn prepare_initial_segment_boot_data(&self) -> ValidaSegmentBootData {
        let initial_register_values = self.initial_register_values();

        let static_data_chip_type = match self.program_table_type() {
            ProgramTableType::Public => Some(StaticDataChipType::Public),
            ProgramTableType::Preprocessed => Some(StaticDataChipType::Preprocessed),
        };

        // If there's static data, and we're in the 0th segment, let's load the static data.
        ValidaSegmentBootData {
            initial_register_values,
            program_rom: self.program_rom().clone(),
            program_table_type: self.program_table_type(),
            segment_number: 0,
            max_trace_height: self.max_trace_height(),
            program_file: self.program_file.clone(),
            static_data: self.static_data.clone(),
            // NOTE: `static_data_chip_type` is currently *NOT USED*. Instead we base the choice of
            // public or preprocessed based on the corresponding type for the program rom
            static_data_chip_type,
            log_enabled: self.log_enabled(),
        }
    }

    fn run_segment_machine(
        state: &mut RunningMachine<F, Self>,
        metrics: &mut BasicMachineMetrics,
        boot_data: ValidaSegmentBootData,
    ) -> (
        bool,
        ValidaSegmentInstanceData,
        ValidaSegmentBootData,
        Vec<u8>,
    ) {
        // create a new segment machine
        let mut segment_machine = BasicMachine::default();
        eprintln!("created new segment machine");
        eprintln!(
            "initial program counter: {:?}",
            boot_data.initial_register_values.pc
        );
        eprintln!("segment number: {:?}", boot_data.segment_number);

        // load the initial data into the machine
        segment_machine.init(boot_data.clone());

        // start the machine, attaching the runtime (which is mutated by the previous segment's execution)
        // The runtime also contains the memory backend, which is the global memory for the entire execution.
        let mut running_segment_machine = segment_machine.start(state.runtime);

        // run the machine
        let (mut segment_instance, segment_output) =
            BasicMachine::run(&mut running_segment_machine, metrics);
        // reset the program rom to not blow up in memory if we have many segments or a large program
        segment_instance.rom = None;
        eprintln!(
            "finished running segment machine, program counter: {:?}",
            segment_instance.pc_final
        );

        // Update the instance data wrt segment number & `is_last_segment` as the `BasicMachine` run
        // function has no way of knowing about these.
        segment_instance.segment_number = boot_data.segment_number;
        if segment_instance.did_stop {
            // update the `is_last_segment`
            running_segment_machine.machine.cpu_mut().is_last_segment = 1;
            segment_instance.is_last_segment = true;
        } else {
            running_segment_machine.machine.cpu_mut().is_last_segment = 0;
            segment_instance.is_last_segment = false;
        }
        let (new_boot_data, mut segment_machine) = BasicMachine::suspend(running_segment_machine);
        eprintln!("suspended segment machine");
        eprintln!(
            "next boot data segment number: {:?}",
            new_boot_data.segment_number
        );
        eprintln!(
            "next boot data start pc: {:?}",
            new_boot_data.initial_register_values.pc
        );

        let boot_data = new_boot_data;
        // assign the final memory state (if any) of the multi segment machine (may still be empty if we are
        // in `run`. In `prove` it will be filled though
        segment_machine.final_memory_state = state.machine.final_memory_state.clone();

        // Assign the segment machine as the last executed one
        state.machine.segment_machine = segment_machine;

        let did_halt = segment_instance.did_stop || segment_instance.did_fail;
        (did_halt, segment_instance, boot_data, segment_output)
    }
}

/// Get the LDEs from the PCS using the prover data for all chips that have a trace
fn gen_lde<'a, SC: StarkConfig>(
    pcs: &'a SC::Pcs,
    pd: &'a ProverData<SC>,
    has_traces: [bool; NUM_CHIPS],
) -> Vec<
    Option<
        <SC::Pcs as UnivariatePcsWithLde<
            SC::Val,
            SC::Challenge,
            RowMajorMatrix<SC::Val>,
            SC::Challenger,
        >>::Lde<'a>,
    >,
> {
    // pull out exactly as many LDEs as we committed
    let mut lde_iter = pcs.get_ldes(pd).into_iter();
    let mut ldes = Vec::with_capacity(NUM_CHIPS);
    for chip_idx in 0..NUM_CHIPS {
        if has_traces[chip_idx] {
            ldes.push(Some(lde_iter.next().unwrap()));
        } else {
            ldes.push(None);
        }
    }
    ldes
}

fn generate_traces<I, T, F>(gen_method: F) -> [Option<T>; NUM_CHIPS]
where
    F: FnOnce() -> I,
    I: IntoIterator<Item = Option<T>>,
{
    let mut result: [Option<T>; NUM_CHIPS] = [const { None }; NUM_CHIPS];
    // Generate traces
    let traces = gen_method();
    for (chip_idx, trace) in traces.into_iter().enumerate() {
        if let Some(t) = trace {
            result[chip_idx] = Some(t);
        }
    }
    result
}

/// Performs the commitment to the public traces and and observes the commitment
/// in the Fiat-Shamir transcript.
fn commit_trace<F: StarkField, SC: StarkConfig<Val = F>, T, Fn>(
    traces: [Option<T>; NUM_CHIPS],
    pcs: &SC::Pcs,
    challenger: &mut SC::Challenger,
    process: Fn,
    observe: bool,
) -> (Com<SC>, ProverData<SC>)
where
    Fn: FnMut(&T) -> RowMajorMatrix<F>,
    //    I: IntoIterator<Item = RowMajorMatrix<F>>,
    T: Clone,
    //    Vec<RowMajorMatrix<F>>: FromIterator<I>,
{
    let processed = traces.iter().flatten().map(process).collect();
    let (commit, data) = pcs.commit_batches(processed);
    if observe {
        // observe can be disabled for second pass over data
        challenger.observe(commit.clone());
    }

    (commit, data)
}

pub fn has_traces<F: TwoAdicField, I, T>(traces: I) -> [bool; NUM_CHIPS]
where
    I: IntoIterator<Item = Option<T>>,
    // If we need `I` to be a reference, use:
    //I: ?Sized,
    //for<'a> &'a I: IntoIterator<Item = &'a Option<T>>,
{
    traces
        .into_iter()
        .map(|opt| opt.is_some())
        .collect::<Vec<bool>>()
        .try_into()
        .unwrap()
}

fn check_ephemeral_sums<SC: StarkConfig>(
    ephemeral_sums: &Vec<Option<SC::Challenge>>,
    segment_idx: u32,
) {
    let sum = ephemeral_sums
        .iter()
        .copied()
        .flatten()
        .sum::<SC::Challenge>();
    assert_eq!(
        sum,
        SC::Challenge::zero(),
        "Sum of ephemeral cumulative sums is not zero in segment {}: {}",
        segment_idx,
        sum
    );
}

fn check_persistent_sums<F: StarkField, SC: StarkConfig>(
    persistent_sums: &Vec<Vec<Option<SC::Challenge>>>,
    interaction_map_guard: Arc<
        Mutex<BTreeMap<BusArgument, (Vec<InteractionVec<F>>, Vec<InteractionVec<F>>)>>,
    >,
) {
    // perform check of interactions. Includes all global/local inter-segment interactions as well
    // as persistent interactions across segments
    check_interactions(&mut interaction_map_guard.lock().unwrap(), false);

    let sum = persistent_sums
        .iter()
        .flat_map(|segment_sums| segment_sums.iter())
        .flatten()
        .copied()
        .sum::<SC::Challenge>();
    assert_eq!(
        sum,
        SC::Challenge::zero(),
        "Sum of persistent cumulative sums is not zero: {}",
        sum
    );
}

fn get_openings<SC: StarkConfig>(
    openings_real: Vec<Vec<Vec<SC::Challenge>>>,
    has_traces: [bool; NUM_CHIPS],
) -> Vec<Vec<Vec<SC::Challenge>>> {
    let mut ops = Vec::with_capacity(NUM_CHIPS);
    let mut iter = openings_real.into_iter(); // turn into iterator so we can use `next`
    for has in has_traces {
        if has {
            // there must be as many elements as `has_traces` has entries, so `unwrap`
            // can never fail
            ops.push(iter.next().unwrap());
        } else {
            ops.push(vec![vec![], vec![]]);
        }
    }
    ops
}

impl<F: StarkField> Machine<F> for MultiSegmentBasicMachine<F> {
    const NUM_CHIPS: usize = NUM_CHIPS;
    type InstanceData = ValidaInstanceData;
    type BootData = ValidaBootData;
    type Runtime = ValidaRuntime;
    type Proof<SC: StarkConfig<Val = F>> = MultiSegmentMachineProof<SC>;
    type Metrics = BasicMachineMetrics;

    fn max_trace_height(&self) -> u32 {
        self.max_trace_height
    }

    fn set_max_trace_height(&mut self, max_trace_height: u32) {
        self.max_trace_height = max_trace_height;
    }

    fn enable_logging(&mut self, log_enable: bool) -> Option<()> {
        self.no_log = !log_enable;
        Some(())
    }

    fn log_enabled(&self) -> bool {
        !self.no_log
    }

    fn run(
        state: &mut RunningMachine<F, Self>,
        metrics: &mut Self::Metrics,
    ) -> (ValidaInstanceData, Vec<u8>) {
        let mut did_halt = false;
        let mut segment_instances: Vec<ValidaSegmentInstanceData> = vec![];
        let mut all_output: Vec<u8> = Vec::new();

        // get initial boot data (will be updated w/ each segment number in loop below)
        let mut segment_boot_data = state.machine.prepare_initial_segment_boot_data();

        // Run all segments
        while !did_halt {
            let (new_did_halt, segment_instance, boot_data, output) =
                MultiSegmentBasicMachine::run_segment_machine(state, metrics, segment_boot_data);
            did_halt = new_did_halt;
            segment_boot_data = boot_data;
            segment_instances.push(segment_instance);
            all_output.extend(output);
        }

        // assign the replay advice
        state.machine.replay_advice = state.runtime.get_replay_advice();

        let pc_init = segment_instances[0].pc_init;
        let fp_init = segment_instances[0].fp_init;

        {
            let last_segment_instance = segment_instances.last_mut().unwrap();
            last_segment_instance.is_last_segment = true;
        }

        let final_memory_state: HashSet<(u32, Word<u8>, u32)> = state
            .runtime
            .memory_backend()
            .into_iter()
            .map(|(addr, record)| {
                (
                    addr,
                    record.value,
                    match record.last_accessed {
                        // NOTE: The values assigned match the logic of `MemoryAccessTimestamp::as_scalar`
                        // It's important we have distinct values for static, for `persistent_sends` of the static
                        // data chip.
                        MemoryAccessTimestamp::ThisSegment => {
                            unreachable!("all ThisSegment memory addresses will have been updated when suspending the last segment")
                        }
                        MemoryAccessTimestamp::PriorSegment(segment) => 3 + segment,
                        MemoryAccessTimestamp::ZeroInitialized => 0,
                        MemoryAccessTimestamp::Static => 2,
                    },
                )
            })
            .collect();

        // TODO: write to all segment machines *OR* write only entries that finish in that segment to a
        // machine?
        // Store final memory state in multi segment machine, so that it can be assigned to each
        // segment machine in the prover (segment machines are dropped!)
        state.machine.final_memory_state = final_memory_state.clone();
        state.machine.segment_machine.final_memory_state = final_memory_state.clone();
        let instance_data = ValidaInstanceData {
            rom: match state.machine.program_table_type() {
                ProgramTableType::Public => Some(state.machine.program_rom().clone()),
                _ => None,
            },
            pc_init,
            fp_init,
            output: all_output.clone(),
            did_fail: segment_instances.last().unwrap().did_fail,
            segments: segment_instances,
        };

        (instance_data, all_output)
    }

    fn init(&mut self, boot_data: Self::BootData) {
        self.program_file = boot_data.program_file;
        self.set_program_rom(
            boot_data.initial_register_values.pc,
            boot_data.program_rom,
            boot_data.program_table_type,
        );
        self.set_initial_register_values(boot_data.initial_register_values);
        self.set_max_trace_height(boot_data.max_trace_height);
        // store static data in multi segment machine until we run it
        self.static_data = if boot_data.static_data.len() > 0 {
            Some(boot_data.static_data)
        } else {
            None
        };
    }

    fn start(mut self, runtime: &mut Self::Runtime) -> RunningMachine<F, Self> {
        RunningMachine {
            machine: Box::new(self),
            runtime,
        }
    }

    fn stop(running_machine: RunningMachine<F, Self>) -> Self {
        *running_machine.machine
    }

    fn step(state: &mut RunningMachine<F, Self>) -> StoppingFlag {
        // temporarily take the current segment machine (we put it back below)
        let current_segment_machine = std::mem::take(&mut state.machine.segment_machine);
        let mut running_segment_machine = RunningMachine {
            machine: Box::new(current_segment_machine),
            runtime: state.runtime,
        };
        let stopping_flag = Machine::step(&mut running_segment_machine);
        state.machine.segment_machine = Machine::stop(running_segment_machine);
        stopping_flag
    }

    fn pre_process<SC>(
        &self,
        config: &SC,
        show_preprocessed: Vec<bool>,
        show_dimensions: bool,
    ) -> (MachineProverKey<SC, Self>, MachineVerifierKey<SC, Self>)
    where
        SC: StarkConfig<Val = F>,
        ProverData<SC>: Clone,
    {
        let mut segment_machine = BasicMachine::default();
        segment_machine.set_max_trace_height(self.max_trace_height);
        segment_machine.set_segment_number(0);
        let (segment_pk, segment_vk) =
            segment_machine.pre_process(config, show_preprocessed, show_dimensions);
        (
            MachineProverKey::<SC, Self>::new(
                segment_pk.preprocessed_traces().clone(),
                segment_pk.preprocessed_commit(),
                segment_pk.preprocessed_prover_data().clone(),
            ),
            MachineVerifierKey::<SC, Self>::new(
                segment_vk.preprocessed_commit(),
                segment_vk.preprocessed_dims(),
            ),
        )
    }

    /// The core proving method for a multi-segment machine.
    ///
    /// In this method we:
    /// - Generate preprocessed traces (per chip and segment), then observe a commitment to them.
    /// - Generate public traces (per chip and segment).
    /// - Generate main traces (per chip and segment).
    /// - Generate multi-segment chip traces.
    /// - Observe a commitment to the main and multi-segment chip traces.
    /// - Sample elements for the permutation challenges.
    /// - Generate permutation traces (per chip and segment).
    /// - Calculate cumulative sums and cumulative products for the permutation traces.
    /// - Observe a commitment to the permutation traces.
    /// - Sample another challenge element `alpha`.
    /// - Generate the quotient polynomials (per chip and segment).
    /// - Observe a commitment to the quotient polynomials.
    /// - Get openings to the preprocessed, main, permutation and quotient polynomials.
    /// - Bundle everything together in a MultiSegmentMachineProof.
    ///
    /// NOTE: All operations relating to the Fiat-Shamir transcript (observations and sampling) are
    /// marked with a comment
    /// // <Number> FIAT-SHAMIR
    /// after the relevant line.
    /// The prefix number can be used to easily find the corresponding operation in the verifier.
    fn prove<SC>(
        &self,
        config: &SC,
        pk: &MachineProverKey<SC, Self>,
        opts: ProverOptions,
        instance_data: &ValidaInstanceData,
    ) -> Self::Proof<SC>
    where
        SC: StarkConfig<Val = F>,
    {
        let ProverOptions {
            show_main,
            show_public,
            show_public_dims,
            show_main_dims,
            show_permutation_dims,
            show_interactions,
        } = opts;

        let sm_len = instance_data.segments.len();
        let mut boot_data = self.prepare_initial_segment_boot_data();

        // runtime so that we can execute individual segments. Using the advice provider of this machine
        let mut runtime = prepare_runtime_from_replay(
            self.replay_advice.clone(),
            WriteCallbackWithDefault::default(),
        )
        .unwrap_or_else(|err| panic!("Failed to prepare runtime: {:?}", err));

        let mut machine = self.clone();
        let mut state = machine.start(&mut runtime);
        let mut metrics = BasicMachineMetrics::initialize();

        let pcs = config.pcs();
        let mut challenger = config.challenger();

        #[cfg(debug_assertions)]
        let interaction_map: BTreeMap<_, (Vec<InteractionVec<F>>, Vec<InteractionVec<F>>)> =
            InteractionMap::new();
        #[cfg(debug_assertions)]
        let interaction_map_guard = Arc::new(Mutex::new(interaction_map));

        // Get the preprocessed traces, commitments and prover data from the ProverKey. These are shared by
        // all segments, hence we do it before the segment loop
        let (preprocessed_traces, preprocessed_commitments, preprocessed_prover_data): (
            &[Option<RowMajorMatrix<F>>; NUM_CHIPS],
            Com<SC>,
            &ProverData<SC>,
        ) = {
            let traces = pk.preprocessed_traces();
            let arr_ref = traces.as_slice().try_into().unwrap_or_else(|_| {
                panic!(
                    "Preprocessed traces length mismatch: expected {}, got {}",
                    NUM_CHIPS,
                    traces.len()
                )
            });
            (
                arr_ref,
                pk.preprocessed_commit(),
                pk.preprocessed_prover_data(),
            )
        };

        let has_preprocessed_traces = has_traces::<F, _, _>(preprocessed_traces.to_vec());

        // 2. Observe the preprocessed commitment
        challenger.observe(preprocessed_commitments); // 1 FIAT-SHAMIR

        // Vector to store the persistent sums of all segments
        let mut cumulative_persistent_sums: Vec<Vec<Option<SC::Challenge>>> =
            vec![vec![None; NUM_CHIPS]; sm_len];

        let mut segment_proofs = Vec::with_capacity(sm_len);

        for seg_idx in 0..sm_len {
            // 1. Absorb initial pc and fp into the transcript
            let seg_inst = &instance_data.segments[seg_idx];
            observe_instance_data::<F, SC>(&mut challenger, seg_inst); // 2 FIAT-SHAMIR

            // execute this segment again to have information for traces
            (_, _, boot_data, _) =
                MultiSegmentBasicMachine::run_segment_machine(&mut state, &mut metrics, boot_data);
            let segment_machine = &state.machine.segment_machine;

            // 3. Public data
            // Get the public traces
            let public_traces = generate_traces(|| {
                segment_machine.generate_public_traces(
                    config,
                    show_public.clone(),
                    show_public_dims,
                )
            });
            // 4. Main trace
            // Get the main traces
            let main_traces = generate_traces(|| {
                segment_machine.generate_main_traces(config, show_main.clone(), show_main_dims)
            });

            // Commit to traces AND observe commitments
            let (public_commitments, public_data) = commit_trace::<F, SC, PublicTrace<F>, _>(
                public_traces.clone(),
                pcs,
                &mut challenger,
                PublicTrace::into_matrix,
                true, // 3 FIAT-SHAMIR
            );
            // Commit AND observe the commitment
            let (main_commitments, main_data): (Com<SC>, ProverData<SC>) =
                commit_trace::<F, SC, RowMajorMatrix<F>, _>(
                    main_traces.clone(),
                    pcs,
                    &mut challenger,
                    |t: &RowMajorMatrix<F>| t.flatten_to_base(),
                    true, // 4 FIAT-SHAMIR
                );
        }

        // sample global challengers for *persistent interactions*. The same for all segments
        let global_perm_challenges = (0..2) // 5 FIAT-SHAMIR
            .map(|_| challenger.sample_ext_element())
            .collect::<Vec<<SC as StarkConfig>::Challenge>>();

        // Reset entire boot data & runtime. The runtime needs to be reset to reset the memory backend.
        // Otherwise memory cells look like they have been used (because they were) & thus it breaks
        // the persistent send/receive logic.
        let mut boot_data = self.prepare_initial_segment_boot_data();
        let mut runtime = prepare_runtime_from_replay(
            self.replay_advice.clone(),
            WriteCallbackWithDefault::default(),
        )
        .unwrap_or_else(|err| panic!("Failed to prepare runtime: {:?}", err));

        let mut machine = self.clone();
        let mut state = machine.start(&mut runtime);
        let mut metrics = BasicMachineMetrics::initialize();

        for seg_idx in 0..sm_len {
            // 1. Get instance data but do *NOT* observe initial state (already done in previous loop!)
            let seg_inst = &instance_data.segments[seg_idx];

            // execute this segment again to have information for traces
            (_, _, boot_data, _) =
                MultiSegmentBasicMachine::run_segment_machine(&mut state, &mut metrics, boot_data);
            let segment_machine = &state.machine.segment_machine;

            // 3. Public data
            // Get the public traces
            let public_traces = generate_traces(|| {
                segment_machine.generate_public_traces(
                    config,
                    show_public.clone(),
                    show_public_dims,
                )
            });
            // 4. Main trace
            // Get the main traces
            let main_traces = generate_traces(|| {
                segment_machine.generate_main_traces(config, show_main.clone(), show_main_dims)
            });

            // Get the public & main data again

            // Commit to traces but do *NOT* observe commitments (already done in previous loop)
            let (_, public_data) = commit_trace::<F, SC, PublicTrace<F>, _>(
                public_traces.clone(),
                pcs,
                &mut challenger,
                PublicTrace::into_matrix,
                false,
            );
            // Commit to traces but do *NOT* observe commitments (already done in previous loop)
            let (main_commitments, main_data): (Com<SC>, ProverData<SC>) =
                commit_trace::<F, SC, RowMajorMatrix<F>, _>(
                    main_traces.clone(),
                    pcs,
                    &mut challenger,
                    |t: &RowMajorMatrix<F>| t.flatten_to_base(),
                    false,
                );

            // Get mask of which chips have a public trace
            let has_public_traces = has_traces::<F, _, _>(public_traces.clone());
            // Get mask of which chips have a public trace
            let has_main_traces = has_traces::<F, _, _>(main_traces.clone());

            // Get the LDEs for the public data
            let mut public_ldes = gen_lde::<SC>(pcs, &public_data, has_public_traces);

            // Get the LDEs for the public data
            let mut main_ldes = gen_lde::<SC>(pcs, &main_data, has_main_traces);

            // generate here as it will be consumed in `generate_quotient_polynomials`
            let mut preprocessed_ldes =
                gen_lde::<SC>(&pcs, &preprocessed_prover_data, has_preprocessed_traces);

            // Get the degrees. Needed for
            let (degrees, log_degrees, g_subgroups) = segment_machine.degrees_and_g_subgroups(
                config,
                &main_traces,
                preprocessed_traces,
                &public_traces,
            );

            // Sample permutation challenges for this segment
            let perm_challenges = (0..2) // 6 FIAT-SHAMIR
                .map(|_| challenger.sample_ext_element())
                .collect::<Vec<<SC as StarkConfig>::Challenge>>();

            // 4. Permutation traces
            // Get the permutation traces for this segment
            let permutation_traces = generate_traces(|| {
                segment_machine.generate_perm_traces(
                    config,
                    preprocessed_traces as &[Option<RowMajorMatrix<F>>],
                    &public_traces as &[Option<PublicTrace<F>>],
                    &main_traces as &[Option<RowMajorMatrix<F>>],
                    &degrees as &[usize],
                    perm_challenges.clone(),
                    global_perm_challenges.clone(),
                    #[cfg(debug_assertions)]
                    &interaction_map_guard,
                    show_permutation_dims,
                )
            });

            // Commit AND observe the commitment
            let (perm_commitments, perm_data): (Com<SC>, ProverData<SC>) =
                commit_trace::<F, SC, RowMajorMatrix<SC::Challenge>, _>(
                    permutation_traces.clone(),
                    pcs,
                    &mut challenger,
                    |t: &RowMajorMatrix<SC::Challenge>| t.flatten_to_base(),
                    true, // 7 FIAT-SHAMIR
                );

            // Sample alpha.
            let alpha: SC::Challenge = challenger.sample_ext_element(); // 8 FIAT-SHAMIR

            // Get mask of which chips have a public trace
            let has_permutation_traces = has_traces::<F, _, _>(permutation_traces.clone());
            // Get the LDEs for the permutation data
            let mut perm_ldes = gen_lde::<SC>(pcs, &perm_data, has_permutation_traces);

            // Get ephemeral & persistent cumulative sums
            let (ephemeral_sums, persistent_sums) =
                segment_machine.cumulative_sums(config, &permutation_traces);

            #[cfg(debug_assertions)]
            {
                check_interactions(&mut interaction_map_guard.lock().unwrap(), true);
                // prune out all interactions that are local to this segment
                interaction_map_guard
                    .lock()
                    .unwrap()
                    .retain(|key, _| matches!(key, BusArgument::Persistent(_)));
                check_ephemeral_sums::<SC>(&ephemeral_sums, seg_idx as u32);
            }

            cumulative_persistent_sums[seg_idx] = persistent_sums;

            // quotient polynomials
            let (quotient_polys, log_quot_degs, coset_shifts_arr) = segment_machine
                .generate_quotient_polynomials(
                    config,
                    preprocessed_traces,
                    &main_traces,
                    &permutation_traces,
                    &degrees,
                    &log_degrees,
                    alpha,
                    perm_challenges.clone(),
                    global_perm_challenges.clone(),
                    &public_traces,
                    &mut preprocessed_ldes,
                    &mut main_ldes,
                    &mut perm_ldes,
                    &mut public_ldes,
                    &ephemeral_sums,
                    &cumulative_persistent_sums[seg_idx][..],
                    show_interactions.clone(),
                );

            let (quot_commitments, quot_data) =
                pcs.commit_shifted_batches(quotient_polys, &coset_shifts_arr);
            challenger.observe(quot_commitments.clone()); // 9 FIAT-SHAMIR

            // Compute all the zeta challenge values
            let zeta = challenger.sample_ext_element(); // 10 FIAT-SHAMIR
            let zeta_and_next_preprocessed =
                calc_zeta::<F, SC>(g_subgroups, has_preprocessed_traces, zeta);
            let zeta_and_next_main = calc_zeta::<F, SC>(g_subgroups, has_main_traces, zeta);
            let zeta_and_next_perm: Vec<Vec<SC::Challenge>> =
                calc_zeta::<F, SC>(g_subgroups, has_permutation_traces, zeta);
            let zeta_exp_quotient_degree: [Vec<SC::Challenge>; NUM_CHIPS] =
                log_quot_degs.map(|log_deg| vec![zeta.exp_power_of_2(log_deg)]);

            let commitments = Commitments {
                main_trace: main_commitments,
                perm_trace: perm_commitments.clone(),
                quotient_chunks: quot_commitments.clone(),
            };

            let segment_rounds: [(&ProverData<SC>, &[Vec<SC::Challenge>]); 4] = [
                (&preprocessed_prover_data, &zeta_and_next_preprocessed),
                (&main_data, &zeta_and_next_main),
                (&perm_data, &zeta_and_next_perm),
                (&quot_data, &zeta_exp_quotient_degree),
            ];

            let (openings, opening_proof) =
                pcs.open_multi_batches(&segment_rounds, &mut challenger);

            let [mut preprocessed_op_real, mut main_op_real, mut perm_op_real, quot_op] = openings
                .try_into()
                .expect("Should have 4 rounds of openings");

            let preprocessed_op = get_openings::<SC>(preprocessed_op_real, has_preprocessed_traces);
            let main_op = get_openings::<SC>(main_op_real, has_main_traces);
            let perm_op = get_openings::<SC>(perm_op_real, has_permutation_traces);

            let mut chip_proofs = Vec::new();
            for chip_idx in 0..NUM_CHIPS {
                let [preprocessed_local, preprocessed_next] = preprocessed_op[chip_idx]
                    .clone()
                    .try_into()
                    .expect("Should have 2 openings");
                let [trace_local, trace_next] = main_op[chip_idx]
                    .clone()
                    .try_into()
                    .expect("Should have 2 openings");
                let [permutation_local, permutation_next] = perm_op[chip_idx]
                    .clone()
                    .try_into()
                    .expect("Should have 2 openings");
                let [quotient_chunks] = quot_op[chip_idx]
                    .clone()
                    .try_into()
                    .expect("Should have 1 opening");

                chip_proofs.push(ChipProof {
                    log_degree: log_degrees[chip_idx],
                    opened_values: OpenedValues {
                        preprocessed_local,
                        preprocessed_next,
                        trace_local,
                        trace_next,
                        permutation_local,
                        permutation_next,
                        quotient_chunks,
                    },
                    cumulative_ephemeral_sum: ephemeral_sums[chip_idx],
                    cumulative_persistent_sum: cumulative_persistent_sums[seg_idx][chip_idx],
                });
            }

            // Observe final state for the segment
            observe_final_state::<F, SC>(&mut challenger, seg_inst); // 11 FIAT-SHAMIR

            // Construct the proof for this segment consisting of its machine proof &
            // instance data required for verification
            let mp = MachineProof {
                commitments,
                opening_proof,
                chip_proofs,
            };
            let id = ProofSegmentInstanceData {
                pc_init: seg_inst.pc_init,
                fp_init: seg_inst.fp_init,
                pc_final: seg_inst.pc_final,
                fp_final: seg_inst.fp_final,
                output: seg_inst.output.clone(),
            };
            let seg_proof = SegmentProof {
                proof: mp,
                instance_data: id,
            };
            segment_proofs.push(seg_proof);
        }

        #[cfg(debug_assertions)]
        check_persistent_sums::<F, SC>(&cumulative_persistent_sums, interaction_map_guard);

        // TODO(jen): Pack up segment proofs

        MultiSegmentMachineProof {
            segment_proofs,
            chip_proofs: vec![],
        }
    }

    fn compute_log_quotient_degrees<SC: StarkConfig<Val = F>>(&self) -> [usize; NUM_CHIPS] {
        panic!("Call `compute_log_quotient_degrees` for a (segment) `BasicMachine` instead.");
        [0; NUM_CHIPS]
    }

    /// NOTE: All operations relating to the Fiat-Shamir transcript (observations and sampling) are
    /// marked with a comment
    /// // <Number> FIAT-SHAMIR
    /// after the relevant line.
    /// The prefix number can be used to easily find the corresponding operation in the verifier.
    fn verify<SC>(
        &self,
        config: &SC,
        proof: &Self::Proof<SC>,            // MultiSegmentMachineProof
        vk: &MachineVerifierKey<SC, Self>,  // Verifier key for MultiSegmentBasicMachine
        instance_data: &Self::InstanceData, // ValidaInstanceData
        show_public: Vec<bool>,
    ) -> Result<(), VerificationError<SC>>
    where
        SC: StarkConfig<Val = F>,
    {
        let pcs = config.pcs();
        let mut challenger = config.challenger();

        // Preprocessed commit and dims are global for all segments, from the VerifierKey
        let (preprocessed_commit, preprocessed_dims_vec_opt) =
            (vk.preprocessed_commit(), vk.preprocessed_dims());

        let preprocessed_dims_flat: Vec<Dimensions> = preprocessed_dims_vec_opt
            .iter()
            .filter_map(|d| *d)
            .collect();

        let has_preprocessed: [bool; NUM_CHIPS] = preprocessed_dims_vec_opt
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let sm_len = proof.segment_proofs.len();
        if instance_data.segments.len() != sm_len {
            return Err(VerificationError::<SC>::InstanceDataSegmentProofNumberMismatch);
        }

        // 1. Observe the preprocessed commitment
        challenger.observe(preprocessed_commit.clone()); // 1 FIAT-SHAMIR

        // 2. Observe initial state, public commitment & main commitment for ALL segments
        for seg_idx in 0..sm_len {
            let mut seg_inst = instance_data.segments[seg_idx].clone();
            // Need to put the program ROM back in. Set to `None` in multi segment `run` to save memory
            seg_inst.rom = instance_data.rom.clone();
            observe_instance_data::<F, SC>(&mut challenger, &seg_inst); // 2 FIAT-SHAMIR

            // Public traces for the current segment
            let public_trace_vec = seg_inst.public_traces(show_public.clone());
            if public_trace_vec.is_empty() {
                return Err(VerificationError::<SC>::Other);
            }
            let public_traces: [Option<PublicTrace<SC::Val>>; NUM_CHIPS] =
                public_trace_vec[0].clone().try_into().unwrap();

            // Commit to public trace & observe commitment. If the verification succeeds, we committed to the
            // same data as the prover
            let (public_commitments, _) = commit_trace::<F, SC, PublicTrace<F>, _>(
                public_traces.clone(),
                pcs,
                &mut challenger,
                PublicTrace::into_matrix,
                true, // 3 FIAT-SHAMIR
            );

            // observe main commitment
            let main_trace = proof.segment_proofs[seg_idx]
                .proof
                .commitments
                .main_trace
                .clone();
            challenger.observe(main_trace.clone()); // 4 FIAT-SHAMIR
        }

        // 3. Sample global permutation challenges
        let global_perm_challenges = (0..2) // 5 FIAT-SHAMIR
            .map(|_| challenger.sample_ext_element())
            .collect::<Vec<SC::Challenge>>();

        // 5. Observe permutation trace commitments for ALL segments
        for segment_idx in 0..proof.segment_proofs.len() {
            let segment_proof = &proof.segment_proofs[segment_idx];
            let mut segment_instance = instance_data.segments[segment_idx].clone();
            // Need to put the program ROM back in. Set to `None` in multi segment `run` to save memory
            segment_instance.rom = instance_data.rom.clone();

            // NOTE: The segment machine here is (and *must*) *ONLY* used for auxiliary functions, which do not depend
            // on its state
            let segment_machine = &self.segment_machine;

            // sample permutation challenge
            let perm_chall = (0..2) // 6 FIAT-SHAMIR
                .map(|_| challenger.sample_ext_element())
                .collect::<Vec<SC::Challenge>>();

            // observe permutation commitment
            challenger.observe(segment_proof.proof.commitments.perm_trace.clone()); // 7 FIAT-SHAMIR

            // 4. Sample alpha
            let alpha: SC::Challenge = challenger.sample_ext_element(); // 8 FIAT-SHAMIR

            // observe quotient commitment
            challenger.observe(segment_proof.proof.commitments.quotient_chunks.clone()); // 9 FIAT-SHAMIR

            // Now do PCS verification for each segment
            let log_quotient_degrees: [usize; NUM_CHIPS] =
                segment_machine.compute_log_quotient_degrees::<SC>();

            // Prepare for PCS verification for this segment
            let chips_for_segment = segment_machine.get_chips::<SC>();

            let has_main_traces_segment: [bool; NUM_CHIPS] =
                core::array::from_fn(|chip_idx| chips_for_segment[chip_idx].main_width() != 0);

            let g_subgroups_segment: [SC::Val; NUM_CHIPS] = segment_proof
                .proof
                .chip_proofs
                .iter()
                .map(|cp| SC::Val::two_adic_generator(cp.log_degree))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            // Sample zeta for this segment
            let zeta: SC::Challenge = challenger.sample_ext_element(); // 10 FIAT-SHAMIR

            // Points for PCS verification
            let zeta_and_next_preprocessed =
                calc_zeta::<F, SC>(g_subgroups_segment, has_preprocessed, zeta);
            let zeta_and_next_main =
                calc_zeta::<F, SC>(g_subgroups_segment, has_main_traces_segment, zeta);
            let zeta_and_next_perm: Vec<Vec<SC::Challenge>> =
                calc_zeta::<F, SC>(g_subgroups_segment, [true; NUM_CHIPS], zeta);
            let zeta_exp_quotient_degree: [Vec<SC::Challenge>; NUM_CHIPS] =
                log_quotient_degrees.map(|log_deg| vec![zeta.exp_power_of_2(log_deg)]);

            // Prepare opened values for PCS verification
            let mut opened_preprocessed_values_segment = vec![];
            let mut opened_main_values_segment = vec![];
            let mut opened_perm_values_segment = vec![];
            let mut opened_quotient_values_segment = vec![];

            for (chip_idx, chip_proof) in segment_proof.proof.chip_proofs.iter().enumerate() {
                let ov = &chip_proof.opened_values;
                if has_preprocessed[chip_idx] {
                    opened_preprocessed_values_segment.push(vec![
                        ov.preprocessed_local.clone(),
                        ov.preprocessed_next.clone(),
                    ]);
                }
                if has_main_traces_segment[chip_idx] {
                    opened_main_values_segment
                        .push(vec![ov.trace_local.clone(), ov.trace_next.clone()]);
                }
                opened_perm_values_segment.push(vec![
                    ov.permutation_local.clone(),
                    ov.permutation_next.clone(),
                ]);
                opened_quotient_values_segment.push(vec![ov.quotient_chunks.clone()]);
            }

            let segment_chips_opening_values = vec![
                opened_preprocessed_values_segment,
                opened_main_values_segment,
                opened_perm_values_segment,
                opened_quotient_values_segment,
            ];

            // Dimensions for PCS verification
            let dims = segment_machine.get_dims(
                preprocessed_dims_vec_opt.clone(),
                chips_for_segment,
                &segment_proof.proof,
                log_quotient_degrees,
            );

            // PCS Verify Call for the segment
            pcs.verify_multi_batches(
                &[
                    (
                        preprocessed_commit.clone(),
                        zeta_and_next_preprocessed.as_slice(),
                    ),
                    (
                        segment_proof.proof.commitments.main_trace.clone(),
                        zeta_and_next_main.as_slice(),
                    ),
                    (
                        segment_proof.proof.commitments.perm_trace.clone(),
                        zeta_and_next_perm.as_slice(),
                    ),
                    (
                        segment_proof.proof.commitments.quotient_chunks.clone(),
                        zeta_exp_quotient_degree.as_slice(),
                    ),
                ],
                &dims,
                segment_chips_opening_values,
                &segment_proof.proof.opening_proof,
                &mut challenger,
            )
            .map_err(PcsError)?;

            // fill the challenges required for chip verification
            let challenges = VerificationChallenges::<SC> {
                perm_challenges: perm_chall.clone(),
                global_perm_challenges: global_perm_challenges.clone(),
                alpha,
                zeta,
            };
            // convert public traces into an array
            // Public traces for the current segment
            let public_trace_vec = segment_instance.public_traces(show_public.clone());
            if public_trace_vec.is_empty() {
                return Err(VerificationError::<SC>::Other);
            }
            let public_traces: [Option<PublicTrace<SC::Val>>; NUM_CHIPS] =
                public_trace_vec[0].clone().try_into().unwrap();
            // Verify the chips
            verify_chip_constraints(
                &segment_machine,
                &segment_proof.proof,
                &public_traces,
                g_subgroups_segment,
                &challenges,
            );

            // Observe final state for the segment
            observe_final_state::<F, SC>(&mut challenger, &segment_instance); // 11 FIAT-SHAMIR
        }

        // Global cumulative sum checks
        for (segment_idx, segment_proof) in proof.segment_proofs.iter().enumerate() {
            let ephemeral_sum_segment: SC::Challenge = segment_proof
                .proof
                .chip_proofs
                .iter()
                .filter_map(|cp| cp.cumulative_ephemeral_sum)
                .sum();
            if ephemeral_sum_segment != SC::Challenge::zero() {
                return Err(VerificationError::<SC>::CumulativeEphemeralSumMismatch);
            }
        }

        // Persistent sums: The sum of ALL persistent interactions across ALL segments must sum to zero.
        let mut total_persistent_sum = SC::Challenge::zero();
        for segment_proof in proof.segment_proofs.iter() {
            for chip_proof in segment_proof.proof.chip_proofs.iter() {
                if let Some(persistent_sum) = chip_proof.cumulative_persistent_sum {
                    total_persistent_sum += persistent_sum;
                }
            }
        }
        // Although we currently don't have any global chips & thus chip proofs, add its
        // persistent cumulative contributions for possible future use
        for chip_proof in proof.chip_proofs.iter() {
            if let Some(persistent_sum) = chip_proof.cumulative_persistent_sum {
                total_persistent_sum += persistent_sum;
            }
        }

        if total_persistent_sum != SC::Challenge::zero() {
            return Err(VerificationError::<SC>::CumulativePersistentSumMismatch);
        }

        Ok(())
    }
}

impl<F: StarkField> MachineWithRegisters<F> for MultiSegmentBasicMachine<F> {
    fn set_initial_register_values(&mut self, reg: Registers) {
        self.initial_register_values = reg;
    }
    fn initial_register_values(&self) -> Registers {
        self.initial_register_values
    }
}

impl<F: StarkField> MachineWithProgramROM<F> for MultiSegmentBasicMachine<F> {
    fn program_rom(&self) -> &ProgramROM<i32> {
        &self.program_table.rom
    }

    fn set_program_rom(
        &mut self,
        init_pc: u32,
        rom: ProgramROM<i32>,
        table_type: ProgramTableType,
    ) {
        self.program_table = ProgramTable {
            init_pc,
            table_type,
            rom,
        };
    }

    fn program_table_type(&self) -> ProgramTableType {
        self.program_table.table_type
    }
}

/// NOTE: This feels a bit hacky. In order to have a lookup chip for the range
/// checks, we use the last segment's lookup chip.
impl<F: StarkField> MachineWithMultiLookupChip<F, BytesTable> for MultiSegmentBasicMachine<F> {
    fn lookup_chip(&self) -> &BytesChip<F> {
        MachineWithMultiLookupChip::lookup_chip(self.get_current_machine())
    }
    fn lookup_chip_mut(&mut self) -> &mut BytesChip<F> {
        MachineWithMultiLookupChip::<F, BytesTable>::lookup_chip_mut(self.get_current_machine_mut())
    }
    fn lookup_chip_bus(&self, receive_index: usize) -> BusArgument {
        MachineWithMultiLookupChip::<F, BytesTable>::lookup_chip_bus(
            self.get_current_machine(),
            receive_index,
        )
    }
}
