use std::sync::{Arc, Mutex};

use crate::config::StarkConfig;
use crate::core::Word;
use crate::indexer::{MachineProverKey, MachineVerifierKey};
use crate::memory_backend::MemoryBackendTrait;
use crate::NUM_CHIPS;
use crate::{
    error::VerificationError, persistence::ChipWithPersistence, AdviceProviderWithDefault,
    InteractionMap, MachineProof, PublicTrace, ReplayAdviceProvider, WriteCallbackWithDefault,
};
use alloc::fmt::Debug;
use p3_commit::Pcs;
use p3_field::{AbstractExtensionField, ExtensionField, Field, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Dimensions, MatrixGet, MatrixRows};
use std::collections::HashSet;

use valida_memory_footprint::MemoryFootprint;

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum StoppingFlag {
    DidStop,
    DidNotStop,
    DidFail,
    SizeLimitReached,
}

pub trait MachineInstanceData<F: Field> {
    // return value outer vector is indexed by segment; inner vector is indexed by chip
    fn public_traces(&self, verbose: Vec<bool>) -> Vec<Vec<Option<PublicTrace<F>>>>;
}

pub trait MachineRuntime {
    type MemoryBackend: MemoryBackendTrait;
    // TODO: cannot be a type here, as `AdviceProviderWithDefault` is not a trait
    //type Adv: AdviceProviderWithDefault;

    /// The memory backend is the global memory for the entire execution.
    fn memory_backend(&self) -> &Self::MemoryBackend;
    fn memory_backend_mut(&mut self) -> &mut Self::MemoryBackend;

    fn write_callback(&mut self) -> &mut WriteCallbackWithDefault;
    #[cfg(feature = "std")]
    fn write_to_file(&mut self, byte: u8) {
        let cb = self.write_callback();
        cb.0(byte);
    }
    fn advice(&mut self) -> &mut AdviceProviderWithDefault;

    /// Pushs a value read from the input advice in the `ReplayAdviceProvider`
    fn push_advice(&mut self, val: Option<u8>);
    /// Copies the replay advice from the runtime
    fn get_replay_advice(&self) -> ReplayAdviceProvider;

    fn read_from_file(&mut self) -> Option<u8> {
        let cb = self.advice();
        let val = cb.0();
        self.push_advice(val);
        val
    }
}

#[derive(Default, Clone)]
pub struct ProverOptions {
    pub show_main: Vec<bool>,
    pub show_public: Vec<bool>,
    pub show_interactions: Vec<bool>,
    pub show_public_dims: bool,
    pub show_main_dims: bool,
    pub show_permutation_dims: bool,
}

pub trait Machine<F: Field>: Sync {
    const NUM_CHIPS: usize;

    //type MemoryBackend: MemoryBackendTrait;

    /// The proof type for the machine
    type Proof<SC: StarkConfig<Val = F>>;

    /// This should include all the static data needed to initialize the machine:
    /// e.g. program ROM, initial register values, etc.
    type BootData;

    /// This should include all dynamic runtime resources needed to run the machine,
    /// such as the memory backend and i/o file handles.
    type Runtime: MachineRuntime;

    /// The public data needed to compute the public traces: e.g. program ROM, output values, static data, etc.
    type InstanceData: MachineInstanceData<F>;

    type Metrics: MachineMetrics;

    /// Sets the internal logging state
    /// Returns None if the state cannot be changed
    fn enable_logging(&mut self, log_enable: bool) -> Option<()>;

    fn log_enabled(&self) -> bool;

    /// Set the maximum trace height for this segment
    fn set_max_trace_height(&mut self, max_trace_height: u32);

    /// Get the maximum trace height for this segment
    fn max_trace_height(&self) -> u32;

    /// Set static machine variables such as the program ROM, initial register values, etc.
    fn init(&mut self, boot_data: Self::BootData);

    fn run(
        state: &mut RunningMachine<F, Self>,
        metrics: &mut Self::Metrics,
    ) -> (Self::InstanceData, Vec<u8>);

    fn step(state: &mut RunningMachine<'_, F, Self>) -> StoppingFlag;

    /// Attach the machine to a runtime and start it, running any dynamic initialization
    /// code that is necessary (e.g. loading static data into the memory backend).
    fn start(self, runtime: &mut Self::Runtime) -> RunningMachine<F, Self>;

    /// Run any cleanup code that is necessary (e.g. saving memory access timestamps)
    /// and detach the runtime.
    fn stop(state: RunningMachine<F, Self>) -> Self;

    /// `show_preprocessed` should be a vector of length NUM_CHIPS, with each element
    /// controlling whether to print a given chip's preprocessed trace.
    fn pre_process<SC>(
        &self,
        config: &SC,
        show_preprocessed: Vec<bool>,
        show_dimensions: bool,
    ) -> (MachineProverKey<SC, Self>, MachineVerifierKey<SC, Self>)
    where
        SC: StarkConfig<Val = F>,
        <SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::ProverData: Clone;

    /// `show_main` and `show public` should each be vectors of length NUM_CHIPS, with each element
    /// controls whether to print the main (resp. public) trace columns for a given chip.
    fn prove<SC>(
        &self,
        config: &SC,
        pk: &MachineProverKey<SC, Self>,
        opts: ProverOptions,
        instance_data: &Self::InstanceData,
    ) -> Self::Proof<SC>
    where
        SC: StarkConfig<Val = F> + Send + Sync + Clone + 'static,
        <SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::ProverData:
            Send + Sync + Clone + 'static,
        <<SC as StarkConfig>::Pcs as Pcs<F, RowMajorMatrix<F>>>::Proof:
            Send + Sync + Clone + 'static,
        <<SC as StarkConfig>::Pcs as Pcs<F, RowMajorMatrix<F>>>::Commitment: Send + Sync + 'static;

    fn compute_log_quotient_degrees<SC: StarkConfig<Val = F>>(&self) -> [usize; NUM_CHIPS];

    // Dimensions for PCS `verify_multi_batches`
    fn get_dims<SC: StarkConfig<Val = F>>(
        &self,
        preprocessed_dims: Vec<Option<Dimensions>>,
        chips: [&dyn ChipWithPersistence<Self, SC, Public = PublicTrace<SC::Val>>; NUM_CHIPS],
        proof: &MachineProof<SC>,
        log_quotient_degrees: [usize; NUM_CHIPS],
    ) -> [Vec<Dimensions>; 4]
    where
        SC: StarkConfig,
        Self: Sized,
        F: TwoAdicField, // <<< Required by the impl PublicValues for PublicTrace
        SC::Challenge: ExtensionField<F> + TwoAdicField, // <<< Required for E in that impl
    {
        [
            preprocessed_dims.into_iter().flatten().collect::<Vec<_>>(),
            chips
                .iter()
                .zip(proof.chip_proofs.iter())
                .flat_map(|(chip, chip_proof)| {
                    let width = chip.main_width();
                    if width == 0 {
                        None
                    } else {
                        Some(Dimensions {
                            width: chip.main_width(),
                            height: 1 << chip_proof.log_degree,
                        })
                    }
                })
                .collect::<Vec<_>>(),
            chips
                .iter()
                .zip(proof.chip_proofs.iter())
                .map(|(chip, chip_proof)| Dimensions {
                    width: chip.permutation_width(self) * SC::Challenge::D,
                    height: 1 << chip_proof.log_degree,
                })
                .collect::<Vec<_>>(),
            proof
                .chip_proofs
                .iter()
                .zip(log_quotient_degrees)
                .map(|(chip_proof, log_quotient_deg)| Dimensions {
                    width: SC::Challenge::D << log_quotient_deg,
                    height: 1 << chip_proof.log_degree,
                })
                .collect::<Vec<_>>(),
        ]
    }

    /// `show_public` should be a vector of length NUM_CHIPS, with each element controlling
    /// whether to print the public trace columns for a given chip.
    fn verify<SC>(
        &self,
        config: &SC,
        proof: &Self::Proof<SC>,
        vk: &MachineVerifierKey<SC, Self>,
        instance_data: &Self::InstanceData,
        show_public: Vec<bool>,
    ) -> Result<(), VerificationError<SC>>
    where
        SC: StarkConfig<Val = F>;

    // NOTE: read and write callbacks are handled via `ValidaRuntime` nowadays!
}

pub trait MachineMetrics {}

pub trait SegmentMachine<F: Field>: Machine<F> {
    fn set_segment_number(&mut self, segment_number: u32);
    fn segment_number(&self) -> u32;
    /// Run any cleanup code that is necessary (e.g. processing memory access
    /// timstamps and setting lookup muiltiplicity counters), detach the runtime,
    /// and return the static initialization data needed for the next segment.
    fn suspend(state: RunningMachine<F, Self>) -> (Self::BootData, Self);

    /// Generate the main traces for the segment.
    fn generate_main_traces<SC>(
        &self,
        config: &SC,
        show_main: Vec<bool>,
        show_main_dims: bool,
    ) -> Vec<Option<RowMajorMatrix<F>>>
    where
        SC: StarkConfig<Val = F>;

    /// Generate the permutation traces for the segment.
    fn generate_perm_traces<SC>(
        &self,
        config: &SC,
        preprocessed_traces: &[Option<RowMajorMatrix<F>>],
        public_traces: &[Option<PublicTrace<SC::Val>>],
        main_traces: &[Option<RowMajorMatrix<F>>],
        degrees: &[usize],
        perm_challenges: Vec<SC::Challenge>,
        global_perm_challenges: Vec<SC::Challenge>,
        #[cfg(debug_assertions)] interaction_map_guard: &Arc<Mutex<InteractionMap<F>>>,
        show_permutation_dims: bool,
    ) -> [Option<RowMajorMatrix<SC::Challenge>>; NUM_CHIPS]
    where
        SC: StarkConfig<Val = F>;

    /// Generate the public traces for the segment.
    fn generate_public_traces<SC>(
        &self,
        config: &SC,
        show_public: Vec<bool>,
        show_public_dims: bool,
    ) -> [Option<PublicTrace<F>>; NUM_CHIPS]
    where
        SC: StarkConfig<Val = F>;

    /// Generate the degrees and g-subgroups for the segment.
    fn degrees_and_g_subgroups<SC>(
        &self,
        config: &SC,
        main_traces: &[Option<RowMajorMatrix<F>>],
        preprocessed_traces: &[Option<RowMajorMatrix<F>>],
        public_traces: &[Option<PublicTrace<F>>],
    ) -> ([usize; NUM_CHIPS], [usize; NUM_CHIPS], [SC::Val; NUM_CHIPS])
    where
        SC: StarkConfig<Val = F>;

    /// Calculate the cumulative sums for the segment.
    fn cumulative_sums<SC>(
        &self,
        config: &SC,
        perm_traces: &[Option<RowMajorMatrix<SC::Challenge>>],
    ) -> (Vec<Option<SC::Challenge>>, Vec<Option<SC::Challenge>>)
    where
        SC: StarkConfig<Val = F>;

    /// Get the quotient polynomials for the segment.
    fn generate_quotient_polynomials<
        SC,
        PreprocessedTraceLde,
        MainTraceLde,
        PermTraceLde,
        PublicTraceLde,
    >(
        &self,
        config: &SC,
        preprocessed_traces: &[Option<RowMajorMatrix<F>>],
        main_traces: &[Option<RowMajorMatrix<F>>],
        perm_traces: &[Option<RowMajorMatrix<SC::Challenge>>],
        degrees: &[usize],
        log_degrees: &[usize],
        alpha: SC::Challenge,
        perm_challenges: Vec<SC::Challenge>,
        global_perm_challenges: Vec<SC::Challenge>,
        public_traces: &[Option<PublicTrace<SC::Val>>],
        preprocessed_trace_ldes: &mut Vec<Option<PreprocessedTraceLde>>,
        main_trace_ldes: &mut Vec<Option<MainTraceLde>>,
        perm_trace_ldes: &mut Vec<Option<PermTraceLde>>,
        public_trace_ldes: &mut Vec<Option<PublicTraceLde>>,
        cumulative_sums: &[Option<SC::Challenge>],
        cumulative_products: &[Option<SC::Challenge>],
        show_interactions: Vec<bool>,
    ) -> (
        Vec<RowMajorMatrix<F>>,
        [usize; NUM_CHIPS],
        [SC::Val; NUM_CHIPS],
    )
    where
        SC: StarkConfig<Val = F>,
        PreprocessedTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync,
        MainTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync,
        PermTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync,
        PublicTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync;
}

pub struct RunningMachine<'a, F: Field, M: Machine<F> + ?Sized> {
    pub machine: Box<M>,
    pub runtime: &'a mut M::Runtime,
}

impl<'a, F: Field, M> MemoryFootprint for RunningMachine<'a, F, M>
where
    M: Machine<F> + ?Sized + MemoryFootprint,
    M::Runtime: MemoryFootprint,
{
    fn memory_footprint(&self) -> usize {
        self.machine.memory_footprint() + self.runtime.memory_footprint()
    }
}

impl<F, M> std::fmt::Debug for RunningMachine<'_, F, M>
where
    F: Field,
    M: Machine<F> + Debug,
    <M::Runtime as MachineRuntime>::MemoryBackend: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.write_fmt(format_args!(
            "BasicRunningMachine {{ machine: {:?}, |cells|: {:?} }}",
            self.machine,
            self.runtime.memory_backend()
        ))
    }
}

pub trait MachineWithFinalMemoryState<F> {
    fn get_final_memory_state(&self) -> HashSet<(u32, Word<u8>, u32)>;
}

// use proptest::strategy::{SBoxedStrategy, Strategy};
// use valida_opcodes::BYTES_PER_INSTR;

// // impl<F, M, Mem> proptest::arbitrary::Arbitrary for RunningMachine<'_, F, M>
// // where
//     F: StarkField,
//     M: Machine<F, MemoryBackend = Mem>
//         + proptest::arbitrary::Arbitrary<Strategy = SBoxedStrategy<M>>,
//     M::Parameters: Send + Sync,
//     Mem: Debug + proptest::arbitrary::Arbitrary<Strategy = SBoxedStrategy<Mem>>,
//     Mem::Parameters: Send + Sync + Clone,
// {
//     type Parameters = (Mem::Parameters, M::Parameters);
//     type Strategy = SBoxedStrategy<Self>;

//     fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
//         use proptest as pt;
//         use pt::collection::btree_map;
//         use pt::sample::SizeRange;
//         use pt::strategy::Strategy;

//         let (memory_params, machine_params) = args;

//         let strat_running_machine =
//             M::arbitrary_with(machine_params).prop_flat_map(move |machine| {
//                 Mem::arbitrary_with(memory_params.clone()).prop_map(|ref mut memory_backend| {
//                     RunningMachine {
//                         machine: Box::new(machine),
//                         memory_backend,
//                     }
//                 })
//             });

//         strat_running_machine.sboxed()
//     }
// }
