#![feature(trait_upcasting)]
use core::marker::PhantomData;

use std::borrow::BorrowMut;
use std::fs::File;
use std::io::Write;
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex};

use p3_air::BaseAir;
use p3_commit::{Pcs, UnivariatePcs, UnivariatePcsWithLde};
use p3_field::{AbstractExtensionField, AbstractField, PrimeField32, TwoAdicField};
use p3_matrix::{
    dense::RowMajorMatrix, Dimensions, Matrix, MatrixGet, MatrixRowSlices, MatrixRows,
};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use valida_alu_u32::{
    add::{Add32Chip, Add32Instruction, MachineWithAdd32Chip},
    bitwise::{
        And32Instruction, Bitwise32Chip, MachineWithBitwise32Chip, Or32Instruction,
        Xor32Instruction,
    },
    com::{Com32Chip, Eq32Instruction, MachineWithCom32Chip, Ne32Instruction},
    div::{Div32Chip, Div32Instruction, MachineWithDiv32Chip, SDiv32Instruction},
    lt::{
        Lt32Chip, Lt32Instruction, Lte32Instruction, MachineWithLt32Chip, Sle32Instruction,
        Slt32Instruction,
    },
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
    MachineWithPersistentMemBus, MachineWithPointerBus, MachineWithProgramBus,
    MachineWithRangeBus8,
};
use valida_bytes::{BytesChip, BytesTable, MachineWithBytesChip, MachineWithRangeCheckeru8};
use valida_cpu::columns::CpuCols;
use valida_cpu::{
    BeqInstruction, BneInstruction, CpuChip, FailInstruction, Imm32Instruction, JalInstruction,
    JalvInstruction, Load32Instruction, LoadFpInstruction, LoadS8Instruction, LoadU8Instruction,
    MachineWithCpuChip, MachineWithRegisters, MemcpyInstruction, Operation, ReadAdviceInstruction,
    StopInstruction, Store32Instruction, StoreU8Instruction,
};
use valida_elliptic::{
    CombSecp256k1Instruction, MulsSecp256k1Instruction, SinvSecp256k1Instruction,
    SmulSecp256k1Instruction,
};
use valida_keccak::{KeccakFChip, KeccakFInstruction, MachineWithKeccakFChip};
use valida_lookups::{
    LookupType, MachineWithLookupChip, MachineWithMultiLookupChip, MultiLookupTableWrapper,
};
use valida_machine::__internal::p3_challenger::{CanObserve, FieldChallenger};
use valida_machine::__internal::{check_constraints, get_log_quotient_degree, quotient};
use valida_machine::{
    check_interactions,
    columns::{PermutationColsView, MAX_PERMUTATION_CONSTRAINT_DEGREE},
    generate_permutation_trace, verify_constraints, BusArgument, Chip, ChipProof, ChipTraceHeight,
    ChipWithPersistence, Commitments, ConstraintError, Instruction, InteractionMap, Machine,
    MachineInstanceData, MachineProof, MachineProverKey, MachineRuntime, MachineVerifierKey,
    MachineWithFinalMemoryState, MemoryAccessTimestamp, MemoryBackendTrait, OpenedValues, Operands,
    PcsError, ProgramROM, ProverOptions, PublicTrace, PublicValues, RunningMachine, SegmentMachine,
    StarkConfig, StarkField, StoppingFlag, StorageBackendType, ValidaStorageBackend,
    VerificationError, Word, NUM_CHIPS,
};
use valida_memory::{
    add_diff_bytes_receives, columns::MemoryCols, MachineWithMemoryChip, MemoryBackend,
    MemoryBackendType, MemoryChip,
};
use valida_opcodes::BYTES_PER_INSTR;
use valida_output::{MachineWithOutputChip, OutputChip, WriteInstruction};
use valida_program::{
    MachineWithProgramChip, MachineWithProgramROM, ProgramChip, ProgramTable, ProgramTableType,
};
use valida_range::{MachineWithRangeChip, RangeCheckerChip, RangeTable};
use valida_static_data::{MachineWithStaticDataChip, StaticDataChip, StaticDataChipType};

use crate::{
    metrics::func_cpu_usage::MemFpRelativeReader, metrics::metrics::BasicMachineMetrics, Registers,
    ValidaInstanceData, ValidaRuntime, ValidaSegmentBootData, ValidaSegmentInstanceData,
};

use std::collections::HashSet;

use ark_std::{end_timer, start_timer};

use valida_memory_footprint::MemoryFootprint;

pub type BasicRunningMachine<'a, F> = RunningMachine<'a, F, BasicMachine<F>>;

impl<F: StarkField> MemoryFootprint for BasicMachine<F> {
    fn memory_footprint(&self) -> usize {
        let mut result = 0;
        result += self.segment_number.memory_footprint();
        // The maximum height of a trace component allowed in a single segment.
        result += self.max_trace_height.memory_footprint();

        result += self.cpu.memory_footprint();
        result += self.program.memory_footprint();
        result += self.mem.memory_footprint();
        result += self.add_u32.memory_footprint();
        result += self.sub_u32.memory_footprint();
        result += self.mul_u32.memory_footprint();
        result += self.div_u32.memory_footprint();
        result += self.shift_u32.memory_footprint();
        result += self.lt_u32.memory_footprint();
        result += self.com_u32.memory_footprint();
        result += self.bitwise_u32.memory_footprint();
        result += self.output.memory_footprint();
        result += self.bytes.memory_footprint();
        result += self.static_data.memory_footprint();
        result += self.keccak_f.memory_footprint();

        //
        result += self.program_file.memory_footprint();

        result += self.final_memory_state.memory_footprint();

        result += self.no_log.memory_footprint();
        result += self.max_segment_size.memory_footprint();

        result
    }
}

#[derive(Default)]
pub struct BasicMachine<F: StarkField> {
    // Which segment of the overall program execution is computed by this `BasicMachine`
    segment_number: u32,
    // The maximum height of a trace component allowed in a single segment.
    max_trace_height: u32,

    // Core instructions
    load32: Load32Instruction,
    loadu8: LoadU8Instruction,
    loads8: LoadS8Instruction,
    store32: Store32Instruction,
    storeu8: StoreU8Instruction,

    jal: JalInstruction,
    jalv: JalvInstruction,
    beq: BeqInstruction,
    bne: BneInstruction,
    imm32: Imm32Instruction,
    stop: StopInstruction,
    fail: FailInstruction,
    loadfp: LoadFpInstruction,

    // ALU instructions
    add32: Add32Instruction,
    sub32: Sub32Instruction,
    mul32: Mul32Instruction,
    mulhs32: Mulhs32Instruction,
    mulhu32: Mulhu32Instruction,
    div32: Div32Instruction,
    sdiv32: SDiv32Instruction,
    shl32: Shl32Instruction,
    shr32: Shr32Instruction,
    sra32: Sra32Instruction,
    lt32: Lt32Instruction,
    lte32: Lte32Instruction,
    and32: And32Instruction,
    or32: Or32Instruction,
    xor32: Xor32Instruction,
    ne32: Ne32Instruction,
    eq32: Eq32Instruction,

    // Input/output instructions
    read: ReadAdviceInstruction,
    write: WriteInstruction,

    // Chips
    pub cpu: CpuChip,
    program: ProgramChip<F>,
    pub mem: MemoryChip,
    add_u32: Add32Chip,
    sub_u32: Sub32Chip,
    mul_u32: Mul32Chip,
    div_u32: Div32Chip,
    shift_u32: Shift32Chip,
    lt_u32: Lt32Chip,
    com_u32: Com32Chip,
    bitwise_u32: Bitwise32Chip,
    output: OutputChip,
    bytes: BytesChip<F>,
    static_data: StaticDataChip,
    keccak_f: KeccakFChip,

    pub file_to_save_stdout: Option<File>,
    // a binary representation of a loaded ELF
    pub program_file: Vec<u8>,

    // Auxiliary data needed for trace generation

    // The final memory state is used to deduce when persistent sends/receives between
    // segments are necessary. The very last update to a memory location never needs to
    // be sent to a next segment.
    pub final_memory_state: HashSet<(u32, Word<u8>, u32)>,

    // store the inverse so the derived default is to log
    no_log: bool,

    max_segment_size: usize,

    _phantom_sc: PhantomData<fn() -> F>,
}

impl<F: StarkField> MemFpRelativeReader for BasicRunningMachine<'_, F> {
    fn get(&self, fp_offset: i32) -> i32 {
        let read_addr_1 = (self.machine.cpu().fp as i32 + fp_offset) as u32;
        self.runtime.memory_backend().get_value(read_addr_1).into()
    }
}

impl<F: StarkField> MachineWithFinalMemoryState<F> for BasicMachine<F> {
    fn get_final_memory_state(&self) -> HashSet<(u32, Word<u8>, u32)> {
        self.final_memory_state.clone()
    }
}

impl<F: StarkField> BasicMachine<F> {
    /// Get the current trace height
    ///
    /// This is done via each chip's internal log variable, usually the length
    /// of the operations field.
    ///
    /// We can exclude the bytes chip, as its height is the size of the corresponding
    /// lookup table (256) which is fixed.
    ///
    /// Similarly, we can exclude the program chip, as its height is the size of the
    /// program which is fixed.
    pub fn current_trace_height(&self) -> u32 {
        self.cpu
            .trace_height()
            .max(self.mem.trace_height())
            .max(self.add_u32.trace_height())
            .max(self.sub_u32.trace_height())
            .max(self.mul_u32.trace_height())
            .max(self.div_u32.trace_height())
            .max(self.shift_u32.trace_height())
            .max(self.lt_u32.trace_height())
            .max(self.com_u32.trace_height())
            .max(self.bitwise_u32.trace_height())
            .max(self.output.trace_height())
            .max(self.static_data.trace_height())
            .max(self.keccak_f.trace_height())
    }

    /// Get an array of trait objects for the chips
    pub fn get_chips<SC: StarkConfig<Val = F>>(
        &self,
    ) -> [&dyn ChipWithPersistence<Self, SC, Public = PublicTrace<SC::Val>>; NUM_CHIPS] {
        [
            &self.cpu,
            &self.program,
            &self.mem,
            &self.add_u32,
            &self.sub_u32,
            &self.mul_u32,
            &self.div_u32,
            &self.shift_u32,
            &self.lt_u32,
            &self.com_u32,
            &self.bitwise_u32,
            &self.output,
            &self.bytes,
            &self.static_data,
            &self.keccak_f,
        ]
    }

    /// Performs the commitment to the public traces and and observes the commitment
    /// in the Fiat-Shamir transcript.
    fn commit_to_public_trace<SC: StarkConfig<Val = F>>(
        public_traces: &[Option<PublicTrace<SC::Val>>; NUM_CHIPS],
        pcs: &SC::Pcs,
        challenger: &mut SC::Challenger,
    ) -> (
        <SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::Commitment,
        <SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::ProverData,
    ) {
        let (public_commit, public_data) = tracing::info_span!("commit to public traces")
            // TODO: Can we avoid this clone?
            .in_scope(|| {
                pcs.commit_batches(
                    public_traces
                        .clone()
                        .iter()
                        .flatten()
                        .map(PublicTrace::into_matrix)
                        .collect(),
                )
            });

        // Observe the public commitment
        challenger.observe(public_commit.clone());
        (public_commit, public_data)
    }
}

// Types from `machine/src/proof.rs`. Needed to properly write some functions

type Val<SC> = <SC as StarkConfig>::Val;
type ValMat<SC> = RowMajorMatrix<Val<SC>>;
type Com<SC> = <<SC as StarkConfig>::Pcs as Pcs<Val<SC>, ValMat<SC>>>::Commitment;
type PcsProof<SC> = <<SC as StarkConfig>::Pcs as Pcs<Val<SC>, ValMat<SC>>>::Proof;

/// Get the vector of the zetas for each trace
pub fn calc_zeta<F: StarkField, SC: StarkConfig<Val = F>>(
    g_subgroups: [F; 15],
    has_traces: [bool; NUM_CHIPS],
    zeta: SC::Challenge,
) -> Vec<Vec<SC::Challenge>> {
    g_subgroups
        .iter()
        .zip(has_traces.iter())
        .flat_map(|(g, has_preprocessed)| {
            if *has_preprocessed {
                Some(vec![zeta, zeta * *g])
            } else {
                None
            }
        })
        .collect()
}

pub fn has_traces<F: TwoAdicField>(traces: &Vec<Option<RowMajorMatrix<F>>>) -> [bool; NUM_CHIPS] {
    traces
        .iter()
        .map(Option::is_some)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

pub struct VerificationChallenges<SC: StarkConfig> {
    // These fields are public, because in the multi segment machine we cannot use
    // `generate_from_challenger`, because we need challenges for each segment
    // (but not the global permutation challenges!).
    pub perm_challenges: Vec<SC::Challenge>,
    pub global_perm_challenges: Vec<SC::Challenge>,
    pub alpha: SC::Challenge,
    pub zeta: SC::Challenge,
}

impl<SC: StarkConfig> VerificationChallenges<SC> {
    /// Generates the permutation and global permutation challenges from the challenger and
    /// observes the permutation trace and quotient chunks.
    fn generate_from_challenger(
        challenger: &mut SC::Challenger,
        commitments: &Commitments<Com<SC>>,
    ) -> Self {
        let mut perm_challenges = Vec::new();
        for _ in 0..2 {
            perm_challenges.push(challenger.sample_ext_element::<SC::Challenge>());
        }
        let global_perm_challenges = (0..2)
            .map(|_| challenger.sample_ext_element())
            .collect::<Vec<SC::Challenge>>();

        challenger.observe(commitments.perm_trace.clone());
        let alpha = challenger.sample_ext_element::<SC::Challenge>();
        challenger.observe(commitments.quotient_chunks.clone());
        let zeta = challenger.sample_ext_element::<SC::Challenge>();

        Self {
            perm_challenges,
            global_perm_challenges,
            alpha,
            zeta,
        }
    }
}

struct OrganizedOpenings<SC: StarkConfig> {
    preprocessed_values: Vec<Vec<Vec<SC::Challenge>>>,
    main_values: Vec<Vec<Vec<SC::Challenge>>>,
    perm_values: Vec<Vec<Vec<SC::Challenge>>>,
    quotient_values: Vec<Vec<Vec<SC::Challenge>>>,
}

impl<SC: StarkConfig> OrganizedOpenings<SC> {
    fn from_chip_proofs(
        chip_proofs: &[ChipProof<SC::Challenge>],
        has_preprocessed: &[bool; NUM_CHIPS],
        has_main_traces: &[bool; NUM_CHIPS],
    ) -> Self {
        let mut preprocessed_values = vec![];
        let mut main_values = vec![];
        let mut perm_values = vec![];
        let mut quotient_values = vec![];

        for ((chip_proof, &has_preprocessed), &has_main) in chip_proofs
            .iter()
            .zip(has_preprocessed.iter())
            .zip(has_main_traces.iter())
        {
            let OpenedValues {
                preprocessed_local,
                preprocessed_next,
                trace_local,
                trace_next,
                permutation_local,
                permutation_next,
                quotient_chunks,
            } = &chip_proof.opened_values;

            if has_preprocessed {
                preprocessed_values
                    .push(vec![preprocessed_local.clone(), preprocessed_next.clone()]);
            }
            if has_main {
                main_values.push(vec![trace_local.clone(), trace_next.clone()]);
            }
            perm_values.push(vec![permutation_local.clone(), permutation_next.clone()]);
            quotient_values.push(vec![quotient_chunks.clone()]);
        }

        Self {
            preprocessed_values,
            main_values,
            perm_values,
            quotient_values,
        }
    }
}

struct ZetaChallenges<SC: StarkConfig> {
    preprocessed: Vec<Vec<SC::Challenge>>,
    main: Vec<Vec<SC::Challenge>>,
    perm: Vec<Vec<SC::Challenge>>,
    quotient: [Vec<SC::Challenge>; NUM_CHIPS],
}

impl<SC: StarkConfig<Val = F>, F: StarkField> ZetaChallenges<SC> {
    fn calculate(
        g_subgroups: [SC::Val; NUM_CHIPS],
        has_preprocessed: [bool; NUM_CHIPS],
        has_main_traces: [bool; NUM_CHIPS],
        log_quotient_degrees: [usize; NUM_CHIPS],
        zeta: SC::Challenge,
    ) -> Self {
        Self {
            preprocessed: calc_zeta::<F, SC>(g_subgroups, has_preprocessed, zeta),
            main: calc_zeta::<F, SC>(g_subgroups, has_main_traces, zeta),
            perm: calc_zeta::<F, SC>(g_subgroups, [true; NUM_CHIPS], zeta),
            quotient: log_quotient_degrees.map(|log_deg| vec![zeta.exp_power_of_2(log_deg)]),
        }
    }
}

fn verify_opening_proof<M, F: StarkField, SC: StarkConfig<Val = F>>(
    machine: &M,
    pcs: &SC::Pcs,
    dims: &[Vec<Dimensions>; 4],
    preprocessed_commit: &Com<SC>,
    commitments: &Commitments<Com<SC>>,
    zeta_challenges: &ZetaChallenges<SC>,
    organized_openings: OrganizedOpenings<SC>,
    opening_proof: &PcsProof<SC>,
    challenger: &mut SC::Challenger,
) -> Result<(), VerificationError<SC>>
where
    M: Machine<F>,
{
    let chips_opening_values = vec![
        organized_openings.preprocessed_values,
        organized_openings.main_values,
        organized_openings.perm_values,
        organized_openings.quotient_values,
    ];

    pcs.verify_multi_batches(
        &[
            (
                preprocessed_commit.clone(),
                zeta_challenges.preprocessed.as_slice(),
            ),
            (
                commitments.main_trace.clone(),
                zeta_challenges.main.as_slice(),
            ),
            (
                commitments.perm_trace.clone(),
                zeta_challenges.perm.as_slice(),
            ),
            (
                commitments.quotient_chunks.clone(),
                zeta_challenges.quotient.as_slice(),
            ),
        ],
        dims,
        chips_opening_values,
        opening_proof,
        challenger,
    )
    .map_err(PcsError)?;
    Ok(())
}

fn verify_chip<SC, M, F, C>(
    machine: &M,
    chip: &C,
    i: usize,
    chip_proof: &ChipProof<SC::Challenge>,
    public_values: &Option<<C as Chip<M, SC>>::Public>,
    g: F,
    challenges: &VerificationChallenges<SC>,
) -> Result<(), ConstraintError<SC>>
where
    M: Machine<F>,
    F: StarkField,
    SC: StarkConfig<Val = F>,
    C: ChipWithPersistence<M, SC>,
{
    verify_constraints::<M, _, SC>(
        &machine,
        chip,
        &chip_proof.opened_values,
        &public_values,
        chip_proof.cumulative_ephemeral_sum,
        chip_proof.cumulative_persistent_sum,
        chip_proof.log_degree,
        g,
        challenges.zeta,
        challenges.alpha,
        &challenges.perm_challenges,
        &challenges.global_perm_challenges,
    )
}

pub fn verify_chip_constraints<F, SC>(
    machine: &BasicMachine<F>,
    proof: &MachineProof<SC>,
    public_traces: &[Option<PublicTrace<SC::Val>>; NUM_CHIPS],
    g_subgroups: [SC::Val; NUM_CHIPS],
    challenges: &VerificationChallenges<SC>,
) -> Result<(), VerificationError<SC>>
where
    F: StarkField,
    SC: StarkConfig<Val = F>,
{
    // macro to avoid writing all parameters for each chip
    macro_rules! verify_chip_at_index {
        ($i:expr, $chip_method:ident) => {
            let cerr = verify_chip(
                machine,
                machine.$chip_method(),
                $i,
                &proof.chip_proofs[$i],
                &public_traces[$i],
                g_subgroups[$i],
                &challenges,
            );
            match cerr {
                Ok(()) => Ok::<(), ConstraintError<SC>>(()),
                Err(ConstraintError::OodEvaluationMismatch { expected, actual }) => {
                    panic!(
                        "Failed to verify constraints on chip {}: expected {} but got {}",
                        $i, expected, actual
                    );
                }
            };
        };
    }

    verify_chip_at_index!(0, cpu);
    verify_chip_at_index!(1, program);
    verify_chip_at_index!(2, mem);
    verify_chip_at_index!(3, add_u32);
    verify_chip_at_index!(4, sub_u32);
    verify_chip_at_index!(5, mul_32);
    verify_chip_at_index!(6, div_u32);
    verify_chip_at_index!(7, shift_u32);
    verify_chip_at_index!(8, lt_u32);
    verify_chip_at_index!(9, com_u32);
    verify_chip_at_index!(10, bitwise_u32);
    verify_chip_at_index!(11, output);
    verify_chip_at_index!(12, bytes);
    verify_chip_at_index!(13, static_data);
    verify_chip_at_index!(14, keccak_f);

    Ok(())
}

fn verify_cumulative_sums<SC: StarkConfig>(
    proof: &MachineProof<SC>,
) -> Result<(), VerificationError<SC>> {
    // Verify ephemeral sums
    let ephemeral_sum: SC::Challenge = proof
        .chip_proofs
        .iter()
        .flat_map(|chip_proof| chip_proof.cumulative_ephemeral_sum)
        .sum();

    if ephemeral_sum != SC::Challenge::zero() {
        return Err(VerificationError::<SC>::CumulativeEphemeralSumMismatch);
    }

    // Verify persistent sums
    let persistent_sum: SC::Challenge = proof
        .chip_proofs
        .iter()
        .flat_map(|chip_proof| chip_proof.cumulative_persistent_sum)
        .sum();

    if persistent_sum != SC::Challenge::zero() {
        return Err(VerificationError::<SC>::CumulativePersistentSumMismatch);
    }

    Ok(())
}

/// Handles observing the initial data from the instance data for this segment.
pub fn observe_instance_data<F: StarkField, SC: StarkConfig<Val = F>>(
    challenger: &mut SC::Challenger,
    instance_data: &ValidaSegmentInstanceData,
) {
    let pc0 = F::from_canonical_u64(instance_data.pc_init as u64);
    let fp0 = F::from_canonical_u64(instance_data.fp_init as u64);
    let is_last_segment = F::from_canonical_u64(instance_data.is_last_segment as u64);
    challenger.observe(pc0);
    challenger.observe(fp0);
    challenger.observe(is_last_segment);
}

/// Handles observing the final state of the segment.
pub fn observe_final_state<F: StarkField, SC: StarkConfig<Val = F>>(
    challenger: &mut SC::Challenger,
    instance_data: &ValidaSegmentInstanceData,
) {
    let pc1 = F::from_canonical_u64(instance_data.pc_final as u64);
    let fp1 = F::from_canonical_u64(instance_data.fp_final as u64);
    challenger.observe(pc1);
    challenger.observe(fp1);
}

pub fn compute_has_preprocessed(preprocessed_dims: &Vec<Option<Dimensions>>) -> [bool; NUM_CHIPS] {
    preprocessed_dims
        .iter()
        .map(Option::is_some)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

pub fn compute_has_main_traces<F: StarkField, SC: StarkConfig<Val = F>>(
    chips: &[&dyn ChipWithPersistence<BasicMachine<F>, SC, Public = PublicTrace<SC::Val>>;
         NUM_CHIPS],
) -> [bool; NUM_CHIPS] {
    chips
        .iter()
        .map(|chip| chip.main_width() != 0)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

pub fn compute_g_subgroups<F: StarkField, SC: StarkConfig<Val = F>>(
    chip_proofs: &[ChipProof<SC::Challenge>],
) -> [SC::Val; NUM_CHIPS] {
    chip_proofs
        .iter()
        .map(|chip_proof| SC::Val::two_adic_generator(chip_proof.log_degree))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Get the openings taking into account only those chips that have a trace.
pub fn get_openings<F: StarkField, SC: StarkConfig<Val = F>>(
    has_traces: [bool; NUM_CHIPS],
    openings: &mut Vec<Vec<Vec<SC::Challenge>>>,
) -> Vec<Vec<Vec<SC::Challenge>>> {
    has_traces
        .iter()
        .map(|&has_perm| {
            if has_perm {
                openings.remove(0)
            } else {
                vec![vec![], vec![]]
            }
        })
        .collect()
}

impl<F: StarkField> Machine<F> for BasicMachine<F> {
    const NUM_CHIPS: usize = NUM_CHIPS;
    type InstanceData = ValidaSegmentInstanceData;
    type BootData = ValidaSegmentBootData;
    type Runtime = ValidaRuntime;
    type Proof<SC: StarkConfig<Val = F>> = MachineProof<SC>;
    type Metrics = BasicMachineMetrics;

    fn enable_logging(&mut self, log_enable: bool) -> Option<()> {
        self.no_log = !log_enable;
        Some(())
    }

    fn log_enabled(&self) -> bool {
        !self.no_log
    }

    /// Get the maximum trace height for this segment
    fn max_trace_height(&self) -> u32 {
        self.max_trace_height
    }

    /// Set the maximum trace height for this segment
    fn set_max_trace_height(&mut self, max_trace_height: u32) {
        let mut mth = max_trace_height;
        if !mth.is_power_of_two() {
            eprintln!("Input max segment size is not a power of two. Rounding up to next.");
            mth = mth.next_power_of_two();
        }
        self.max_trace_height = mth;
    }

    fn init(&mut self, boot_data: Self::BootData) {
        self.program_file = boot_data.program_file;
        self.set_segment_number(boot_data.segment_number);
        // possibly disable trace generation
        self.enable_logging(boot_data.log_enabled);
        // and set the segment number in the ephemeral memory chip
        self.mem.segment_number = boot_data.segment_number as usize;
        self.set_max_trace_height(boot_data.max_trace_height);
        self.set_program_rom(boot_data.program_rom, boot_data.program_table_type);
        self.set_initial_register_values(boot_data.initial_register_values);
        let static_data_chip_type = match boot_data.program_table_type {
            ProgramTableType::Public => StaticDataChipType::Public,
            ProgramTableType::Preprocessed => StaticDataChipType::Preprocessed,
        };

        // If there's static data, and we're in the 0th segment, let's load the static data.
        if self.segment_number == 0 {
            if let Some(static_data) = boot_data.static_data {
                self.static_data_mut()
                    .load(static_data, static_data_chip_type);
            }
        }
    }

    fn start(mut self, runtime: &mut Self::Runtime) -> BasicRunningMachine<F> {
        let mut res = BasicRunningMachine {
            machine: Box::new(self),
            runtime,
        };

        // Call initialize_memory on the RunningMachine so that the static data
        // is copied into memory
        Self::initialize_memory(&mut res);

        res
    }
    fn stop(mut running_machine: BasicRunningMachine<'_, F>) -> Self {
        *running_machine.machine
    }

    fn run(
        state: &mut BasicRunningMachine<F>,
        metrics: &mut Self::Metrics,
    ) -> (ValidaSegmentInstanceData, Vec<u8>) {
        let mut final_stop_flag = StoppingFlag::DidNotStop;

        let mut step_did_stop = StoppingFlag::DidNotStop;
        loop {
            let pc = state.machine.cpu().pc;
            let instruction = *state.machine.program_rom().get_instruction(pc);

            metrics.register_instruction(&instruction, state);

            let step_did_stop = Self::step(state);
            // If we halted or reached the size limit (need to continue execution
            // in the next segment), we can stop the execution at this point.
            if step_did_stop != StoppingFlag::DidNotStop {
                final_stop_flag = step_did_stop;
                break;
            }
        }

        let log = state.machine.log_enabled();
        let (pc_init, pc_final, fp_init, fp_final) = {
            if log {
                debug_assert!(
                    !state.machine.cpu.registers.is_empty(),
                    "register state has not been initialized"
                );
            }
            let Registers {
                pc: pc_init,
                fp: fp_init,
            } = state.machine.initial_register_values();
            (pc_init, state.machine.cpu.pc, fp_init, state.machine.cpu.fp)
        };

        // Add receives for `diff_bytes` in the memory columns
        add_diff_bytes_receives(&mut *state.machine);

        // Extract the "final" memory state for this segment. This is needed to avoid sending
        // persistent sends/receives, which do not make sense for a single segment.
        let final_memory_state = state
            .runtime
            .memory_backend()
            .into_iter()
            .map(|(addr, record)| {
                (
                    addr,
                    record.value,
                    // NOTE: Values follow `MemoryAccessTimestamp::as_scalar` (but as a usize) with exception of
                    // `ThisSegment`. `3` so that it yields the value expected by a multi segment machine for
                    // segment zero.
                    match record.last_accessed {
                        MemoryAccessTimestamp::ThisSegment => 3,
                        MemoryAccessTimestamp::PriorSegment(segment) => 3 + segment,
                        MemoryAccessTimestamp::ZeroInitialized => 0,
                        MemoryAccessTimestamp::Static => 2,
                    },
                )
            })
            .collect();
        // For a multi segment machine, the `final_memory_state` field for each segment will be
        // overwritten after all segments have finished execution
        state.machine.final_memory_state = final_memory_state;

        // the rom and static_data are only needed when the machine is run in universal setup mode
        let rom = {
            match state.machine.program_table_type() {
                ProgramTableType::Public => Some(state.machine.program_rom().clone()),
                _ => None,
            }
        };
        let static_data = match state.machine.static_data().chip_type() {
            StaticDataChipType::Public => Some(state.machine.static_data().get_cells()),
            _ => None,
        };

        let did_stop = final_stop_flag == StoppingFlag::DidStop;
        // if the machine stopped, it means it was the last segment
        state.machine.cpu.is_last_segment = did_stop as u32;
        (
            ValidaSegmentInstanceData {
                rom, // Will be reset to `None` in multi segment machine. This way the basic machine continues to work
                output: state.machine.output.bytes().to_vec(),
                static_data,
                pc_init,
                pc_final,
                fp_init,
                fp_final,
                did_stop,
                did_fail: final_stop_flag == StoppingFlag::DidFail,
                is_last_segment: did_stop, // if it stopped, it's the last segment
                segment_number: state.machine.segment_number,
            },
            state.machine.output.bytes().to_vec(),
        )
    }

    fn pre_process<SC>(
        &self,
        config: &SC,
        show_preprocessed: Vec<bool>,
        show_dims: bool,
    ) -> (MachineProverKey<SC, Self>, MachineVerifierKey<SC, Self>)
    where
        SC: StarkConfig<Val = F>,
    {
        let pcs = config.pcs();
        let chips = self.get_chips::<SC>();

        let (preprocessed_traces, preprocessed_dims, preprocessed_trace_prints): (
            [Option<RowMajorMatrix<SC::Val>>; NUM_CHIPS],
            [Option<Dimensions>; NUM_CHIPS],
            [(Option<Vec<String>>, String); NUM_CHIPS],
        ) = tracing::info_span!("generate preprocessed traces").in_scope(|| {
            let ((traces, dims), prints): ((Vec<_>, Vec<_>), Vec<_>) = chips
                .par_iter()
                .zip(show_preprocessed.into_par_iter())
                .map(|(chip, verbose)| {
                    let (trace, log) = chip.get_preprocessed_trace(verbose);
                    let claimed_width =
                        <dyn Chip<_, _, Public = _> as BaseAir<SC::Val>>::preprocessed_width(
                            *chip,
                        );
                    let (chip_dims, chip_prints_and_name) = match trace {
                        None => {
                            debug_assert_eq!(
                                claimed_width,
                                0,
                                "Chip {} does not have a preprocessed trace but claims width as {}",
                                chip.name(),
                                claimed_width
                            );
                            debug_assert!(log.is_none());
                            (None, (log, chip.name()))
                        }
                        Some(ref trace) => {
                            debug_assert_eq!(
                                claimed_width,
                                trace.width(),
                                "Chip {} has a preprocessed trace with width {} but claims width as {}",
                                chip.name(),
                                trace.width(),
                                claimed_width
                            );
                            let dims = Dimensions {
                                width: claimed_width,
                                height: trace.height(),
                            };

                            if show_dims {
                                let mut prints = log.unwrap_or_default();
                                prints.push(format!(
                                    "Preprocessed trace dimensions for Chip: {}",
                                    chip.name()
                                ));
                                prints.push("-".repeat(80).to_string());
                                prints.push(format!(
                                    "{} columns and {} rows",
                                    dims.width, dims.height
                                ));
                                prints.push(format!("{} total cells", dims.width * dims.height));
                                (Some(dims), (Some(prints), chip.name()))
                            } else {
                                (Some(dims), (log, chip.name()))
                            }
                        }
                    };
                    ((trace, chip_dims), chip_prints_and_name)
                })
                .collect();

            (
                traces.try_into().unwrap(),
                dims.try_into().unwrap(),
                prints.try_into().unwrap(),
            )
        });

        let has_preprocessed_traces = has_traces(&preprocessed_traces.to_vec());

        for ((chip_prints, chip_name), dims) in preprocessed_trace_prints
            .iter()
            .zip(preprocessed_dims.iter())
        {
            if let Some(prints) = chip_prints {
                println!("Preprocessed trace for Chip: {}", chip_name);
                println!("{}", "-".repeat(80));
                for print in prints.iter() {
                    if !print.is_empty() {
                        println!("{print}");
                    }
                }
                println!("{}", "-".repeat(80));
            }
        }
        if show_dims {
            let total_size = preprocessed_dims
                .iter()
                .map(|d| d.map(|d| d.width * d.height).unwrap_or(0))
                .sum::<usize>();
            println!("Total pre-processed trace size: {}", total_size);
        }

        // TODO: can we avoid this clone?
        let (preprocessed_commit, preprocessed_data) =
            tracing::info_span!("commit to preprocessed traces").in_scope(|| {
                pcs.commit_batches(preprocessed_traces.clone().into_iter().flatten().collect())
            });
        let pk = MachineProverKey::new(
            preprocessed_traces.to_vec(),
            preprocessed_commit.clone(),
            preprocessed_data,
        );
        let vk = MachineVerifierKey::new(preprocessed_commit, preprocessed_dims.to_vec());
        (pk, vk)
    }

    /// The core proving method for a single segment machine.
    ///
    /// In this method we:
    /// - Generate preprocessed traces (per chip), then observe a commitment to them.
    /// - Generate public traces (per chip).
    /// - Generate main traces (per chip), then observe a commitment to them.
    /// - Sample elements for the permutation challenges.
    /// - Generate permutation traces (per chip).
    /// - Calculate cumulative sums and cumulative products for the permutation traces.
    /// - Observe a commitment to the permutation traces.
    /// - Sample another challenge element `alpha`.
    /// - Generate the quotient polynomials (per chip).
    /// - Observe a commitment to the quotient polynomials.
    /// - Get openings to the preprocessed, main, permutation and quotient polynomials.
    /// - Bundle everything together in a ChipProof.
    fn prove<SC>(
        &self,
        config: &SC,
        pk: &MachineProverKey<SC, Self>,
        opts: ProverOptions,
        instance: &ValidaSegmentInstanceData,
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

        let mut challenger = config.challenger();
        // TODO: Seed challenger with digest of all constraints & trace lengths.
        let pcs = config.pcs();
        // observe initial state
        observe_instance_data::<F, SC>(&mut challenger, instance);

        // Generate preprocessed traces.
        let (preprocessed_traces, preprocessed_commit, preprocessed_data) = (
            pk.preprocessed_traces(),
            pk.preprocessed_commit(),
            pk.preprocessed_prover_data(),
        );

        let has_preprocessed_traces = has_traces(&preprocessed_traces.to_vec());

        // Observe a commitment to the preprocessed traces.
        challenger.observe(preprocessed_commit.clone());

        let t_preprocessed_trace_ldes =
            start_timer!(|| "valida >machine.prove(..) | preprocessed_trace_ldes");
        let mut preprocessed_trace_ldes_real = pcs.get_ldes(preprocessed_data).into_iter();

        // add the None's back in so we can iterate through this as we do with the other lde arrays.
        let mut preprocessed_trace_ldes: Vec<_> = has_preprocessed_traces
            .iter()
            .map(|&has_trace| {
                if has_trace {
                    Some(preprocessed_trace_ldes_real.next().unwrap())
                } else {
                    None
                }
            })
            .collect();
        end_timer!(t_preprocessed_trace_ldes);

        // Generate public traces.
        let t_public_traces = start_timer!(|| "valida >machine.prove(..) | public_traces");
        let public_traces = self.generate_public_traces(config, show_public, show_public_dims);
        end_timer!(t_public_traces);

        // Commit to the public trace
        let (public_commit, public_data) =
            BasicMachine::<F>::commit_to_public_trace::<SC>(&public_traces, pcs, &mut challenger);
        // Observe the public commitment
        challenger.observe(public_commit);

        let t_public_trace_ldes = start_timer!(|| "valida >machine.prove(..) | public_trace_ldes");
        let mut public_trace_ldes: Vec<_> = public_traces
            .iter()
            .map(|opt| opt.as_ref().map(|trace| trace.get_ldes(config)))
            .collect();
        end_timer!(t_public_trace_ldes);

        // Generate main traces.
        let t_main_traces = start_timer!(|| "valida >machine.prove(..) | main_traces");
        let main_traces = self.generate_main_traces(config, show_main, show_main_dims);
        end_timer!(t_main_traces);

        let has_main_traces = has_traces(&main_traces);

        // Commit to main traces.
        let t_main_commit = start_timer!(|| "valida >machine.prove(..) | main_commit");
        let (main_commit, main_data) = tracing::info_span!("commit to main traces")
            // TODO: Can we avoid this clone?
            .in_scope(|| pcs.commit_batches(main_traces.clone().into_iter().flatten().collect()));
        end_timer!(t_main_commit);
        let t_main_trace_ldes = start_timer!(|| "valida >machine.prove(..) | main_trace_ldes");
        challenger.observe(main_commit.clone());

        let mut main_trace_ldes_real = pcs.get_ldes(&main_data).into_iter();

        // add the None's back in
        let mut main_trace_ldes: Vec<_> = has_main_traces
            .iter()
            .map(|&has_trace| {
                if has_trace {
                    Some(main_trace_ldes_real.next().unwrap())
                } else {
                    None
                }
            })
            .collect();
        end_timer!(t_main_trace_ldes);

        let t_g_subgroups = start_timer!(|| "valida >machine.prove(..) | g_subgroups");
        let (degrees, log_degrees, g_subgroups) =
            self.degrees_and_g_subgroups(config, &main_traces, preprocessed_traces, &public_traces);
        end_timer!(t_g_subgroups);

        let t_perm_traces = start_timer!(|| "valida >machine.prove(..) | perm_traces");

        // TODO(jen): why 2?
        // sample permutation challenges for ephemeral interactions
        let perm_challenges = (0..2)
            .map(|_| challenger.sample_ext_element())
            .collect::<Vec<<SC as StarkConfig>::Challenge>>();
        // now sample global challengers for *persistent interactions*
        let global_perm_challenges = (0..2)
            .map(|_| challenger.sample_ext_element())
            .collect::<Vec<<SC as StarkConfig>::Challenge>>();

        #[cfg(debug_assertions)]
        let interaction_map = InteractionMap::new();
        #[cfg(debug_assertions)]
        let interaction_map_guard = Arc::new(Mutex::new(interaction_map));
        let perm_traces = self.generate_perm_traces(
            config,
            &preprocessed_traces,
            &public_traces,
            &main_traces,
            &degrees,
            perm_challenges.clone(),
            global_perm_challenges.clone(),
            #[cfg(debug_assertions)]
            &interaction_map_guard,
            show_permutation_dims,
        );
        end_timer!(t_perm_traces);

        // Calculate cumulative sums.
        let (cumulative_ephemeral_sums, cumulative_persistent_sums) =
            self.cumulative_sums(config, &perm_traces);

        let t_perm_commit = start_timer!(|| "valida >machine.prove(..) | perm_commit");
        let (perm_commit, perm_data) = tracing::info_span!("commit to permutation traces")
            .in_scope(|| {
                let flattened_perm_traces = perm_traces
                    .iter()
                    .filter_map(|opt| opt.as_ref().map(|trace| trace.flatten_to_base()))
                    .collect::<Vec<_>>();
                pcs.commit_batches(flattened_perm_traces)
            });
        end_timer!(t_perm_commit);

        let has_perm_traces = has_traces(&perm_traces.to_vec());

        challenger.observe(perm_commit.clone());

        let t_perm_trace_ldes = start_timer!(|| "valida >machine.prove(..) | perm_trace_ldes");
        let mut perm_trace_ldes_real = pcs.get_ldes(&perm_data).into_iter();

        // add the None's back in
        let mut perm_trace_ldes: Vec<_> = has_perm_traces
            .iter()
            .map(|&has_perm| {
                if has_perm {
                    Some(perm_trace_ldes_real.next().unwrap())
                } else {
                    None
                }
            })
            .collect();
        end_timer!(t_perm_trace_ldes);

        let alpha: SC::Challenge = challenger.sample_ext_element();

        let t_check_constraints_and_quotients =
            start_timer!(|| "valida >machine.prove(..) | check_constraints_and_quotients");
        let (quotients, log_quotient_degrees, coset_shifts) = self.generate_quotient_polynomials(
            config,
            &preprocessed_traces,
            &main_traces,
            &perm_traces,
            &degrees,
            &log_degrees,
            alpha,
            perm_challenges,
            global_perm_challenges,
            &public_traces,
            &mut preprocessed_trace_ldes,
            &mut main_trace_ldes,
            &mut perm_trace_ldes,
            &mut public_trace_ldes,
            &cumulative_ephemeral_sums,
            &cumulative_persistent_sums,
            show_interactions,
        );
        end_timer!(t_check_constraints_and_quotients);

        let t_quotient_commit = start_timer!(|| "valida >machine.prove(..) | quotient_commit");
        let (quotient_commit, quotient_data) = tracing::info_span!("commit to quotient chunks")
            .in_scope(|| pcs.commit_shifted_batches(quotients.to_vec(), &coset_shifts));
        end_timer!(t_quotient_commit);

        challenger.observe(quotient_commit.clone());

        #[cfg(debug_assertions)]
        {
            check_interactions(&mut interaction_map_guard.lock().unwrap(), false);
            let sum = cumulative_ephemeral_sums
                .iter()
                .copied()
                .flatten()
                .sum::<SC::Challenge>();
            assert_eq!(
                sum,
                SC::Challenge::zero(),
                "Sum of cumulative sums is not zero: {}",
                sum
            );
        }
        // Compute all the zeta challenge values
        let zeta: SC::Challenge = challenger.sample_ext_element();
        let zeta_and_next_preprocessed =
            calc_zeta::<F, SC>(g_subgroups, has_preprocessed_traces, zeta);
        let zeta_and_next_main = calc_zeta::<F, SC>(g_subgroups, has_main_traces, zeta);
        let zeta_and_next_perm: Vec<Vec<SC::Challenge>> =
            calc_zeta::<F, SC>(g_subgroups, has_perm_traces, zeta);
        let zeta_exp_quotient_degree: [Vec<SC::Challenge>; NUM_CHIPS] =
            log_quotient_degrees.map(|log_deg| vec![zeta.exp_power_of_2(log_deg)]);

        let prover_data_and_points = [
            (preprocessed_data, zeta_and_next_preprocessed.as_slice()),
            (&main_data, zeta_and_next_main.as_slice()),
            (&perm_data, zeta_and_next_perm.as_slice()),
            (&quotient_data, zeta_exp_quotient_degree.as_slice()),
        ];
        let t_multiopen = start_timer!(|| "valida >machine.prove(..) | multiopen");
        let (openings, opening_proof) =
            pcs.open_multi_batches(&prover_data_and_points, &mut challenger);
        end_timer!(t_multiopen);

        let [mut preprocessed_openings_real, mut main_openings_real, mut perm_openings_real, quotient_openings] =
            openings
                .try_into()
                .expect("Should have 4 rounds of openings");

        let perm_openings = get_openings::<F, SC>(has_perm_traces, &mut perm_openings_real);
        let main_openings = get_openings::<F, SC>(has_main_traces, &mut main_openings_real);
        let preprocessed_openings =
            get_openings::<F, SC>(has_preprocessed_traces, &mut preprocessed_openings_real);

        let chip_proofs = log_degrees
            .iter()
            .zip(preprocessed_openings)
            .zip(main_openings)
            .zip(perm_openings)
            .zip(quotient_openings)
            .zip(cumulative_ephemeral_sums)
            .zip(cumulative_persistent_sums)
            .map(
                |(
                    (
                        ((((log_degree, preprocessed), main), perm), quotient),
                        cumulative_ephemeral_sum,
                    ),
                    cumulative_persistent_sum,
                )| {
                    let [preprocessed_local, preprocessed_next] =
                        preprocessed.try_into().expect("Should have 2 openings");

                    let [main_local, main_next] = main.try_into().expect("Should have 2 openings");
                    let [perm_local, perm_next] = perm.try_into().expect("Should have 2 openings");
                    let [quotient_chunks] = quotient.try_into().expect("Should have 1 opening");

                    let opened_values = OpenedValues {
                        preprocessed_local,
                        preprocessed_next,
                        trace_local: main_local,
                        trace_next: main_next,
                        permutation_local: perm_local,
                        permutation_next: perm_next,
                        quotient_chunks,
                    };

                    ChipProof {
                        log_degree: *log_degree,
                        opened_values,
                        cumulative_ephemeral_sum,
                        cumulative_persistent_sum,
                    }
                },
            )
            .collect::<Vec<_>>();

        // observe the final state
        observe_final_state::<F, SC>(&mut challenger, instance);

        let commitments = Commitments {
            main_trace: main_commit,
            perm_trace: perm_commit,
            quotient_chunks: quotient_commit,
        };

        MachineProof {
            commitments,
            opening_proof,
            chip_proofs,
        }
    }

    fn compute_log_quotient_degrees<SC: StarkConfig<Val = F>>(&self) -> [usize; NUM_CHIPS] {
        [
            get_log_quotient_degree::<Self, SC, _>(self, self.cpu()),
            get_log_quotient_degree::<Self, SC, _>(self, self.program()),
            get_log_quotient_degree::<Self, SC, _>(self, self.mem()),
            get_log_quotient_degree::<Self, SC, _>(self, self.add_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.sub_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.mul_32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.div_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.shift_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.lt_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.com_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.bitwise_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.output()),
            get_log_quotient_degree::<Self, SC, _>(self, self.bytes()),
            get_log_quotient_degree::<Self, SC, _>(self, self.static_data()),
            get_log_quotient_degree::<Self, SC, _>(self, self.keccak_f()),
        ]
    }

    fn verify<SC>(
        &self,
        config: &SC,
        proof: &Self::Proof<SC>,
        vk: &MachineVerifierKey<SC, Self>,
        instance_data: &Self::InstanceData,
        show_public: Vec<bool>,
    ) -> Result<(), VerificationError<SC>>
    where
        SC: StarkConfig<Val = F>,
    {
        // Setup phase - challenger and basic data
        let mut challenger = config.challenger();
        observe_instance_data::<F, SC>(&mut challenger, instance_data);

        let pcs = config.pcs();
        let chips = self.get_chips::<SC>();
        let log_quotient_degrees = self.compute_log_quotient_degrees::<SC>();

        // Extract commitments and dimensions
        let (preprocessed_commit, preprocessed_dims) =
            (vk.preprocessed_commit(), vk.preprocessed_dims());
        challenger.observe(preprocessed_commit.clone());

        let has_preprocessed = compute_has_preprocessed(&preprocessed_dims);
        let has_main_traces = compute_has_main_traces(&chips);
        let g_subgroups = compute_g_subgroups::<F, SC>(&proof.chip_proofs);

        // Generate public traces
        let public_traces: [Option<PublicTrace<SC::Val>>; NUM_CHIPS] = instance_data
            .public_traces(show_public)[0]
            .clone()
            .try_into()
            .unwrap();

        // Commit to the public trace to get the public commitment
        let (public_commit, _) =
            BasicMachine::<F>::commit_to_public_trace::<SC>(&public_traces, pcs, &mut challenger);
        // Observe the public commitment
        challenger.observe(public_commit);

        // Challenge generation
        challenger.observe(proof.commitments.main_trace.clone());
        let challenges: VerificationChallenges<SC> =
            VerificationChallenges::generate_from_challenger(&mut challenger, &proof.commitments);

        // Opening verification
        let organized_openings: OrganizedOpenings<SC> = OrganizedOpenings::from_chip_proofs(
            &proof.chip_proofs,
            &has_preprocessed,
            &has_main_traces,
        );

        let zeta_challenges: ZetaChallenges<SC> = ZetaChallenges::calculate(
            g_subgroups,
            has_preprocessed,
            has_main_traces,
            log_quotient_degrees,
            challenges.zeta,
        );

        let dims = self.get_dims(
            preprocessed_dims.clone(),
            chips,
            &proof,
            log_quotient_degrees,
        );
        verify_opening_proof(
            self,
            pcs,
            &dims,
            &preprocessed_commit,
            &proof.commitments,
            &zeta_challenges,
            organized_openings,
            &proof.opening_proof,
            &mut challenger,
        )?;

        // Final observations
        observe_final_state::<F, SC>(&mut challenger, instance_data);

        // Core verification
        verify_chip_constraints(self, proof, &public_traces, g_subgroups, &challenges)?;
        verify_cumulative_sums(proof)?;

        Ok(())
    }

    fn step(state: &mut RunningMachine<'_, F, Self>) -> StoppingFlag {
        // Fetch
        let pc = state.machine.cpu().pc;
        let instruction = state.machine.program_rom().get_instruction(pc);
        let opcode = instruction.opcode;
        let ops = instruction.operands;

        // Execute
        match opcode {
            <Load32Instruction as Instruction<Self, F>>::OPCODE => {
                Load32Instruction::execute(state, ops)
            }
            <LoadU8Instruction as Instruction<Self, F>>::OPCODE => {
                LoadU8Instruction::execute(state, ops)
            }
            <LoadS8Instruction as Instruction<Self, F>>::OPCODE => {
                LoadS8Instruction::execute(state, ops)
            }
            <Store32Instruction as Instruction<Self, F>>::OPCODE => {
                Store32Instruction::execute(state, ops)
            }
            <StoreU8Instruction as Instruction<Self, F>>::OPCODE => {
                StoreU8Instruction::execute(state, ops)
            }
            <JalInstruction as Instruction<Self, F>>::OPCODE => JalInstruction::execute(state, ops),
            <JalvInstruction as Instruction<Self, F>>::OPCODE => {
                JalvInstruction::execute(state, ops)
            }
            <BeqInstruction as Instruction<Self, F>>::OPCODE => BeqInstruction::execute(state, ops),
            <BneInstruction as Instruction<Self, F>>::OPCODE => BneInstruction::execute(state, ops),
            <Imm32Instruction as Instruction<Self, F>>::OPCODE => {
                Imm32Instruction::execute(state, ops)
            }
            <StopInstruction as Instruction<Self, F>>::OPCODE => {
                StopInstruction::execute(state, ops)
            }
            <FailInstruction as Instruction<Self, F>>::OPCODE => {
                FailInstruction::execute(state, ops)
            }
            <LoadFpInstruction as Instruction<Self, F>>::OPCODE => {
                LoadFpInstruction::execute(state, ops)
            }
            <Add32Instruction as Instruction<Self, F>>::OPCODE => {
                Add32Instruction::execute(state, ops)
            }
            <Sub32Instruction as Instruction<Self, F>>::OPCODE => {
                Sub32Instruction::execute(state, ops)
            }
            <Mul32Instruction as Instruction<Self, F>>::OPCODE => {
                Mul32Instruction::execute(state, ops)
            }
            <Mulhs32Instruction as Instruction<Self, F>>::OPCODE => {
                Mulhs32Instruction::execute(state, ops)
            }
            <Mulhu32Instruction as Instruction<Self, F>>::OPCODE => {
                Mulhu32Instruction::execute(state, ops)
            }
            <Div32Instruction as Instruction<Self, F>>::OPCODE => {
                Div32Instruction::execute(state, ops)
            }
            <SDiv32Instruction as Instruction<Self, F>>::OPCODE => {
                SDiv32Instruction::execute(state, ops)
            }
            <Shl32Instruction as Instruction<Self, F>>::OPCODE => {
                Shl32Instruction::execute(state, ops)
            }
            <Shr32Instruction as Instruction<Self, F>>::OPCODE => {
                Shr32Instruction::execute(state, ops)
            }
            <Sra32Instruction as Instruction<Self, F>>::OPCODE => {
                Sra32Instruction::execute(state, ops)
            }
            <Lt32Instruction as Instruction<Self, F>>::OPCODE => {
                Lt32Instruction::execute(state, ops)
            }
            <Lte32Instruction as Instruction<Self, F>>::OPCODE => {
                Lte32Instruction::execute(state, ops)
            }
            <Slt32Instruction as Instruction<Self, F>>::OPCODE => {
                Slt32Instruction::execute(state, ops)
            }
            <Sle32Instruction as Instruction<Self, F>>::OPCODE => {
                Sle32Instruction::execute(state, ops)
            }
            <And32Instruction as Instruction<Self, F>>::OPCODE => {
                And32Instruction::execute(state, ops)
            }
            <Or32Instruction as Instruction<Self, F>>::OPCODE => {
                Or32Instruction::execute(state, ops)
            }
            <Xor32Instruction as Instruction<Self, F>>::OPCODE => {
                Xor32Instruction::execute(state, ops)
            }
            <Ne32Instruction as Instruction<Self, F>>::OPCODE => {
                Ne32Instruction::execute(state, ops)
            }
            <Eq32Instruction as Instruction<Self, F>>::OPCODE => {
                Eq32Instruction::execute(state, ops)
            }
            <ReadAdviceInstruction as Instruction<Self, F>>::OPCODE => {
                ReadAdviceInstruction::execute(state, ops)
            }
            <WriteInstruction as Instruction<Self, F>>::OPCODE => {
                WriteInstruction::execute(state, ops)
            }
            <KeccakFInstruction as Instruction<Self, F>>::OPCODE => {
                KeccakFInstruction::execute(state, ops)
            }
            <MemcpyInstruction as Instruction<Self, F>>::OPCODE => {
                MemcpyInstruction::execute(state, ops)
            }
            <CombSecp256k1Instruction as Instruction<Self, F>>::OPCODE => {
                CombSecp256k1Instruction::execute(state, ops)
            }
            <MulsSecp256k1Instruction as Instruction<Self, F>>::OPCODE => {
                MulsSecp256k1Instruction::execute(state, ops)
            }
            <SinvSecp256k1Instruction as Instruction<Self, F>>::OPCODE => {
                SinvSecp256k1Instruction::execute(state, ops)
            }
            <SmulSecp256k1Instruction as Instruction<Self, F>>::OPCODE => {
                SmulSecp256k1Instruction::execute(state, ops)
            }
            _ => panic!("Unrecognized opcode: {}, pc = {}", opcode, pc),
        };
        let log = state.machine.log_enabled();
        state.machine.read_word(pc, log);

        // A STOP instruction signals the end of the program
        if opcode == <StopInstruction as Instruction<Self, F>>::OPCODE {
            StoppingFlag::DidStop
        } else if opcode == <FailInstruction as Instruction<Self, F>>::OPCODE {
            StoppingFlag::DidFail
        } else if state.machine.current_trace_height() >= state.machine.max_trace_height() {
            StoppingFlag::SizeLimitReached
        } else {
            StoppingFlag::DidNotStop
        }
    }
}

impl<F: StarkField> MachineWithProgramROM<F> for BasicMachine<F>
where
    F: PrimeField32,
{
    fn program_rom(&self) -> &ProgramROM<i32> {
        &self.program().table.0.rom
    }

    fn set_program_rom(&mut self, rom: ProgramROM<i32>, table_type: ProgramTableType) {
        let table = ProgramTable { table_type, rom };
        self.program_mut().set_table(MultiLookupTableWrapper(table));
    }

    fn program_table_type(&self) -> ProgramTableType {
        self.program().table.0.table_type
    }
}

impl<F: StarkField> SegmentMachine<F> for BasicMachine<F> {
    fn segment_number(&self) -> u32 {
        self.segment_number
    }
    fn set_segment_number(&mut self, segment_number: u32) {
        self.segment_number = segment_number;
    }

    /// Suspend the current segment machine, returning the boot data for the next segment machine.
    fn suspend(mut state: RunningMachine<F, Self>) -> (Self::BootData, Self) {
        Self::suspend_memory_state(&mut state);
        // NOTE: We *cannot* use the values from `state.machine.cpu().registers` as the starting point for the
        // next segment. These represent the register values of the *LAST EXECUTED* instruction. But this last
        // instruction will *modify* the program counter to a different value to point to the *next* instruction
        // that needs to be executed (in a non trivial way, if the last instruction contained was a branching / jumping
        // instruction).
        let reg = Registers {
            pc: state.machine.cpu().pc,
            fp: state.machine.cpu().fp,
        };
        (
            ValidaSegmentBootData {
                initial_register_values: reg,
                // Increment the segment number for the next segment.
                segment_number: state.machine.segment_number() + 1,
                max_trace_height: state.machine.max_trace_height(),
                program_rom: state.machine.program_rom().clone(),
                program_table_type: state.machine.program_table_type(),
                program_file: state.machine.program_file.clone(),
                static_data: Some(state.machine.static_data().get_cells().clone()), // TODO: Should we always use `Some` here?
                static_data_chip_type: Some(state.machine.static_data().chip_type()),
                log_enabled: state.machine.log_enabled(),
            },
            *state.machine,
        )
    }

    /// Generate main traces for all the chips in a machine.
    ///
    /// Returns a tuple of:
    /// - A vector of main traces.
    /// - A vector of dimensions for each trace.
    /// - A vector of logs for each trace (for debugging purposes).
    fn generate_main_traces<SC>(
        &self,
        config: &SC,
        show_main: Vec<bool>,
        show_main_dims: bool,
    ) -> Vec<Option<RowMajorMatrix<F>>>
    where
        SC: StarkConfig<Val = F>,
    {
        let chips = self.get_chips::<SC>();

        // TODO(jen): complex types here, required for the chip ordering situation currently but should be refactored
        let (main_traces, main_trace_dimensions, main_trace_prints): (
            Vec<Option<RowMajorMatrix<F>>>,
            Vec<Option<Dimensions>>,
            Vec<Option<Vec<String>>>,
        ) = tracing::info_span!("generate main trace").in_scope(|| {
            let (traces_and_dims, logs_and_names) = chips
                .par_iter()
                .zip(show_main.into_par_iter())
                .enumerate()
                .map(|(ident, (chip, verbose))| {
                    let (trace, log) = chip.generate_main_trace(&self, verbose);
                    let (dims, prints) = match trace {
                        Some(ref trace) => {
                            let dims = Dimensions {
                                width: trace.width(),
                                height: trace.height(),
                            };
                            debug_assert_eq!(
                                dims.width,
                                chip.main_width(),
                                "Chip {} claims main trace width as {} but actual width is {}",
                                ident,
                                chip.main_width(),
                                dims.width,
                            );
                            let prints = if show_main_dims {
                                let mut prints = log.unwrap_or_default();
                                prints.push(format!(
                                    "Main trace dimensions for Chip: {}",
                                    chip.name()
                                ));
                                prints.push("-".repeat(80).to_string());
                                prints.push(format!(
                                    "{} columns and {} rows",
                                    dims.width, dims.height
                                ));
                                prints.push(format!("{} total cells", dims.width * dims.height));
                                Some(prints)
                            } else {
                                log
                            };
                            (Some(dims), prints)
                        }
                        None => {
                            debug_assert_eq!(
                                chip.main_width(),
                                0,
                                "Chip {} does not have a main trace but claims width as {}",
                                ident,
                                chip.main_width()
                            );
                            (None, log)
                        }
                    };
                    ((trace, dims), prints)
                })
                .collect::<(Vec<_>, Vec<_>)>();

            let (traces, dims): (Vec<_>, Vec<_>) = traces_and_dims.into_iter().unzip();
            (traces, dims, logs_and_names)
        });

        // Print trace information
        for (i, chip_prints) in main_trace_prints.iter().enumerate() {
            if let Some(prints) = chip_prints {
                println!("Main trace for Chip: {}", chips[i].name());
                println!("{}", "-".repeat(80));
                for print in prints {
                    if !print.is_empty() {
                        println!("{print}");
                    }
                }
                println!("{}", "-".repeat(80));
            }
        }

        if show_main_dims {
            let total_size = main_trace_dimensions
                .iter()
                .map(|d| d.map(|d| d.width * d.height).unwrap_or(0))
                .sum::<usize>();
            println!("Total main trace size: {}", total_size);
            println!("{}", "-".repeat(80));
            println!();
        }

        main_traces
    }

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
        SC: StarkConfig<Val = F>,
    {
        let chips = self.get_chips::<SC>();

        let (perm_traces, perm_trace_dims): ([Option<RowMajorMatrix<SC::Challenge>>; NUM_CHIPS], [Dimensions; NUM_CHIPS]) =
            tracing::info_span!("generate permutation traces").in_scope(|| {
                let (traces, dims) =
                chips
                    .iter()
                    .enumerate()
                    .map(|(i, chip)| {
                        let trace = generate_permutation_trace(
                            self,
                            *chip,
                            &preprocessed_traces[i],
                            &public_traces[i],
                            &main_traces[i],
                            degrees[i],
                            perm_challenges.clone(),
                            global_perm_challenges.clone(),
                            #[cfg(debug_assertions)]
                            interaction_map_guard.clone()
                        );
                        let dims = trace.as_ref().map_or(
                            Dimensions { width: 0, height: 0 },
                            |t| Dimensions {
                                width: t.width(),
                                height: t.height(),
                            }
                        );
                        if show_permutation_dims {
                            println!("Permutation trace dimensions for Chip: {}", chip.name());
                            println!("{}", "-".repeat(80));
                            println!("{} *extension field* columns and {} rows", dims.width, dims.height);
                            println!("{} total *extension field* cells", dims.width * dims.height);
                            println!("{} total commitment cost", dims.width * dims.height * SC::Challenge::D);
                        }
                        debug_assert_eq!(dims.width, chip.permutation_width(self),
                        "Chip {:?}: The width of the permutation trace should be {:?} (number of interactions + 1), but actual width is {:?}",
                        i,
                        dims.width,
                        chip.permutation_width(self)
                    );
                        (trace, dims)
                    })
                    .collect::<(Vec<_>, Vec<_>)>();
                (traces
                    .try_into().unwrap(), dims.try_into().unwrap())
            });
        if show_permutation_dims {
            let total_size = perm_trace_dims
                .iter()
                .map(|d| d.width * d.height)
                .sum::<usize>();
            println!(
                "Total permutation trace size: {} extension field elements",
                total_size
            );
            println!("{}", "-".repeat(80));
            println!();
        }

        perm_traces
    }

    fn degrees_and_g_subgroups<SC>(
        &self,
        config: &SC,
        main_traces: &[Option<RowMajorMatrix<F>>],
        preprocessed_traces: &[Option<RowMajorMatrix<F>>],
        public_traces: &[Option<PublicTrace<F>>],
    ) -> ([usize; NUM_CHIPS], [usize; NUM_CHIPS], [SC::Val; NUM_CHIPS])
    where
        SC: StarkConfig<Val = F>,
    {
        let degrees: [usize; NUM_CHIPS] = main_traces
            .iter().zip(preprocessed_traces.iter()).zip(public_traces.iter())
            .map(|((main, preprocessed), public)| {
                // The public trace only has a "height" when it is a matrix.
                let public_height =  public.as_ref().and_then(|public_values| {
                    match public_values {
                        PublicTrace::PublicMatrix(matrix) => Some(matrix.height()),
                        PublicTrace::PublicVector(_) => None,
                    }
                });

                let (main_height, preprocessed_height) = (main.as_ref().map(|t| t.height()), preprocessed.as_ref().map(|t| t.height()));
                let heights = vec![main_height, preprocessed_height, public_height].into_iter().flatten().collect::<Vec<_>>();

                let first_height = heights.first().expect("all trace components are empty");
                debug_assert!(heights.iter().all(|&h| h == *first_height), "Trace components do not all have the same size. Main: {:?}, Preprocessed: {:?}, Public: {:?}", main_height, preprocessed_height, public_height);

                *first_height
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let log_degrees = degrees.map(log2_strict_usize);
        let g_subgroups = log_degrees.map(|log_deg| SC::Val::two_adic_generator(log_deg));

        (degrees, log_degrees, g_subgroups)
    }

    fn generate_public_traces<SC>(
        &self,
        config: &SC,
        show_public: Vec<bool>,
        show_public_dims: bool,
    ) -> [Option<PublicTrace<F>>; NUM_CHIPS]
    where
        SC: StarkConfig<Val = F>,
    {
        let chips = self.get_chips::<SC>();

        let (public_traces, public_dimensions, public_value_prints_and_names): (
            [_; NUM_CHIPS],
            [_; NUM_CHIPS],
            [_; NUM_CHIPS],
        ) = tracing::info_span!("generate public traces").in_scope(|| {
            let (traces_and_dims, logs_and_names) = chips
                .par_iter()
                .zip(show_public.into_par_iter())
                .enumerate()
                .map(|(ident, (chip, verbose))| {
                    let (trace, log) = chip.generate_public_values(verbose);
                    trace
                        .map(|trace| {
                            let dims = Dimensions {
                                width: trace.width(),
                                height: trace.height(),
                            };
                            let claimed_width = <dyn ChipWithPersistence<
                                _,
                                _,
                                Public = PublicTrace<F>,
                            > as BaseAir<F>>::public_width(
                                *chip
                            );

                            debug_assert_eq!(
                                dims.width, claimed_width,
                                "Chip {:?}: Public trace width claimed as {:?} but actually {:?}",
                                ident, claimed_width, dims.width
                            );

                            let prints = if show_public_dims {
                                let mut prints = log.unwrap_or_default();
                                prints.push(format!(
                                    "Public trace dimensions for Chip: {}",
                                    chip.name()
                                ));
                                prints.push("-".repeat(80).to_string());
                                prints.push(format!(
                                    "{} columns and {} rows",
                                    dims.width, dims.height
                                ));
                                Some(prints)
                            } else {
                                log
                            };

                            ((Some(trace), Some(dims)), (prints, chip.name()))
                        })
                        .unwrap_or(((None, None), (None, chip.name())))
                })
                .collect::<(Vec<_>, Vec<_>)>();

            let (traces, dims): (Vec<_>, Vec<_>) = traces_and_dims.into_iter().unzip();

            (
                traces.try_into().unwrap(),
                dims.try_into().unwrap(),
                logs_and_names.try_into().unwrap(),
            )
        });

        // Rest of the printing logic remains the same...
        for (chip_prints, chip_name) in public_value_prints_and_names.iter() {
            if let Some(prints) = chip_prints {
                println!("Public values for Chip: {}", chip_name);
                println!("{}", "-".repeat(80));
                for print in prints.iter() {
                    if !print.is_empty() {
                        println!("{print}");
                    }
                }
                println!("{}", "-".repeat(80));
            }
        }

        if show_public_dims {
            let total_size = public_dimensions
                .iter()
                .map(|d| d.map(|d| d.width * d.height).unwrap_or(0))
                .sum::<usize>();
            println!("Total public trace size: {}", total_size);
            println!("{}", "-".repeat(80));
            println!();
        }

        public_traces
    }

    fn cumulative_sums<SC>(
        &self,
        config: &SC,
        perm_traces: &[Option<RowMajorMatrix<SC::Challenge>>],
    ) -> (Vec<Option<SC::Challenge>>, Vec<Option<SC::Challenge>>)
    where
        SC: StarkConfig<Val = F>,
    {
        let chips = self.get_chips::<SC>();

        chips
            .iter()
            .zip(perm_traces.iter())
            .map(|(chip, trace_opt)| {
                if let Some(trace) = trace_opt {
                    let (num_ephemeral, num_persistent_sends, num_persistent_receives) = (
                        chip.ephemeral_interactions(self).len(),
                        chip.persistent_sends(self).len(),
                        chip.persistent_receives(self).len(),
                    );
                    let PermutationColsView {
                        ephemeral_cols,
                        persistent_cols,
                    } = PermutationColsView::as_view::<MAX_PERMUTATION_CONSTRAINT_DEGREE>(
                        num_ephemeral,
                        num_persistent_sends,
                        num_persistent_receives,
                        trace.row_slice(trace.height() - 1),
                    );
                    (
                        ephemeral_cols.last().copied(),
                        persistent_cols.last().copied(),
                    )
                } else {
                    (None, None)
                }
            })
            .collect::<(Vec<_>, Vec<_>)>()
    }

    // TODO(jen): way too many arguments here
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
        cumulative_ephemeral_sums: &[Option<SC::Challenge>],
        cumulative_persistent_sums: &[Option<SC::Challenge>],
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
        PublicTraceLde: MatrixRows<SC::Val> + MatrixGet<SC::Val> + Sync,
    {
        let pcs = config.pcs();

        let mut quotients: Vec<RowMajorMatrix<SC::Val>> = vec![];

        let mut interaction_prints: [Option<Vec<String>>; NUM_CHIPS] =
            vec![None; NUM_CHIPS].try_into().unwrap();

        macro_rules! chip {
            ($method:ident, $i:literal) => {
                let t_chip = start_timer!(|| format!(
                    "valida >machine.prove(..) | chip: {}",
                    stringify!($method)
                ));
                let chip = self.$method();
                #[cfg(debug_assertions)]
                {
                    let t_constraint = start_timer!(|| format!(
                        "valida >machine.prove(..) | constraint for {}",
                        stringify!($method)
                    ));
                    let interaction_prints = check_constraints::<Self, _, SC>(
                        self,
                        chip,
                        &preprocessed_traces[$i],
                        &main_traces[$i],
                        &perm_traces[$i],
                        degrees[$i],
                        &perm_challenges.clone(),
                        &global_perm_challenges.clone(),
                        &public_traces[$i],
                        show_interactions[$i],
                    );
                    if let Some(prints) = interaction_prints {
                        println!("Interactions for Chip: {}", Chip::<Self, SC>::name(chip));
                        println!("{}", "-".repeat(80));
                        for print in prints.iter() {
                            if !print.is_empty() {
                                println!("{print}");
                            }
                        }
                        println!("{}", "-".repeat(80));
                    }
                    end_timer!(t_constraint);
                }
                let t_quotient = start_timer!(|| format!(
                    "valida >machine.prove(..) | quotient for {}",
                    stringify!($method)
                ));

                quotients.push(quotient(
                    self,
                    config,
                    chip,
                    log_degrees[$i],
                    preprocessed_trace_ldes.remove(0),
                    main_trace_ldes.remove(0),
                    perm_trace_ldes.remove(0),
                    public_trace_ldes.remove(0),
                    cumulative_ephemeral_sums[$i],
                    cumulative_persistent_sums[$i],
                    &perm_challenges,
                    &global_perm_challenges,
                    alpha,
                ));
                end_timer!(t_quotient);
                end_timer!(t_chip);
            };
        }

        chip!(cpu, 0);
        chip!(program, 1);
        chip!(mem, 2);
        chip!(add_u32, 3);
        chip!(sub_u32, 4);
        chip!(mul_32, 5);
        chip!(div_u32, 6);
        chip!(shift_u32, 7);
        chip!(lt_u32, 8);
        chip!(com_u32, 9);
        chip!(bitwise_u32, 10);
        chip!(output, 11);
        chip!(bytes, 12);
        chip!(static_data, 13);
        chip!(keccak_f, 14);

        let log_quotient_degrees: [usize; NUM_CHIPS] = [
            get_log_quotient_degree::<Self, SC, _>(self, self.cpu()),
            get_log_quotient_degree::<Self, SC, _>(self, self.program()),
            get_log_quotient_degree::<Self, SC, _>(self, self.mem()),
            get_log_quotient_degree::<Self, SC, _>(self, self.add_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.sub_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.mul_32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.div_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.shift_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.lt_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.com_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.bitwise_u32()),
            get_log_quotient_degree::<Self, SC, _>(self, self.output()),
            get_log_quotient_degree::<Self, SC, _>(self, self.bytes()),
            get_log_quotient_degree::<Self, SC, _>(self, self.static_data()),
            get_log_quotient_degree::<Self, SC, _>(self, self.keccak_f()),
        ];
        assert_eq!(quotients.len(), NUM_CHIPS);
        assert_eq!(log_quotient_degrees.len(), NUM_CHIPS);
        let coset_shifts = tracing::debug_span!("coset shift").in_scope(|| {
            let pcs_coset_shift = pcs.coset_shift();
            log_quotient_degrees.map(|log_d| pcs_coset_shift.exp_power_of_2(log_d))
        });
        assert_eq!(coset_shifts.len(), NUM_CHIPS);

        (quotients, log_quotient_degrees, coset_shifts)
    }
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

impl<F: StarkField> MachineWithPersistentMemBus<F> for BasicMachine<F> {
    fn persistent_mem_bus(&self) -> BusArgument {
        BusArgument::Persistent(0)
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

impl<F: StarkField> MachineWithRangeBus8<F> for BasicMachine<F> {
    fn range_bus_8(&self) -> BusArgument {
        BusArgument::Global(5)
    }
}

impl<F: StarkField> MachineWithPointerBus<F> for BasicMachine<F> {
    fn pointer_bus(&self) -> BusArgument {
        BusArgument::Global(6)
    }
}

impl<F: StarkField> MachineWithRegisters<F> for BasicMachine<F> {
    fn set_initial_register_values(&mut self, reg: Registers) {
        self.cpu.set_initial_register_values(reg);
    }
    fn initial_register_values(&self) -> Registers {
        Registers {
            pc: self.cpu.pc_init,
            fp: self.cpu.fp_init,
        }
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
        let log = !self.no_log;
        if log {
            self.cpu.push_bus_op(imm, opcode, operands);
        }
    }

    fn push_pointer_op(&mut self, opcode: u32, operands: Operands<i32>) {
        let log = !self.no_log;
        if log {
            self.cpu.push_pointer_op(opcode, operands);
        }
    }
    fn push_left_imm_bus_op(
        &mut self,
        imm: Option<Word<u8>>,
        opcode: u32,
        operands: Operands<i32>,
    ) {
        let log = !self.no_log;
        if log {
            self.cpu.push_left_imm_bus_op(imm, opcode, operands);
        }
    }
    fn push_op(&mut self, op: Operation, opcode: u32, operands: Operands<i32>) {
        let log = !self.no_log;
        if log {
            self.cpu.push_op(op, opcode, operands);
        }
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

impl<F: StarkField> MachineWithMemoryChip<F> for BasicMachine<F> {
    fn mem(&self) -> &MemoryChip {
        &self.mem
    }

    fn mem_mut(&mut self) -> &mut MemoryChip {
        &mut self.mem
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
        &self.mul_u32
    }

    fn mul_32_mut(&mut self) -> &mut Mul32Chip {
        &mut self.mul_u32
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

impl<F: StarkField> MachineWithKeccakFChip<F> for BasicMachine<F> {
    fn keccak_f(&self) -> &KeccakFChip {
        &self.keccak_f
    }

    fn keccak_f_mut(&mut self) -> &mut KeccakFChip {
        &mut self.keccak_f
    }

    fn set_preimage(&mut self, preimage: [Word<u8>; 50]) {
        self.keccak_f_mut().preimage.push(preimage);
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

impl<F: StarkField> std::fmt::Debug for BasicMachine<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.write_fmt(format_args!(
            "BasicMachine {{ pc: {:x}, fp: {:x} }}",
            self.cpu.pc, self.cpu.fp,
        ))
    }
}

pub fn fp_strategy<F>() -> RangeInclusive<u32>
where
    F: StarkField,
{
    let max_addr = ((-F::one()).as_canonical_u32() >> 1) - 1;
    (0..=max_addr)
}

pub fn pc_strategy<F>() -> RangeInclusive<u32>
where
    F: StarkField,
{
    let field_size = (-F::one()).as_canonical_u32() + 1;
    0..=field_size / BYTES_PER_INSTR
}
