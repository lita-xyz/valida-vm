use crate::config::StarkConfig;
use alloc::vec::Vec;
use p3_commit::Pcs;
use p3_matrix::dense::RowMajorMatrix;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

type Val<SC> = <SC as StarkConfig>::Val;
type ValMat<SC> = RowMajorMatrix<Val<SC>>;
type Com<SC> = <<SC as StarkConfig>::Pcs as Pcs<Val<SC>, ValMat<SC>>>::Commitment;
type PcsProof<SC> = <<SC as StarkConfig>::Pcs as Pcs<Val<SC>, ValMat<SC>>>::Proof;

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "SC::Challenge: Serialize + DeserializeOwned")]
pub struct MachineProof<SC: StarkConfig> {
    pub commitments: Commitments<Com<SC>>,
    pub opening_proof: PcsProof<SC>,
    pub chip_proofs: Vec<ChipProof<SC::Challenge>>,
}

/// Segment proof is the `MachineProof` of a single segment in a multi segment
/// machine with additional auxiliary data needed for verification, i.e. some of
/// the instance data (program counters & frame pointers in particular)
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "SC::Challenge: Serialize + DeserializeOwned")]
pub struct SegmentProof<SC: StarkConfig> {
    pub proof: MachineProof<SC>,
    pub instance_data: ProofSegmentInstanceData,
}

/// Variation of the `ValidaSegmentInstanceData` of all the fields that are required to
/// verify the proof.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ProofSegmentInstanceData {
    // Initial program counter for this segment
    pub pc_init: u32,
    // Final program counter for this segment
    pub pc_final: u32,
    // Initial frame pointer for this segment
    pub fp_init: u32,
    // Final frame pointer for this segment
    pub fp_final: u32,
    // Output produced in this segment
    pub output: Vec<u8>,
}

/// The `MultiSegmentMachineProof` is a proof for the [`MultiSegmentBasicMachine`].
///
/// It is composed of multiple proofs for each segment.
#[derive(Serialize, Deserialize)]
#[serde(bound = "SC::Challenge: Serialize + DeserializeOwned")]
pub struct MultiSegmentMachineProof<SC: StarkConfig> {
    /// The proofs for each segment.
    pub segment_proofs: Vec<SegmentProof<SC>>,
    /// The multi-segment chip proofs.
    pub chip_proofs: Vec<ChipProof<SC::Challenge>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Commitments<Com> {
    pub main_trace: Com,
    pub perm_trace: Com,
    pub quotient_chunks: Com,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChipProof<Challenge> {
    pub log_degree: usize,
    pub opened_values: OpenedValues<Challenge>,
    pub cumulative_ephemeral_sum: Option<Challenge>,
    pub cumulative_persistent_sum: Option<Challenge>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OpenedValues<Challenge> {
    pub preprocessed_local: Vec<Challenge>,
    pub preprocessed_next: Vec<Challenge>,
    pub trace_local: Vec<Challenge>,
    pub trace_next: Vec<Challenge>,
    pub permutation_local: Vec<Challenge>,
    pub permutation_next: Vec<Challenge>,
    pub quotient_chunks: Vec<Challenge>,
}
