use core::fmt::{Debug, Display, Formatter};

use p3_commit::Pcs;
use p3_matrix::dense::RowMajorMatrix;

use crate::StarkConfig;

#[derive(Debug)]
pub struct PcsError<E>(pub E);

#[derive(Debug)]
pub enum VerificationErrorGeneric<E: Debug> {
    /// The shape of opennings does not match the chip shapes.
    InvalidProofShape(ProofShapeError),
    /// Openning proof is invalid.
    InvalidOpeningArgument(PcsError<E>),
    /// Out-of-domain evaluation mismatch.
    ///
    /// `constraints(zeta)` did not match `quotient(zeta) Z_H(zeta)`.
    OodEvaluationMismatch,

    /// Cumulative ephemeral sum mismatch
    ///
    /// The running logarithmic derivative sum across all permutation traces is nonzero.
    CumulativeEphemeralSumMismatch,

    /// Cumulative persistent sum mismatch
    ///
    /// The running logarithmic derivative sum across all persistent traces across all chips and segments is not zero.
    CumulativePersistentSumMismatch,

    /// Mismatch of the number of segment machines and segment proofs
    ///
    /// This error indicates a bug in the verifier, as setting up one segment `BasicMachine` for
    /// each segment proof is our task.
    SegmentMachineProofNumberMismatch,

    /// Mismatch of the number of segments in the `ValidaInstanceData` and segment proofs.
    ///
    /// This error indicates a bug in the verifier, as adding the instance data segments to the
    /// `ValidaInstanceData` based on each segment is our task.
    InstanceDataSegmentProofNumberMismatch,

    /// The vector of public traces is empty
    EmptyPublicTraces,

    /// Other error. If you use this error, better create a new enum element
    Other,
}

pub type VerificationError<SC> = VerificationErrorGeneric<
    <<SC as StarkConfig>::Pcs as Pcs<
        <SC as StarkConfig>::Val,
        RowMajorMatrix<<SC as StarkConfig>::Val>,
    >>::Error,
>;

#[derive(Debug)]
pub struct OodEvaluationMismatch;

#[derive(Debug)]
pub enum ProofShapeError {
    Preprocessed,
    MainTrace,
    Permutation,
    Quotient,
}

impl<E: Display + Debug> Display for VerificationErrorGeneric<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            VerificationErrorGeneric::InvalidProofShape(err) => {
                write!(f, "Invalid proof shape: for {} opening", err)
            }
            VerificationErrorGeneric::InvalidOpeningArgument(err) => {
                write!(f, "Invalid opening argument: {:?}", err)
            }
            VerificationErrorGeneric::OodEvaluationMismatch => {
                write!(f, "Invalid quotient argument")
            }
            VerificationErrorGeneric::CumulativeEphemeralSumMismatch => {
                write!(f, "Invalid permutation argument for ephemeral sum")
            }
            VerificationErrorGeneric::CumulativePersistentSumMismatch => {
                write!(f, "Invalid permutation argument for persistent sum")
            }
            VerificationErrorGeneric::SegmentMachineProofNumberMismatch => {
                write!(
                    f,
                    "Number of segment machines does not match number of segment proofs"
                )
            }
            VerificationErrorGeneric::InstanceDataSegmentProofNumberMismatch => {
                write!(
                    f,
                    "Number of instance data segments does not match number of segment proofs"
                )
            }
            VerificationErrorGeneric::EmptyPublicTraces => {
                write!(f, "Public traces are empty")
            }

            VerificationErrorGeneric::Other => {
                write!(f, "Other error encountered, multi segment related")
            }
        }
    }
}

impl Display for ProofShapeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            ProofShapeError::Preprocessed => {
                write!(f, "Preprocessed opening mismatch")
            }
            ProofShapeError::MainTrace => {
                write!(f, "Main trace opening mismatch")
            }
            ProofShapeError::Permutation => {
                write!(f, "Permutation opening mismatch")
            }
            ProofShapeError::Quotient => {
                write!(f, "Quotient opening mismatch")
            }
        }
    }
}

impl<E: Debug> From<ProofShapeError> for VerificationErrorGeneric<E> {
    fn from(err: ProofShapeError) -> Self {
        VerificationErrorGeneric::<E>::InvalidProofShape(err)
    }
}

impl<E: Debug> From<OodEvaluationMismatch> for VerificationErrorGeneric<E> {
    fn from(_: OodEvaluationMismatch) -> Self {
        VerificationErrorGeneric::<E>::OodEvaluationMismatch
    }
}

impl<E: Debug> From<PcsError<E>> for VerificationErrorGeneric<E> {
    fn from(err: PcsError<E>) -> Self {
        VerificationErrorGeneric::<E>::InvalidOpeningArgument(err)
    }
}
