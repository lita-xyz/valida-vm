use core::marker::PhantomData;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, UnivariatePcsWithLde};
use p3_field::{AbstractExtensionField, ExtensionField, PackedField, PrimeField32, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

use valida_memory_footprint::MemoryFootprint;

/// We define a custom trait for the field we use for two reasons:
/// 1. easier to extend with custom traits like `MemoryFootprint`
/// 2. less typing needed across the board. Typing `PrimeField32 + TwoAdicField`
///    gets old very fast.
pub trait StarkField: PrimeField32 + TwoAdicField + MemoryFootprint {}

/// We need to implement the stark field for baby bear
impl StarkField for p3_baby_bear::BabyBear {}

pub trait StarkConfig {
    /// The field over which trace data is encoded.
    type Val: StarkField; //StarkField; // TODO: Relax to Field?
    type PackedVal: PackedField<Scalar = Self::Val>;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Self::Val> + TwoAdicField + MemoryFootprint;
    type PackedChallenge: AbstractExtensionField<Self::PackedVal, F = Self::Challenge> + Copy;

    /// The PCS used to commit to trace polynomials.
    type Pcs: UnivariatePcsWithLde<
        Self::Val,
        Self::Challenge,
        RowMajorMatrix<Self::Val>,
        Self::Challenger,
    >;

    /// The challenger (Fiat-Shamir) implementation used.
    type Challenger: FieldChallenger<Self::Val>
        + CanObserve<<Self::Pcs as Pcs<Self::Val, RowMajorMatrix<Self::Val>>>::Commitment>;

    fn pcs(&self) -> &Self::Pcs;

    fn challenger(&self) -> Self::Challenger;
}

#[derive(Clone)]
pub struct StarkConfigImpl<Val, Challenge, PackedChallenge, Pcs, Challenger> {
    pcs: Pcs,
    init_challenger: Challenger,
    _phantom: PhantomData<(Val, Challenge, PackedChallenge, Challenger)>,
}

impl<Val, Challenge, PackedChallenge, Pcs, Challenger>
    StarkConfigImpl<Val, Challenge, PackedChallenge, Pcs, Challenger>
{
    pub fn new(pcs: Pcs, init_challenger: Challenger) -> Self {
        Self {
            pcs,
            init_challenger,
            _phantom: PhantomData,
        }
    }
}

impl<Val, Challenge, PackedChallenge, Pcs, Challenger> StarkConfig
    for StarkConfigImpl<Val, Challenge, PackedChallenge, Pcs, Challenger>
where
    Val: StarkField, // TODO: Relax to Field?
    Challenge: ExtensionField<Val> + TwoAdicField + MemoryFootprint,
    PackedChallenge: AbstractExtensionField<Val::Packing, F = Challenge> + Copy,
    Pcs: UnivariatePcsWithLde<Val, Challenge, RowMajorMatrix<Val>, Challenger>,
    Challenger: FieldChallenger<Val>
        + Clone
        + CanObserve<<Pcs as p3_commit::Pcs<Val, RowMajorMatrix<Val>>>::Commitment>,
{
    type Val = Val;
    type PackedVal = Val::Packing;
    type Challenge = Challenge;
    type PackedChallenge = PackedChallenge;
    type Pcs = Pcs;
    type Challenger = Challenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn challenger(&self) -> Self::Challenger {
        self.init_challenger.clone()
    }
}
