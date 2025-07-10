use core::marker::PhantomData;

use p3_commit::Pcs;
use p3_matrix::{dense::RowMajorMatrix, Dimensions};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::{config::StarkConfig, Machine};

type Val<SC> = <SC as StarkConfig>::Val;
type ProverData<SC> =
    <<SC as StarkConfig>::Pcs as Pcs<Val<SC>, RowMajorMatrix<Val<SC>>>>::ProverData;

#[derive(Deserialize, Serialize)]
#[serde(bound(serialize = "ProverData<SC>: Serialize"))]
#[serde(bound(deserialize = "ProverData<SC>: DeserializeOwned"))]
pub struct MachineProverKey<SC, M>
where
    SC: StarkConfig,
    M: Machine<SC::Val> + ?Sized,
{
    preprocessed_traces: Vec<Option<RowMajorMatrix<SC::Val>>>,
    preprocessed_commit: <SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::Commitment,
    preprocessed_prover_data: <SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::ProverData,
    _phantom_data: PhantomData<M>,
}

impl<SC: StarkConfig, M: Machine<SC::Val>> MachineProverKey<SC, M> {
    pub fn new(
        preprocessed_traces: Vec<Option<RowMajorMatrix<SC::Val>>>,
        preprocessed_commit: <SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::Commitment,
        preprocessed_prover_data: <SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::ProverData,
    ) -> Self {
        assert_eq!(
            preprocessed_traces.len(),
            M::NUM_CHIPS,
            "Preprocessed traces length does not match machine chip count"
        );
        Self {
            preprocessed_traces,
            preprocessed_commit,
            preprocessed_prover_data,
            _phantom_data: PhantomData,
        }
    }

    pub fn preprocessed_traces(&self) -> &Vec<Option<RowMajorMatrix<SC::Val>>> {
        &self.preprocessed_traces
    }

    pub fn preprocessed_commit(
        &self,
    ) -> <SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::Commitment {
        self.preprocessed_commit.clone()
    }

    pub fn preprocessed_prover_data(
        &self,
    ) -> &<SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::ProverData {
        &self.preprocessed_prover_data
    }
}

#[derive(Deserialize, Serialize)]
#[serde(bound(serialize = "ProverData<SC>: Serialize"))]
#[serde(bound(deserialize = "ProverData<SC>: DeserializeOwned"))]
pub struct MachineVerifierKey<SC, M>
where
    SC: StarkConfig,
    M: Machine<SC::Val> + ?Sized,
{
    preprocessed_commit: <SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::Commitment,
    preprocessed_dims: Vec<Option<Dimensions>>,
    _phantom_data: PhantomData<M>,
}

impl<SC: StarkConfig, M: Machine<SC::Val>> MachineVerifierKey<SC, M> {
    pub fn new(
        preprocessed_commit: <SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::Commitment,
        preprocessed_dims: Vec<Option<Dimensions>>,
    ) -> Self {
        assert_eq!(
            preprocessed_dims.len(),
            M::NUM_CHIPS,
            "Preprocessed dims length does not match machine chip count"
        );
        Self {
            preprocessed_commit,
            preprocessed_dims,
            _phantom_data: PhantomData,
        }
    }
    pub fn preprocessed_commit(
        &self,
    ) -> <SC::Pcs as Pcs<SC::Val, RowMajorMatrix<SC::Val>>>::Commitment {
        self.preprocessed_commit.clone()
    }
    pub fn preprocessed_dims(&self) -> Vec<Option<Dimensions>> {
        self.preprocessed_dims.clone()
    }
}
