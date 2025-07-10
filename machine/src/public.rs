use alloc::slice;
use core::iter::Cloned;
use p3_commit::UnivariatePcsWithLde;
use p3_field::{ExtensionField, PrimeField32, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Matrix, MatrixGet, MatrixRowSlices, MatrixRows};
use p3_util::log2_strict_usize;
use valida_memory_footprint::MemoryFootprint;

use crate::{StarkConfig, StarkField};

pub trait PublicValues<F, E>: MatrixRowSlices<F> + MatrixGet<F> + Sized
where
    F: TwoAdicField,
    E: ExtensionField<F> + TwoAdicField,
{
    fn interpolate(&self, zeta: E, offset: usize) -> Vec<E>
    where
        Self: core::marker::Sized,
    {
        let height = self.height();
        let log_height = log2_strict_usize(height);
        let g = F::two_adic_generator(log_height);
        let point = zeta * g.powers().nth(offset).unwrap();

        p3_interpolation::interpolate_coset::<F, E, _>(self, F::one(), point)
    }

    fn get_ldes<SC>(&self, config: &SC) -> Self
    where
        SC: StarkConfig<Val = F, Challenge = E>;
}

impl<F, E, T> PublicValues<F, E> for T
where
    F: TwoAdicField,
    E: ExtensionField<F> + TwoAdicField,
    T: From<RowMajorMatrix<F>> + MatrixRowSlices<F> + MatrixGet<F> + Sized + Clone,
{
    fn get_ldes<SC>(&self, config: &SC) -> Self
    where
        SC: StarkConfig<Val = F, Challenge = E>,
    {
        let pcs = config.pcs();
        let mat = self.clone().to_row_major_matrix();
        pcs.compute_lde_batch(mat).into()
    }
}

// In the case that the public values are a vector rather than a matrix,
// we view it as a matrix with a single row repeated as many times as desired.
#[derive(Clone, Debug, Default)]
pub struct PublicRow<F>(pub Vec<F>);

impl<F: StarkField + MemoryFootprint> MemoryFootprint for PublicRow<F> {
    fn memory_footprint(&self) -> usize {
        self.0.memory_footprint()
    }
}

impl<T> Matrix<T> for PublicRow<T> {
    fn width(&self) -> usize {
        self.0.len()
    }
    fn height(&self) -> usize {
        1
    }
}

impl<T: Clone> MatrixRows<T> for PublicRow<T> {
    type Row<'a>
        = Cloned<slice::Iter<'a, T>>
    where
        T: 'a,
        Self: 'a;

    fn row(&self, _r: usize) -> Self::Row<'_> {
        self.0.iter().cloned()
    }
}

impl<T: Clone> MatrixRowSlices<T> for PublicRow<T> {
    fn row_slice(&self, _r: usize) -> &[T] {
        self.0.iter().as_slice()
    }
}

impl<T: Clone> MatrixGet<T> for PublicRow<T> {
    fn get(&self, _r: usize, c: usize) -> T {
        self.0[c].clone()
    }
}

impl<F, E> PublicValues<F, E> for PublicRow<F>
where
    F: TwoAdicField,
    E: ExtensionField<F> + TwoAdicField,
{
    fn interpolate(&self, _zeta: E, _offset: usize) -> Vec<E> {
        self.0.iter().map(|v| E::from_base(*v)).collect()
    }

    fn get_ldes<SC>(&self, _config: &SC) -> Self
    where
        SC: StarkConfig<Val = F>,
    {
        self.clone()
    }
}

#[derive(Clone, Debug)]
pub enum PublicTrace<F> {
    PublicMatrix(RowMajorMatrix<F>),
    PublicVector(PublicRow<F>),
}

impl<F: StarkField + MemoryFootprint> MemoryFootprint for PublicTrace<F> {
    fn memory_footprint(&self) -> usize {
        match self {
            PublicTrace::PublicMatrix(m) => m.memory_footprint(),
            PublicTrace::PublicVector(r) => r.memory_footprint(),
        }
    }
}

impl<F: Clone> PublicTrace<F> {
    pub fn from_matrix(mat: RowMajorMatrix<F>) -> Self {
        Self::PublicMatrix(mat)
    }

    pub fn from_vec(vec: Vec<F>) -> Self {
        Self::PublicVector(PublicRow(vec))
    }

    /// Helper to convert a public trace into a `RowMajorMatrix`. This is used in the prover and verifier
    /// to commit to the public trace
    pub fn into_matrix(&self) -> RowMajorMatrix<F> {
        match self {
            PublicTrace::PublicMatrix(m) => m.clone(),
            PublicTrace::PublicVector(v) => RowMajorMatrix::new(v.0.clone(), v.0.len()),
        }
    }
}

impl<F> Matrix<F> for PublicTrace<F> {
    fn width(&self) -> usize {
        match self {
            PublicTrace::PublicMatrix(mat) => mat.width(),
            PublicTrace::PublicVector(row) => row.width(),
        }
    }
    fn height(&self) -> usize {
        match self {
            PublicTrace::PublicMatrix(mat) => mat.height(),
            PublicTrace::PublicVector(row) => row.height(),
        }
    }
}

impl<F: Clone> MatrixRows<F> for PublicTrace<F> {
    type Row<'a>
        = Cloned<slice::Iter<'a, F>>
    where
        F: 'a,
        Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        match self {
            PublicTrace::PublicMatrix(mat) => mat.row(r),
            PublicTrace::PublicVector(row) => row.row(r),
        }
    }
}

impl<F: Clone> MatrixGet<F> for PublicTrace<F> {
    fn get(&self, r: usize, c: usize) -> F {
        match self {
            PublicTrace::PublicMatrix(mat) => mat.get(r, c),
            PublicTrace::PublicVector(row) => row.get(r, c),
        }
    }
}

impl<F: Clone> MatrixRowSlices<F> for PublicTrace<F> {
    fn row_slice(&self, r: usize) -> &[F] {
        match self {
            PublicTrace::PublicMatrix(mat) => mat.row_slice(r),
            PublicTrace::PublicVector(row) => row.row_slice(r),
        }
    }
}

impl<F, E> PublicValues<F, E> for PublicTrace<F>
where
    F: TwoAdicField,
    E: ExtensionField<F> + TwoAdicField,
{
    fn interpolate(&self, zeta: E, offset: usize) -> Vec<E> {
        match self {
            PublicTrace::PublicMatrix(mat) => mat.interpolate(zeta, offset),
            PublicTrace::PublicVector(row) => row.interpolate(zeta, offset),
        }
    }
    fn get_ldes<SC>(&self, config: &SC) -> Self
    where
        SC: StarkConfig<Val = F, Challenge = E>,
    {
        match self {
            PublicTrace::PublicMatrix(mat) => PublicTrace::PublicMatrix(mat.get_ldes(config)),
            PublicTrace::PublicVector(row) => PublicTrace::PublicVector(row.get_ldes(config)),
        }
    }
}
