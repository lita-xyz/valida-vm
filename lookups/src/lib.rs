use core::marker::PhantomData;

use p3_air::VirtualPairCol;
use p3_field::{Field, PrimeField};
use p3_matrix::{dense::RowMajorMatrix, Matrix, MatrixRowSlices, MatrixRows};
use p3_maybe_rayon::prelude::*;
use smallvec::{smallvec, SmallVec};
use valida_bus::MachineWithRangeBus8;
use valida_machine::{
    BusArgument, Chip, ChipTraceHeight, ChipWithPersistence, Interaction, Machine, PublicTrace,
    StarkConfig, __internal::p3_field::AbstractField, SMALLVEC_SIZE,
};
use valida_memory_footprint::MemoryFootprint;

extern crate alloc;
use alloc::collections::BTreeMap;

pub mod stark;
pub trait LookupTable<F>: Default
where
    F: Field,
{
    type M<'a>: MatrixRowSlices<F> + Sync
    where
        Self: 'a;

    fn lookup_type(&self) -> LookupType;
    fn lookup_matrix(&self, verbose: bool) -> (Self::M<'_>, Option<Vec<String>>);
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn name(&self) -> String;
}

#[derive(Clone, Copy)]
pub enum LookupType {
    Public,
    Preprocessed,
    Private,
}

pub trait MultiLookupTable<F>: Default
where
    F: Field,
{
    type M<'a>: MatrixRowSlices<F> + Sync
    where
        Self: 'a;

    fn num_receives(&self) -> usize;

    fn private_columns(&self, _verbose: bool) -> (Option<Self::M<'_>>, Option<Vec<String>>) {
        (None, None)
    }
    fn public_columns(&self, _verbose: bool) -> (Option<Self::M<'_>>, Option<Vec<String>>) {
        (None, None)
    }
    fn preprocessed_columns(&self, _verbose: bool) -> (Option<Self::M<'_>>, Option<Vec<String>>) {
        (None, None)
    }
    fn num_private_columns(&self) -> usize {
        0
    }
    fn num_public_columns(&self) -> usize {
        0
    }
    fn num_preprocessed_columns(&self) -> usize {
        0
    }
    fn height(&self) -> usize;

    fn fields_for_receive(&self, i: usize) -> SmallVec<[VirtualPairCol<F>; SMALLVEC_SIZE]>;

    fn name(&self) -> String;
}

#[derive(Default)]
pub struct MultiLookupTableWrapper<L>(pub L);

impl<L: MemoryFootprint> MemoryFootprint for MultiLookupTableWrapper<L> {
    fn memory_footprint(&self) -> usize {
        self.0.memory_footprint()
    }
}

impl<F: Field, L: LookupTable<F>> MultiLookupTable<F> for MultiLookupTableWrapper<L> {
    type M<'a>
        = L::M<'a>
    where
        L: 'a;

    fn num_receives(&self) -> usize {
        1
    }

    fn private_columns(&self, verbose: bool) -> (Option<Self::M<'_>>, Option<Vec<String>>) {
        match self.0.lookup_type() {
            LookupType::Private => {
                let (matrix, log) = self.0.lookup_matrix(verbose);
                (Some(matrix), log)
            }
            _ => (None, None),
        }
    }
    fn preprocessed_columns(&self, verbose: bool) -> (Option<Self::M<'_>>, Option<Vec<String>>) {
        match self.0.lookup_type() {
            LookupType::Preprocessed => {
                let (matrix, log) = self.0.lookup_matrix(verbose);
                (Some(matrix), log)
            }
            _ => (None, None),
        }
    }
    fn public_columns(&self, verbose: bool) -> (Option<Self::M<'_>>, Option<Vec<String>>) {
        match self.0.lookup_type() {
            LookupType::Public => {
                let (matrix, log) = self.0.lookup_matrix(verbose);
                (Some(matrix), log)
            }
            _ => (None, None),
        }
    }
    fn num_private_columns(&self) -> usize {
        match self.0.lookup_type() {
            LookupType::Private => self.0.width(),
            _ => 0,
        }
    }
    fn num_preprocessed_columns(&self) -> usize {
        match self.0.lookup_type() {
            LookupType::Preprocessed => self.0.width(),
            _ => 0,
        }
    }
    fn num_public_columns(&self) -> usize {
        match self.0.lookup_type() {
            LookupType::Public => self.0.width(),
            _ => 0,
        }
    }

    fn fields_for_receive(&self, i: usize) -> SmallVec<[VirtualPairCol<F>; SMALLVEC_SIZE]> {
        debug_assert!(i == 0, "Only one receive is supported for a LookupTable");
        (0..self.0.width())
            .map(|j| match self.0.lookup_type() {
                LookupType::Preprocessed => VirtualPairCol::single_preprocessed(j),
                LookupType::Private => VirtualPairCol::single_main(j),
                LookupType::Public => VirtualPairCol::single_public(j),
            })
            .collect()
    }

    fn height(&self) -> usize {
        self.0.height()
    }

    fn name(&self) -> String {
        self.0.name()
    }
}

impl<F: PrimeField, L: MultiLookupTable<F>> ChipTraceHeight for LookupChip<L, F> {
    fn trace_height(&self) -> u32 {
        self.table.height() as u32
    }
}

#[derive(Clone, Default)]
pub struct LookupChip<L, F>
where
    F: PrimeField,
    L: MultiLookupTable<F>,
{
    pub table: L,
    pub counts: BTreeMap<usize, BTreeMap<SmallVec<[F; SMALLVEC_SIZE]>, u32>>,
    pub _phantom: PhantomData<F>,
}

impl<L: MemoryFootprint + MultiLookupTable<F>, F: PrimeField> MemoryFootprint for LookupChip<L, F> {
    fn memory_footprint(&self) -> usize {
        // Vec's heap allocation
        self.table.memory_footprint() + self.counts.memory_footprint()
    }
}

impl<L, F> LookupChip<L, F>
where
    F: PrimeField,
    L: MultiLookupTable<F>,
{
    pub fn set_table(&mut self, table: L) {
        self.table = table;
    }

    fn vector_multi_lookup(
        &mut self,
        fields: SmallVec<[F; SMALLVEC_SIZE]>,
        log: bool,
        receive_index: usize,
    ) {
        debug_assert!(receive_index < self.table.num_receives());
        if log {
            self.counts
                .entry(receive_index)
                .or_default()
                .entry(fields)
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }
    }
    fn scalar_multi_lookup(&mut self, value: F, log: bool, receive_index: usize) {
        self.vector_multi_lookup(smallvec![value], log, receive_index);
    }
}

impl<M, SC, L> Chip<M, SC> for LookupChip<L, SC::Val>
where
    M: MachineWithMultiLookupChip<SC::Val, L> + MachineWithRangeBus8<SC::Val>,
    SC: StarkConfig,
    L: MultiLookupTable<SC::Val> + Sync,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        format!("LookupChip for {}", self.table.name())
    }

    fn generate_main_trace(
        &self,
        _machine: &M,
        verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>) {
        let height = self.table.height();

        let (private_table, private_table_prints) = self.table.private_columns(verbose);
        let num_private_cols = self.table.num_private_columns();

        // we only need the public and preprocessed table to calculate the lookup rows to compute counts:
        // we don't need to print them here.
        let (public_table, _) = self.table.public_columns(false);
        let public_width = self.table.num_public_columns();
        let (preprocessed_table, _) = self.table.preprocessed_columns(false);
        let preprocessed_width = self.table.num_preprocessed_columns();

        debug_assert_eq!(
            preprocessed_width,
            preprocessed_table.as_ref().map_or(0, |t| t.width())
        );
        debug_assert_eq!(public_width, public_table.as_ref().map_or(0, |t| t.width()));
        debug_assert_eq!(
            num_private_cols,
            private_table.as_ref().map_or(0, |t| t.width())
        );

        let width = num_private_cols + self.table.num_receives();

        let row_from_trace_opts = |n, trace_opt: &Option<L::M<'_>>| {
            SmallVec::<[SC::Val; SMALLVEC_SIZE]>::from_vec(
                trace_opt
                    .as_ref()
                    .map(|trace| trace.row_slice(n).to_vec())
                    .unwrap_or_default(),
            )
        };

        let rows = (0..height)
            .into_par_iter()
            .map(|n| {
                let (private_row, public_row, preprocessed_row) = (
                    row_from_trace_opts(n, &private_table),
                    row_from_trace_opts(n, &public_table),
                    row_from_trace_opts(n, &preprocessed_table),
                );
                let mut argument_row = vec![SC::Val::zero(); self.table.num_receives()];
                argument_row
                    .iter_mut()
                    .enumerate()
                    .for_each(|(index, entry)| {
                        let lookup_vec: SmallVec<[SC::Val; SMALLVEC_SIZE]> = self
                            .table
                            .fields_for_receive(index)
                            .into_iter()
                            .map(|field| field.apply(&preprocessed_row, &public_row, &private_row))
                            .collect::<SmallVec<_>>();
                        if let Some(count) = self
                            .counts
                            .get(&index)
                            .and_then(|counts_for_index| counts_for_index.get(&lookup_vec))
                        {
                            *entry = SC::Val::from_canonical_u32(*count)
                        }
                    });
                (private_row, argument_row)
            })
            .collect::<Vec<_>>();

        let log = if verbose {
            let mut log_prints = Vec::with_capacity(height);
            let private_prints = private_table_prints.unwrap_or(Vec::with_capacity(height));
            for (index, ((_private_row, argument_row), private_row_print)) in
                rows.iter().zip(private_prints).enumerate()
            {
                let mut row_printed = String::new();
                if (argument_row != &vec![SC::Val::zero(); self.table.num_receives()])
                    || !private_row_print.is_empty()
                {
                    row_printed = format!("Main row {index} for {}: ", self.table.name());
                    row_printed.push_str(&private_row_print);
                    row_printed.push_str(&format!("Receive counts: {:?}:", argument_row));
                }
                log_prints.push(row_printed);
            }
            Some(log_prints)
        } else {
            None
        };

        let mut values = rows
            .into_iter()
            .flat_map(|(private_row, argument_row)| private_row.into_iter().chain(argument_row))
            .collect::<Vec<_>>();

        debug_assert_eq!(values.len() % width, 0);
        values.resize(height.next_power_of_two() * width, SC::Val::zero());
        (Some(RowMajorMatrix::new(values, width)), log)
    }

    fn main_width(&self) -> usize {
        self.table.num_private_columns() + self.table.num_receives()
    }

    fn generate_public_values(&self, verbose: bool) -> (Option<Self::Public>, Option<Vec<String>>) {
        if let (Some(public_columns), log_opt) = self.table.public_columns(verbose) {
            let table = public_columns.to_row_major_matrix();
            let width = table.width();
            let height = table.height();
            let mut values = table.values;
            debug_assert_eq!(values.len() % width, 0);
            values.resize(height.next_power_of_two() * width, SC::Val::zero());
            (
                Some(PublicTrace::PublicMatrix(RowMajorMatrix::new(
                    values, width,
                ))),
                log_opt,
            )
        } else {
            (None, None)
        }
    }

    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let mult_column =
            |index| VirtualPairCol::single_main(index + self.table.num_private_columns());
        (0..self.table.num_receives())
            .map(|index| {
                let fields = self.table.fields_for_receive(index).to_vec();
                Interaction {
                    fields,
                    count: mult_column(index),
                    argument_index: machine.lookup_chip_bus(index),
                }
            })
            .collect()
    }

    fn get_preprocessed_trace(
        &self,
        verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>) {
        if let (Some(preprocessed_columns), log_opt) = self.table.preprocessed_columns(verbose) {
            let table = preprocessed_columns.to_row_major_matrix();
            let width = table.width();
            let height = table.height();
            let mut values = table.values;
            debug_assert_eq!(values.len() % width, 0);
            values.resize(height.next_power_of_two() * width, SC::Val::zero());
            (Some(RowMajorMatrix::new(values, width)), log_opt)
        } else {
            (None, None)
        }
    }
}

impl<M, SC, L> ChipWithPersistence<M, SC> for LookupChip<L, SC::Val>
where
    M: MachineWithMultiLookupChip<SC::Val, L> + MachineWithRangeBus8<SC::Val>,
    SC: StarkConfig,
    L: MultiLookupTable<SC::Val> + Sync,
{
}

pub trait MachineWithMultiLookupChip<F: PrimeField, L: MultiLookupTable<F>>: Machine<F> {
    fn lookup_chip(&self) -> &LookupChip<L, F>;
    fn lookup_chip_mut(&mut self) -> &mut LookupChip<L, F>;

    fn vector_multi_lookup(
        &mut self,
        fields: SmallVec<[F; SMALLVEC_SIZE]>,
        log: bool,
        receive_index: usize,
    ) {
        self.lookup_chip_mut()
            .vector_multi_lookup(fields, log, receive_index);
    }
    fn scalar_multi_lookup(&mut self, value: F, log: bool, receive_index: usize) {
        self.lookup_chip_mut()
            .scalar_multi_lookup(value, log, receive_index);
    }

    fn lookup_chip_bus(&self, receive_index: usize) -> BusArgument;
}

pub trait MachineWithLookupChip<F: PrimeField, L: LookupTable<F>>: Machine<F> {
    fn lookup_chip(&self) -> &LookupChip<MultiLookupTableWrapper<L>, F>;
    fn lookup_chip_mut(&mut self) -> &mut LookupChip<MultiLookupTableWrapper<L>, F>;

    fn vector_lookup(&mut self, fields: SmallVec<[F; SMALLVEC_SIZE]>, log: bool) {
        self.lookup_chip_mut().vector_multi_lookup(fields, log, 0);
    }
    fn scalar_lookup(&mut self, value: F, log: bool) {
        self.lookup_chip_mut().scalar_multi_lookup(value, log, 0);
    }

    fn lookup_chip_bus(&self) -> BusArgument;
}

impl<F, M, L> MachineWithMultiLookupChip<F, MultiLookupTableWrapper<L>> for M
where
    F: PrimeField,
    M: MachineWithLookupChip<F, L>,
    L: LookupTable<F>,
{
    fn lookup_chip(&self) -> &LookupChip<MultiLookupTableWrapper<L>, F> {
        self.lookup_chip()
    }
    fn lookup_chip_mut(&mut self) -> &mut LookupChip<MultiLookupTableWrapper<L>, F> {
        self.lookup_chip_mut()
    }
    fn lookup_chip_bus(&self, receive_index: usize) -> BusArgument {
        debug_assert_eq!(receive_index, 0);
        self.lookup_chip_bus()
    }
    fn vector_multi_lookup(
        &mut self,
        fields: SmallVec<[F; SMALLVEC_SIZE]>,
        log: bool,
        receive_index: usize,
    ) {
        assert_eq!(
            receive_index, 0,
            "Only one receive is supported for a LookupTable"
        );
        self.vector_lookup(fields, log);
    }
}
