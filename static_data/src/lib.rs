#![no_std]

extern crate alloc;

use crate::columns::{
    StaticDataLookupCols, NUM_STATIC_DATA_LOOKUP_COLS, STATIC_DATA_LOOKUP_COL_MAP,
};
use alloc::{collections::BTreeMap, format, string::String, vec, vec::Vec};
use core::{borrow::Borrow, iter, mem::transmute};
use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use valida_bus::{MachineWithMemBus, MachineWithPersistentMemBus};
use valida_machine::{
    Chip, ChipTraceHeight, ChipWithPersistence, Interaction, Machine, MachineRuntime,
    MemoryAccessTimestamp, MemoryRecord, PublicTrace, RunningMachine, StarkConfig,
    StorageBackendTrait, Word,
};
use valida_memory_footprint::MemoryFootprint;

pub mod columns;
pub mod stark;

#[derive(Clone, Copy, Debug, Default)]
pub enum StaticDataChipType {
    #[default]
    Public,
    Preprocessed,
}

impl MemoryFootprint for StaticDataChipType {
    fn memory_footprint(&self) -> usize {
        core::mem::size_of::<Self>()
    }
}

#[derive(Default, Clone, Debug)]
pub struct StaticDataChip {
    chip_type: StaticDataChipType,
    cells: BTreeMap<u32, Word<u8>>,
}

impl MemoryFootprint for StaticDataChip {
    fn memory_footprint(&self) -> usize {
        self.chip_type.memory_footprint() + self.cells.memory_footprint()
    }
}

impl StaticDataChip {
    fn cell_to_row<F: PrimeField32>(
        (addr, value): (&u32, &Word<u8>),
    ) -> [F; NUM_STATIC_DATA_LOOKUP_COLS] {
        let mut row = [F::zero(); NUM_STATIC_DATA_LOOKUP_COLS];
        let cols: &mut StaticDataLookupCols<F> = unsafe { transmute(&mut row) };
        cols.addr = F::from_canonical_u32(*addr);
        cols.value = value.transform(F::from_canonical_u8);
        cols.is_real = F::one();

        row
    }

    pub fn cells_to_table<F: PrimeField32>(
        cells: &BTreeMap<u32, Word<u8>>,
        verbose: bool,
    ) -> (RowMajorMatrix<F>, Option<Vec<String>>) {
        let n = cells.len();

        let mut values = cells
            .par_iter()
            .flat_map(|cell| Self::cell_to_row(cell))
            .collect::<Vec<_>>();

        let log = if verbose {
            let mut log_prints = Vec::with_capacity(n);
            for (i, row) in values.chunks(NUM_STATIC_DATA_LOOKUP_COLS).enumerate() {
                let cols: &StaticDataLookupCols<F> = row[..].borrow();
                log_prints.push(format!("StaticData row {i}: {:?}", cols));
            }
            Some(log_prints)
        } else {
            None
        };

        // Pad the ROM to a power of two.
        values.resize(
            n.next_power_of_two() * NUM_STATIC_DATA_LOOKUP_COLS,
            F::zero(),
        );
        (
            RowMajorMatrix::new(values, NUM_STATIC_DATA_LOOKUP_COLS),
            log,
        )
    }

    pub fn chip_type(&self) -> StaticDataChipType {
        self.chip_type
    }

    pub fn new(table_type: StaticDataChipType) -> Self {
        StaticDataChip {
            chip_type: table_type,
            cells: BTreeMap::default(),
        }
    }

    pub fn load(&mut self, cells: BTreeMap<u32, Word<u8>>, chip_type: StaticDataChipType) {
        assert!(self.cells.is_empty(), "Static data table already loaded");
        self.chip_type = chip_type;
        self.cells = cells;
    }

    pub fn set_type(&mut self, chip_type: StaticDataChipType) {
        self.chip_type = chip_type;
    }

    pub fn write(&mut self, address: u32, value: Word<u8>) {
        assert!(
            !self.cells.contains_key(&address),
            "Address already written"
        );
        self.cells.insert(address, value);
    }

    pub fn get_cells(&self) -> BTreeMap<u32, Word<u8>> {
        self.cells.clone()
    }
}

pub trait MachineWithStaticDataChip<F: PrimeField>: Machine<F> {
    fn static_data(&self) -> &StaticDataChip;
    fn static_data_mut(&mut self) -> &mut StaticDataChip;
    fn initialize_memory(state: &mut RunningMachine<'_, F, Self>) {
        let static_data = state.machine.static_data().get_cells();
        let initial_state = static_data.iter().map(|(&addr, &value)| {
            (
                addr,
                MemoryRecord {
                    value,
                    last_accessed: MemoryAccessTimestamp::Static,
                },
            )
        });
        for (addr, record) in initial_state {
            state.runtime.memory_backend_mut().set(addr, record);
        }
    }
}

impl ChipTraceHeight for StaticDataChip {
    fn trace_height(&self) -> u32 {
        self.cells.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for StaticDataChip
where
    SC: StarkConfig,
    M: Machine<SC::Val>,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        let type_str = match self.chip_type {
            StaticDataChipType::Public => "Public",
            StaticDataChipType::Preprocessed => "Preprocessed",
        };
        format!("StaticData ({type_str})")
    }

    /// There are no private columns needed in this trace
    fn generate_main_trace(
        &self,
        _machine: &M,
        _verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>) {
        (None, None)
    }

    fn generate_public_values(&self, verbose: bool) -> (Option<Self::Public>, Option<Vec<String>>) {
        match self.chip_type {
            StaticDataChipType::Public => {
                // If no static data, return None. Matches expectation for the verifier
                let (trace, log) = Self::cells_to_table(&self.cells, verbose);
                (Some(PublicTrace::from_matrix(trace)), log)
            }
            _ => (None, None),
        }
    }

    fn get_preprocessed_trace(
        &self,
        verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>) {
        match self.chip_type() {
            StaticDataChipType::Preprocessed => {
                let (table, log) = Self::cells_to_table(&self.cells, verbose);
                (Some(table), log)
            }
            _ => (None, None),
        }
    }
}

impl<M, SC> ChipWithPersistence<M, SC> for StaticDataChip
where
    SC: StarkConfig,
    M: MachineWithPersistentMemBus<SC::Val>,
{
    /// Persistent send for the static data chip sends _all elements_ of the static data chip. These are
    /// received in the ephemeral memory chip in segment 0, by having the ephemeral memory chip read
    /// the static data chip cells to add all of them to its main trace (see ephemeral memory chip's
    /// `generate_main_trace` function).
    fn persistent_sends(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let make_column = |i: usize| match self.chip_type() {
            StaticDataChipType::Public => VirtualPairCol::single_public(i),
            StaticDataChipType::Preprocessed => VirtualPairCol::single_preprocessed(i),
        };

        let addr = make_column(STATIC_DATA_LOOKUP_COL_MAP.addr);
        let value = STATIC_DATA_LOOKUP_COL_MAP.value.transform(make_column);
        let is_real = make_column(STATIC_DATA_LOOKUP_COL_MAP.is_real);
        let timestamp = VirtualPairCol::constant(MemoryAccessTimestamp::Static.as_scalar());
        let fields = vec![addr]
            .into_iter()
            .chain(value.into_iter_le())
            .chain(iter::once(timestamp))
            .collect();
        let receive = Interaction {
            fields,
            count: is_real,
            argument_index: machine.persistent_mem_bus(),
        };
        vec![receive]
    }
}
