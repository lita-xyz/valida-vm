//#![no_std]

extern crate alloc;

use crate::columns::{
    MemoryCols, AS_SCALAR_SEGMENT_OFFSET, MEM_COL_MAP, MEM_PUBLIC_VECTOR_MAP, NUM_MEM_COLS,
};
use alloc::{
    collections::BTreeMap,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::borrow::{Borrow, BorrowMut};
use core::hash::BuildHasherDefault;
use core::iter;
use core::mem::transmute;
use fxhash::FxHasher32;
use hashbrown::HashMap;
use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use smallvec::SmallVec;
use valida_bus::{
    MachineWithBytesBus, MachineWithMemBus, MachineWithPersistentMemBus, MachineWithRangeBus8,
};
use valida_bytes::{half_baby_bear_range_sends, BytesTable, MachineWithBytesChip};
use valida_lookups::MachineWithMultiLookupChip;
use valida_machine::{
    Chip, ChipTraceHeight, ChipWithPersistence, Interaction, Machine, MachineRuntime,
    MachineWithFinalMemoryState, MemoryAccessTimestamp, MemoryBackendTrait, MemoryRecord,
    PublicTrace, RunningMachine, SegmentMachine, StarkConfig, StorageBackendTrait, Word,
};
use valida_memory_footprint::MemoryFootprint;
use valida_static_data::MachineWithStaticDataChip;
use valida_util::batch_multiplicative_inverse_allowing_zero;

use std::collections::HashSet;

use bitvec::prelude::BitVec;

pub mod columns;
pub mod stark;

// #[derive(Copy, Clone, Debug, Eq, PartialEq)]
// pub enum MemoryTimestamp {
//     Static,
//     Ordinary(u32),
// }

// impl PartialOrd for MemoryTimestamp {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         match (self, other) {
//             (MemoryTimestamp::Static, MemoryTimestamp::Static) => Some(Ordering::Equal),
//             (MemoryTimestamp::Static, MemoryTimestamp::Ordinary(_clk)) => Some(Ordering::Less),
//             (MemoryTimestamp::Ordinary(_clk), MemoryTimestamp::Static) => Some(Ordering::Greater),
//             (MemoryTimestamp::Ordinary(clk1), MemoryTimestamp::Ordinary(clk2)) => {
//                 Some(clk1.cmp(clk2))
//             }
//         }
//     }
// }
// impl Ord for MemoryTimestamp {
//     fn cmp(&self, other: &Self) -> Ordering {
//         self.partial_cmp(other).unwrap()
//     }
// }
// impl Into<u32> for MemoryTimestamp {
//     fn into(self) -> u32 {
//         match self {
//             MemoryTimestamp::Static => 0,
//             MemoryTimestamp::Ordinary(clk) => clk + 1,
//         }
//     }
// }

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PersistentMemoryTimestamp<T> {
    #[default]
    ZeroInitialized,
    Static,
    PriorSegment(T),
}

#[derive(Debug, Clone, Default, Copy, Eq, PartialEq)]
pub struct PersistentMemoryRecord {
    pub value: Word<u8>,
    pub last_accessed: PersistentMemoryTimestamp<u32>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Operation {
    // if the first operation to an address is a write,
    // we log a dummy read to record the cell's prior value
    DummyRead(u32, MemoryRecord),
    Read(u32, MemoryRecord),
    Write(u32, Word<u8>),
}

impl MemoryFootprint for Operation {
    fn memory_footprint(&self) -> usize {
        match self {
            Operation::DummyRead(a, b) => a.memory_footprint() + b.memory_footprint(),
            Operation::Read(a, b) => a.memory_footprint() + b.memory_footprint(),
            Operation::Write(a, b) => a.memory_footprint() + b.memory_footprint(),
        }
    }
}

impl Operation {
    pub fn get_address(&self) -> u32 {
        match self {
            Operation::DummyRead(addr, _) => *addr,
            Operation::Read(addr, _) => *addr,
            Operation::Write(addr, _) => *addr,
        }
    }
    pub fn get_value(&self) -> Word<u8> {
        match self {
            Operation::DummyRead(_, MemoryRecord { value, .. }) => *value,
            Operation::Read(_, MemoryRecord { value, .. }) => *value,
            Operation::Write(_, value) => *value,
        }
    }
    pub fn get_timestamp(&self) -> MemoryAccessTimestamp<u32> {
        match self {
            Operation::DummyRead(_, MemoryRecord { last_accessed, .. }) => *last_accessed,
            Operation::Read(_, MemoryRecord { last_accessed, .. }) => *last_accessed,
            Operation::Write(_, _) => MemoryAccessTimestamp::ThisSegment,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryBackendType {
    Unordered,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryBackend<R> {
    Unordered(HashMap<u32, R, BuildHasherDefault<FxHasher32>>),
}

pub type MemoryChipBackend = MemoryBackend<MemoryRecord>;
pub type PersistentMemoryState = BTreeMap<u32, PersistentMemoryRecord>;

impl<R> Default for MemoryBackend<R> {
    fn default() -> Self {
        Self::Unordered(Default::default())
    }
}

#[derive(Clone, Debug)]
pub enum Iter<'a, R> {
    Ordered(alloc::collections::btree_map::Iter<'a, u32, R>),
    Unordered(hashbrown::hash_map::Iter<'a, u32, R>),
    Array {
        next_addr: u32,
        num_remaining: usize,
        bv: &'a BitVec<usize>,
        v: &'a [R],
    },
}

impl<'a, R> Iterator for Iter<'a, R> {
    type Item = (u32, &'a R);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Iter::Ordered(it) => it.next().map(|(k, v)| (*k, v)),
            Iter::Unordered(it) => it.next().map(|(k, v)| (*k, v)),
            Iter::Array {
                next_addr,
                num_remaining,
                bv,
                v,
            } => {
                if *num_remaining == 0 {
                    None
                } else {
                    let mut dst = (*next_addr as usize) >> 2;
                    dst += 1;
                    while !bv[dst] {
                        dst += 1;
                        debug_assert!(dst < bv.len(), "index should not extend past end of buffer");
                    }
                    *next_addr = (dst << 2) as u32;
                    *num_remaining -= 1;
                    Some((*next_addr, &v[dst]))
                }
            }
        }
    }
}

impl<R> ExactSizeIterator for Iter<'_, R> {
    fn len(&self) -> usize {
        match self {
            Iter::Ordered(it) => it.len(),
            Iter::Unordered(it) => it.len(),
            Iter::Array { num_remaining, .. } => *num_remaining,
        }
    }
}

impl<R> MemoryBackend<R> {
    pub fn new_with_backend(backend_type: MemoryBackendType) -> Self {
        match backend_type {
            MemoryBackendType::Unordered => Self::Unordered(Default::default()),
        }
    }

    pub fn transform<T>(&self, f: impl Fn(&R) -> T) -> MemoryBackend<T> {
        match self {
            MemoryBackend::Unordered(hm) => {
                MemoryBackend::Unordered(hm.iter().map(|(k, v)| (*k, f(v))).collect())
            }
        }
    }

    pub fn get(&self, addr: &u32) -> Option<&R> {
        match self {
            MemoryBackend::Unordered(hm) => hm.get(addr),
        }
    }

    pub fn insert(&mut self, addr: u32, record: R) {
        match self {
            MemoryBackend::Unordered(hm) => {
                hm.insert(addr, record);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            MemoryBackend::Unordered(hm) => hm.is_empty(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            MemoryBackend::Unordered(hm) => hm.len(),
        }
    }

    pub fn iter(&self) -> Iter<R> {
        match self {
            MemoryBackend::Unordered(hm) => Iter::Unordered(hm.iter()),
        }
    }
}

/// The maximum number of memory operations that occur in an instruction.
/// This is used to size a SmallVec to avoid a heap allocation per clock cycle.
const INSTR_MAX_MEM_OPS: usize = 4;

#[derive(Default)]
pub struct MemoryChip {
    pub operations: BTreeMap<u32, SmallVec<[Operation; INSTR_MAX_MEM_OPS]>>,
    pub segment_number: usize,
}

impl MemoryFootprint for MemoryChip {
    fn memory_footprint(&self) -> usize {
        self.operations.memory_footprint() + self.segment_number.memory_footprint()
    }
}

pub trait MachineWithMemoryChip<F: PrimeField>:
    Machine<F> + MachineWithBytesChip<F> + SegmentMachine<F>
{
    fn mem(&self) -> &MemoryChip;
    fn mem_mut(&mut self) -> &mut MemoryChip;
    fn increment_memory_operation_count(&mut self, count: u32);

    fn read(state: &mut RunningMachine<'_, F, Self>, clk: u32, address: u32) -> Word<u8> {
        let log = state.machine.log_enabled();

        // Attempt to get the memory record from the current segment's memory backend
        let record: MemoryRecord = state
            .runtime
            .memory_backend()
            .get(&address)
            .copied()
            .unwrap_or_default();

        // Insert the new operation
        let new_record = MemoryRecord {
            value: record.value,
            last_accessed: MemoryAccessTimestamp::ThisSegment,
        };
        // Update the (global) memory backend with the new record
        state.runtime.memory_backend_mut().set(address, new_record);

        if log {
            // Record the read operation
            state
                .machine
                .mem_mut()
                .operations
                .entry(clk)
                .or_default()
                .push(Operation::Read(address, record));
            // Check the address is within the valid range for the field
            state.machine.check_half_baby_bear_range(&address.into());
        }

        // Always increment memory operation count for each read, regardless of log setting
        state.machine.increment_memory_operation_count(1);

        record.value
    }

    fn write(state: &mut RunningMachine<'_, F, Self>, clk: u32, address: u32, value: Word<u8>) {
        let log = state.machine.log_enabled();
        let mut dummy_read_addresses = Vec::new();
        let mut memory_ops_count = 0;

        // Check the global memory backend to see if the cell has been initialized
        let MemoryRecord {
            value: old_value,
            last_accessed,
        } = state
            .runtime
            .memory_backend()
            .get(&address)
            .copied()
            .unwrap_or_default();

        // Count memory operations consistently regardless of log setting
        if !matches!(last_accessed, MemoryAccessTimestamp::ThisSegment) {
            memory_ops_count += 1; // Count the dummy read
        }
        memory_ops_count += 1; // Count the write operation

        if log {
            let operations = &mut state.machine.mem_mut().operations;

            // If the cell is uninitialized for this execution segment, log a dummy read
            // operation to record the cell's prior value and last access time.
            if !matches!(last_accessed, MemoryAccessTimestamp::ThisSegment) {
                operations
                    .entry(clk)
                    .or_default()
                    .push(Operation::DummyRead(
                        address,
                        MemoryRecord {
                            value: old_value,
                            last_accessed,
                        },
                    ));
                dummy_read_addresses.push(address);
            }

            // Insert the new write operation into the current segment's MemoryChip operations
            operations
                .entry(clk)
                .or_default()
                .push(Operation::Write(address, value));

            // Range check the address to the valid range in the field
            let word: Word<u8> = address.into();
            state.machine.check_half_baby_bear_range(&word);
            for addr in dummy_read_addresses {
                state.machine.check_half_baby_bear_range(&addr.into());
            }
        }

        // Always increment memory operation count, regardless of log setting
        state
            .machine
            .increment_memory_operation_count(memory_ops_count);

        // Update the (global) memory backend with the new record, and mark that
        // we accessed this address in this segment.
        state.runtime.memory_backend_mut().set(
            address,
            MemoryRecord {
                value,
                last_accessed: MemoryAccessTimestamp::ThisSegment,
            },
        );
    }

    /// At the end of execution, sort the memory operations and update the
    /// range check counter for the `diff` values.
    fn suspend_memory_state(state: &mut RunningMachine<'_, F, Self>) {
        let machine = &mut state.machine;
        if machine.log_enabled() {
            // First, flatten all operations into a list of (clock cycle, operation) pairs
            let mut ops = machine
                .mem()
                .operations
                .par_iter()
                .map(|(clk, ops)| {
                    ops.iter()
                        .map(|op| (*clk, *op))
                        .collect::<Vec<(u32, Operation)>>()
                })
                .flatten()
                .collect::<Vec<_>>();

            if ops.len() == 0 {
                // nothing to do if no operations. This can happen in a multi segment machine, if e.g.
                // the last segment just happens to only contain the STOP instruction (or any that would
                // not lead to memory operations)
                return;
            }

            // Then, sort the operations by address, then by clock cycle.
            ops.sort_by_key(|(timestamp, op)| (op.get_address(), *timestamp));

            // Compute the diffs for pairs of subsequent operations
            let mut diffs_and_addrs = ops
                .par_windows(2)
                .map(|window| {
                    let ((clk1, op1), (clk2, op2)) = (window[0], window[1]);
                    let addr2 = op2.get_address();
                    let addr1 = op1.get_address();

                    let diff = if addr1 == addr2 {
                        clk2 - clk1
                    } else {
                        addr2 - addr1
                    };
                    (diff, addr1)
                })
                .collect::<Vec<_>>();
            // include the diff of 1 for the last row
            diffs_and_addrs.push((1, ops.last().unwrap().1.get_address()));
        }

        // Finally, update the `MemoryAccessTimestamp` for each memory record that was accessed in this
        // segment. We modify their timestamp from `ThisSegment` to `PriorSegment(segment_number)`.
        for (_addr, MemoryRecord { last_accessed, .. }) in
            state.runtime.memory_backend_mut().iter_mut()
        {
            if let MemoryAccessTimestamp::ThisSegment = last_accessed {
                *last_accessed = MemoryAccessTimestamp::PriorSegment(machine.segment_number());
            }
        }
    }
}

impl MemoryChip {
    pub fn new(segment_number: usize) -> Self {
        Self {
            operations: BTreeMap::new(),
            segment_number,
        }
    }

    pub fn get_sorted_operations<F: PrimeField32>(
        &self,
        final_memory_state: &HashSet<(u32, Word<u8>, u32)>,
        static_data: &BTreeMap<u32, Word<u8>>,
    ) -> Vec<[F; NUM_MEM_COLS]> {
        let ops = self
            .operations
            .par_iter()
            .map(|(clk, ops)| {
                ops.iter()
                    .map(|op| (*clk, *op))
                    .collect::<Vec<(u32, Operation)>>()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let mut rows = ops
            .par_iter()
            .map(|(clk, op)| self.op_to_row(*clk, *op, final_memory_state))
            .collect::<Vec<_>>();

        if self.segment_number == 0 {
            // Now add all the static data *IFF* this is the first segment
            rows.extend(
                static_data
                    .par_iter()
                    .map(|(addr, val)| self.static_data_to_row(*addr, *val))
                    .collect::<Vec<_>>(),
            );
        }

        // TODO: also sort by `is_static_write` to make sure all static writes always appear *before*
        // any real accesses? They are at clock 0, but a read could also be at clock zero in theory,
        // in which case the order would be random. Issue is that adding it would put
        // `is_static_write == 0` *before* the 1 case, which is not what we want
        rows.sort_by_key(|row| (row[MEM_COL_MAP.addr], row[MEM_COL_MAP.clk]));
        self.set_sorted_trace_cols(&mut rows);
        rows
    }
}

// pub fn load_state(&mut self, state: PersistentMemoryState) {
//     for (addr, record) in state.machine.into_iter() {
//         // TODO: optimize allocations here
//         self.cells.insert(
//             addr,
//             MemoryRecord {
//                 value: record.value,
//                 last_accessed: MemoryAccessTimestamp::Persistent(record.last_accessed),
//             },
//         );
//     }
// }

// pub fn save_state(&self, state: &mut PersistentMemoryState) {
//     self.cells.iter().for_each(|(addr, &record)| {
//         state.machine.insert((
//             addr,
//             PersistentMemoryRecord {
//                 value: record.value,
//                 last_accessed: match record.last_accessed {
//                     MemoryAccessTimestamp::ThisSegment => {
//                         PersistentMemoryTimestamp::PriorSegment(self.segment_number as u32)
//                     }
//                     MemoryAccessTimestamp::Persistent(persistent_timestamp) => {
//                         persistent_timestamp
//                     }
//                 },
//             },
//         ));
//     })
// }
//}

/// Helper function to add receives for `diff_bytes` range checks
///
/// Note that the corresponding sends are added in `Chip::global_sends`
/// The corresponding receives are added here, and require special handling
/// due to the lookup chip requiring mutable access to the `Machine`, and
/// requiring that all memory operations have been generated.
pub fn add_diff_bytes_receives<F: PrimeField32, M>(machine: &mut M)
where
    M: MachineWithRangeBus8<F>
        + MachineWithBytesChip<F>
        + MachineWithMultiLookupChip<F, BytesTable>
        + MachineWithMemoryChip<F>
        + MachineWithFinalMemoryState<F>
        + MachineWithStaticDataChip<F>,
{
    // NOTE: `add_diff_bytes_receives` is called after execution of a `BasicMachine` .
    // (i.e. a single segment) finishes. This means the final memory state here is still
    // empty. As a result the trace built in `get_sorted_operations` is slightly different
    // than the final trace. The columns `is_final` and `skip_persistent_send` will always
    // be zero. They are however not used for anything related to the range checks.
    let final_mem_state = machine.get_final_memory_state();
    let static_data = machine.static_data().get_cells();
    let memory = machine.mem_mut();

    // Iterate through the sorted memory operations, add receives for `diff_bytes`

    for row in memory.get_sorted_operations::<F>(&final_mem_state, &static_data) {
        let cols: &MemoryCols<F> = row.as_ref().borrow();
        if cols.is_read == F::one()
            || cols.is_write == F::one()
            || cols.is_dummy_read == F::one()
            || cols.is_static_write == F::one()
        {
            // Check that the difference between successive addresses/clocks is within range
            machine.check_half_baby_bear_range(&cols.diff_bytes.to_u8_word());
        }
    }
}

impl ChipTraceHeight for MemoryChip {
    fn trace_height(&self) -> u32 {
        self.operations.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for MemoryChip
where
    M: MachineWithMemBus<SC::Val>
        + MachineWithBytesBus<SC::Val>
        + MachineWithBytesChip<SC::Val>
        + MachineWithRangeBus8<SC::Val>
        + MachineWithFinalMemoryState<SC::Val>
        + MachineWithStaticDataChip<SC::Val>,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "Memory".to_string()
    }

    fn generate_main_trace(
        &self,
        machine: &M,
        verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>) {
        let mut rows = self.get_sorted_operations::<SC::Val>(
            &machine.get_final_memory_state(),
            &machine.static_data().get_cells(),
        );

        let log = if verbose {
            let mut log_prints = Vec::with_capacity(rows.len());
            for (i, row) in rows.iter().enumerate() {
                let cols: &MemoryCols<SC::Val> = unsafe { transmute(row) };
                log_prints.push(format!("Memory row {}: {:?}", i, cols));
            }
            Some(log_prints)
        } else {
            None
        };

        // Make sure the table length is a power of two
        let padding_row = [SC::Val::zero(); NUM_MEM_COLS];
        rows.resize(rows.len().next_power_of_two(), padding_row);

        let trace = RowMajorMatrix::new(
            rows.clone().into_iter().flatten().collect::<Vec<_>>(),
            NUM_MEM_COLS,
        );

        (Some(trace), log)
    }

    fn generate_public_values(&self, verbose: bool) -> (Option<Self::Public>, Option<Vec<String>>) {
        let log = if verbose {
            Some(vec![format!(
                "Public values for memory chip: segment number {}",
                self.segment_number
            )])
        } else {
            None
        };
        (
            Some(PublicTrace::from_vec(vec![SC::Val::from_canonical_usize(
                self.segment_number,
            )])),
            log,
        )
    }

    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let is_read: VirtualPairCol<SC::Val> = VirtualPairCol::single_main(MEM_COL_MAP.is_read);
        let clk = VirtualPairCol::single_main(MEM_COL_MAP.clk);
        let addr = VirtualPairCol::single_main(MEM_COL_MAP.addr);
        let value = MEM_COL_MAP.value.transform(VirtualPairCol::single_main);

        let mut fields = vec![is_read, clk, addr.clone()];
        fields.extend(value.clone().into_iter_le());

        let is_real = VirtualPairCol::sum_main(vec![MEM_COL_MAP.is_read, MEM_COL_MAP.is_write]);
        let cpu_receive = Interaction {
            fields,
            count: is_real,
            argument_index: machine.mem_bus(),
        };

        vec![cpu_receive]
    }

    fn global_sends(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let is_real_addr = VirtualPairCol::sum_main(vec![
            MEM_COL_MAP.is_read,
            MEM_COL_MAP.is_write,
            MEM_COL_MAP.is_dummy_read,
        ]);

        let is_real_diff = VirtualPairCol::sum_main(vec![
            MEM_COL_MAP.is_read,
            MEM_COL_MAP.is_write,
            MEM_COL_MAP.is_dummy_read,
            MEM_COL_MAP.is_static_write,
        ]);

        // Range check the address bytes to be under half of the baby bear field size
        // Static data excluded (there are no corresponding receives)
        // TODO: Check if we want to change that?
        let addr_range_sends = {
            let addr_bytes_cols = MEM_COL_MAP
                .addr_bytes
                .transform(VirtualPairCol::single_main);
            half_baby_bear_range_sends(machine, &addr_bytes_cols, is_real_addr)
        };

        // Range check the diff bytes to be under half of the baby bear field size
        // Static data included
        let diff_range_sends = {
            let diff_bytes_cols = MEM_COL_MAP
                .diff_bytes
                .transform(VirtualPairCol::single_main);
            half_baby_bear_range_sends(machine, &diff_bytes_cols, is_real_diff)
        };

        addr_range_sends
            .into_iter()
            .chain(diff_range_sends)
            .collect()
    }
}

impl<M, SC> ChipWithPersistence<M, SC> for MemoryChip
where
    SC: StarkConfig,
    M: MachineWithPersistentMemBus<SC::Val>
        + MachineWithMemBus<SC::Val>
        + MachineWithBytesBus<SC::Val>
        + MachineWithRangeBus8<SC::Val>
        + MachineWithBytesChip<SC::Val>
        + MachineWithFinalMemoryState<SC::Val>
        + MachineWithStaticDataChip<SC::Val>,
{
    /// NOTE: Persistend sends / receives terminology used for the persistent memory bus
    ///
    /// A persistent *send* is a send, which contains the *last value* written to a memory
    /// location at the end of a segment. It is denoted a 'send', because we think about it
    /// as sending the value over from segment `i` to segment `i+1`. All persistent sends
    /// for addresses that will be accessed again in a future segment will be received by
    /// this `MemoryChip`. All those sends, which are never accessed again in the entire
    /// program execution, are pruned by comparing with the final memory state at the end of
    /// execution and thus never produced (see `skip_persistent_send`).
    ///
    /// A persistent *receive* is a receive, which contains the *initial value* contained in a
    /// memory location. Either the very first time an address is read (across all segments)
    /// or for the first time within a segment. It is denoted a 'receive', because we either
    /// receive the value from segment `i-1` *or* from the static data chip, if we access
    /// an address written by the static data chip for the first time. Any access to a memory
    /// location for the very first time across all segments does _not_ produce a persistent
    /// receive; we recognize this situation and avoid the receive (`is_zero_initialized`).
    /// Further, we do *not* generate persistent receives for first accesses to static data
    /// cells that happen after the first segment (`skip_persistent_receive`). All static data
    /// cells are received in the *first* segment.
    ///
    /// TODO: Add example sends/receives
    fn persistent_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let addr_col = VirtualPairCol::single_main(MEM_COL_MAP.addr);
        let value_cols = MEM_COL_MAP.value.transform(VirtualPairCol::single_main);
        let persistent_read_set_receive = {
            let previous_segment = VirtualPairCol::single_main(MEM_COL_MAP.prior_timestamp);
            // We only receive if is_initial is true AND it's not zero-initialized
            // This will be 1 only when is_initial is 1 and is_zero_initialized is 0
            let receive_condition = VirtualPairCol::<SC::Val>::new_main(
                vec![
                    (MEM_COL_MAP.is_initial, SC::Val::one()),
                    (MEM_COL_MAP.is_zero_initialized, -SC::Val::one()),
                    (MEM_COL_MAP.skip_persistent_receive, -SC::Val::one()),
                ],
                SC::Val::zero(),
            );
            let fields = vec![addr_col.clone()]
                .into_iter()
                .chain(value_cols.clone().into_iter_le())
                .chain(iter::once(previous_segment))
                .collect();

            Interaction {
                fields,
                count: receive_condition,
                argument_index: machine.persistent_mem_bus(),
            }
        };
        vec![persistent_read_set_receive]
    }

    fn persistent_sends(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let addr_col = VirtualPairCol::single_main(MEM_COL_MAP.addr);
        let value_cols = MEM_COL_MAP.value.transform(VirtualPairCol::single_main);
        // We perform a send iff `skip_persistent_send` is zero. This will be the case if:
        // `addr_equal == 0 && is_final == 0`, i.e. not a padding row (would put `addr_equaly` possibly to 1)
        // and *NOT* the final access to this memory location (in which case no need to send).
        let is_final_col = VirtualPairCol::new_main(
            vec![
                (MEM_COL_MAP.skip_persistent_send, -SC::Val::one()),
                (MEM_COL_MAP.is_static_write, SC::Val::one()),
                (MEM_COL_MAP.is_read, SC::Val::one()),
                (MEM_COL_MAP.is_write, SC::Val::one()),
                (MEM_COL_MAP.is_dummy_read, SC::Val::one()),
            ],
            SC::Val::zero(),
        );

        // NOTE: We add `AS_SCALAR_SEGMENT_OFFSET == 3`, because the `last_accessesd.as_scalar()`
        // in `op_to_row` turns the `MemoryAccessTimestamp` into different "integer classes".
        // (0 zero init, 1 this segment, 2 static, >=3 segment).
        // 3 corresponds to segment index 0 and then just incrementing.
        // This is neeeded to get matching persistent sends & receives.
        let segment_number_col = VirtualPairCol::new_public(
            vec![(MEM_PUBLIC_VECTOR_MAP.segment_number, SC::Val::one())],
            SC::Val::from_canonical_u32(AS_SCALAR_SEGMENT_OFFSET as u32),
        );
        let fields = vec![addr_col]
            .into_iter()
            .chain(value_cols.into_iter_le())
            .chain(iter::once(segment_number_col))
            .collect();
        vec![Interaction {
            fields,
            count: is_final_col,
            argument_index: machine.persistent_mem_bus(),
        }]
    }
}

fn is_final_state(
    addr: u32,
    value: Word<u8>,
    segment: u32,
    final_memory_state: &HashSet<(u32, Word<u8>, u32)>,
) -> bool {
    final_memory_state.contains(&(addr, value, segment))
}

impl MemoryChip {
    fn op_to_row<F: PrimeField32>(
        &self,
        clk: u32,
        op: Operation,
        final_memory_state: &HashSet<(u32, Word<u8>, u32)>,
    ) -> [F; NUM_MEM_COLS] {
        let mut row = [F::zero(); NUM_MEM_COLS];
        let cols: &mut MemoryCols<F> = unsafe { transmute(&mut row) };

        cols.clk = F::from_canonical_u32(clk);

        // Segment number of this segment. Needed to check if operation is final memory op. Uses
        // `as_scalar` offset for segment numbers.
        let segment = self.segment_number + (AS_SCALAR_SEGMENT_OFFSET as usize);

        match op {
            Operation::DummyRead(
                addr,
                MemoryRecord {
                    value,
                    last_accessed,
                },
            ) => {
                debug_assert!(2 * (addr as u64) < F::ORDER_U32 as u64);
                cols.addr = F::from_canonical_u32(addr);
                cols.addr_bytes = Word::from(addr).transform(F::from_canonical_u8);
                cols.value = value.transform(F::from_canonical_u8);
                cols.prior_timestamp = last_accessed.as_scalar();
                cols.is_zero_initialized = F::from_bool(matches!(
                    last_accessed,
                    MemoryAccessTimestamp::ZeroInitialized
                ));
                cols.is_read = F::from_bool(false);
                cols.is_write = F::from_bool(false);
                cols.is_dummy_read = F::from_bool(true);
                // dummy reads are always the initial read to the address *unless* this address
                // contains static chip data (handled in `set_sorted_trace_cols`
                cols.is_initial = F::from_bool(true);

                // Determine if this is a final memory location (never used again)
                cols.is_final = F::from_bool(is_final_state(
                    addr,
                    value,
                    segment as u32,
                    final_memory_state,
                ));
            }
            Operation::Read(
                addr,
                MemoryRecord {
                    value,
                    last_accessed,
                },
            ) => {
                debug_assert!(2 * (addr as u64) < F::ORDER_U32 as u64);
                cols.addr = F::from_canonical_u32(addr);
                cols.addr_bytes = Word::from(addr).transform(F::from_canonical_u8);
                cols.value = value.transform(F::from_canonical_u8);
                cols.prior_timestamp = last_accessed.as_scalar();
                cols.is_zero_initialized = F::from_bool(matches!(
                    last_accessed,
                    MemoryAccessTimestamp::ZeroInitialized
                ));
                cols.is_read = F::from_bool(true);
                cols.is_write = F::from_bool(false);
                cols.is_dummy_read = F::from_bool(false);
                // is_initial is true if the last access was not in this segment
                cols.is_initial = F::from_bool(last_accessed.is_initial());

                // Determine if this is a final memory location (never used again)
                cols.is_final = F::from_bool(is_final_state(
                    addr,
                    value,
                    segment as u32,
                    final_memory_state,
                ));
            }
            Operation::Write(addr, value) => {
                debug_assert!(2 * (addr as u64) < F::ORDER_U32 as u64);
                cols.addr = F::from_canonical_u32(addr);
                cols.addr_bytes = Word::from(addr).transform(F::from_canonical_u8);
                cols.value = value.transform(F::from_canonical_u8);
                cols.prior_timestamp = MemoryAccessTimestamp::ThisSegment.as_scalar();
                cols.is_zero_initialized = F::from_bool(false);
                cols.is_read = F::from_bool(false);
                cols.is_write = F::from_bool(true);
                cols.is_dummy_read = F::from_bool(false);
                // if a write would be the initial operation at an address, a dummy read is inserted ahead of it
                cols.is_initial = F::from_bool(false);

                // Determine if this is a final memory location (never used again)
                cols.is_final = F::from_bool(is_final_state(
                    addr,
                    value,
                    segment as u32,
                    final_memory_state,
                ));
            }
        }
        row
    }

    fn static_data_to_row<F: PrimeField>(&self, addr: u32, value: Word<u8>) -> [F; NUM_MEM_COLS] {
        let mut row = [F::zero(); NUM_MEM_COLS];
        let cols: &mut MemoryCols<F> = unsafe { transmute(&mut row) };
        cols.clk = F::zero();
        cols.addr = F::from_canonical_u32(addr);
        cols.addr_bytes = Word::from(addr).transform(F::from_canonical_u8);
        cols.value = value.transform(F::from_canonical_u8);
        cols.is_write = F::from_bool(false);
        cols.is_read = F::from_bool(false);
        cols.is_dummy_read = F::from_bool(false);
        cols.is_static_write = F::from_bool(true);

        cols.prior_timestamp = MemoryAccessTimestamp::Static.as_scalar();
        cols.is_zero_initialized = F::from_bool(false);
        // if a write would be the initial operation at an address, a dummy read is inserted ahead of it
        cols.is_initial = F::from_bool(false);

        row
    }

    fn set_sorted_trace_cols<F: PrimeField + PrimeField32>(&self, rows: &mut [[F; NUM_MEM_COLS]]) {
        let n = rows.len();
        let mut is_initials = vec![F::zero(); n];
        // the first row should always have is_initial set
        if n > 0 {
            is_initials[0] = F::one()
        }
        let (diffs, addr_equals) = rows
            .par_windows(2)
            .zip(is_initials.par_iter_mut().skip(1))
            .map(move |(window, next_is_initial)| {
                let cols_local: &MemoryCols<F> = window[0][..].borrow();
                let cols_next: &MemoryCols<F> = window[1][..].borrow();
                // NOTE: This condition is *also* true, if we go from a static write to a memory cell
                // to a regular write or read. This is the one case where `is_dummy_read == true`
                // does *NOT* imply `is_initial == true`!
                if cols_local.addr == cols_next.addr {
                    (cols_next.clk - cols_local.clk, F::one())
                } else {
                    *next_is_initial = F::one();
                    (cols_next.addr - cols_local.addr, F::zero())
                }
            })
            .collect::<(Vec<_>, Vec<_>)>();

        // NOTE: `addr_equal` has length n-1 (rows.len() == n). Thus last row is not
        // covered by this.
        let skip_persistent_sends: Vec<F> = rows
            .iter()
            .zip(&addr_equals)
            .map(|(row, addr_equal)| {
                // If `addr_equal` is `true`, we must *NOT* send a `skip_persistent_send`.
                // If `is_final`   is `true`, we must *NOT* send a `skip_persistent_send`
                //    (to skip `persistent_receives` required from the PersistentMemoryChip)
                // If `is_static_write` is `true`, we must *NOT* send a `skip_persistent_send`
                // If `is_dummy_read`   is `true`, we must *NOT* send a `skip_persistent_send`
                // If `addr_equal` is `false`, we MIGHT send a `skip_persistent_send` (assuming `is_final` is `false`)
                // If `is_final`   is `false`, we MIGHT send a `skip_persistent_send` (assuming `addr_equal` is `false`)

                let cols: &MemoryCols<F> = row[..].borrow();
                let mut skip_persistent_send =
                    *addr_equal + cols.is_final + cols.is_static_write + cols.is_dummy_read;
                if skip_persistent_send > F::one() {
                    // if more than one is true, clamp to 1
                    skip_persistent_send = F::one();
                }
                skip_persistent_send
            })
            .collect();

        let diff_invs = batch_multiplicative_inverse_allowing_zero(diffs.clone());

        let n = rows.len();
        if n > 0 {
            rows.par_iter_mut()
                .take(n)
                .zip(diffs)
                .zip(diff_invs)
                .zip(addr_equals)
                .zip(is_initials.par_iter().take(n))
                .zip(skip_persistent_sends.clone())
                .for_each(
                    |(
                        ((((row, diff), diff_inv), addr_equal), is_initial),
                        skip_persistent_send,
                    )| {
                        let cols: &mut MemoryCols<F> = row[..].borrow_mut();
                        cols.diff = diff;
                        cols.diff_bytes = Word::from(cols.diff.as_canonical_u32())
                            .transform(F::from_canonical_u8);
                        cols.diff_inv = diff_inv;
                        cols.addr_equal = addr_equal;
                        cols.skip_persistent_send = skip_persistent_send;
                        cols.is_initial = *is_initial;
                        if self.segment_number > 0
                            && cols.prior_timestamp == F::from_canonical_u32(2)
                        {
                            cols.skip_persistent_receive = F::one();
                        }
                    },
                );

            let last_cols: &mut MemoryCols<F> = rows[n - 1][..].borrow_mut();
            last_cols.diff = F::one();
            last_cols.diff_bytes = Word::from(F::one());
            last_cols.diff_inv = F::one();
            last_cols.addr_equal = F::zero();
            last_cols.is_initial = is_initials[n - 1];
            // NOTE: Have to assign `skip_persistent_send`, because `addr_equal` has length n-1, so
            // above code misses skip persistent send for last row. `addr_equal` is zero here, so ignored.
            last_cols.skip_persistent_send =
                last_cols.is_final + last_cols.is_dummy_read + last_cols.is_static_write;
            if last_cols.skip_persistent_send > F::one() {
                // if more than one is true, clamp to 1
                last_cols.skip_persistent_send = F::one();
            }

            if self.segment_number > 0 && last_cols.prior_timestamp == F::from_canonical_u32(2) {
                last_cols.skip_persistent_receive = F::one();
            }
        }
    }
}
