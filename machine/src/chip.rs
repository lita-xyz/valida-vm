use crate::__internal::{DebugConstraintBuilder, ProverConstraintFolder};
use crate::debug_builder::AirBuilderWithGlobalPermutationChallenges;
use crate::folding_builder::VerifierConstraintFolder;
use crate::public::PublicValues;
use crate::{Machine, SMALLVEC_SIZE};
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::hash::{Hash, Hasher};
use smallvec::SmallVec;
use valida_memory_footprint::MemoryFootprint;

use crate::config::StarkConfig;
use crate::symbolic::symbolic_builder::SymbolicAirBuilder;
use p3_air::{Air, AirBuilderWithPublicValues, PairBuilder, PermutationAirBuilder, VirtualPairCol};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

#[derive(Clone, Debug)]
pub struct InteractionMetadata {
    pub chip_name: String,
    pub row: usize,
}

impl MemoryFootprint for InteractionMetadata {
    fn memory_footprint(&self) -> usize {
        self.chip_name.memory_footprint() + self.row.memory_footprint()
    }
}

#[derive(Clone, Debug)]
pub struct InteractionVec<F> {
    pub fields: SmallVec<[F; SMALLVEC_SIZE]>,
    // used for Debug implementation only
    #[allow(dead_code)]
    pub metadata: InteractionMetadata,
}

impl<T: MemoryFootprint> MemoryFootprint for InteractionVec<T> {
    fn memory_footprint(&self) -> usize {
        self.fields.memory_footprint() + self.metadata.memory_footprint()
    }
}

//to test that the sends and receives match, we want to ignore the metadata
impl<F: PartialEq> PartialEq for InteractionVec<F> {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}
impl<F: Eq> Eq for InteractionVec<F> {}
impl<F: PartialOrd> PartialOrd for InteractionVec<F> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.fields.partial_cmp(&other.fields)
    }
}
impl<F: Ord> Ord for InteractionVec<F> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.fields.cmp(&other.fields)
    }
}
impl<F: Hash> Hash for InteractionVec<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fields.hash(state);
    }
}
pub type InteractionMap<F> =
    BTreeMap<BusArgument, (Vec<InteractionVec<F>>, Vec<InteractionVec<F>>)>;

#[cfg(feature = "std")]
pub fn check_interactions<F: Eq + Ord + Hash + Debug>(
    map: &mut InteractionMap<F>,
    skip_persistent_interactions: bool,
) {
    use std::collections::HashMap;
    for (bus, (sends, receives)) in map.iter_mut() {
        if skip_persistent_interactions && matches!(bus, BusArgument::Persistent(_)) {
            continue; // skip this key
        }

        let mut sends_map: HashMap<_, i32> = HashMap::new();
        let mut receives_map: HashMap<_, i32> = HashMap::new();

        for send in sends {
            *sends_map.entry(send).or_insert(0) += 1;
        }

        for receive in receives {
            *receives_map.entry(receive).or_insert(0) += 1;
        }

        let mut unmatched = Vec::new();

        for (send, send_count) in &sends_map {
            match receives_map.get(send) {
                Some(recv_count) if recv_count != send_count => {
                    unmatched.push(format!(
                        "Mismatch for {:?}: {} sends but {} receives",
                        send, send_count, recv_count
                    ));
                }
                None => {
                    unmatched.push(format!(
                        "Unmatched send {:?}: {} sends but 0 receives",
                        send, send_count
                    ));
                }
                _ => {}
            }
        }

        for (receive, recv_count) in &receives_map {
            if !sends_map.contains_key(receive) {
                unmatched.push(format!(
                    "Unmatched receive {:?}: 0 sends but {} receives",
                    receive, recv_count
                ));
            }
        }

        if !unmatched.is_empty() {
            let mut out = format!("Sends and receives do not match for bus {:?}\n", bus);
            for msg in unmatched {
                out.push_str(&format!("{}\n", msg));
            }
            panic!("{}", out);
        }
    }
}

pub trait ChipTraceHeight {
    fn trace_height(&self) -> u32;
}

pub trait Chip<M, SC>:
    for<'a> Air<ProverConstraintFolder<'a, M, SC>>
    + for<'a> Air<VerifierConstraintFolder<'a, M, SC>>
    + for<'a> Air<SymbolicAirBuilder<'a, M, SC>>
    + for<'a> Air<DebugConstraintBuilder<'a, M, SC>>
where
    M: Machine<SC::Val>,
    SC: StarkConfig,
{
    type Public: PublicValues<SC::Val, SC::Challenge>;

    fn name(&self) -> String;

    /// Generate the main trace for the chip given the provided machine.
    fn generate_main_trace(
        &self,
        _machine: &M,
        _verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>) {
        (None, None)
    }
    fn main_width(&self) -> usize {
        self.width()
    }

    fn generate_public_values(
        &self,
        _verbose: bool,
    ) -> (Option<Self::Public>, Option<Vec<String>>) {
        (None, None)
    }

    fn get_preprocessed_trace(
        &self,
        verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>) {
        let trace = self.preprocessed_trace();
        if !verbose {
            return (trace, None);
        }
        let log = if let Some(ref trace) = trace {
            let mut log_prints = Vec::with_capacity(trace.height());
            for (index, row) in trace.rows().enumerate() {
                log_prints.push(format!("row {index} {:?}:", row));
            }
            Some(log_prints)
        } else {
            None
        };
        (trace, log)
    }

    /// Sends that are interactions happening within the same chip
    fn local_sends(&self) -> Vec<Interaction<SC::Val>> {
        vec![]
    }

    /// Receives that are interactions happening within the same chip
    fn local_receives(&self) -> Vec<Interaction<SC::Val>> {
        vec![]
    }

    /// Sends that are interactions happening across multiple chips in the same segment
    fn global_sends(&self, _machine: &M) -> Vec<Interaction<SC::Val>> {
        vec![]
    }

    /// Receives that are interactions happening across multiple chips in the same segment
    fn global_receives(&self, _machine: &M) -> Vec<Interaction<SC::Val>> {
        vec![]
    }

    /// Return all "ephemeral" interactions, i.e. those that are not persistent
    /// across execution segments. Includes all local and global sends and receives.
    fn ephemeral_interactions(&self, machine: &M) -> Vec<(Interaction<SC::Val>, InteractionType)> {
        let mut interactions: Vec<(Interaction<SC::Val>, InteractionType)> = vec![];
        interactions.extend(
            self.local_sends()
                .into_iter()
                .map(|i| (i, InteractionType::LocalSend)),
        );
        interactions.extend(
            self.local_receives()
                .into_iter()
                .map(|i| (i, InteractionType::LocalReceive)),
        );
        interactions.extend(
            self.global_sends(machine)
                .into_iter()
                .map(|i| (i, InteractionType::GlobalSend)),
        );
        interactions.extend(
            self.global_receives(machine)
                .into_iter()
                .map(|i| (i, InteractionType::GlobalReceive)),
        );
        interactions
    }
}

pub trait ValidaAirBuilder:
    PairBuilder
    + PermutationAirBuilder
    + AirBuilderWithPublicValues
    + AirBuilderWithGlobalPermutationChallenges
{
    type Machine;

    fn machine(&self) -> &Self::Machine;
}

#[derive(Debug, Clone)]
pub struct Interaction<F: Field> {
    pub fields: Vec<VirtualPairCol<F>>,
    pub count: VirtualPairCol<F>,
    pub argument_index: BusArgument,
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub enum InteractionType {
    LocalSend,
    LocalReceive,
    GlobalSend,
    GlobalReceive,
    PersistentSend,
    PersistentReceive,
}

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum BusArgument {
    // permutation is within a single chip
    Local(usize),
    // permutation is across multiple chips in the same segment
    Global(usize),
    // permutation is across multiple segments and multiple chips
    Persistent(usize),
}

impl MemoryFootprint for BusArgument {
    fn memory_footprint(&self) -> usize {
        match self {
            BusArgument::Local(_v) => core::mem::size_of::<usize>(),
            BusArgument::Global(_v) => core::mem::size_of::<usize>(),
            BusArgument::Persistent(_v) => core::mem::size_of::<usize>(),
        }
    }
}

impl BusArgument {
    pub fn identifier(&self) -> usize {
        match self {
            Self::Local(n) => 3 * n,
            Self::Global(n) => (3 * n) + 1,
            Self::Persistent(n) => (3 * n) + 2,
        }
    }
}

impl<F: Field> Interaction<F> {
    pub fn is_local(&self) -> bool {
        matches!(self.argument_index, BusArgument::Persistent(_))
    }

    pub fn is_global(&self) -> bool {
        matches!(self.argument_index, BusArgument::Global(_))
    }

    pub fn is_persistent(&self) -> bool {
        matches!(self.argument_index, BusArgument::Persistent(_))
    }

    pub fn argument_index(&self) -> usize {
        match self.argument_index {
            BusArgument::Local(i) => i,
            BusArgument::Global(i) => i,
            BusArgument::Persistent(i) => i,
        }
    }
}

#[macro_export]
macro_rules! instructions {
    ($($t:ident),*) => {
        $(
            #[derive(Default)]
            pub struct $t {}
        )*
    }
}
