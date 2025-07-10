extern crate alloc;

use alloc::vec::Vec;

use alloc::collections::BTreeMap;
use smallvec::{Array, SmallVec};
use std::collections::HashSet;

/// NOTE: This is written without any macros mostly because I didn't feel like trying to figure
/// out how to write `derive` macros that walk over the fields for the generic cases. Would make
/// this a bit more concise, but whatever

pub trait MemoryFootprint {
    fn memory_footprint(&self) -> usize;

    fn in_bytes(&self) -> usize {
        self.memory_footprint()
    }

    fn in_kilo_bytes(&self) -> f64 {
        self.memory_footprint() as f64 / 1000.0
    }

    fn in_mega_bytes(&self) -> f64 {
        self.memory_footprint() as f64 / 1e6
    }

    fn in_giga_bytes(&self) -> f64 {
        self.memory_footprint() as f64 / 1e9
    }
}

//pub trait SimplePodTypes: MemoryFootprint {}
//
//impl SimplePodTypes for u8 {}
//impl SimplePodTypes for u32 {}
//impl SimplePodTypes for bool {}
//impl SimplePodTypes for usize {}

use core::mem;
impl<T: MemoryFootprint> MemoryFootprint for Vec<T> {
    fn memory_footprint(&self) -> usize {
        // NOTE: Currently ignoring overhead of `self.capacity()`
        self.iter()
            .map(|item| item.memory_footprint())
            .sum::<usize>()
    }
}

impl MemoryFootprint for u8 {
    fn memory_footprint(&self) -> usize {
        mem::size_of::<u8>()
    }
}

impl MemoryFootprint for bool {
    fn memory_footprint(&self) -> usize {
        mem::size_of::<bool>()
    }
}

impl MemoryFootprint for u32 {
    fn memory_footprint(&self) -> usize {
        mem::size_of::<u32>()
    }
}

impl MemoryFootprint for usize {
    fn memory_footprint(&self) -> usize {
        mem::size_of::<usize>()
    }
}

impl MemoryFootprint for String {
    fn memory_footprint(&self) -> usize {
        self.capacity() * mem::size_of::<u8>()
    }
}

// SmallVec<[Operation; INSTR_MAX_MEM_OPS]>>,
impl<T: Array> MemoryFootprint for SmallVec<T>
where
    T::Item: MemoryFootprint,
{
    fn memory_footprint(&self) -> usize {
        // Stack size of the SmallVec struct itself
        let stack_size = mem::size_of::<Self>();

        // Heap allocation (only if spilled)
        let heap_size = if self.spilled() {
            // If spilled to heap, count the heap allocation
            self.capacity() * mem::size_of::<T::Item>()
        } else {
            // If inline, no heap allocation
            0
        };

        stack_size + heap_size
    }
}

impl<K, V: MemoryFootprint> MemoryFootprint for BTreeMap<K, V> {
    fn memory_footprint(&self) -> usize {
        // Vec's heap allocation
        self.len() * mem::size_of::<K>()
            + self.iter().map(|kv| kv.1.memory_footprint()).sum::<usize>()
    }
}

// pub final_memory_state: HashSet<(u32, Word<u8>, u32)>,

impl<T1, T2, T3> MemoryFootprint for (T1, T2, T3)
where
    T1: MemoryFootprint,
    T2: MemoryFootprint,
    T3: MemoryFootprint,
{
    fn memory_footprint(&self) -> usize {
        self.0.memory_footprint() + self.1.memory_footprint() + self.2.memory_footprint()
    }
}

impl<T1, T2> MemoryFootprint for (T1, T2)
where
    T1: MemoryFootprint,
    T2: MemoryFootprint,
{
    fn memory_footprint(&self) -> usize {
        self.0.memory_footprint() + self.1.memory_footprint()
    }
}

impl<K: MemoryFootprint> MemoryFootprint for HashSet<K> {
    fn memory_footprint(&self) -> usize {
        mem::size_of::<Self>() + self.iter().map(|v| v.memory_footprint()).sum::<usize>()
    }
}

impl<T: MemoryFootprint, const N: usize> MemoryFootprint for [T; N] {
    fn memory_footprint(&self) -> usize {
        self.iter().map(|v| v.memory_footprint()).sum::<usize>()
    }
}

impl<T: MemoryFootprint> MemoryFootprint for Option<T> {
    fn memory_footprint(&self) -> usize {
        let mut result = mem::size_of::<Self>();
        match self {
            Some(v) => result += v.memory_footprint(),
            None => (),
        }
        result
    }
}

impl<T: MemoryFootprint + Sized> MemoryFootprint for Box<T> {
    fn memory_footprint(&self) -> usize {
        mem::size_of::<Box<T>>() + self.as_ref().memory_footprint()
    }
}

use bitvec::vec::BitVec;
use core::hash::BuildHasherDefault;
use hashbrown::HashMap;

impl MemoryFootprint for BitVec<usize> {
    fn memory_footprint(&self) -> usize {
        mem::size_of::<Self>() + self.capacity() / 8
    }
}

impl<K, V, FxHasher32> MemoryFootprint for HashMap<K, V, BuildHasherDefault<FxHasher32>> {
    fn memory_footprint(&self) -> usize {
        // HashMap struct on stack
        mem::size_of::<Self>() +
        // Heap allocation for the hash table
        // HashMap allocates space for both keys and values plus metadata
        self.capacity() * (mem::size_of::<K>() + mem::size_of::<V>()) +
        // Additional overhead for hash table metadata (buckets, control bytes, etc.)
        // hashbrown uses ~1 byte per bucket for control information
        self.capacity()
    }
}

use p3_baby_bear::BabyBear;
use p3_field::{PrimeField32, TwoAdicField};

//impl MemoryFootprint for Vec<BabyBear> {
//    fn memory_footprint(&self) -> usize {
//        // Vec's heap allocation
//        self.capacity() * self.memory_footprint()
//    }
//}

impl MemoryFootprint for BabyBear {
    fn memory_footprint(&self) -> usize {
        mem::size_of::<Self>()
    }
}

//impl<F: StarkField> MemoryFootprint for F {
//    fn memory_footprint(&self) -> usize {
//        mem::size_of::<F>()
//    }
//}

//impl<F: StarkField> MemoryFootprint for F {
//    fn memory_footprint(&self) -> usize {}
//}

use p3_matrix::dense::RowMajorMatrix;
// For the traces (regular field and extension field / challenge)
impl<F: MemoryFootprint> MemoryFootprint for RowMajorMatrix<F> {
    //impl MemoryFootprint for RowMajorMatrix<BabyBear> {
    fn memory_footprint(&self) -> usize {
        self.values.memory_footprint() + self.width.memory_footprint()
    }
}

// /// A dense matrix stored in row-major form.
// #[derive(Clone, Debug, PartialEq, Eq)]
// #[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
// pub struct RowMajorMatrix<T> {
//     /// All values, stored in row-major order.
//     pub values: Vec<T>,
//     pub width: usize,
// }

use p3_field::extension::BinomialExtensionField;
//impl<F: MemoryFootprint> MemoryFootprint for BinomialExtensionField<F, 5> {
//    fn memory_footprint(&self) -> usize {
//        5 * mem::size_of::<F>()
//    }
//}

// For `SC::Challenge`
impl<T: TwoAdicField + MemoryFootprint> MemoryFootprint for BinomialExtensionField<T, 5> {
    fn memory_footprint(&self) -> usize {
        5 * mem::size_of::<T>()
    }
}

//impl<T: TwoAdicField + MemoryFootprint> MemoryFootprint
//    for RowMajorMatrix<BinomialExtensionField<T, 5>>
//{
//    fn memory_footprint(&self) -> usize {
//        5 * mem::size_of::<T>()
//    }
//}

//impl<F: TwoAdicField + TwoAdicField + MemoryFootprint> MemoryFootprint for RowMajorMatrix<F> {
//    //impl MemoryFootprint for RowMajorMatrix<BabyBear> {
//    fn memory_footprint(&self) -> usize {
//        self.values.memory_footprint() + self.width.memory_footprint()
//    }
//}
