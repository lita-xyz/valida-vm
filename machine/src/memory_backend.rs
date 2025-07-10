use alloc::alloc::alloc_zeroed;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use bitvec::vec::BitVec;
use core::alloc::Layout;
use core::fmt::Debug;
use core::hash::BuildHasherDefault;
use core::iter::ExactSizeIterator;
use core::mem;
use fxhash::FxHasher32;
use hashbrown::HashMap;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};

use valida_memory_footprint::MemoryFootprint;

use crate::Word;

#[derive(Debug, Clone, Default, Copy, Eq, PartialEq)]
pub enum MemoryAccessTimestamp<T> {
    #[default]
    ZeroInitialized,
    Static,
    PriorSegment(T),
    ThisSegment,
}

impl<T> MemoryFootprint for MemoryAccessTimestamp<T> {
    fn memory_footprint(&self) -> usize {
        mem::size_of::<Self>() as usize
    }
}

#[derive(Debug, Clone, Default, Copy, Eq, PartialEq)]
pub struct MemoryRecord {
    pub value: Word<u8>,
    pub last_accessed: MemoryAccessTimestamp<u32>,
}

impl MemoryFootprint for MemoryRecord {
    fn memory_footprint(&self) -> usize {
        self.value.memory_footprint() + self.last_accessed.memory_footprint()
    }
}

impl From<u32> for MemoryRecord {
    fn from(value: u32) -> Self {
        Self {
            value: Word::from(value),
            last_accessed: MemoryAccessTimestamp::ThisSegment,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StorageBackendType {
    #[default]
    Ordered,
    Unordered,
    Array,
    //LeanArray,
}

#[derive(Clone, Eq, PartialEq)]
pub enum ValidaStorageBackend<R> {
    Ordered(BTreeMap<u32, R>),
    Unordered(HashMap<u32, R, BuildHasherDefault<FxHasher32>>),
    Array(usize, BitVec<usize>, Vec<R>),
    //LeanArray(Vec<R>),
}

impl<R: MemoryFootprint> MemoryFootprint for ValidaStorageBackend<R> {
    fn memory_footprint(&self) -> usize {
        match self {
            ValidaStorageBackend::Ordered(t) => t.memory_footprint(),
            ValidaStorageBackend::Unordered(t) => t.memory_footprint(),
            ValidaStorageBackend::Array(a, b, c) => {
                a.memory_footprint() + b.memory_footprint() + c.memory_footprint()
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ValidaMemoryBackend {
    pub storage: ValidaStorageBackend<MemoryRecord>,
    memory_size: u32,
}

impl MemoryFootprint for ValidaMemoryBackend {
    fn memory_footprint(&self) -> usize {
        self.storage.memory_footprint() + self.memory_size.memory_footprint()
    }
}

impl<R: Debug> Debug for ValidaStorageBackend<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ValidaStorageBackend::Ordered(bt) => write!(f, "{:?}", bt),
            ValidaStorageBackend::Unordered(hm) => write!(f, "{:?}", hm),
            ValidaStorageBackend::Array(_n, _bv, _v) => {
                for (addr, record) in self.iter() {
                    write!(f, "{}: {:?}", addr, record)?;
                }
                Ok(())
            } //MemoryBackend::LeanArray(v) => write!(f, "{:?}", v),
        }
    }
}

impl ValidaMemoryBackend {
    pub fn default_for_field<F: PrimeField32>() -> Self {
        Self {
            storage: ValidaStorageBackend::new(StorageBackendType::Ordered),
            memory_size: F::ORDER_U32 >> 1,
        }
    }
    pub fn default_with_size<F: PrimeField32>(size: u32) -> Self {
        assert!(size <= F::ORDER_U32 >> 1);
        Self {
            storage: ValidaStorageBackend::new(StorageBackendType::Ordered),
            memory_size: size,
        }
    }
}

//pub type PersistentMemoryState = BTreeMap<u32, PersistentMemoryRecord>;

pub trait StorageBackendTrait<R: Debug> {
    type Iter<'a>: ExactSizeIterator<Item = (u32, &'a R)>
    where
        R: 'a;
    type IterMut<'a>: ExactSizeIterator<Item = (u32, &'a mut R)>
    where
        R: 'a;
    fn new(backend_type: StorageBackendType) -> Self;
    //fn transform<T>(&self, f: impl Fn(&R) -> T) -> Self;
    fn get(&self, addr: &u32) -> Option<&R>;
    fn set(&mut self, addr: u32, record: R);
    fn iter(&self) -> Iter<'_, R>;
    fn iter_mut(&mut self) -> IterMut<'_, R>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}
pub trait MemoryBackendTrait: StorageBackendTrait<MemoryRecord> {
    /// Return "---------------------" if uninitialized, else, return the cell's value.
    /// Used in debugger mode
    fn examine(&self, address: u32) -> String {
        let value = self.get(&address);
        match value {
            Some(MemoryRecord {
                value: raw_value,
                last_accessed,
            }) => {
                let u32val: u32 = (*raw_value).into();
                format!("value: {}, last accessed: {:?}", u32val, last_accessed)
            }
            None => String::from("--------"),
        }
    }

    /// Read from a cell. Used for debugging purposes
    fn get_value(&self, address: u32) -> Word<u8> {
        self.get(&address).copied().unwrap_or_default().value
    }

    /// Get the timestamp of the last access to a cell.
    fn get_last_accessed_timestamp(&self, address: u32) -> MemoryAccessTimestamp<u32> {
        self.get(&address)
            .copied()
            .unwrap_or_default()
            .last_accessed
    }

    /// Get the size of the memory
    fn memory_size(&self) -> u32;
}

impl<R> Default for ValidaStorageBackend<R> {
    fn default() -> Self {
        Self::Ordered(Default::default())
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
pub enum IterMut<'a, R> {
    Ordered(alloc::collections::btree_map::IterMut<'a, u32, R>),
    Unordered(hashbrown::hash_map::IterMut<'a, u32, R>),
    Array {
        next_addr: u32,
        num_remaining: usize,
        bv: &'a mut BitVec<usize>,
        v_tail: &'a mut [R],
        v_tail_start: usize,
    },
}

impl<'a, R: Debug> Iterator for Iter<'a, R> {
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

                    while !bv[dst] {
                        dst += 1;
                        debug_assert!(dst < bv.len(), "index should not extend past end of buffer");
                    }
                    *next_addr = ((dst + 1) << 2) as u32;
                    *num_remaining -= 1;
                    Some((*next_addr, &v[dst]))
                }
            }
        }
    }
}

impl<'a, R> Iterator for IterMut<'a, R> {
    type Item = (u32, &'a mut R);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IterMut::Ordered(it) => it.next().map(|(k, v)| (*k, v)),
            IterMut::Unordered(it) => it.next().map(|(k, v)| (*k, v)),
            IterMut::Array {
                next_addr,
                num_remaining,
                bv,
                v_tail,
                v_tail_start,
            } => {
                if *num_remaining == 0 {
                    None
                } else {
                    let mut dst = (*next_addr as usize) >> 2;
                    while !bv[dst] {
                        dst += 1;
                        debug_assert!(dst < bv.len(), "index should not extend past end of buffer");
                    }
                    *next_addr = ((dst + 1) << 2) as u32;
                    *num_remaining -= 1;
                    let _v_tail = mem::replace(v_tail, &mut []);
                    // head is v[v_tail_start..=dst], tail is v[dst+1..]
                    let (head, tail) = _v_tail.split_at_mut(dst + 1 - *v_tail_start);
                    let (next, _) = head.split_last_mut().unwrap();
                    *v_tail = tail;
                    *v_tail_start = dst + 1;
                    Some((*next_addr, next))
                }
            }
        }
    }
}

impl<R: Debug> ExactSizeIterator for Iter<'_, R> {
    fn len(&self) -> usize {
        match self {
            Iter::Ordered(it) => it.len(),
            Iter::Unordered(it) => it.len(),
            Iter::Array { num_remaining, .. } => *num_remaining,
        }
    }
}

impl<R> ExactSizeIterator for IterMut<'_, R> {
    fn len(&self) -> usize {
        match self {
            IterMut::Ordered(it) => it.len(),
            IterMut::Unordered(it) => it.len(),
            IterMut::Array { num_remaining, .. } => *num_remaining,
        }
    }
}

impl<R: Debug> StorageBackendTrait<R> for ValidaStorageBackend<R> {
    type Iter<'a>
        = Iter<'a, R>
    where
        R: 'a;
    type IterMut<'a>
        = IterMut<'a, R>
    where
        R: 'a;
    fn new(backend_type: StorageBackendType) -> Self {
        match backend_type {
            StorageBackendType::Ordered => Self::Ordered(Default::default()),
            StorageBackendType::Unordered => Self::Unordered(Default::default()),
            StorageBackendType::Array => {
                let mem_size = ((BabyBear::ORDER_U32 - 1) as usize >> 3) + 1;
                let num_elems = (mem_size - 1 + usize::BITS as usize) / usize::BITS as usize;
                let bv_align = core::mem::align_of::<usize>();
                let word_align = core::mem::align_of::<R>();
                let bv_layout = Layout::from_size_align(num_elems, bv_align).expect("ok");
                let vec_layout = Layout::from_size_align(mem_size, word_align).expect("ok");
                // SAFETY: both ptrs are from safely constructed layouts, and bv_ptr has num_elems *
                // usize::BITS number of bits. vec and bv are then safely constructed within those
                // bounds. the two pointers are also asserted to be non-null.
                unsafe {
                    let bv_ptr = alloc_zeroed(bv_layout) as *mut usize;
                    let vec_ptr = alloc_zeroed(vec_layout) as *mut R;
                    let bitptr = bv_ptr.try_into().expect("ok");

                    assert!(!bv_ptr.is_null());
                    assert!(!vec_ptr.is_null());

                    let bv =
                        BitVec::from_raw_parts(bitptr, mem_size, num_elems * usize::BITS as usize);
                    let mem = Vec::from_raw_parts(vec_ptr, mem_size, mem_size);
                    Self::Array(0, bv, mem)
                }
            } // MemoryBackendType::LeanArray => {
              //     let mem_size = ((-BabyBear::one()).as_canonical_u32() as usize >> 3) + 1;
              //     let word_align = core::mem::align_of::<R>();
              //     let vec_layout = Layout::from_size_align(mem_size, word_align).expect("ok");
              //     unsafe {
              //         let vec_ptr = alloc_zeroed(vec_layout) as *mut R;
              //         let mem = Vec::from_raw_parts(vec_ptr, mem_size, mem_size);
              //         Self::LeanArray(mem)
              //     }
              // }
        }
    }

    // fn transform<T>(&self, f: impl Fn(&R) -> T) -> ValidaStorageBackend<T> {
    //     match self {
    //         ValidaStorageBackend::Ordered(bt) => {
    //             ValidaStorageBackend::Ordered(bt.iter().map(|(k, v)| (*k, f(v))).collect())
    //         }
    //         ValidaStorageBackend::Unordered(hm) => {
    //             ValidaStorageBackend::Unordered(hm.iter().map(|(k, v)| (*k, f(v))).collect())
    //         }
    //         ValidaStorageBackend::Array(n, bv, v) => {
    //             let mut transformed = ValidaStorageBackend::new(StorageBackendType::Array);
    //             let ValidaStorageBackend::Array(ref mut new_n, ref mut new_bv, ref mut new_v) =
    //                 transformed
    //             else {
    //                 unreachable!()
    //             };
    //             *new_n = *n;
    //             new_bv.clone_from_bitslice(bv);
    //             for i in 0..v.len() {
    //                 if bv[i >> 2] {
    //                     new_v[i] = f(&v[i]);
    //                 }
    //             }
    //             transformed
    //         } //MemoryBackend::LeanArray(v) => MemoryBackend::LeanArray(v.iter().map(f).collect()),
    //     }
    // }

    fn get(&self, addr: &u32) -> Option<&R> {
        match self {
            ValidaStorageBackend::Ordered(bt) => bt.get(addr),
            ValidaStorageBackend::Unordered(hm) => hm.get(addr),
            ValidaStorageBackend::Array(_, bv, v) => {
                if bv[*addr as usize >> 2] {
                    Some(&v[*addr as usize >> 2])
                } else {
                    None
                }
            } //MemoryBackend::LeanArray(v) => Some(&v[*addr as usize >> 2]),
        }
    }

    fn set(&mut self, addr: u32, record: R) {
        match self {
            ValidaStorageBackend::Ordered(bt) => {
                bt.insert(addr, record);
            }
            ValidaStorageBackend::Unordered(hm) => {
                hm.insert(addr, record);
            }
            ValidaStorageBackend::Array(n, bv, v) => {
                let dst = addr as usize >> 2;
                if !bv[dst] {
                    *n += 1;
                    bv.set(dst, true);
                }
                v[dst] = record;
            } // MemoryBackend::LeanArray(v) => {
              //     v[addr as usize >> 2] = record;
              // }
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            ValidaStorageBackend::Ordered(bt) => bt.is_empty(),
            ValidaStorageBackend::Unordered(hm) => hm.is_empty(),
            ValidaStorageBackend::Array(n, _, _) => *n == 0,
            //MemoryBackend::LeanArray(v) => v.is_empty(),
        }
    }

    fn len(&self) -> usize {
        match self {
            ValidaStorageBackend::Ordered(bt) => bt.len(),
            ValidaStorageBackend::Unordered(hm) => hm.len(),
            ValidaStorageBackend::Array(n, _, _) => *n,
            //MemoryBackend::LeanArray(v) => v.len(),
        }
    }
    fn iter(&self) -> Iter<'_, R> {
        match self {
            ValidaStorageBackend::Ordered(bt) => Iter::Ordered(bt.iter()),
            ValidaStorageBackend::Unordered(hm) => Iter::Unordered(hm.iter()),
            ValidaStorageBackend::Array(n, bv, v) => Iter::Array {
                next_addr: 0,
                num_remaining: *n,
                bv,
                v,
            },
            //}, //MemoryBackend::LeanArray(_) => Iter::Array,
        }
    }
    fn iter_mut(&mut self) -> IterMut<'_, R> {
        match self {
            ValidaStorageBackend::Ordered(bt) => IterMut::Ordered(bt.iter_mut()),
            ValidaStorageBackend::Unordered(hm) => IterMut::Unordered(hm.iter_mut()),
            ValidaStorageBackend::Array(n, bv, v) => IterMut::Array {
                next_addr: 0,
                num_remaining: *n,
                bv,
                v_tail: v,
                v_tail_start: 0,
            },
            //}, //MemoryBackend::LeanArray(_) => Iter::Array,
        }
    }
}
impl StorageBackendTrait<MemoryRecord> for ValidaMemoryBackend {
    type Iter<'a> = Iter<'a, MemoryRecord>;
    type IterMut<'a> = IterMut<'a, MemoryRecord>;
    fn new(backend_type: StorageBackendType) -> Self {
        Self {
            storage: ValidaStorageBackend::new(backend_type),
            memory_size: 0,
        }
    }

    fn get(&self, addr: &u32) -> Option<&MemoryRecord> {
        self.storage.get(addr)
    }

    fn set(&mut self, addr: u32, record: MemoryRecord) {
        self.storage.set(addr, record);
    }

    fn iter(&self) -> Iter<'_, MemoryRecord> {
        self.storage.iter()
    }

    fn iter_mut(&mut self) -> IterMut<'_, MemoryRecord> {
        self.storage.iter_mut()
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }
}
impl MemoryBackendTrait for ValidaMemoryBackend {
    fn memory_size(&self) -> u32 {
        self.memory_size
    }
}

impl<T> MemoryAccessTimestamp<T> {
    pub fn is_initial(&self) -> bool {
        !matches!(self, MemoryAccessTimestamp::ThisSegment)
    }
}

impl MemoryAccessTimestamp<u32> {
    pub fn as_scalar<F: AbstractField>(self) -> F {
        match self {
            MemoryAccessTimestamp::ThisSegment => F::one(),
            // Special case to exclude zero-initialized memory from the persistent sends in the memory chip.
            MemoryAccessTimestamp::ZeroInitialized => F::zero(),
            MemoryAccessTimestamp::Static => F::two(),
            MemoryAccessTimestamp::PriorSegment(s) => {
                (F::two() + F::one()) + F::from_canonical_u32(s)
            }
        }
    }
}

impl<'a> IntoIterator for &'a ValidaMemoryBackend {
    type Item = (u32, &'a MemoryRecord);
    type IntoIter = Iter<'a, MemoryRecord>;

    fn into_iter(self) -> Self::IntoIter {
        self.storage.iter()
    }
}
