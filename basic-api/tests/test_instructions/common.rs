use crate::testmachine::{TestMachine, TestMachineState};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32, TwoAdicField};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::sync::LazyLock;
use std::vec;
use valida_alu_u32::{
    add::MachineWithAdd32Chip, bitwise::MachineWithBitwise32Chip, com::MachineWithCom32Chip,
    div::MachineWithDiv32Chip, lt::MachineWithLt32Chip, mul::MachineWithMul32Chip,
    sub::MachineWithSub32Chip,
};
use valida_basic_api::{
    machine::basic::pc_strategy, BasicMachine, BasicRunningMachine, ValidaRuntime,
};
use valida_cpu::MachineWithCpuChip;
use valida_machine::{
    Instruction, Machine, Operands, StarkField, StorageBackendTrait, ValidaMemoryBackend, Word,
};
use valida_machine::{MemoryAccessTimestamp, MemoryRecord};
use valida_memory::Operation::{DummyRead, Read, Write};

use proptest::prelude::*;
use proptest::sample::Selector;
use proptest::sample::SizeRange;
use proptest::strategy::Strategy;

pub static MAX_ADDR: LazyLock<u32> =
    LazyLock::new(|| ((-BabyBear::one()).as_canonical_u32() >> 1) - 1);

pub type BBMachine = BasicMachine<BabyBear>;
pub type BBState = TestMachineState<BBMachine>;
pub type BBRunningMachine<'a> = BasicRunningMachine<'a, BabyBear>;

#[derive(Debug)]
pub struct BBMemoryBackend(pub ValidaMemoryBackend);

impl Default for BBState {
    fn default() -> Self {
        Self {
            machine: BasicMachine::default(),
            memory_backend: ValidaMemoryBackend::default_for_field::<BabyBear>(),
        }
    }
}

pub fn max_addr<F: StarkField>() -> u32 {
    ((-F::one()).as_canonical_u32() >> 1) - 1
}

pub fn address_strategy(max_addr: u32) -> impl Clone + Strategy<Value = u32> {
    (0..=max_addr / 4).prop_map(|x| 4 * x)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DataStrategy {
    #[default]
    Numerical,
    Address,
    Bitwise,
    Program,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemorySize(pub u32);

impl Default for MemorySize {
    fn default() -> Self {
        MemorySize(2)
    }
}

impl From<u32> for MemorySize {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

// impl TestMachine<BabyBear> for BBState {
//     type M = BasicMachine<BabyBear>;
//     fn registers(&self) -> (u32, u32) {
//         (self.machine.pc, self.machine.fp)
//     }

//     fn clk(&self) -> u32 {
//         self.machine.clk()
//     }

//     fn num_cells(&self) -> usize {
//         self.memory.len()
//     }

//     fn assigned_cells(&self) -> impl ExactSizeIterator<Item = (u32, Word<BabyBear>)> {
//         self.memory
//             .iter()
//             .map(|(addr, &MemoryRecord { value, .. })| (addr, value))
//     }

//     fn read(&self, addr: u32) -> Option<Word<BabyBear>> {
//         self.memory.get(&addr).copied().map(|record| record.value)
//     }

//     fn set(&mut self, addr: u32, val: u32) {
//         self.memory.set(
//             addr,
//             MemoryRecord {
//                 value: val.into(),
//                 last_accessed: MemoryAccessTimestamp::ThisSegment,
//             },
//         );
//     }

//     fn memory_log_size(&self) -> usize {
//         self.state.memory_log_size()
//     }

//     fn memory_log(&self, clk: u32) -> Vec<Operation> {
//         self.state.memory_log(clk)
//     }
// }

impl proptest::arbitrary::Arbitrary for BBMemoryBackend {
    type Parameters = (MemorySize, DataStrategy);

    type Strategy = SBoxedStrategy<Self>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use proptest as pt;
        use pt::collection::btree_map;
        let (memory_size, data_strategy) = args;

        let strat_addr = address_strategy(memory_size.0 * 4);
        let strat_pc = pc_strategy::<BabyBear>();
        let strat_size = SizeRange::default() + memory_size.0 as usize;
        let strat_data = match data_strategy {
            DataStrategy::Numerical => pt::num::u32::ANY.sboxed(),
            DataStrategy::Address => strat_addr.clone().sboxed(),
            DataStrategy::Bitwise => pt::bits::u32::ANY.sboxed(),
            DataStrategy::Program => strat_pc.clone().sboxed(),
        }
        .prop_map(|x| MemoryRecord {
            value: x.into(),
            last_accessed: MemoryAccessTimestamp::ThisSegment,
        });

        let strat_backend = btree_map(strat_addr, strat_data, strat_size).prop_map(move |map| {
            let mut backend = ValidaMemoryBackend::default_with_size::<BabyBear>(memory_size.0);
            let mut has_nonzero = false;
            for (addr, record) in map {
                if !has_nonzero && record.value != 0.into() {
                    has_nonzero = true;
                }
                backend.set(addr, record);
            }
            // If no non-zero values, add one
            if !has_nonzero {
                backend.set(
                    0,
                    MemoryRecord {
                        value: 1.into(),
                        last_accessed: MemoryAccessTimestamp::ThisSegment,
                    },
                );
            }
            BBMemoryBackend(backend)
        });

        strat_backend.sboxed()
    }
}

pub trait MachineWithTestChips<F: PrimeField32>:
    Machine<F>
    + MachineWithAdd32Chip<F>
    + MachineWithBitwise32Chip<F>
    + MachineWithCom32Chip<F>
    + MachineWithDiv32Chip<F>
    + MachineWithLt32Chip<F>
    + MachineWithMul32Chip<F>
    + MachineWithSub32Chip<F>
    + Default
{
}

impl<M, F> MachineWithTestChips<F> for M
where
    F: PrimeField32,
    M: Machine<F>
        + MachineWithAdd32Chip<F>
        + MachineWithBitwise32Chip<F>
        + MachineWithCom32Chip<F>
        + MachineWithDiv32Chip<F>
        + MachineWithLt32Chip<F>
        + MachineWithMul32Chip<F>
        + MachineWithSub32Chip<F>
        + Default,
{
}

#[inline]
fn clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

// calculates a random target and corresponding offset from the given fp
fn new_offset<R: Rng + ?Sized>(fp: u32, max_addr: u32, rng: &mut R) -> (u32, i32) {
    // 0xbfffffff = 0x3fffffff | 0x80000000
    // this keeps the sign bit available but the available range of offsets are
    // allowed to span a wide range centered on fp
    let offset = rng.gen::<i32>() & i32::from_le_bytes([0xbf, 0xff, 0xff, 0xfc]);
    let target = clamp(fp as i64 + offset as i64, 0, i32::MAX as i64);
    let target = clamp(target, 0, max_addr as i64) & !3;
    let new_offset = target - fp as i64;
    assert!(new_offset <= i32::MAX as i64);
    assert!(new_offset >= i32::MIN as i64);
    debug_assert!(target % 4 == 0);
    (target as u32, new_offset as i32)
}

pub fn new_addr<R: Rng + ?Sized>(fp: u32, max_addr: u32, rng: &mut R) -> (u32, i32) {
    let addr = rng.gen_range(0..=max_addr) & !3;
    (addr, calc_offset(fp, addr))
}

pub fn calc_offset(fp: u32, addr: u32) -> i32 {
    let offset = addr as i64 - fp as i64;
    if offset >= i32::MIN as i64 && offset <= i32::MAX as i64 {
        Some(offset as i32)
    } else {
        None
    }
    .expect("good offset")
}

pub fn test_arith<I, O>(
    op: O,
    mut state: TestMachineState<BasicMachine<BabyBear>>,
    nonzero: bool,
) -> Result<(), TestCaseError>
where
    I: Instruction<BasicMachine<BabyBear>, BabyBear>,
    O: Fn(u32, u32) -> u32,
{
    let mut rng = rand::thread_rng();
    let max_addr = max_addr::<BabyBear>();
    let clk = state.clk();

    let keys: Vec<u32> = state.assigned_cells().map(|x| x.0).collect();
    println!("keys: {:?}", keys);
    let nonzero_keys: Vec<u32> = state
        .assigned_cells()
        .filter_map(|x| if x.1 != 0.into() { Some(x.0) } else { None })
        .collect();
    println!("nonzero_keys: {:?}", nonzero_keys);
    let (old_pc, old_fp) = state.registers();

    // calculate source operand offsets
    let src1 = *keys.choose(&mut rng).expect("nonempty");
    println!("src1: {:?}", src1);
    let src2 = *if nonzero { nonzero_keys } else { keys }
        .choose(&mut rng)
        .expect("nonempty");
    println!("src2: {:?}", src2);
    let off1 = calc_offset(old_fp, src1);
    println!("off1: {:?}", off1);
    let off2 = calc_offset(old_fp, src2);
    println!("off2: {:?}", off2);
    let (dst_addr, dst) = new_offset(old_fp, max_addr, &mut rng);

    println!("dst_addr: {:?}", dst_addr);
    println!("dst: {:?}", dst);

    let val1 = state.memory_backend.get(&src1).expect("nonempty");
    let val2 = state.memory_backend.get(&src2).expect("nonempty");
    let expected: Word<u8> = op(val1.value.into(), val2.value.into()).into();

    let expected_ops = match state.memory_backend.get(&dst_addr) {
        None => vec![
            Read(src1, *val1),
            Read(src2, *val2),
            DummyRead(dst_addr, Default::default()),
            Write(dst_addr, expected),
        ],
        _ => vec![
            Read(src1, *val1),
            Read(src2, *val2),
            Write(dst_addr, expected),
        ],
    };

    println!("cells before execute: {:?}", state.memory_backend);
    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend.clone();
    let mut running_machine = state.machine.start(&mut runtime);

    I::execute(&mut running_machine, Operands([dst, off1, off2, 0, 0]));
    state.machine = Machine::stop(running_machine);
    state.memory_backend = runtime.memory_backend;

    println!("cells after execute: {:?}", state.memory_backend);
    println!("executed");
    let result = state.get(dst_addr).expect("nonempty");
    println!("result");
    let (new_pc, new_fp) = state.registers();

    prop_assert_eq!(old_pc + 1, new_pc);
    prop_assert_eq!(old_fp, new_fp);
    prop_assert_eq!(expected, result);
    prop_assert_eq!(state.memory_log_size(), 1);
    prop_assert_eq!(state.memory_log(clk), expected_ops);

    Ok(())
}

pub fn test_mem<I>(lhs: u32, rhs: u32, expected: u32)
where
    I: Instruction<BasicMachine<BabyBear>, BabyBear>,
    TestMachineState<BasicMachine<BabyBear>>: Default,
{
    let mut state = TestMachineState::<BasicMachine<BabyBear>>::default();
    let clk = state.clk();
    let expected_ops = match state.memory_backend.get(&8) {
        None => vec![
            Read(
                0,
                MemoryRecord {
                    value: lhs.into(),
                    last_accessed: MemoryAccessTimestamp::ThisSegment,
                },
            ),
            Read(
                4,
                MemoryRecord {
                    value: rhs.into(),
                    last_accessed: MemoryAccessTimestamp::ThisSegment,
                },
            ),
            DummyRead(8, Default::default()),
            Write(8, expected.into()),
        ],
        _ => vec![
            Read(
                0,
                MemoryRecord {
                    value: lhs.into(),
                    last_accessed: MemoryAccessTimestamp::ThisSegment,
                },
            ),
            Read(
                4,
                MemoryRecord {
                    value: rhs.into(),
                    last_accessed: MemoryAccessTimestamp::ThisSegment,
                },
            ),
            Write(8, expected.into()),
        ],
    };

    state.set(0, lhs);
    state.set(4, rhs);
    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);

    I::execute(&mut running_machine, Operands([8, 0, 4, 0, 0]));
    let state = TestMachineState {
        machine: Machine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let result = state.get(8).expect("nonempty");

    assert_eq!((1, 0), state.registers());
    let result_u32: u32 = result.into();
    assert_eq!(expected, result_u32);
    for cell in state.assigned_cells() {
        println!("{:?}", cell);
    }
    assert_eq!(3, state.assigned_cells().len());
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

pub fn test_limm<I>(imm: u32, rhs: u32, expected: u32)
where
    I: Instruction<BasicMachine<BabyBear>, BabyBear>,
{
    let mut state = TestMachineState::<BasicMachine<BabyBear>>::default();
    let clk = state.clk();
    let expected_ops = match state.memory_backend.get(&12) {
        None => vec![
            Read(
                0,
                MemoryRecord {
                    value: rhs.into(),
                    last_accessed: MemoryAccessTimestamp::ThisSegment,
                },
            ),
            DummyRead(12, Default::default()),
            Write(12, expected.into()),
        ],
        _ => vec![
            Read(
                0,
                MemoryRecord {
                    value: rhs.into(),
                    last_accessed: MemoryAccessTimestamp::ThisSegment,
                },
            ),
            Write(12, expected.into()),
        ],
    };

    state.set(0, rhs);
    state.machine_mut().set_fp(4);

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);

    I::execute(&mut running_machine, Operands([8, imm as i32, -4, 1, 0]));
    state = TestMachineState {
        machine: Machine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let result = state.get(12).expect("nonempty");

    assert_eq!((1, 4), state.registers());
    let result_u32: u32 = result.into();
    assert_eq!(expected, result_u32);
    assert_eq!(2, state.assigned_cells().len());
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

pub fn test_rimm<I>(lhs: u32, imm: u32, expected: u32)
where
    I: Instruction<BasicMachine<BabyBear>, BabyBear>,
    TestMachineState<BasicMachine<BabyBear>>: Default,
{
    let mut state = TestMachineState::<BasicMachine<BabyBear>>::default();
    let clk = state.clk();
    let expected_ops = match state.memory_backend.get(&12) {
        None => vec![
            Read(
                0,
                MemoryRecord {
                    value: lhs.into(),
                    last_accessed: MemoryAccessTimestamp::ThisSegment,
                },
            ),
            DummyRead(12, Default::default()),
            Write(12, expected.into()),
        ],
        _ => vec![
            Read(
                0,
                MemoryRecord {
                    value: lhs.into(),
                    last_accessed: MemoryAccessTimestamp::ThisSegment,
                },
            ),
            Write(12, expected.into()),
        ],
    };

    state.set(0, lhs);
    state.machine_mut().set_fp(4);

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    runtime.memory_backend = state.memory_backend;
    let mut running_machine = state.machine.start(&mut runtime);
    I::execute(&mut running_machine, Operands([8, -4, imm as i32, 0, 1]));
    state = TestMachineState {
        machine: Machine::stop(running_machine),
        memory_backend: runtime.memory_backend,
    };
    let result = state.get(12).expect("nonempty");

    assert_eq!((1, 4), state.registers());
    let result_u32: u32 = result.into();
    assert_eq!(expected, result_u32);
    assert_eq!(2, state.assigned_cells().len());
    assert_eq!(state.memory_log_size(), 1);
    assert_eq!(state.memory_log(clk), expected_ops);
}

pub fn extract<T: Copy + Ord>(sel: &Selector, v: &mut BTreeSet<T>) -> T {
    v.take(&sel.select(v.iter().copied())).expect("nonempty")
}

pub fn with_unique<const N: usize, TM, F, G, T>(sel: Selector, state: TM, g: G) -> T
where
    TM: TestMachine<F>,
    TM::M: MachineWithTestChips<F>,
    F: StarkField,
    G: FnOnce(TM, u32, u32, u32, [u32; N]) -> T,
{
    let mut entries: BTreeSet<_> = state.assigned_cells().map(|x| x.0).collect();
    let (old_pc, old_fp) = state.registers();
    let clk = state.clk();

    let mut v = [0; N];
    let mut i = 0;
    while i < N {
        v[i] = extract(&sel, &mut entries);
        i += 1;
    }

    g(state, old_pc, old_fp, clk, v)
}

pub fn new_addr_strat<F: StarkField>(
    strategy: DataStrategy,
    size: MemorySize,
) -> impl Clone + Strategy<Value = (BBState, u32)> {
    let base_strat = bb_state_strategy(strategy, size);
    let max_addr = max_addr::<F>();
    (base_strat, address_strategy(max_addr))
        .prop_filter("claimed addr", |(state, addr)| state.get(*addr).is_none())
}

pub fn bb_state_strategy(
    strategy: DataStrategy,
    size: MemorySize,
) -> impl Clone + Strategy<Value = BBState> {
    let strat_pc = pc_strategy::<BabyBear>();
    let strat_fp: RangeInclusive<u32> = fp_strategy::<BabyBear>();

    let strat_addr = address_strategy_field::<BabyBear>();
    let strat_size = SizeRange::default() + size.0 as usize;
    let strat_data = match strategy {
        DataStrategy::Numerical => proptest::num::u32::ANY.sboxed(),
        DataStrategy::Address => strat_addr.clone().sboxed(),
        DataStrategy::Bitwise => proptest::bits::u32::ANY.sboxed(),
        DataStrategy::Program => strat_pc.clone().sboxed(),
    }
    .prop_map(|x| MemoryRecord {
        value: x.into(),
        last_accessed: MemoryAccessTimestamp::ThisSegment,
    });
    let strat_mem = hash_map(strat_addr, strat_data, strat_size);

    // Create machine strategy
    let strat_machine = (strat_mem, strat_pc, strat_fp).prop_map(|(_, pc, fp)| {
        let mut result = BasicMachine::default();
        result.cpu.pc = pc;
        result.cpu.fp = fp;
        result
    });

    // Create memory backend strategy
    let strat_backend = BBMemoryBackend::arbitrary_with((size, strategy));

    // Combine into BBState
    (strat_machine, strat_backend).prop_map(|(machine, backend)| BBState {
        machine,
        memory_backend: backend.0,
    })
}

use proptest::collection::hash_map;
use proptest::strategy::SBoxedStrategy;
use std::ops::RangeInclusive;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BasicMachineStrategy {
    #[default]
    Numerical,
    Address,
    Bitwise,
    Program,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MachineSize(pub usize);

impl Default for MachineSize {
    fn default() -> Self {
        MachineSize(2)
    }
}

impl From<usize> for MachineSize {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

pub fn address_strategy_field<F>() -> impl Clone + Strategy<Value = u32>
where
    F: StarkField,
{
    let max_addr = max_addr::<F>();
    (0..=max_addr / 4).prop_map(|x| 4 * x)
}

pub fn fp_strategy<F>() -> RangeInclusive<u32>
where
    F: StarkField,
{
    let max_addr = max_addr::<F>();
    0..=max_addr
}
