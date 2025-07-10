use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;
use valida_cpu::MachineWithCpuChip;
use valida_machine::{
    Machine, MemoryAccessTimestamp, MemoryRecord, StorageBackendTrait, ValidaMemoryBackend, Word,
};
use valida_memory::{MachineWithMemoryChip, Operation};

pub trait TestMachine<F>
where
    F: PrimeField32,
{
    type M: MachineWithCpuChip<F> + MachineWithMemoryChip<F>;
    fn machine(&self) -> &Self::M;
    fn machine_mut(&mut self) -> &mut Self::M;

    fn registers(&self) -> (u32, u32);
    fn clk(&self) -> u32;

    fn num_cells(&self) -> usize;
    fn assigned_cells(&self) -> impl ExactSizeIterator<Item = (u32, Word<u8>)>;
    fn get(&self, addr: u32) -> Option<Word<u8>>;
    fn set(&mut self, addr: u32, val: u32);
    fn memory_log_size(&self) -> usize;
    fn memory_log(&self, clk: u32) -> Vec<Operation>;
}

#[derive(Debug, Clone)]
pub struct TestMachineState<M: Machine<BabyBear>> {
    pub machine: M,
    pub memory_backend: ValidaMemoryBackend,
}

impl<M> TestMachine<BabyBear> for TestMachineState<M>
where
    M: MachineWithCpuChip<BabyBear> + MachineWithMemoryChip<BabyBear>,
{
    type M = M;

    fn machine(&self) -> &M {
        &self.machine
    }
    fn machine_mut(&mut self) -> &mut M {
        &mut self.machine
    }

    fn registers(&self) -> (u32, u32) {
        (self.machine.cpu().pc, self.machine.cpu().fp)
    }

    fn clk(&self) -> u32 {
        self.machine.cpu().clock
    }

    fn num_cells(&self) -> usize {
        self.memory_backend.len()
    }

    fn assigned_cells(&self) -> impl ExactSizeIterator<Item = (u32, Word<u8>)> {
        self.memory_backend
            .iter()
            .map(|(addr, &MemoryRecord { value, .. })| (addr, value))
    }

    fn get(&self, addr: u32) -> Option<Word<u8>> {
        self.memory_backend
            .get(&addr)
            .copied()
            .map(|record| record.value)
    }

    fn set(&mut self, addr: u32, val: u32) {
        self.memory_backend.set(
            addr,
            MemoryRecord {
                value: val.into(),
                last_accessed: MemoryAccessTimestamp::ThisSegment,
            },
        );
    }

    fn memory_log_size(&self) -> usize {
        self.machine.mem().operations.len()
    }

    fn memory_log(&self, clk: u32) -> Vec<Operation> {
        self.machine
            .mem()
            .operations
            .get(&clk)
            .cloned()
            .unwrap_or_default()
            .into_vec()
    }
}
