use p3_field::PrimeField32;
use valida_machine::{
    AdviceProviderWithDefault, MachineRuntime, ReplayAdviceProvider, ValidaMemoryBackend,
    WriteCallbackWithDefault,
};
use valida_memory_footprint::MemoryFootprint;

pub struct ValidaRuntime {
    pub memory_backend: ValidaMemoryBackend,
    pub write_callback: WriteCallbackWithDefault,
    pub advice_provider: AdviceProviderWithDefault,
    pub replay_advice: ReplayAdviceProvider,
}

impl MemoryFootprint for ValidaRuntime {
    fn memory_footprint(&self) -> usize {
        // skippping over the write callback & advice provider for now, they are just fn pointers in a box
        self.memory_backend.memory_footprint() + self.replay_advice.memory_footprint()
    }
}

impl ValidaRuntime {
    pub fn default_for_field<F: PrimeField32>() -> Self {
        Self {
            memory_backend: ValidaMemoryBackend::default_for_field::<F>(),
            write_callback: WriteCallbackWithDefault::default(),
            advice_provider: AdviceProviderWithDefault::default(),
            replay_advice: ReplayAdviceProvider::default(),
        }
    }
}

impl MachineRuntime for ValidaRuntime {
    type MemoryBackend = ValidaMemoryBackend;
    //type Adv = AdviceProviderWithDefault;
    fn memory_backend(&self) -> &ValidaMemoryBackend {
        &self.memory_backend
    }

    fn memory_backend_mut(&mut self) -> &mut ValidaMemoryBackend {
        &mut self.memory_backend
    }

    fn write_callback(&mut self) -> &mut WriteCallbackWithDefault {
        &mut self.write_callback
    }

    /// Pushs a value read from the input advice in the `ReplayAdviceProvider`
    fn push_advice(&mut self, val: Option<u8>) {
        self.replay_advice.push_advice(val);
    }

    /// Copies the replay advice from the runtime
    fn get_replay_advice(&self) -> ReplayAdviceProvider {
        self.replay_advice.clone()
    }

    fn advice(&mut self) -> &mut AdviceProviderWithDefault {
        &mut self.advice_provider
    }
}
