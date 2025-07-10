use core::slice;
use std::fs::File;
use std::io;
use std::io::Read;

use valida_memory_footprint::MemoryFootprint;

/// Get the next byte from the advice tape, if any.
pub type AdviceProvider = Box<dyn FnMut() -> Option<u8> + Send + Sync>;

pub struct AdviceProviderWithDefault(pub AdviceProvider);

/// A `ReplayAdviceProvider` simply stores the entirety of the advice (i.e. input) of the
/// program recorded during the initial execution pass. This allows us to replay the input
/// in further execution passes done in the prover.
#[derive(Clone, Debug, Default)]
pub struct ReplayAdviceProvider {
    pub advice: Vec<u8>,
}

impl MemoryFootprint for ReplayAdviceProvider {
    fn memory_footprint(&self) -> usize {
        self.advice.memory_footprint()
    }
}

impl Default for AdviceProviderWithDefault {
    fn default() -> Self {
        Self(Box::new(|| None))
    }
}

impl ReplayAdviceProvider {
    pub fn push_advice(&mut self, val: Option<u8>) {
        match val {
            Some(v) => self.advice.push(v),
            None => (),
        }
    }
}

pub fn get_fixed_advice_provider(advice: Vec<u8>) -> AdviceProvider {
    let mut index: usize = 0;
    Box::new(move || {
        if index < advice.len() {
            let advice_byte = advice[index];
            index += 1;
            Some(advice_byte)
        } else {
            None
        }
    })
}

/// Constructs an advice provider from a `ReplayAdviceProvider` to get the same execution
/// behavior during subsequent execution passes
pub fn get_replay_advice_provider(replay_advice: ReplayAdviceProvider) -> AdviceProvider {
    get_fixed_advice_provider(replay_advice.advice)
}

pub fn get_fixed_advice_provider_from_file(file: &mut File) -> Result<AdviceProvider, String> {
    let mut advice = Vec::new();
    file.read_to_end(&mut advice)
        .map_err(|err| format!("Cannot read the advice file: {:?}", err))?;
    Ok(get_fixed_advice_provider(advice))
}

#[cfg(feature = "std")]
pub fn get_stdin_advice_provider() -> AdviceProvider {
    Box::new(move || {
        let mut advice_byte = 0u8;
        match io::stdin().read_exact(slice::from_mut(&mut advice_byte)) {
            Ok(_) => Some(advice_byte),
            Err(_) => None,
        }
    })
}
