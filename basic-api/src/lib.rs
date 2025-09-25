#![allow(unused)]
#![feature(trait_upcasting)]

use std::env;
use std::fs::File;

extern crate alloc;

pub mod commands;
pub mod instance_data;
pub use instance_data::{ValidaInstanceData, ValidaSegmentInstanceData};

pub mod machine;
// Re-export the core types from the crate root.
pub use machine::{
    basic::{BasicMachine, BasicRunningMachine},
    boot::{ValidaBootData, ValidaSegmentBootData},
    multi_segment::MultiSegmentBasicMachine,
    runtime::ValidaRuntime,
};

pub(crate) mod metrics;
pub use metrics::metrics::BasicMachineMetrics;

// Re-exports for convenience
pub use valida_cpu::Registers;
// This was previously `pub use valida_elf::*;``
pub use valida_elf::load_elf_object_file;
pub use valida_machine::Word;

pub mod embdebug;
