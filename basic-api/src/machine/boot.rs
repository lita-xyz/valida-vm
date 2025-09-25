use std::collections::BTreeMap;

use valida_cpu::{Operation, Registers};
use valida_machine::{ProgramROM, ValidaMemoryBackend, Word};
use valida_memory_footprint::MemoryFootprint;
use valida_program::ProgramTableType;
use valida_static_data::StaticDataChipType;

/// The boot data for a machine.
pub struct ValidaBootData {
    /// Program ROM
    pub program_rom: ProgramROM<i32>,
    /// Program table type
    pub program_table_type: ProgramTableType,
    /// Static data
    pub static_data: BTreeMap<u32, Word<u8>>,
    /// Static data chip type
    pub static_data_chip_type: StaticDataChipType,
    /// Initial register values
    pub initial_register_values: Registers,
    /// Maximum allowed trace height
    pub max_trace_height: u32,
    /// A binary representation of a loaded ELF
    pub program_file: Vec<u8>,
}

impl MemoryFootprint for ValidaBootData {
    fn memory_footprint(&self) -> usize {
        self.program_rom.memory_footprint()
            + self.program_table_type.memory_footprint()
            + self.static_data.memory_footprint()
            + self.static_data_chip_type.memory_footprint()
            + self.initial_register_values.memory_footprint()
            + self.max_trace_height.memory_footprint()
            + self.program_file.memory_footprint()
    }
}

/// The same as `ValidaBootData`, but with the segment number and static data is optional.
#[derive(Default, Debug, Clone)]
pub struct ValidaSegmentBootData {
    pub initial_register_values: Registers,
    /// Program ROM
    pub program_rom: ProgramROM<i32>,
    /// Program table type
    pub program_table_type: ProgramTableType,
    /// Segment number
    pub segment_number: u32,
    /// Maximum allowed trace height
    pub max_trace_height: u32,
    /// a binary representation of a loaded ELF
    pub program_file: Vec<u8>,
    /// Optional static data
    pub static_data: Option<BTreeMap<u32, Word<u8>>>,
    /// Optional static data chip type
    pub static_data_chip_type: Option<StaticDataChipType>,
    /// If trace generation is enabled (`--fast` command line arg / `log_enabled` fn)
    /// Passed from the `MultiSegmentBasicMachine` to the child `BasicMachines`
    pub log_enabled: bool,
    /// Initial memory state for this segment
    pub initial_memory_state: ValidaMemoryBackend,
}

impl MemoryFootprint for ValidaSegmentBootData {
    fn memory_footprint(&self) -> usize {
        self.initial_register_values.memory_footprint()
            + self.program_rom.memory_footprint()
            + self.program_table_type.memory_footprint()
            + self.segment_number.memory_footprint()
            + self.max_trace_height.memory_footprint()
            + self.program_file.memory_footprint()
            + self.static_data.memory_footprint()
            + self.static_data_chip_type.memory_footprint()
            + self.log_enabled.memory_footprint()
            + self.initial_memory_state.memory_footprint()
    }
}
