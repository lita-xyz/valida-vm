use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::Write;

use valida_machine::{InstructionWord, MachineMetrics};
use valida_opcodes::Opcode;

use crate::metrics::func_cpu_usage::{FuncCpuUsage, MemFpRelativeReader};

pub struct BasicMachineMetrics {
    func_cpu_usage: FuncCpuUsage,
    opcode_usage: BTreeMap<Opcode, usize>,
    profiling_file: Option<File>,
}

impl MachineMetrics for BasicMachineMetrics {}

fn get_profiling_file() -> Option<File> {
    let file_path = env::var("PROFILING_PATH").ok()?;
    Some(File::create(file_path).expect("Cannot create profiling file"))
}

impl BasicMachineMetrics {
    pub fn initialize() -> Self {
        Self {
            func_cpu_usage: FuncCpuUsage::initialize(),
            opcode_usage: Default::default(),
            profiling_file: get_profiling_file(),
        }
    }

    // Dump profiling information to env file "PROFILING_PATH"
    // and opcode usage metrics to stdout
    pub fn finalize(mut self, program_file: &Vec<u8>) {
        if let Some(mut file) = self.profiling_file {
            eprintln!("Opcode counts:");
            let mut total = 0;
            for (opcode, count) in &self.opcode_usage {
                eprintln!("{:?}: {}", opcode, count);
                total += count;
            }
            eprintln!("Total: {}", total);

            let as_flamegraph = self.func_cpu_usage.as_flamegraph(program_file);
            file.write_all(as_flamegraph.as_bytes())
                .expect("Unable to write data");

            eprintln!(
                "Profiling info stored at: \"{}\"",
                env::var("PROFILING_PATH").expect("File path should be valid")
            );
        }
    }

    fn register_instruction_impl(
        &mut self,
        inst: &InstructionWord<i32>,
        state: &impl MemFpRelativeReader,
    ) {
        if let Ok(opcode) = Opcode::try_from(inst.opcode) {
            *self.opcode_usage.entry(opcode).or_insert(0) += 1;
        }

        self.func_cpu_usage.on_step(inst, state);
    }

    /// Register an instruction to our metrics tracker
    // The first part, checking whether metrics tracking is on
    // is always inlined to limit overhead of function calls
    #[inline(always)]
    pub fn register_instruction(
        &mut self,
        inst: &InstructionWord<i32>,
        state: &impl MemFpRelativeReader,
    ) {
        if let Some(ref profiling_file1) = self.profiling_file {
            self.register_instruction_impl(inst, state)
        }
    }
}
