extern crate alloc;
use alloc::alloc::{alloc_zeroed, Layout};
use std::io::{stdin, Read};
use std::process;

use p3_baby_bear::BabyBear;
use postcard::from_bytes;

use valida_basic_api::commands::common::{default_config, prepare_basic_machine};
use valida_basic_api::instance_data::ValidaSegmentInstanceData;
use valida_basic_api::{BasicMachine, Registers};
use valida_elf::load_elf_object_file;
use valida_machine::Machine;
use valida_program::MachineWithProgramROM;
use valida_program::ProgramTableType;
use valida_static_data::{MachineWithStaticDataChip, StaticDataChipType};

fn zero_vec<T: Sized>(size: usize) -> Vec<T> {
    if size == 0 {
        Vec::with_capacity(0)
    } else {
        let elem_size = core::mem::size_of::<T>();
        let elem_align = core::mem::size_of::<T>();
        let layout = Layout::from_size_align(elem_size * size, elem_align).expect("ok");
        unsafe {
            let vec_ptr = alloc_zeroed(layout) as *mut T;
            assert!(!vec_ptr.is_null());
            Vec::from_raw_parts(vec_ptr, elem_size * size, elem_size * size)
        }
    }
}

fn read_component<R: Read>(reader: &mut R) -> Vec<u8> {
    let mut u32_buf = [0u8; 4];
    reader.read_exact(&mut u32_buf).expect("ok");
    let mut v = zero_vec(u32::from_le_bytes(u32_buf) as usize);
    reader.read_exact(&mut v).expect("ok");
    v
}

pub fn recursive_main(setup: ProgramTableType) {
    let mut handle = stdin();
    let config = default_config();

    let elf_vec = read_component(&mut handle);
    let output_vec = read_component(&mut handle);
    let vk_vec = read_component(&mut handle);
    let proof_vec = read_component(&mut handle);

    let program = load_elf_object_file(&elf_vec);
    let machine = prepare_basic_machine(
        program.code.clone(),
        program.data.clone(),
        Registers {
            pc: program.initial_program_counter,
            fp: 1006632952, // TODO: default initial frame pointer value. It's last 8 byte aligned value in memory address space: 0x3bfffff8
        },
        setup == ProgramTableType::Preprocessed,
        1_048_576, // 2**20
        elf_vec,
    );

    let show_public = vec![false; BasicMachine::<BabyBear>::NUM_CHIPS];

    let (rom, static_data) = match (
        machine.program_table_type(),
        machine.static_data().chip_type(),
    ) {
        (ProgramTableType::Public, StaticDataChipType::Public) => {
            (Some(program.code), Some(program.data))
        }
        (ProgramTableType::Preprocessed, StaticDataChipType::Preprocessed) => (None, None),
        // These two options are at present unreachable
        (ProgramTableType::Public, StaticDataChipType::Preprocessed) => (Some(program.code), None),
        (ProgramTableType::Preprocessed, StaticDataChipType::Public) => (None, Some(program.data)),
    };

    let vk = from_bytes(&vk_vec).expect("good vk");
    let proof = ciborium::from_reader(proof_vec.as_slice()).expect("good proof");
    let instance_data = ValidaSegmentInstanceData {
        rom,
        output: output_vec,
        pc_init: program.initial_program_counter,
        fp_init: 1006632952, // TODO: default initial frame pointer value. It's last 8 byte aligned value in memory address space: 0x3bfffff8
        did_fail: false,
        did_stop: true,
        pc_final: 0, // Not used by verify method
        fp_final: 0, // Not used by verify method
        static_data,
        is_last_segment: true,
        segment_number: 0,
    };
    match machine.verify(&config, &proof, &vk, &instance_data, show_public) {
        Ok(_) => println!("success"),
        Err(_) => println!("failure"),
    }
}
