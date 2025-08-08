#![no_std]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use elf::abi::{PF_X, PT_LOAD, PT_TLS};
use elf::endian::LittleEndian;
use elf::segment::ProgramHeader;
use elf::ElfBytes;
use elf::ParseError;
use valida_machine::{ProgramROM, Word};

pub struct Program {
    pub code: ProgramROM<i32>,
    pub data: BTreeMap<u32, Word<u8>>,
    pub initial_program_counter: u32,
}

pub fn create_func_address_to_name_map(
    file: &ElfBytes<'_, LittleEndian>,
) -> Result<BTreeMap<u64, String>, ParseError> {
    let (symbol_table, string_table) = file.symbol_table()?.unwrap();
    #[allow(non_snake_case)]
    let STT_FUNC = 2;
    let map: BTreeMap<_, _> = symbol_table
        .iter()
        .filter(|symbol| symbol.st_symtype() == STT_FUNC)
        .map(|symbol| {
            (
                symbol.st_value,
                string_table
                    .get(symbol.st_name as usize)
                    .unwrap()
                    .to_string(),
            )
        })
        .collect();

    Ok(map)
}

pub fn minimal_parse_elf(file: &Vec<u8>) -> ElfBytes<'_, LittleEndian> {
    ElfBytes::<LittleEndian>::minimal_parse(file.as_slice()).unwrap()
}

fn load_text<'a>(
    file: &ElfBytes<'a, LittleEndian>,
    loadable_segments: &[ProgramHeader],
) -> &'a [u8] {
    let text_segments: Vec<_> = loadable_segments
        .iter()
        .filter(|ph| ph.p_flags & PF_X != 0)
        .collect();

    if text_segments.len() != 1 {
        panic!("Unexpected number of text segments");
    }

    let text_segment = text_segments[0];

    if text_segment.p_paddr != 0 || text_segment.p_vaddr != 0 {
        panic!("Unexpected text segment physical address");
    }

    if text_segment.p_filesz != text_segment.p_memsz {
        panic!("text_segment.p_filesz != text_segment.p_memsz");
    }

    file.segment_data(text_segment)
        .expect("Could not find segment data")
}

fn load_data(
    file: &ElfBytes<'_, LittleEndian>,
    loadable_segments: &[ProgramHeader],
) -> BTreeMap<u32, Word<u8>> {
    let data_segments: Vec<_> = loadable_segments
        .iter()
        .filter(|ph| ph.p_flags & PF_X == 0)
        .collect();

    let mut data: BTreeMap<u32, Word<u8>> = BTreeMap::new();
    for ph in data_segments {
        if ph.p_memsz < ph.p_filesz {
            panic!("ph.p_memsz < ph.p_filesz");
        }

        if ph.p_paddr != ph.p_vaddr {
            panic!("ph.p_paddr != ph.p_vaddr");
        }

        if ph.p_paddr % 4 != 0 {
            panic!("ph.p_paddr % 4 != 0");
        }

        // `ph.p_memsz - ph.p_filesz` corresponds to bss data
        // which is supposed to be zero initialized in the vm.
        // In valida vm the whole data space is zero initialized by default.
        // Thus bss data requires no explicit handing.
        let mut section_data =
            Vec::from(file.segment_data(ph).expect("Could not find segment data"));

        section_data.resize(((ph.p_filesz + 3) & !3) as usize, 0);

        for i in 0..(section_data.len() / 4) {
            data.insert(
                <u64 as TryInto<u32>>::try_into(ph.p_paddr).unwrap()
                    + <usize as TryInto<u32>>::try_into(i * 4).unwrap(),
                u32::from_le_bytes([
                    section_data[i * 4],
                    section_data[i * 4 + 1],
                    section_data[i * 4 + 2],
                    section_data[i * 4 + 3],
                ])
                .into(),
            );
        }
    }
    data
}

pub fn load_elf_object_file(file: &Vec<u8>, should_convert_opcode: bool) -> Program {
    let file = minimal_parse_elf(file);

    let segments = file.segments().unwrap();

    let loadable_segments: Vec<_> = segments.iter().filter(|ph| ph.p_type == PT_LOAD).collect();

    let has_tls = !segments
        .iter()
        .filter(|ph| ph.p_type == PT_TLS)
        .collect::<Vec<_>>()
        .is_empty();

    if has_tls {
        panic!("ELF file is not supposed to have PT_TLS segments");
    };

    let code = load_text(&file, &loadable_segments);
    let data = load_data(&file, &loadable_segments);

    Program {
        code: ProgramROM::from_machine_code(code, should_convert_opcode),
        data,
        initial_program_counter: 0,
    }
}
