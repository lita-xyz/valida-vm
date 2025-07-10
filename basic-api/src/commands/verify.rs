use std::collections::BTreeMap;
use std::fs::read;
use std::io::{stdout, Write};

use p3_baby_bear::BabyBear;
use p3_field::{PrimeField32, TwoAdicField};
use postcard::from_bytes;

use valida_cpu::Registers;
use valida_elf::Program;
use valida_machine::{
    Machine, MachineProof, MultiSegmentMachineProof, RunningMachine, StarkConfig, StarkField, Word,
};
use valida_program::{MachineWithProgramChip, MachineWithProgramROM, ProgramTableType};
use valida_static_data::{MachineWithStaticDataChip, StaticDataChipType};

use crate::commands::common::{prepare_basic_machine, prepare_machine};
use crate::instance_data::{ValidaInstanceData, ValidaSegmentInstanceData};
use crate::{BasicMachine, MultiSegmentBasicMachine, ValidaRuntime};

use ark_std::{end_timer, start_timer};

use super::common::{MyConfig, MyPK, MyVK};

/// A verification function for the basic machine (i.e. for a single segment)
pub fn verify_basic_machine(
    program: Program,
    config: MyConfig,
    output: Vec<u8>,
    proof_bytes: Vec<u8>,
    option_vk: Option<Vec<u8>>,
    show_public: bool,
    show_preprocessed: bool,
    initial_fp: u32,
    specific: bool,
    max_trace_height: u32,
    program_file: Vec<u8>,
) -> Result<(), String> {
    let t_prep = start_timer!(|| "valida | Prepare machine");
    let mut machine = prepare_basic_machine(
        program.code.clone(),
        program.data.clone(),
        Registers {
            pc: program.initial_program_counter,
            fp: initial_fp,
        },
        specific,
        max_trace_height,
        program_file,
    );
    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    // TODO: Do we even need a running machine for the verifier?
    let mut state = machine.start(&mut runtime);

    end_timer!(t_prep);

    let show_public = vec![show_public; BasicMachine::<BabyBear>::NUM_CHIPS];
    let show_preprocessed = vec![show_preprocessed; BasicMachine::<BabyBear>::NUM_CHIPS];

    let t_deser = start_timer!(|| "valida > run | Input deserialization");

    let (rom, static_data) = match (
        state.machine.program_table_type(),
        state.machine.static_data().chip_type(),
    ) {
        (ProgramTableType::Public, StaticDataChipType::Public) => {
            (Some(program.code), Some(program.data))
        }
        (ProgramTableType::Preprocessed, StaticDataChipType::Preprocessed) => (None, None),
        // These two options are at present unreachable
        (ProgramTableType::Public, StaticDataChipType::Preprocessed) => (Some(program.code), None),
        (ProgramTableType::Preprocessed, StaticDataChipType::Public) => (None, Some(program.data)),
    };
    let instance_data = ValidaSegmentInstanceData {
        rom,
        static_data,
        output,
        pc_init: program.initial_program_counter,
        fp_init: initial_fp,
        pc_final: 0, // WARNING: We set the final program counter and frame pointer to 0
        fp_final: 0, //   At the moment neither of them is used in the verification!
        did_fail: false,
        did_stop: true,
        is_last_segment: true,
        segment_number: 0,
    };

    let proof: MachineProof<MyConfig> = ciborium::from_reader(proof_bytes.as_slice())
        .map_err(|err| format!("Proof deserialization failed: {:?}", err))?;
    end_timer!(t_deser);

    let t_verif = start_timer!(|| "valida > verify | state.machine.verify(..)");
    let vk = option_vk
        .map_or_else(
            || {
                Ok(state
                    .machine
                    .pre_process(&config, show_preprocessed, false)
                    .1)
            },
            |vk| from_bytes(&vk),
        )
        .map_err(|err| format! {"Incorrect vk: {:?}", err})?;

    let verification_result = state
        .machine
        .verify(&config, &proof, &vk, &instance_data, show_public)
        .map_err(|err| format!("Verification error: {:?}", err));
    end_timer!(t_verif);
    verification_result

    //Ok(())
}

fn get_segments<SC: StarkConfig, F>(
    state: &RunningMachine<'_, F, MultiSegmentBasicMachine<F>>,
    proof: &MultiSegmentMachineProof<SC>,
    program: &Program,
) -> (Vec<ValidaSegmentInstanceData>, Vec<u8>)
where
    F: StarkField,
{
    // TODO: for now we reuse the same value for public / preprocessed for the static data chip
    // as for the program table, because in `basic.rs::init` we map them directly at the moment
    let (rom, static_data) = match (state.machine.program_table_type()) {
        ProgramTableType::Public => (Some(program.code.clone()), Some(program.data.clone())),
        ProgramTableType::Preprocessed => (None, None),
    };
    let mut segments: Vec<ValidaSegmentInstanceData> = vec![];

    let mut full_segments_output = vec![];

    for (i, s) in proof.segment_proofs.iter().enumerate() {
        let is_last_segment = if i >= proof.segment_proofs.len() - 1 {
            true
        } else {
            false
        };

        // Only segment 0 has well defined static data
        let static_data = if i == 0 {
            static_data.clone()
        } else {
            // In other cases we *MUST* still have `Some` here. This is required to match the behavior of the
            // prover. And in the prover we need _something_ for the static data chip, because otherwise for segments
            // larger 1, we would get a height of zero in `degrees_and_g_subgroups`
            Some(BTreeMap::<u32, Word<u8>>::default())
        };

        let seg_output = s.instance_data.output.clone();
        full_segments_output.extend(seg_output.clone());

        // TODO: Consider if we want/need each segment instance to have rom & output
        let seg = ValidaSegmentInstanceData {
            pc_init: s.instance_data.pc_init,
            fp_init: s.instance_data.fp_init,
            pc_final: s.instance_data.pc_final,
            fp_final: s.instance_data.fp_final,
            rom: rom.clone(),
            static_data,
            output: seg_output,
            did_fail: false,
            did_stop: is_last_segment, // last segment also stops
            is_last_segment,
            segment_number: i.try_into().expect("More than 2^32 segments"),
        };

        segments.push(seg);
    }

    (segments, full_segments_output)
}

pub fn read_proof_build_instance_data<F: StarkField>(
    state: &mut RunningMachine<'_, F, MultiSegmentBasicMachine<F>>,
    proof_bytes: Vec<u8>,
    output: Vec<u8>,
    program: Program,
    initial_fp: u32,
) -> Result<(MultiSegmentMachineProof<MyConfig>, ValidaInstanceData), String> {
    let t_deser = start_timer!(|| "valida > run | Input deserialization");

    let proof: MultiSegmentMachineProof<MyConfig> =
        ciborium::from_reader(proof_bytes.as_slice())
            .map_err(|err| format!("Proof deserialization failed: {:?}", err))?;
    end_timer!(t_deser);

    // construct the segments with their pc,fp values from the machine prooof
    let (segments, full_segments_output) = get_segments(&state, &proof, &program);
    state.machine.segment_machine = BasicMachine::default(); // just need a default machine to access chips etc

    // verify the segment's output matches the claimed output the user provided
    if output != full_segments_output {
        return Err(
            "Given claimed output does not match output of all segments stored in the proof"
                .to_string(),
        );
    }

    let instance_data = ValidaInstanceData {
        rom: segments[0].rom.clone(),
        output,
        pc_init: program.initial_program_counter,
        fp_init: initial_fp,
        did_fail: false,
        segments,
    };
    Ok((proof, instance_data))
}

pub fn verify(
    program: Program,
    config: MyConfig,
    output: Vec<u8>,
    proof_bytes: Vec<u8>,
    option_vk: Option<Vec<u8>>,
    show_public: bool,
    show_preprocessed: bool,
    initial_fp: u32,
    specific: bool,
    max_trace_height: u32,
    program_file: Vec<u8>,
) -> Result<(), String> {
    let t_prep = start_timer!(|| "valida | Prepare machine");
    let mut machine = prepare_machine(
        program.code.clone(),
        program.data.clone(),
        Registers {
            pc: program.initial_program_counter,
            fp: initial_fp,
        },
        specific,
        max_trace_height,
        program_file,
    );
    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    let mut state = machine.start(&mut runtime);

    end_timer!(t_prep);

    let show_public = vec![show_public; BasicMachine::<BabyBear>::NUM_CHIPS];
    let show_preprocessed = vec![show_preprocessed; BasicMachine::<BabyBear>::NUM_CHIPS];

    let (proof, instance_data) =
        read_proof_build_instance_data(&mut state, proof_bytes, output, program, initial_fp)?;

    let t_verif = start_timer!(|| "valida > verify | state.machine.verify(..)");
    let vk = option_vk
        .map_or_else(
            || {
                Ok(state
                    .machine
                    .pre_process(&config, show_preprocessed, false)
                    .1)
            },
            |vk| from_bytes(&vk),
        )
        .map_err(|err| format! {"Incorrect vk: {:?}", err})?;

    let verification_result = state
        .machine
        .verify(&config, &proof, &vk, &instance_data, show_public)
        .map_err(|err| format!("Verification error: {:?}", err));
    end_timer!(t_verif);
    verification_result
}
