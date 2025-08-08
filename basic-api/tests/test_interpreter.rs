use p3_baby_bear::BabyBear;
use std::fs::read_to_string;
use valida_assembler::assemble;
use valida_basic_api::commands::common::prepare_runtime;
use valida_basic_api::{BasicMachine, BasicMachineMetrics};
use valida_cpu::MachineWithRegisters;
use valida_machine::{
    get_fixed_advice_provider, AdviceProvider, AdviceProviderWithDefault, Machine, ProgramROM,
    ReplayAdviceProvider, SegmentMachine, WriteCallbackWithDefault,
};
use valida_program::MachineWithProgramROM;

fn run_program(asm_path: &str, advice: AdviceProvider) -> Vec<u8> {
    let mut machine = BasicMachine::<BabyBear>::default();
    machine.set_segment_number(0); // `SegmentMachine`
    machine.set_max_trace_height(65536);
    let asm = read_to_string(asm_path).expect("Failed to read asm");
    let rom = ProgramROM::from_machine_code(&assemble(&asm).unwrap(), false);
    machine.set_program_rom(rom, valida_program::ProgramTableType::Public);
    let fp_init = 16777216; // default stack height
    machine.set_initial_register_values(valida_cpu::Registers { pc: 0, fp: fp_init });

    let mut runtime = prepare_runtime(
        AdviceProviderWithDefault(advice),
        WriteCallbackWithDefault::default(),
    )
    .unwrap_or_else(|err| panic!("Failed to runtime: {:?}", err));

    let mut state = machine.start(&mut runtime);
    let mut metrics = BasicMachineMetrics::initialize();

    let (_instance_data, output) = BasicMachine::run(&mut state, &mut metrics);
    output
}

#[test]
fn run_fibonacci() {
    let fib_number = 25;
    // Put the desired fib number in the advice tape.
    let advice = get_fixed_advice_provider(vec![fib_number]);

    // Run the program
    let output = run_program("tests/programs/assembly/fibonacci.val", advice);
    assert_eq!(output.len(), 4);
    let actual_result = u32::from_le_bytes(output.try_into().unwrap());

    let expected_result = fibonacci(fib_number);
    assert_eq!(actual_result, expected_result);
}

fn fibonacci(n: u8) -> u32 {
    let mut a = 0u32;
    let mut b = 1u32;
    for _ in 0..n {
        let temp = a;
        a = b;
        (b, _) = temp.overflowing_add(b);
    }
    a
}

#[test]
fn endianness_and_loadu8_storeu8() {
    let output = run_program(
        "tests/programs/assembly/endianess_and_loadu8_storeu8.val",
        get_fixed_advice_provider(vec![]),
    );
    // verifies that two LSBs from 0x01020304 word immediate are 4 and 3
    // the immediate is loaded with `imm32` and LSBs do not change after `storeu8` and `loadu8`
    assert_eq!(output, vec![1, 2]);
}
