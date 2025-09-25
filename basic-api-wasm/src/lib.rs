mod utils;

use valida_basic_api::commands::common::default_config;
use valida_basic_api::commands::{get_fixed_advice_provider, AdviceProviderWithDefault};
use valida_basic_api::load_elf_object_file;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

const INITIAL_FP: u32 = 1006632952;

#[wasm_bindgen]
pub fn run(
    program_bytes: Vec<u8>,
    stdin: Vec<u8>,
    max_trace_height: u32,
) -> Result<Vec<u8>, String> {
    let program = load_elf_object_file(&program_bytes);
    let specific = false;

    let (status, stdout) = valida_basic_api::commands::run(
        program,
        true,
        Default::default(),
        AdviceProviderWithDefault(get_fixed_advice_provider(stdin)),
        INITIAL_FP,
        specific,
        max_trace_height,
        program_bytes,
    );

    if status {
        Ok(stdout)
    } else {
        Err("Terminated with FAIL".to_string())
    }
}

#[wasm_bindgen]
pub fn prove(
    program_bytes: Vec<u8>,
    stdin: Vec<u8>,
    max_trace_height: u32,
) -> Result<Vec<u8>, String> {
    let program = load_elf_object_file(&program_bytes);
    let specific = false;

    valida_basic_api::commands::prove(
        program,
        Default::default(),
        AdviceProviderWithDefault(get_fixed_advice_provider(stdin)),
        None,
        default_config(),
        INITIAL_FP,
        specific,
        max_trace_height,
        program_bytes,
        1, // max_parallel_segments - use sequential execution for WASM
    )
}

#[wasm_bindgen]
pub fn verify(
    program_bytes: Vec<u8>,
    stdout: Vec<u8>,
    proof: Vec<u8>,
    max_trace_height: u32,
) -> Result<(), String> {
    let program = load_elf_object_file(&program_bytes);
    let specific = false;

    valida_basic_api::commands::verify(
        program,
        default_config(),
        stdout,
        proof,
        None,
        false,
        false,
        INITIAL_FP,
        specific,
        max_trace_height,
        program_bytes,
    )
}

#[cfg(test)]
pub mod tests {
    use rust_embed::Embed;
    use wasm_bindgen_test::*;

    use crate::{prove, run, verify};

    // TODO: Just putting a constant. 2^20 is the default value we currently use
    // as part of the `Shared` config struct in `basic/src/bin/args/mod.rs`.
    pub const MAX_TRACE_HEIGHT: u32 = 1 << 20;

    #[derive(Embed)]
    #[folder = "$CARGO_MANIFEST_DIR/binary/"]
    struct Asset;

    fn wasm_bindgen_test(max_trace_height: u32) {
        let file = Asset::get("fibonacci")
            .expect("fibonacci example in Rust examples should be already built in release mode");

        let input = 6;
        let output = 8;

        let mut expected_output = format!(
            "Please enter a number from 0 to 46:\nThe {}-th fibonacci number is: {}",
            input, output
        )
        .as_bytes()
        .to_vec();
        expected_output.push(10);

        let run_result = run(
            file.data.to_vec(),
            input.to_string().as_bytes().to_vec(),
            max_trace_height,
        );

        assert_eq!(run_result, Ok(expected_output));

        let prove_result = prove(
            file.data.to_vec(),
            input.to_string().as_bytes().to_vec(),
            max_trace_height,
        );

        assert!(prove_result.is_ok());

        let verify_result = verify(
            file.data.to_vec(),
            run_result.unwrap(),
            prove_result.unwrap(),
            max_trace_height,
        );

        assert!(verify_result.is_ok());
    }

    #[wasm_bindgen_test]
    #[test]
    fn wasm_bindings_test_max_trace_size() {
        wasm_bindgen_test(MAX_TRACE_HEIGHT);
    }

    #[wasm_bindgen_test]
    #[test]
    fn wasm_bindings_test_small_trace_size() {
        wasm_bindgen_test(8192);
    }
}
