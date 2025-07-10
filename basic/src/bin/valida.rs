use ark_std::{end_timer, start_timer};
use clap::Parser;

use std::fs::{self, read, File};
use std::io::Write;
use valida_basic_api::commands::common::default_config;
use valida_basic_api::commands::prove::ProveDebugOptions;
use valida_basic_api::commands::{
    get_file_write_callback, get_fixed_advice_provider_from_file, get_stdin_advice_provider,
    preprocess, preprocess_basic_machine, prove, prove_basic_machine, run, run_basic_machine,
    verify, verify_basic_machine, AdviceProviderWithDefault, WriteCallbackWithDefault,
};
use valida_elf::load_elf_object_file;

mod args;
use args::*;

mod interactive;
use interactive::repl_run;

fn create_write_callback_from_arg(path: String) -> Result<WriteCallbackWithDefault, String> {
    File::create(path)
        .map_err(|err| format!("Cannot create output file: {:?}", err))
        .map(get_file_write_callback)
        .map(WriteCallbackWithDefault)
}

fn create_advice_provider_from_arg(
    option_path: Option<String>,
) -> Result<AdviceProviderWithDefault, String> {
    match option_path {
        Some(file_name) => {
            let mut file = File::open(file_name)
                .map_err(|err| format!("Cannot open advice file: {:?}", err))?;
            get_fixed_advice_provider_from_file(&mut file).map(AdviceProviderWithDefault)
        }
        None => Ok(AdviceProviderWithDefault(get_stdin_advice_provider())),
    }
}

fn main() {
    let args = Args::parse();

    let (program, initial_fp, max_trace_height) = match args.action {
        Commands::Run(ref cmd) => (
            &cmd.shared.program,
            cmd.shared.initial_fp,
            cmd.shared.max_trace_height,
        ),
        // initial_fp is not relevant for preprocess command
        Commands::Preprocess(ref cmd) => (&cmd.program, 0, cmd.max_trace_height),
        Commands::Prove(ref cmd) => (
            &cmd.shared.program,
            cmd.shared.initial_fp,
            cmd.shared.max_trace_height,
        ),
        Commands::Verify(ref cmd) => (
            &cmd.shared.program,
            cmd.shared.initial_fp,
            cmd.shared.max_trace_height,
        ),
        Commands::Interactive(ref cmd) => (
            &cmd.shared.program,
            cmd.shared.initial_fp,
            cmd.shared.max_trace_height,
        ),
    };

    let t_deser = start_timer!(|| format!(
        "valida | Program deserialization ({} bytes)",
        fs::metadata(&program)
            .unwrap_or_else(|_| panic!("Failed to get executable file size: {}", &program))
            .len()
    ));

    let program_file =
        fs::read(program).unwrap_or_else(|_| panic!("Failed to read executable file: {}", program));
    let program = load_elf_object_file(&program_file);
    end_timer!(t_deser);

    let t_action = start_timer!(|| format!("valida | {}", args.action));
    let config = default_config();

    // TODO: set segment_number appropriately when proving multi-segment executions
    let segment_number = 0;

    // NOTE: Currently we call the run/preprocess/prove/verify functions that internally use
    // the `BasicMachine` instead of the MultiSegmentBasicMachine. Once the latter is fully
    // implemented, we'll switch over.
    match args.action {
        Commands::Run(cmd) => {
            let (status, _) = run(
                program,
                cmd.fast,
                create_write_callback_from_arg(cmd.stdout).unwrap(),
                create_advice_provider_from_arg(cmd.stdin).unwrap(),
                initial_fp,
                args.specific,
                max_trace_height,
                program_file.clone(),
            );
            if !status {
                eprintln!("Program terminated with FAIL");
                std::process::exit(1);
            }
        }
        Commands::Prove(cmd) => {
            let opts = ProveDebugOptions {
                show_public: cmd.show_public,
                show_preprocessed: cmd.show_preprocessed,
                show_main: cmd.show_main,
                show_interactions: cmd.show_interactions,
                show_dims: cmd.show_dims,
            };
            let result: Result<(), String> = (|| {
                let mut action_file = File::create(cmd.proof.clone())
                    .map_err(|err| format!("File creation failed: {:?}", err))?;

                let option_pk = cmd
                    .clone()
                    .pk
                    .map(|file| {
                        read(file).map_err(|err| format!("Failed reading pk file: {:?}", err))
                    })
                    .transpose()?;

                let proof = prove(
                    program,
                    opts,
                    create_advice_provider_from_arg(Some(cmd.stdin.clone())).unwrap(),
                    option_pk,
                    config,
                    initial_fp,
                    args.specific,
                    max_trace_height,
                    program_file,
                )?;
                action_file
                    .write_all(&proof)
                    .map_err(|err| format! {"Writing proof failed: {:?}", err})?;

                Ok(())
            })();

            match result {
                Ok(_) => {
                    println!("Proof successful");
                }
                Err(err) => {
                    eprintln!("Proof creation failed: {}", err);
                    std::process::exit(1);
                }
            }
        }
        Commands::Verify(cmd) => {
            let status = verify(
                program,
                config,
                read(&cmd.stdout).expect("failed to read claimed output file"),
                read(&cmd.proof).expect("File reading failed"),
                cmd.clone().vk.map(|file| read(file).expect("good file")),
                cmd.show_public,
                cmd.show_preprocessed,
                initial_fp,
                args.specific,
                max_trace_height,
                program_file,
            );
            match status {
                Ok(_) => {
                    println!("Proof verified");
                }
                Err(err) => {
                    eprintln!("Proof verification failed: {:?}", err);
                    std::process::exit(1);
                }
            }
        }
        Commands::Preprocess(cmd) => {
            let (pk_bytes, vk_bytes) = preprocess(
                program,
                config,
                cmd.show_preprocessed,
                cmd.show_dims,
                initial_fp,
                args.specific,
                max_trace_height,
                program_file,
            );
            let pk_path = cmd.name.clone() + ".pk";
            let vk_path = cmd.name.clone() + ".vk";

            std::fs::write(pk_path, pk_bytes).unwrap();
            std::fs::write(vk_path, vk_bytes).unwrap();
        }
        Commands::Interactive(ref cmd) => {
            repl_run(&cmd);
        }
    };
    end_timer!(t_action);
}
