use std::fs::{self, File};

use super::args::InteractiveCommand;
use reedline_repl_rs::clap::{Arg, Command};
use reedline_repl_rs::Repl;

use p3_baby_bear::BabyBear;
use valida_basic_api::commands::interactive::{
    init_context, last_frame, list_instrs, run_until, set_bp, show_frame, show_memory, status,
    step, Context,
};
use valida_basic_api::commands::{
    get_fixed_advice_provider_from_file, get_stdin_advice_provider, AdviceProviderWithDefault,
};
use valida_basic_api::ValidaRuntime;

fn prepare_repl<'a>(
    runtime: &'a mut ValidaRuntime,
    cmd: &'a InteractiveCommand,
) -> Repl<Context<'a>, reedline_repl_rs::Error> {
    let repl = Repl::new(Context::new(
        runtime,
        cmd.shared.initial_fp,
        //cmd.stdin.clone(),
        fs::read(&cmd.shared.program)
            .unwrap_or_else(|_| panic!("Failed to read executable file: {}", &cmd.shared.program)),
        AdviceProviderWithDefault(
            cmd.stdin
                .as_ref()
                .map(|file_name| {
                    get_fixed_advice_provider_from_file(&mut File::open(file_name).unwrap())
                        .unwrap()
                })
                .unwrap_or(get_stdin_advice_provider()),
        ),
    ))
    .with_name("REPL")
    .with_version("v0.1.0")
    .with_description("Valida VM REPL")
    .with_banner("Start by using keywords")
    .with_command(
        Command::new("x").about("read machine state"),
        |_args, context| Ok(Some(status(context))),
    )
    .with_command(
        Command::new("s")
            .arg(Arg::new("num_steps").required(false))
            .about("step assembly"),
        |_args, context| Ok(step(context)),
    )
    .with_command(
        Command::new("f")
            .arg(Arg::new("size").required(false))
            .about("show frame"),
        |args, context| {
            let size: i32 = match args.contains_id("size") {
                true => args
                    .get_one::<String>("size")
                    .unwrap()
                    .parse::<i32>()
                    .unwrap(),
                false => 6,
            };
            Ok(Some(show_frame(size, context)))
        },
    )
    .with_command(
        Command::new("lf").about("show last frame and current frame"),
        |_args, context| Ok(Some(last_frame(context))),
    )
    .with_command(
        Command::new("b")
            .arg(Arg::new("pc").required(false))
            .about("set break point at"),
        |args, context| {
            let pc = args
                .get_one::<String>("pc")
                .unwrap()
                .parse::<u32>()
                .unwrap();

            Ok(Some(set_bp(pc, context)))
        },
    )
    .with_command(
        Command::new("r").about("run until stop or breakpoint"),
        |_args, context| Ok(Some(run_until(context))),
    )
    .with_command(
        Command::new("l")
            .about("list instruction at current PC")
            .arg(Arg::new("size").required(false)),
        |args, context| Ok(Some(list_instrs(args.get_one::<String>("size"), context))),
    )
    .with_command(
        Command::new("m")
            .arg(Arg::new("addr").required(true))
            .about("show memory at address"),
        |args, context| {
            let addr = args
                .get_one::<String>("addr")
                .unwrap()
                .parse::<u32>()
                .unwrap();
            Ok(Some(show_memory(addr, context)))
        },
    )
    .with_command(
        Command::new("reset").about("reset machine state!"),
        |_args, context| Ok(Some(init_context(context))),
    );

    repl
}

pub fn repl_run(cmd: &InteractiveCommand) {
    // instantiate repl

    // We instantiate the runtime *before* the context, as to make sure it lives as long
    // as the context. Otherwise the borrowing rules will make the code much more complicated
    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();

    let mut repl = prepare_repl(&mut runtime, cmd);

    let _ = repl.run();
}
