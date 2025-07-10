use clap::Parser;
use std::fmt;
use std::path::PathBuf;

use reedline_repl_rs::clap::Subcommand;

#[derive(Parser, Clone)]
pub struct Args {
    #[command(subcommand)]
    pub action: Commands,

    /// Whether the machine has a program specific setup: default is false for universal setup. If
    /// true, the setup is program-specific.
    #[arg(
        global = true,
        name = "Program Specific Setup",
        long = "program-specific",
        default_value = "false"
    )]
    pub specific: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Subcommand)]
pub enum Commands {
    Run(RunCommand),
    Preprocess(PreprocessCommand),
    Prove(ProveCommand),
    Verify(VerifyCommand),
    Interactive(InteractiveCommand),
}

impl fmt::Display for Commands {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Commands::Run(_) => write!(f, "run"),
            Commands::Preprocess(_) => write!(f, "preprocess"),
            Commands::Prove(_) => write!(f, "prove"),
            Commands::Verify(_) => write!(f, "verify"),
            Commands::Interactive(_) => write!(f, "interactive"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Parser)]
pub struct Shared {
    /// Program binary file.
    #[arg(name = "PROGRAM")]
    pub program: String,

    /// The initial frame pointer value.
    /// It's last 8 byte aligned value in memory address space: 0x3bfffff8.
    #[arg(long, default_value = "1006632952")]
    pub initial_fp: u32,

    /// The max segment size, default is 2^20 to prove a block in less than 64GB
    #[arg(
        long = "max-segment-size",
        name = "Max Segment Size",
        default_value = "1048576"
    )]
    pub max_trace_height: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Parser)]
pub struct InteractiveCommand {
    #[clap(flatten)]
    pub shared: Shared,

    /// Path to a file from which the standard input for the <PROGRAM> will be read.
    /// If not present then the standard input for the <PROGRAM> is the same as for `valida` process.
    #[arg(name = "STDIN")]
    pub stdin: Option<String>,

    /// Memory backend to use
    #[arg( name = "Memory Backend", long = "backend", value_parser = ["btree", "hashmap", "array", "lean"], default_value = "btree")]
    pub backend_type: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Parser)]
pub struct PreprocessCommand {
    /// Program binary file.
    #[arg(name = "PROGRAM")]
    pub program: String,

    /// The proving and verifying keys will be saved to NAME.pk and NAME.vk.
    #[arg(name = "NAME")]
    pub name: String,

    /// Whether to print the preprocessed traces.
    #[arg(long = "show-preprocessed", action = clap::ArgAction::SetTrue, name = "Show Preprocessed")]
    pub show_preprocessed: bool,

    /// Whether to print the sizes of the various traces
    #[arg(long = "show-dimensionss", action = clap::ArgAction::SetTrue, name = "Show Dimensions")]
    pub show_dims: bool,

    /// The max segment size
    #[arg(long = "max-segment-size", name = "Max Segment Size")]
    pub max_trace_height: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Parser)]
pub struct RunCommand {
    #[clap(flatten)]
    pub shared: Shared,

    /// Path to a file where the standard output of the <PROGRAM> will be saved.
    #[arg(name = "STDOUT")]
    pub stdout: String,

    /// Path to a file from which the standard input for the <PROGRAM> will be read.
    /// If not present then the standard input for the <PROGRAM> is the same as for `valida` process.
    #[arg(name = "STDIN")]
    pub stdin: Option<String>,

    /// Disable trace generation for "valida run".
    #[arg(name = "Disable traces", long = "fast", default_value = "false")]
    pub fast: bool,

    /// Memory backend to use
    #[arg(name = "Memory Backend", long = "backend", value_parser = ["btree", "hashmap", "array", "lean"], default_value = "btree")]
    pub backend_type: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Parser)]
pub struct ProveCommand {
    #[clap(flatten)]
    pub shared: Shared,

    /// Path to a file where the proof will be saved.
    #[arg(name = "PROOF")]
    pub proof: String,

    /// Path to a file from which the standard input for the <PROGRAM> will be read.
    #[arg(name = "STDIN")]
    pub stdin: String,

    /// Proving key to use.
    #[arg(long = "proving-key", name = "Proving Key")]
    pub pk: Option<PathBuf>,

    /// Whether to print the public inputs.
    #[arg(long = "show-public", action = clap::ArgAction::SetTrue, name = "Show Public Values")]
    pub show_public: bool,

    /// Whether to print the preprocessed traces during proving, assuming they are not read from file.
    #[arg(long = "show-preprocessed", action = clap::ArgAction::SetTrue, name = "Show Preprocessed Traces")]
    pub show_preprocessed: bool,

    /// Whether to print the main traces.
    #[arg(long = "show-main", action = clap::ArgAction::SetTrue, name = "Show Main Traces")]
    pub show_main: bool,

    /// Whether to print the interactions (lookup vectors) for the permutation argument.
    #[arg(long = "show-interactions", action = clap::ArgAction::SetTrue, name = "Show Interactions")]
    pub show_interactions: bool,

    /// Whether to print the sizes of the various traces.
    #[arg(long = "show-dimensions", action = clap::ArgAction::SetTrue, name = "Show Dimensions of traces")]
    pub show_dims: bool,

    /// Memory backend to use
    #[arg(name = "Memory Backend", long = "backend", value_parser = ["btree", "hashmap", "array", "lean"], default_value = "btree")]
    pub backend_type: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Parser)]
pub struct VerifyCommand {
    #[clap(flatten)]
    pub shared: Shared,

    /// Path to a file containing a proof that needs to be verified.
    #[arg(name = "PROOF")]
    pub proof: String,

    /// Path to a file where the claimed standard output for the <PROGRAM> is stored.
    /// The content of that file is a part of the statement being verified.
    #[arg(name = "CLAIMED_STDOUT")]
    pub stdout: String,

    /// Verifying key to use.
    #[arg(long = "verifying-key", name = "Verifying Key")]
    pub vk: Option<PathBuf>,

    /// Whether to print the public input during verification.
    #[arg(long = "show-public", action = clap::ArgAction::SetTrue, name = "Show Public Values")]
    pub show_public: bool,

    /// Whether to print the preprocessed input during verification, assuming it is not read from file.
    #[arg(long = "show-preprocessed", action = clap::ArgAction::SetTrue, name = "Show Preprocessed Traces")]
    pub show_preprocessed: bool,
}
