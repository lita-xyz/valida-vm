#![no_main]
valida_rs::entrypoint!(main);

use recursive_verifier_api::recursive_main;
use valida_program::ProgramTableType::Public;

fn main() {
    recursive_main(Public)
}
