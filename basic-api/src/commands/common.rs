use std::fs::File;
use std::io::Write;

use p3_baby_bear::BabyBear;

use p3_fri::{FriConfig, TwoAdicFriPcs, TwoAdicFriPcsConfig};
use valida_cpu::{MachineWithCpuChip, Registers};
use valida_machine::{
    get_file_write_callback, get_replay_advice_provider, AdviceProviderWithDefault, Machine,
    MachineProverKey, MachineVerifierKey, ProgramROM, ReplayAdviceProvider, RunningMachine,
    SegmentMachine, StorageBackendTrait, ValidaMemoryBackend, WriteCallbackWithDefault,
};

use valida_program::{MachineWithProgramChip, MachineWithProgramROM, ProgramTableType};
use valida_static_data::{MachineWithStaticDataChip, StaticDataChipType};

use alloc::collections::BTreeMap;
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_keccak::Keccak256Hash;
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon::Poseidon;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use valida_cpu::MachineWithRegisters;
use valida_machine::__internal::p3_commit::ExtensionMmcs;
use valida_machine::{StarkConfigImpl, StorageBackendType, Word};

use ark_std::{end_timer, start_timer};

use crate::{
    BasicMachine, BasicRunningMachine, MultiSegmentBasicMachine, ValidaBootData, ValidaRuntime,
};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 5>;
type PackedChallenge = BinomialExtensionField<<Val as Field>::Packing, 5>;
type Mds16 = CosetMds<Val, 16>;
type Perm16 = Poseidon<Val, Mds16, 16, 5>;
type Challenger = DuplexChallenger<Val, Perm16, 16>;
type MyHash = SerializingHasher32<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Val, MyHash, 2, 8>;
type ValMmcs = FieldMerkleTreeMmcs<Val, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel;
type MyFriConfig = TwoAdicFriPcsConfig<Val, Challenge, Challenger, Dft, ValMmcs, ChallengeMmcs>;

type Pcs = TwoAdicFriPcs<MyFriConfig>;

pub type MyConfig = StarkConfigImpl<Val, Challenge, PackedChallenge, Pcs, Challenger>;
pub type MyPK = MachineProverKey<MyConfig, BasicMachine<BabyBear>>;
pub type MyVK = MachineVerifierKey<MyConfig, BasicMachine<BabyBear>>;

pub type MyState<'a> = RunningMachine<'a, BabyBear, MultiSegmentBasicMachine<BabyBear>>;
pub type MyMachine = MultiSegmentBasicMachine<BabyBear>;

pub fn prepare_machine(
    code: ProgramROM<i32>,
    data: BTreeMap<u32, Word<u8>>,
    initial_register_values: Registers,
    specific: bool,
    // program_table_type: ProgramTableType,
    // static_data_chip_type: StaticDataChipType,
    max_trace_height: u32,
    program_file: Vec<u8>,
) -> MyMachine {
    let (program_table_type, static_data_chip_type) = if !specific {
        (ProgramTableType::Public, StaticDataChipType::Public)
    } else {
        (
            ProgramTableType::Preprocessed,
            StaticDataChipType::Preprocessed,
        )
    };

    let boot_data = ValidaBootData {
        program_rom: code,
        static_data: data,
        initial_register_values,
        program_table_type,
        static_data_chip_type,
        max_trace_height,
        program_file,
    };
    let t_machine_setup = start_timer!(|| "valida > common | machine.set_program_rom(..)");
    let mut machine = MyMachine::default();
    machine.init(boot_data);
    end_timer!(t_machine_setup);

    machine
}

/// Prepares a single `BasicMachine` (segment number 0)
pub fn prepare_basic_machine(
    code: ProgramROM<i32>,
    data: BTreeMap<u32, Word<u8>>,
    initial_register_values: Registers,
    specific: bool,
    max_trace_height: u32,
    program_file: Vec<u8>,
) -> BasicMachine<BabyBear> {
    let (program_table_type, static_data_chip_type) = if !specific {
        (ProgramTableType::Public, StaticDataChipType::Public)
    } else {
        (
            ProgramTableType::Preprocessed,
            StaticDataChipType::Preprocessed,
        )
    };

    let mut machine = BasicMachine::<BabyBear>::default();
    machine.set_segment_number(0); // single machine == segment number 0
    machine.set_max_trace_height(max_trace_height);
    machine.set_program_rom(initial_register_values.pc, code, program_table_type);
    machine.set_initial_register_values(initial_register_values);
    machine.static_data_mut().load(data, static_data_chip_type);

    machine
}

pub fn prepare_runtime(
    stdin: AdviceProviderWithDefault,
    stdout: WriteCallbackWithDefault,
) -> Result<ValidaRuntime, ()> {
    let memory_backend = ValidaMemoryBackend::default_for_field::<BabyBear>();

    Ok(ValidaRuntime {
        memory_backend,
        write_callback: stdout,
        advice_provider: stdin,
        replay_advice: ReplayAdviceProvider::default(),
    })
}

pub fn prepare_runtime_from_replay(
    replay_advice: ReplayAdviceProvider,
    stdout: WriteCallbackWithDefault,
) -> Result<ValidaRuntime, ()> {
    let memory_backend = ValidaMemoryBackend::default_for_field::<BabyBear>();

    Ok(ValidaRuntime {
        memory_backend,
        write_callback: stdout,
        advice_provider: AdviceProviderWithDefault(get_replay_advice_provider(replay_advice)),
        replay_advice: ReplayAdviceProvider::default(),
    })
}

pub fn prepare_runtime_default() -> Result<ValidaRuntime, ()> {
    let memory_backend = ValidaMemoryBackend::default_for_field::<BabyBear>();
    let advice_provider = AdviceProviderWithDefault::default();
    let write_callback = WriteCallbackWithDefault::default();
    let replay_advice = ReplayAdviceProvider::default();
    Ok(ValidaRuntime {
        memory_backend,
        write_callback,
        advice_provider,
        replay_advice,
    })
}

pub fn default_config() -> MyConfig {
    let mds16 = Mds16::default();
    let mut rng: Pcg64 = Seeder::from("valida seed").make_rng();
    let perm16 = Perm16::new_from_rng(4, 22, mds16, &mut rng);
    let hash = MyHash::new(Keccak256Hash {});

    let compress = MyCompress::new(hash);
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 40,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    let t_pcs_init =
        start_timer!(|| "valida > common | Polynomial Commitment Scheme initialization");
    let pcs = Pcs::new(fri_config, dft, val_mmcs);
    let challenger = Challenger::new(perm16);
    let config = MyConfig::new(pcs, challenger);
    end_timer!(t_pcs_init);

    config
}
