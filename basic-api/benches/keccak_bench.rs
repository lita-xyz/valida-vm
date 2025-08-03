use criterion::{criterion_group, criterion_main, Criterion};
use p3_baby_bear::BabyBear;
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2Bowers;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeField32, TwoAdicField};
use p3_fri::{FriConfig, TwoAdicFriPcs, TwoAdicFriPcsConfig};
use p3_keccak::Keccak256Hash;
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon::Poseidon;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
use rand::thread_rng;
use valida_basic_api::{BasicMachine, ValidaRuntime};
use valida_cpu::{MachineWithRegisters, StopInstruction};
use valida_keccak::KeccakFInstruction;
use valida_machine::{
    Instruction, InstructionWord, Machine, MachineProof, Operands, ProgramROM, ProverOptions,
    StarkConfigImpl, StarkField,
};
use valida_program::{MachineWithProgramROM, ProgramTableType};

fn prove_program(
    program: Vec<InstructionWord<i32>>,
    program_table_type: ProgramTableType,
) -> BasicMachine<BabyBear> {
    let mut machine = BasicMachine::<Val>::default();
    let rom = ProgramROM::new(program);
    machine.set_program_rom(0, rom, program_table_type);
    // Set max trace height to the default on the command line: 2**27.
    machine.set_max_trace_height(2 * 2u32.pow(27));
    machine.set_initial_register_values(valida_cpu::Registers { pc: 0, fp: 0x1000 });

    let mut runtime = ValidaRuntime::default_for_field::<BabyBear>();
    let mut running_machine = machine.start(&mut runtime);
    let (instance_data, _output) = BasicMachine::<BabyBear>::run(&mut running_machine);
    let finalized_machine = BasicMachine::<BabyBear>::stop(running_machine);

    type Val = BabyBear;

    type Challenge = BinomialExtensionField<Val, 5>;
    type PackedChallenge = BinomialExtensionField<<Val as Field>::Packing, 5>;

    type Mds16 = CosetMds<Val, 16>;
    let mds16 = Mds16::default();

    type Perm16 = Poseidon<Val, Mds16, 16, 5>;
    let perm16 = Perm16::new_from_rng(4, 22, mds16, &mut thread_rng()); // TODO: Use deterministic RNG

    type MyHash = SerializingHasher32<Keccak256Hash>;
    let hash = MyHash::new(Keccak256Hash {});

    type MyCompress = CompressionFunctionFromHasher<Val, MyHash, 2, 8>;
    let compress = MyCompress::new(hash);

    type ValMmcs = FieldMerkleTreeMmcs<Val, MyHash, MyCompress, 8>;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2Bowers;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm16, 16>;

    type MyFriConfig = TwoAdicFriPcsConfig<Val, Challenge, Challenger, Dft, ValMmcs, ChallengeMmcs>;
    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 40,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    type Pcs = TwoAdicFriPcs<MyFriConfig>;
    type MyConfig = StarkConfigImpl<Val, Challenge, PackedChallenge, Pcs, Challenger>;

    let pcs = Pcs::new(fri_config, dft, val_mmcs);

    let challenger = Challenger::new(perm16);
    let config = MyConfig::new(pcs, challenger);

    // Have each trace print once: only shown if the test fails anyway.
    // skip preprocessed, which includes some long traces
    let show_preprocessed = vec![false; BasicMachine::<Val>::NUM_CHIPS];
    let show_public_prover = vec![true; BasicMachine::<Val>::NUM_CHIPS];
    let show_main = vec![true; BasicMachine::<Val>::NUM_CHIPS];
    let show_interactions = vec![true; BasicMachine::<Val>::NUM_CHIPS];
    let show_public_verifier = vec![false; BasicMachine::<Val>::NUM_CHIPS];
    let show_public_dims = true;
    let show_main_dims = true;
    let show_permutation_dims = true;
    let show_preprocessed_dims = true;

    let prover_opts = ProverOptions {
        show_main,
        show_public: show_public_prover,
        show_interactions,
        show_public_dims,
        show_main_dims,
        show_permutation_dims,
    };

    let (pk, vk) =
        finalized_machine.pre_process(&config, show_preprocessed, show_preprocessed_dims);
    let proof = finalized_machine.prove(&config, &pk, prover_opts, &instance_data);

    let mut bytes = vec![];
    ciborium::into_writer(&proof, &mut bytes).expect("serialization failed");
    println!("Proof size: {} bytes", bytes.len());
    let deserialized_proof: MachineProof<MyConfig> =
        ciborium::from_reader(bytes.as_slice()).expect("deserialization failed");

    finalized_machine
        .verify(
            &config,
            &proof,
            &vk,
            &instance_data,
            show_public_verifier.clone(),
        )
        .expect("verification failed");
    finalized_machine
        .verify(
            &config,
            &deserialized_proof,
            &vk,
            &instance_data,
            show_public_verifier,
        )
        .expect("verification failed");

    finalized_machine
}

fn keccak_program<Val: StarkField>() -> Vec<InstructionWord<i32>> {
    let mut program = vec![];

    for i in 0..1000 {
        program.extend([
            //Keccak hash
            InstructionWord {
                opcode: <KeccakFInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
                operands: Operands([0, 0, 0, 0, 0]),
            },
        ]);
    }

    program.extend([
        // Stop
        InstructionWord {
            opcode: <StopInstruction as Instruction<BasicMachine<Val>, Val>>::OPCODE,
            operands: Operands::default(),
        },
    ]);

    program
}

pub fn keccak_proving_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("keccak_proving");

    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(50));
    let program = keccak_program::<BabyBear>();

    group.bench_function("prove_keccak", |b| {
        b.iter(|| {
            prove_program(program.clone(), ProgramTableType::Public);
        })
    });

    group.finish();
}

criterion_group!(benches, keccak_proving_benchmark);
criterion_main!(benches);
