use p3_baby_bear::BabyBear;
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2Bowers;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeField32, TwoAdicField};
use p3_fri::FriConfig;
use p3_fri::{TwoAdicFriPcs, TwoAdicFriPcsConfig};
use p3_keccak::Keccak256Hash;
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon::Poseidon;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
use rand::thread_rng;
use valida_basic_api::BasicMachine;
use valida_machine::__internal::p3_commit::ExtensionMmcs;
use valida_machine::{Machine, ProverOptions, StarkConfigImpl};

pub type Val = BabyBear;
pub type Challenge = BinomialExtensionField<Val, 5>;
pub type PackedChallenge = BinomialExtensionField<<Val as Field>::Packing, 5>;
pub type Mds16 = CosetMds<Val, 16>;
pub type Perm16 = Poseidon<Val, Mds16, 16, 5>;
pub type MyHash = SerializingHasher32<Keccak256Hash>;
pub type MyCompress = CompressionFunctionFromHasher<Val, MyHash, 2, 8>;
pub type ValMmcs = FieldMerkleTreeMmcs<Val, MyHash, MyCompress, 8>;
pub type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
pub type Dft = Radix2Bowers;
pub type Challenger = DuplexChallenger<Val, Perm16, 16>;
pub type MyFriConfig = TwoAdicFriPcsConfig<Val, Challenge, Challenger, Dft, ValMmcs, ChallengeMmcs>;
pub type Pcs = TwoAdicFriPcs<MyFriConfig>;
pub type MyConfig = StarkConfigImpl<Val, Challenge, PackedChallenge, Pcs, Challenger>;

pub fn get_machine_config() -> MyConfig {
    let mds16 = Mds16::default();
    let perm16 = Perm16::new_from_rng(4, 22, mds16, &mut thread_rng()); // TODO: Use deterministic RNG
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

    let pcs = Pcs::new(fri_config, dft, val_mmcs);

    let challenger = Challenger::new(perm16);
    let config = MyConfig::new(pcs, challenger);
    config
}

/// Returns the prover options used in all the tests as well as the a vector for
/// `show_preprocessed`, bool for `show_preprocessed_dims` and vector for
/// `show_public_verifier`.
pub fn prover_options() -> (ProverOptions, Vec<bool>, bool, Vec<bool>) {
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

    (
        prover_opts,
        show_preprocessed,
        show_preprocessed_dims,
        show_public_verifier,
    )
}
