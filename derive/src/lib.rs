extern crate alloc;

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{spanned::Spanned, Data, Field, Fields, Ident};

// TODO: now trivial with a single field
struct MachineFields {
    val: Ident,
}

impl Parse for MachineFields {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        syn::parenthesized!(content in input);
        let val = content.parse()?;
        Ok(MachineFields { val })
    }
}

#[proc_macro_derive(
    Machine,
    attributes(machine_fields, bus, chip, static_data_chip, instruction)
)]
pub fn machine_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_machine(&ast)
}

fn impl_machine(machine: &syn::DeriveInput) -> TokenStream {
    if let Data::Struct(struct_) = &machine.data {
        let fields = match &struct_.fields {
            Fields::Named(named) => named.named.iter().collect(),
            Fields::Unnamed(unnamed) => unnamed.unnamed.iter().collect(),
            Fields::Unit => vec![],
        };

        let instructions = fields
            .iter()
            .filter(|f| f.attrs.iter().any(|a| a.path.is_ident("instruction")))
            .copied()
            .collect::<Vec<_>>();
        let chips = fields
            .iter()
            .filter(|f| f.attrs.iter().any(|a| a.path.is_ident("chip")))
            .copied()
            .collect::<Vec<_>>();

        let machine_fields = machine
            .attrs
            .iter()
            .find(|a| a.path.segments.len() == 1 && a.path.segments[0].ident == "machine_fields")
            .expect("machine_fields attribute required to derive Machine");
        let machine_fields: MachineFields = syn::parse2(machine_fields.tokens.clone())
            .expect("Invalid machine_fields attribute, expected #[machine_fields(<Val>)]");
        let val = &machine_fields.val;

        let static_data_chip: Option<Ident> = chips
            .iter()
            .filter(|f| f.attrs.iter().any(|a| a.path.is_ident("static_data_chip")))
            .map(|f| {
                f.ident
                    .clone()
                    .expect("static data chip requires an identifier")
            })
            .next();

        let name = &machine.ident;
        let preprocess = preprocess_method(&chips);
        let run = run_method(&static_data_chip);
        let step = step_method(machine, &instructions, val);
        let prove = prove_method(&chips);
        let verify = verify_method(&chips);

        let (impl_generics, ty_generics, where_clause) = machine.generics.split_for_impl();

        let num_chips = chips.len();

        let stream = quote! {
            impl #impl_generics Machine<#val> for #name #ty_generics #where_clause {
                const NUM_CHIPS: usize = #num_chips;
                type InstanceData = ValidaInstanceData;
                fn enable_logging(&mut self, _: bool) -> Option<()> { None }
                fn log_enabled(&self) -> bool { true }
                #step
                #run
                #preprocess
                #prove
                #verify
            }
        };

        stream.into()
    } else {
        panic!("Machine derive only supports structs");
    }
}

#[deprecated] // Planning manual impls for now.
#[allow(dead_code)]
fn impl_machine_chip_impl_given_chips(
    machine: &syn::DeriveInput,
    chips: &[&Field],
) -> TokenStream2 {
    let chip_impls = chips.iter().map(|chip| {
        let chip_ty = &chip.ty;
        let tokens = quote!(#chip_ty);
        let chip_impl_name = Ident::new(&format!("MachineWith{}", tokens), chip.span());
        let chip_methods = chip_methods(chip);

        let name = &machine.ident;
        let (impl_generics, ty_generics, where_clause) = machine.generics.split_for_impl();

        quote! {
            impl #impl_generics #chip_impl_name for #name #ty_generics #where_clause {
                #chip_methods
            }
        }
    });
    quote! {
        #(#chip_impls)*
    }
}

#[allow(dead_code)]
fn chip_methods(chip: &Field) -> TokenStream2 {
    let mut methods = vec![];
    let chip_name = chip.ident.as_ref().unwrap();
    let chip_name_mut = Ident::new(&format!("{}_mut", chip_name), chip_name.span());
    let chip_type = &chip.ty;
    methods.push(quote! {
        fn #chip_name(&self) -> &#chip_type {
            &self.#chip_name
        }
        fn #chip_name_mut(&mut self) -> &mut #chip_type {
            &mut self.#chip_name
        }
    });
    quote! {
        #(#methods)*
    }
}

fn run_method(static_data_chip: &Option<Ident>) -> TokenStream2 {
    let init_static_data: TokenStream2 = match static_data_chip {
        Some(_static_data_chip) => quote! {
            self.initialize_memory();
        },
        None => quote! {},
    };

    quote! {
        fn run<Adv: ::valida_machine::AdviceProvider>(&mut self, program: &ProgramROM<i32>, advice: &mut Adv) -> ValidaInstanceData {
            #init_static_data

            loop {
                let step_did_stop = self.step(advice);
                if step_did_stop == StoppingFlag::DidStop {
                    self.finalize_memory();
                    break;
                }
            }

            // Record padded STOP instructions
            let n = self.cpu().clock.next_power_of_two() - self.cpu().clock;
            for _ in 0..n {
                let pc = self.cpu().pc;
                self.read_word(pc, true);
            }

            let (pc_init, fp_init) = {
                debug_assert!(!self.cpu.registers.is_empty(), "register state has not been initialized");
                (self.cpu.pc_init, self.cpu.fp_init)
            };

            // Add receives for `diff_bytes` in the memory columns
            // Required for soundness of the memory argument
            ::valida_memory::add_diff_bytes_receives(self);

            ValidaInstanceData {
                rom: {
                    match self.program_table_type() {
                        ProgramTableType::Public => Some(self.program_rom().clone()),
                        _ => None,
                    }
                },

                static_data: match self.static_data().chip_type() {
                    StaticDataChipType::Public => Some(self.static_data().get_cells()),
                    _ => None,
                },
                output: self.output.bytes().to_vec(),
                pc_init,
                fp_init,
            }
        }
    }
}

fn step_method(machine: &syn::DeriveInput, instructions: &[&Field], val: &Ident) -> TokenStream2 {
    // TODO: combine this with run
    let name = &machine.ident;
    let (_, ty_generics, _) = machine.generics.split_for_impl();

    let opcode_arms = instructions
        .iter()
        .map(|inst| {
            let ty = &inst.ty;
            quote! {
                // TODO: Self instead of #name #ty_generics?
                <#ty as Instruction<#name #ty_generics, #val>>::OPCODE =>
                    #ty::execute(self, ops, advice),
            }
        })
        .collect::<TokenStream2>();

    quote! {
       fn step<Adv: ::valida_machine::AdviceProvider>(&mut self, advice: &mut Adv) -> StoppingFlag {
           let pc = self.cpu().pc;
           let instruction = self.program_rom().get_instruction(pc);
           let opcode = instruction.opcode;
           let ops = instruction.operands;

           match opcode {
               #opcode_arms
               _ => panic!("Unrecognized opcode: {}", opcode),
           };
           self.read_word(pc, true);

           if opcode == <StopInstruction as Instruction<Self, #val>>::OPCODE {
              StoppingFlag::DidStop
           } else {
              StoppingFlag::DidNotStop
            }
       }
    }
}

fn preprocess_method(chips: &[&Field]) -> TokenStream2 {
    let num_chips = chips.len();
    let chip_list = chips
        .iter()
        .map(|chip| {
            let chip_name = chip.ident.as_ref().unwrap();
            quote! {
                alloc::boxed::Box::new(self.#chip_name()),
            }
        })
        .collect::<TokenStream2>();
    quote! {
        fn pre_process<SC>(
            &mut self,
            config: &SC,
            show_preprocessed: Vec<bool>,
            _show_dims: bool,
        ) -> (MachineProverKey<SC, Self>, MachineVerifierKey<SC, Self>)
        where
            SC: StarkConfig<Val = F>,
        {
            let pcs = config.pcs();

            let mut chips: [Box<&dyn Chip<Self, SC, Public=PublicTrace<SC::Val>>>; #num_chips] = [ #chip_list ];

            let preprocessed_traces: [Option<RowMajorMatrix<SC::Val>>; #num_chips] =
                tracing::info_span!("generate preprocessed traces").in_scope(|| {
                    chips
                        .par_iter()
                        .map(|chip| chip.get_preprocessed_trace(false))
                        .map(|(trace, _)| trace)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                });

            let preprocessed_dims: [Option<Dimensions>; #num_chips] = preprocessed_traces
                .iter()
                .map(|trace| {
                    trace.as_ref().map(|trace| Dimensions {
                        width: trace.width(),
                        height: trace.height()
                     })
            })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            let (preprocessed_commit, preprocessed_data) =
                tracing::info_span!("commit to preprocessed traces").in_scope(|| {
                    pcs.commit_batches(preprocessed_traces.clone().into_iter().flatten().collect())
                });

            let pk = MachineProverKey::new(
                preprocessed_traces.to_vec(),
                preprocessed_commit.clone(),
                preprocessed_data,
            );

            let vk = MachineVerifierKey::new(preprocessed_commit, preprocessed_dims.to_vec());

            (pk, vk)
        }
    }
}

fn prove_method(chips: &[&Field]) -> TokenStream2 {
    let num_chips = chips.len();
    let chip_list = chips
        .iter()
        .map(|chip| {
            let chip_name = chip.ident.as_ref().unwrap();
            quote! {
                alloc::boxed::Box::new(self.#chip_name()),
            }
        })
        .collect::<TokenStream2>();

    let quotient_degree_calls = chips
        .iter()
        .map(|chip| {
            let chip_name = chip.ident.as_ref().unwrap();
            quote! {
                get_log_quotient_degree::<Self, SC, _>(self, self.#chip_name()),
            }
        })
        .collect::<TokenStream2>();

    let compute_quotients = chips
        .iter()
        .enumerate()
        .map(|(i, chip)| {
            let chip_name = chip.ident.as_ref().unwrap();
            quote! {
                #[cfg(debug_assertions)]
                let _ = check_constraints::<Self, _, SC>(
                    self,
                    self.#chip_name(),
                    &preprocessed_traces[#i],
                    &main_traces[#i],
                    &perm_traces[#i],
                    degrees[#i],
                    &perm_challenges,
                    &public_traces[#i],
                    false
                );

                quotients.push(quotient(
                    self,
                    config,
                    self.#chip_name(),
                    log_degrees[#i],
                    preprocessed_trace_ldes.remove(0),
                    main_trace_ldes.remove(0),
                    perm_trace_ldes.remove(0),
                    public_trace_ldes.remove(0),
                    cumulative_sums[#i],
                    &perm_challenges,
                    alpha,
                ));
            }
        })
        .collect::<TokenStream2>();

    quote! {
        #[tracing::instrument(name = "prove machine execution", skip_all)]
        fn prove<SC: StarkConfig<Val = F>>(&self, config: &SC, pk: &MachineProverKey<SC, Self>, prover_opts: ProverOptions) -> ::valida_machine::MachineProof<SC>
        {
            use ::valida_machine::__internal::*;
            use ::valida_machine::__internal::p3_air::{BaseAir};
            use ::valida_machine::__internal::p3_field::{AbstractField};
            use ::valida_machine::__internal::p3_challenger::{CanObserve, FieldChallenger};
            use ::valida_machine::__internal::p3_commit::{Pcs, UnivariatePcs, UnivariatePcsWithLde};
            use ::valida_machine::__internal::p3_matrix::{Matrix, MatrixRowSlices, dense::RowMajorMatrix};
            use ::valida_machine::__internal::p3_util::log2_strict_usize;
            use ::valida_machine::{generate_permutation_trace, Chip, MachineProof, ChipProof, Commitments, PublicTrace, PublicValues};

            use ::valida_machine::OpenedValues;
            use alloc::vec;
            use alloc::vec::Vec;
            use alloc::boxed::Box;

            let mut chips: [Box<&dyn Chip<Self, SC, Public=PublicTrace<SC::Val>>>; #num_chips] = [ #chip_list ];
            let log_quotient_degrees: [usize; #num_chips] = [ #quotient_degree_calls ];

            let mut challenger = config.challenger();
            // TODO: Seed challenger with digest of all constraints & trace lengths.
            let pcs = config.pcs();

            let (preprocessed_traces, preprocessed_commit, preprocessed_data) = (
                pk.preprocessed_traces(),
                pk.preprocessed_commit(),
                pk.preprocessed_prover_data()
            );

            let has_preprocessed_traces = preprocessed_traces
                .iter()
                .map(Option::is_some);

            challenger.observe(preprocessed_commit.clone());
            let mut preprocessed_trace_ldes_real = pcs.get_ldes(&preprocessed_data);

            // add the None's back in
            let mut preprocessed_trace_ldes: Vec<_> = has_preprocessed_traces.clone()
                    .map(|has_trace| {
                         if has_trace {
                             Some(preprocessed_trace_ldes_real.remove(0))
                         } else {
                             None
                         }
                    })
                    .collect();

            let mut public_traces: [Option<PublicTrace<SC::Val>>; #num_chips] =
                tracing::info_span!("generate public traces").in_scope(|| {
                        chips.par_iter()
                            .map(|chip| chip.generate_public_values(false))
                            .map(|(trace, _)| trace)
                            .collect::<Vec<_>>()
                            .try_into().unwrap()
            });

            let mut public_trace_ldes: Vec<_> = public_traces
                .iter()
                .map(|opt| opt.as_ref().map(|trace| trace.get_ldes(config)))
                .collect();

            let main_traces: [Option<RowMajorMatrix<SC::Val>>; #num_chips] =
                tracing::info_span!("generate main traces")
                    .in_scope(||
                        chips.par_iter()
                            .map(|chip| chip.generate_main_trace(self, false))
                            .map(|(trace, _)| trace)
                            .collect::<Vec<_>>()
                            .try_into().unwrap()
                    );

            let has_main_traces = main_traces
                .iter()
                .map(Option::is_some);

            let degrees: [usize; #num_chips] = main_traces
            .iter().zip(preprocessed_traces.iter()).zip(public_traces.iter())
            .map(|((main, preprocessed), public)| {
                let public_height = public.as_ref().and_then(|public_values| {
                    match public_values {
                        PublicTrace::PublicMatrix(matrix) => Some(matrix.height()),
                        PublicTrace::PublicVector(_) => None,
                    }
                });

                let (main_height, preprocessed_height) = (main.as_ref().map(|m| m.height()), preprocessed.as_ref().map(|p| p.height()));

                let heights = vec![main_height, preprocessed_height, public_height].into_iter().flatten().collect::<Vec<_>>();

                let first_height = heights.first().expect("all trace components are empty");
                debug_assert!(heights.iter().all(|&h| h == *first_height), "Trace components do not all have the same size. Main: {:?}, Preprocessed: {:?}, Public: {:?}", main_height, preprocessed_height, public_height);

                *first_height
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
            let log_degrees = degrees.map(|d| log2_strict_usize(d));
            let g_subgroups = log_degrees.map(|log_deg| SC::Val::two_adic_generator(log_deg));

            let (main_commit, main_data) = tracing::info_span!("commit to main traces")
                .in_scope(|| pcs.commit_batches(main_traces.clone().into_iter().flatten().collect()));
            challenger.observe(main_commit.clone());
            let mut main_trace_ldes_real = pcs.get_ldes(&main_data).into_iter();

                // add the None's back in
            let mut main_trace_ldes: Vec<_> = has_main_traces.clone()
                .map(|has_trace| {
                    if has_trace {
                        Some(main_trace_ldes_real.next().unwrap())
                    } else {
                        None
                    }
                })
            .collect();

            let mut perm_challenges = Vec::new();
            for _ in 0..2 {
                perm_challenges.push(challenger.sample_ext_element());
            }

            let perm_traces = tracing::info_span!("generate permutation traces")
                .in_scope(||
                    chips.into_par_iter().enumerate().map(|(i, chip)| {
                        generate_permutation_trace(self, *chip, &preprocessed_traces[i], &public_traces[i], &main_traces[i], degrees[i], perm_challenges.clone())
                    }).collect::<Vec<_>>()
                );

            let cumulative_sums = perm_traces.iter()
                .map(|trace| trace.row_slice(trace.height() - 1).last().unwrap().clone())
                .collect::<Vec<_>>();

            let (perm_commit, perm_data) = tracing::info_span!("commit to permutation traces")
                .in_scope(|| {
                    let flattened_perm_traces = perm_traces.iter()
                        .map(|trace| trace.flatten_to_base())
                        .collect::<Vec<_>>();
                    pcs.commit_batches(flattened_perm_traces)
                });
            challenger.observe(perm_commit.clone());
            let mut perm_trace_ldes = pcs.get_ldes(&perm_data);

            let alpha: SC::Challenge = challenger.sample_ext_element();

            let mut quotients: Vec<RowMajorMatrix<SC::Val>> = vec![];
            #compute_quotients
            assert_eq!(quotients.len(), #num_chips);
            assert_eq!(log_quotient_degrees.len(), #num_chips);
            let coset_shifts = tracing::debug_span!("coset shift").in_scope(|| {
                let pcs_coset_shift = pcs.coset_shift();
                log_quotient_degrees.map(|log_d| pcs_coset_shift.exp_power_of_2(log_d))
            });
            assert_eq!(coset_shifts.len(), #num_chips);
            let (quotient_commit, quotient_data) = tracing::info_span!("commit to quotient chunks")
                .in_scope(|| pcs.commit_shifted_batches(quotients.to_vec(), &coset_shifts));

            challenger.observe(quotient_commit.clone());

            #[cfg(debug_assertions)]
            check_cumulative_sums(&perm_traces[..]);

            let zeta: SC::Challenge = challenger.sample_ext_element();
            let zeta_and_next: [Vec<SC::Challenge>; #num_chips] =
                g_subgroups.map(|g| vec![zeta, zeta * g]);
            let zeta_exp_quotient_degree: [Vec<SC::Challenge>; #num_chips] =
                log_quotient_degrees.map(|log_deg| vec![zeta.exp_power_of_2(log_deg)]);
            let prover_data_and_points = [
                // TODO: Causes some errors, probably related to the fact that not all chips have preprocessed traces?
                (preprocessed_data, zeta_and_next.as_slice()),
                (&main_data, zeta_and_next.as_slice()),
                (&perm_data, zeta_and_next.as_slice()),
                (&quotient_data, zeta_exp_quotient_degree.as_slice()),
            ];
            let (openings, opening_proof) = pcs.open_multi_batches(
               &prover_data_and_points, &mut challenger);

            // TODO: add preprocessed openings
            let [mut preprocessed_openings_real, mut main_openings_real, perm_openings, quotient_openings] =
                openings.try_into().expect("Should have 4 rounds of openings");

            // add None's back in so we can iterate through this as we do with the other openings
            let mut preprocessed_openings: Vec<_> = has_preprocessed_traces
                .map(|has_trace| {
                    if has_trace {
                        preprocessed_openings_real.remove(0)
                    } else {
                        vec![vec![], vec![]]
                    }
                })
                .collect();
            let mut main_openings: Vec<_> = has_main_traces
            .map(|has_trace| {
                if has_trace {
                    main_openings_real.remove(0)
                } else {
                    vec![vec![], vec![]]
                }
            })
            .collect();


            let commitments = Commitments {
                main_trace: main_commit,
                perm_trace: perm_commit,
                quotient_chunks: quotient_commit,
            };


            // TODO: add preprocessed openings
            let chip_proofs = log_degrees
                .iter()
                .zip(preprocessed_openings)
                .zip(main_openings)
                .zip(perm_openings)
                .zip(quotient_openings)
                .zip(perm_traces)
                .map(|(((((log_degree, preprocessed), main), perm), quotient), perm_trace)| {
                    let [preprocessed_local, preprocessed_next] =
                        preprocessed.try_into().expect("Should have 2 openings");

                    let [main_local, main_next] = main.try_into().expect("Should have 2 openings");
                    let [perm_local, perm_next] = perm.try_into().expect("Should have 2 openings");
                    let [quotient_chunks] = quotient.try_into().expect("Should have 1 opening");

                    let opened_values = OpenedValues {
                        preprocessed_local,
                        preprocessed_next,
                        trace_local: main_local,
                        trace_next: main_next,
                        permutation_local: perm_local,
                        permutation_next: perm_next,
                        quotient_chunks,
                    };

                    let cumulative_sum = perm_trace.row_slice(perm_trace.height() - 1).last().unwrap().clone();
                    ChipProof {
                        log_degree: *log_degree,
                        opened_values,
                        cumulative_sum,
                    }
                })
                .collect::<Vec<_>>();

            MachineProof {
                commitments,
                opening_proof,
                chip_proofs,
            }
        }
    }
}

fn verify_method(chips: &[&Field]) -> TokenStream2 {
    let num_chips = chips.len();
    let chip_list = chips
        .iter()
        .map(|chip| {
            let chip_name = chip.ident.as_ref().unwrap();
            quote! {
                alloc::boxed::Box::new(self.#chip_name()),
            }
        })
        .collect::<TokenStream2>();

    let quotient_degree_calls = chips
        .iter()
        .map(|chip| {
            let chip_name = chip.ident.as_ref().unwrap();
            quote! {
                get_log_quotient_degree::<Self, SC, _>(self, self.#chip_name()),
            }
        })
        .collect::<TokenStream2>();

    let verify_constraints = chips
        .iter()
        .enumerate()
        .map(|(i, chip)| {
            let chip_name = chip.ident.as_ref().unwrap();
            quote! {
                let chip = self.#chip_name();
                verify_constraints::<Self, _, SC>(
                    self,
                    self.#chip_name(),
                    &proof.chip_proofs[#i].opened_values,
                    &public_traces[#i],
                    proof.chip_proofs[#i].cumulative_sum,
                    proof.chip_proofs[#i].log_degree,
                    g_subgroups[#i],
                    zeta,
                    alpha,
                    &perm_challenges
                ).expect(&alloc::format!("Failed to verify constraints on chip {}", #i));
            }
        })
        .collect::<TokenStream2>();

    quote! {
        fn verify<SC: StarkConfig<Val = F>>(
            &self,
            config: &SC,
            proof: &::valida_machine::MachineProof<SC>,
            vk: &MachineVerifierKey<SC, Self>,
            instance_data: &Self::InstanceData,
            show_public: Vec<bool>,
        ) -> core::result::Result<(), VerificationError<SC>>
        {
            use ::valida_machine::__internal::*;
            use ::valida_machine::__internal::p3_air::{BaseAir};
            use ::valida_machine::__internal::p3_field::{AbstractField, AbstractExtensionField};
            use ::valida_machine::__internal::p3_challenger::{CanObserve, FieldChallenger};
            use ::valida_machine::__internal::p3_commit::{Pcs, UnivariatePcs, UnivariatePcsWithLde};
            use ::valida_machine::__internal::p3_matrix::Dimensions;
            use ::valida_machine::__internal::p3_util::log2_strict_usize;
            use ::valida_machine::{verify_constraints, MachineProof, ChipProof, Commitments, PublicTrace};
            use ::valida_machine::OpenedValues;
            use ::valida_machine::{VerificationError, ProofShapeError, OodEvaluationMismatch};
            use alloc::vec;
            use alloc::vec::Vec;
            use alloc::boxed::Box;


            let mut chips: [Box<&dyn Chip<Self, SC, Public=PublicTrace<SC::Val>>>; #num_chips] = [ #chip_list ];
            let log_quotient_degrees: [usize; #num_chips] = [ #quotient_degree_calls ];
            let mut challenger = config.challenger();
            // TODO: Seed challenger with digest of all constraints & trace lengths.
            let pcs = config.pcs();
            let (preprocessed_commit, preprocessed_dims) = (vk.preprocessed_commit(), vk.preprocessed_dims());

            let chips_interactions = chips
            .iter()
            .map(|chip| chip.all_interactions(self))
            .collect::<Vec<_>>();

            let show_public = vec![false; #num_chips];
            let public_traces: [Option<PublicTrace<SC::Val>>; #num_chips] =
                instance_data.public_traces(show_public).try_into().unwrap();

            let dims = &[
                preprocessed_dims.into_iter().flatten().collect::<Vec<_>>(),
                chips
                    .iter()
                    .zip(proof.chip_proofs.iter())

                    .flat_map(|(chip, chip_proof)| {
                        let width = chip.main_width();
                        if width == 0 {
                            None
                        } else {
                          Some(Dimensions {
                            width: chip.main_width(),
                            height: 1 << chip_proof.log_degree,
                            })
                        }
                    })
                    .collect::<Vec<_>>(),
                chips_interactions.iter()
                  .zip(proof.chip_proofs.iter())
                    .map(|(interactions, chip_proof)| Dimensions {
                        width: (interactions.len() + 1) * SC::Challenge::D,
                        height: 1 << chip_proof.log_degree,
                    })
                    .collect::<Vec<_>>(),
                proof.chip_proofs.iter()
                    .zip(log_quotient_degrees)
                    .map(|(chip_proof, log_quotient_deg)| Dimensions {
                        width: log_quotient_deg << SC::Challenge::D,
                        height: 1 << chip_proof.log_degree,
                    })
                    .collect::<Vec<_>>(),
            ];

            challenger.observe(preprocessed_commit.clone());

            // Get the generators of the trace subgroups for each chip.
            let g_subgroups: [SC::Val; #num_chips] = proof.chip_proofs
                .iter()
                .map(|chip_proof| SC::Val::two_adic_generator(chip_proof.log_degree))
                .collect::<Vec<_>>().try_into().unwrap();

            // TODO: maybe avoid cloning opened values (not sure if possible)
            let mut preprocessed_values = vec![];
            let mut main_values = vec![];
            let mut perm_values = vec![];
            let mut quotient_values = vec![];

            for chip_proof in proof.chip_proofs.iter() {
                let OpenedValues {
                    preprocessed_local,
                    preprocessed_next,
                    trace_local,
                    trace_next,
                    permutation_local,
                    permutation_next,
                    quotient_chunks,
                } = &chip_proof.opened_values;
                if !preprocessed_local.is_empty() {
                    preprocessed_values
                    .push(vec![preprocessed_local.clone(), preprocessed_next.clone()]);
                }
                if !trace_local.is_empty() {
                main_values.push(vec![trace_local.clone(), trace_next.clone()]);}
                perm_values.push(vec![permutation_local.clone(), permutation_next.clone()]);
                quotient_values.push(vec![quotient_chunks.clone()]);
            }

            let chips_opening_values = vec![preprocessed_values,
                main_values,
                perm_values,
                quotient_values
                ];

            // Observe commitments and get challenges.
            let Commitments {
                main_trace,
                perm_trace,
                quotient_chunks,
            } = &proof.commitments;


            challenger.observe(main_trace.clone());

            let mut perm_challenges = Vec::new();
            for _ in 0..2 {
                perm_challenges.push(challenger.sample_ext_element::<SC::Challenge>());
            }

            challenger.observe(perm_trace.clone());

            let alpha = challenger.sample_ext_element::<SC::Challenge>();

            challenger.observe(quotient_chunks.clone());

            // Verify the opening proof.
            let zeta: SC::Challenge = challenger.sample_ext_element();
            let zeta_and_next: [Vec<SC::Challenge>; #num_chips] =
                g_subgroups.map(|g| vec![zeta, zeta * g]);
            let zeta_exp_quotient_degree: [Vec<SC::Challenge>; #num_chips] =
                log_quotient_degrees.map(|log_deg| vec![zeta.exp_power_of_2(log_deg)]);
            pcs
                .verify_multi_batches(
                    &[
                        (preprocessed_commit.clone(), zeta_and_next.as_slice()),
                        (main_trace.clone(), zeta_and_next.as_slice()),
                        (perm_trace.clone(), zeta_and_next.as_slice()),
                        (quotient_chunks.clone(), zeta_exp_quotient_degree.as_slice()),
                    ],
                    dims,
                    chips_opening_values,
                    &proof.opening_proof,
                    &mut challenger,
                )
                .map_err(PcsError)?;

            // Verify the constraints.
            #verify_constraints
            // Verify that the cumulative_sum sums add up to zero.
            let sum: SC::Challenge = proof
                .chip_proofs
                .iter()
                .map(|chip_proof| chip_proof.cumulative_sum)
                .sum();

            if sum != SC::Challenge::zero() {
                return Err(VerificationError::<SC>::CumulativeSumMismatch);
            }

            Ok(())
        }
    }
}

#[proc_macro_derive(AlignedBorrow)]
pub fn aligned_borrow_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    // Get struct name from ast
    let name = &ast.ident;
    let methods = quote! {
        impl<T> Borrow<#name<T>> for [T] {
            fn borrow(&self) -> &#name<T> {
                debug_assert_eq!(self.len(), size_of::<#name<u8>>());
                let (prefix, shorts, _suffix) = unsafe { self.align_to::<#name<T>>() };
                debug_assert!(prefix.is_empty(), "Alignment should match");
                debug_assert_eq!(shorts.len(), 1);
                &shorts[0]
            }
        }

        impl<T> BorrowMut<#name<T>> for [T] {
            fn borrow_mut(&mut self) -> &mut #name<T> {
                debug_assert_eq!(self.len(), size_of::<#name<u8>>());
                let (prefix, shorts, _suffix) = unsafe { self.align_to_mut::<#name<T>>() };
                debug_assert!(prefix.is_empty(), "Alignment should match");
                debug_assert_eq!(shorts.len(), 1);
                &mut shorts[0]
            }
        }
    };
    methods.into()
}
