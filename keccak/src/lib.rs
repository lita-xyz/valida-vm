extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use columns::{indexes, WrapKeccakCols, NUM_WRAP_KECCAK_COLS, WRAPKECCAK_COL_MAP};
use core::{borrow::Borrow, mem::transmute};
use p3_symmetric::Permutation;
use valida_bus::{MachineWithMemBus, MachineWithPointerBus};
use valida_cpu::MachineWithCpuChip;
use valida_machine::{
    instructions, Chip, ChipTraceHeight, ChipWithPersistence, Instruction, Interaction, Operands,
    PublicTrace, RunningMachine, Word, MEMORY_CELL_BYTES,
};
use valida_opcodes::KECCAKF;

use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_keccak::KeccakF as KeccakFExecute;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use valida_machine::StarkConfig;

use valida_memory_footprint::MemoryFootprint;

use constants::rc_value_limb;
use logic::{andn, xor};

use crate::columns::indexes_x;
use spin::Mutex;

pub mod columns;
pub mod constants;
pub mod logic;
pub mod stark;

pub const NUM_ROUNDS: usize = 24;
pub const BITS_PER_LIMB: usize = 8;
pub const U64_LIMBS: usize = 64 / BITS_PER_LIMB;
const HASH_STATE_LIMBS: usize = 200;

#[derive(Clone)]
pub struct Operation {
    base_adddress: Word<u8>,
    clk: u32,
}

impl MemoryFootprint for Operation {
    fn memory_footprint(&self) -> usize {
        self.base_adddress.memory_footprint() + self.clk.memory_footprint()
    }
}

pub struct KeccakFChip {
    pub operations: Vec<Operation>,
    pub preimage: Vec<[Word<u8>; 50]>,
}

impl MemoryFootprint for KeccakFChip {
    fn memory_footprint(&self) -> usize {
        self.operations.memory_footprint() + self.preimage.memory_footprint()
    }
}

impl Default for KeccakFChip {
    fn default() -> Self {
        Self {
            operations: Vec::new(),
            preimage: Vec::new(),
        }
    }
}

impl ChipTraceHeight for KeccakFChip {
    fn trace_height(&self) -> u32 {
        self.operations.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for KeccakFChip
where
    M: MachineWithMemBus<SC::Val> + MachineWithPointerBus<SC::Val>,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "Keccak".to_string()
    }

    fn generate_main_trace(
        &self,
        _machine: &M,
        verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>)
    where
        M: MachineWithMemBus<SC::Val> + MachineWithPointerBus<SC::Val>,
        SC: StarkConfig,
    {
        let num_ops = self.operations.len();
        let real_rows = num_ops * NUM_ROUNDS;
        let num_padded_rows = real_rows.next_power_of_two();
        let padding_rows = num_padded_rows - real_rows;

        let values = Mutex::new(vec![
            SC::Val::zero();
            num_padded_rows * NUM_WRAP_KECCAK_COLS
        ]);
        let mut log_prints = if verbose {
            Some(Vec::with_capacity(num_padded_rows))
        } else {
            None
        };

        // Process each real operation
        self.operations
            .par_iter()
            .enumerate()
            .for_each(|(op_idx, op)| {
                // Initialize the input state for this operation
                let mut input = [[[SC::Val::zero(); U64_LIMBS]; 5]; 5];

                // Convert preimage bytes to the initial state for this operation
                let preimage_bytes: [u8; 200] = self.preimage[op_idx]
                    .into_iter()
                    .flat_map(Word::into_iter_le)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap_or_else(|_| panic!("Failed to convert preimage to bytes"));

                // Initialize the input state from preimage bytes
                for i in 0..HASH_STATE_LIMBS {
                    let (x, y, k) = indexes_x(i);
                    input[x][y][k] = SC::Val::from_canonical_u8(preimage_bytes[i]);
                }

                // Process each round for this operation
                for round in 0..NUM_ROUNDS {
                    let row_idx = op_idx * NUM_ROUNDS + round;
                    let row_start = row_idx * NUM_WRAP_KECCAK_COLS;
                    let row_end = row_start + NUM_WRAP_KECCAK_COLS;

                    // Generate row data and get updated state
                    let (row_data, new_state) =
                        self.op_to_row::<<SC as StarkConfig>::Val>(op, round, input, true);

                    // Copy row data to the values vector
                    let mut values = values.lock();
                    values[row_start..row_end].copy_from_slice(&row_data);

                    // Update input state for next round
                    input = new_state;
                }
            });

        if let Some(log) = &mut log_prints {
            for (i, row) in values
                .lock()
                .chunks(NUM_WRAP_KECCAK_COLS)
                .take(real_rows)
                .enumerate()
            {
                let cols: &WrapKeccakCols<SC::Val> = row.borrow();
                log.push(format!("Keccak row {}: {:?}", i, cols));
            }
        }

        // Handle padding with dummy operations
        if padding_rows > 0 {
            let dummy_op = Operation {
                base_adddress: Word::default(),
                clk: 0,
            };

            let mut input = [[[SC::Val::zero(); U64_LIMBS]; 5]; 5]; // Zero state
            let full_dummy_ops = padding_rows / NUM_ROUNDS;
            let remaining_rows = padding_rows % NUM_ROUNDS;

            // Add complete dummy operations
            for op_idx in 0..full_dummy_ops {
                for round in 0..NUM_ROUNDS {
                    let row_idx = real_rows + (op_idx * NUM_ROUNDS) + round;
                    let row_start = row_idx * NUM_WRAP_KECCAK_COLS;
                    let row_end = row_start + NUM_WRAP_KECCAK_COLS;

                    let (row_data, new_state) =
                        self.op_to_row::<<SC as StarkConfig>::Val>(&dummy_op, round, input, false);

                    let mut values = values.lock();
                    values[row_start..row_end].copy_from_slice(&row_data);
                    input = new_state;

                    if let Some(log) = &mut log_prints {
                        let cols: &WrapKeccakCols<SC::Val> = row_data[..].borrow();
                        log.push(format!("KeccakF padding row {}: {:?}", row_idx, cols));
                    }
                }
                // Reset input state to zero for next dummy operation
                input = [[[SC::Val::zero(); U64_LIMBS]; 5]; 5];
            }

            // Add partial dummy operation if needed
            if remaining_rows > 0 {
                for round in 0..remaining_rows {
                    let row_idx = real_rows + (full_dummy_ops * NUM_ROUNDS) + round;
                    let row_start = row_idx * NUM_WRAP_KECCAK_COLS;
                    let row_end = row_start + NUM_WRAP_KECCAK_COLS;

                    let (row_data, new_state) =
                        self.op_to_row::<<SC as StarkConfig>::Val>(&dummy_op, round, input, false);

                    let mut values = values.lock();
                    values[row_start..row_end].copy_from_slice(&row_data);
                    input = new_state;

                    if let Some(log) = &mut log_prints {
                        let cols: &WrapKeccakCols<SC::Val> = row_data[..].borrow();
                        log.push(format!("KeccakF padding row {}: {:?}", row_idx, cols));
                    }
                }
            }
        }

        (
            Some(RowMajorMatrix {
                values: values.into_inner(),
                width: NUM_WRAP_KECCAK_COLS,
            }),
            log_prints,
        )
    }

    fn global_sends(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let mut interactions = Vec::<Interaction<SC::Val>>::new();

        for is_postimage in [false, true] {
            for word_idx in 0..HASH_STATE_LIMBS / MEMORY_CELL_BYTES {
                let base_limb_idx = word_idx * MEMORY_CELL_BYTES;
                let mut limbs = [0; MEMORY_CELL_BYTES];

                for j in 0..MEMORY_CELL_BYTES {
                    let i = base_limb_idx + j;
                    if is_postimage {
                        let (x, y, limb_index) = indexes(i);
                        limbs[j] = WRAPKECCAK_COL_MAP.a_prime_prime_prime(x, y, limb_index)
                    } else {
                        let (x, y, limb_index) = indexes(i);
                        limbs[j] = WRAPKECCAK_COL_MAP.a[y][x][limb_index]
                    };
                }

                let value = Word::from_components_le(limbs)
                    .transform(|byte| VirtualPairCol::single_main(byte));

                let is_read = VirtualPairCol::constant(if is_postimage {
                    SC::Val::zero()
                } else {
                    SC::Val::one()
                });

                let clk = VirtualPairCol::single_main(WRAPKECCAK_COL_MAP.clk);

                let addr = VirtualPairCol::new_main(
                    vec![
                        (*WRAPKECCAK_COL_MAP.base_address.index_le(0), SC::Val::one()),
                        (
                            *WRAPKECCAK_COL_MAP.base_address.index_le(1),
                            SC::Val::from_canonical_u32(1 << 8),
                        ),
                        (
                            *WRAPKECCAK_COL_MAP.base_address.index_le(2),
                            SC::Val::from_canonical_u32(1 << 16),
                        ),
                        (
                            *WRAPKECCAK_COL_MAP.base_address.index_le(3),
                            SC::Val::from_canonical_u32(1 << 24),
                        ),
                    ],
                    SC::Val::from_canonical_usize(word_idx * 4),
                );

                let mut fields = vec![is_read, clk, addr];
                fields.extend(value.into_iter_le());

                let interaction = if is_postimage {
                    Interaction {
                        fields: fields,
                        count: VirtualPairCol::single_main(WRAPKECCAK_COL_MAP.export_output),
                        argument_index: machine.mem_bus(),
                    }
                } else {
                    Interaction {
                        fields: fields,
                        count: VirtualPairCol::single_main(WRAPKECCAK_COL_MAP.export_input),
                        argument_index: machine.mem_bus(),
                    }
                };
                interactions.push(interaction);
            }
        }
        interactions
    }

    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let opcode = VirtualPairCol::constant(SC::Val::from_canonical_u32(KECCAKF));
        let mut fields = vec![opcode];

        let base_address = WRAPKECCAK_COL_MAP
            .base_address
            .iter_le()
            .map(|&elem| VirtualPairCol::single_main(elem))
            .collect::<Vec<_>>();

        fields.extend(base_address);

        let receive = Interaction {
            fields,
            count: VirtualPairCol::single_main(WRAPKECCAK_COL_MAP.is_real),
            argument_index: machine.pointer_bus(),
        };
        vec![receive]
    }
}

impl<M, SC> ChipWithPersistence<M, SC> for KeccakFChip
where
    SC: StarkConfig,
    M: MachineWithMemBus<SC::Val> + MachineWithPointerBus<SC::Val>,
{
}

impl KeccakFChip {
    fn op_to_row<F>(
        &self,
        op: &Operation,
        round: usize,
        input: [[[F; U64_LIMBS]; 5]; 5],
        export: bool,
    ) -> ([F; NUM_WRAP_KECCAK_COLS], [[[F; U64_LIMBS]; 5]; 5])
    where
        F: PrimeField + PrimeField32,
    {
        let round_24 = round % NUM_ROUNDS;
        let mut row = [F::zero(); NUM_WRAP_KECCAK_COLS];
        let cols: &mut WrapKeccakCols<F> = unsafe { transmute(&mut row) };
        cols.step_flags[round_24] = F::one();

        if round == 0 {
            cols.export_input = F::from_bool(export);
            cols.is_real = F::from_bool(export);
        }
        if round == NUM_ROUNDS - 1 {
            cols.export_output = F::from_bool(export);
        }

        for y in 0..5 {
            for x in 0..5 {
                for limb in 0..U64_LIMBS {
                    cols.a[y][x][limb] = input[y][x][limb];
                }
            }
        }

        // Populate C[x] = xor(A[x, 0], A[x, 1], A[x, 2], A[x, 3], A[x, 4]).
        for x in 0..5 {
            for z in 0..64 {
                let limb = z / BITS_PER_LIMB;
                let bit_in_limb = z % BITS_PER_LIMB;
                let a = (0..5).map(|y| {
                    let a_limb = cols.a[y][x][limb].as_canonical_u32();
                    ((a_limb >> bit_in_limb) & 1) != 0
                });
                cols.c[x][z] = F::from_bool(a.fold(false, |acc, x| acc ^ x));
            }
        }

        // Populate C'[x, z] = xor(C[x, z], C[x - 1, z], C[x + 1, z - 1]).
        for x in 0..5 {
            for z in 0..64 {
                cols.c_prime[x][z] = xor([
                    cols.c[x][z],
                    cols.c[(x + 4) % 5][z],
                    cols.c[(x + 1) % 5][(z + 63) % 64],
                ]);
            }
        }

        // Populate A'. To avoid shifting indices, we rewrite
        //     A'[x, y, z] = xor(A[x, y, z], C[x - 1, z], C[x + 1, z - 1])
        // as
        //     A'[x, y, z] = xor(A[x, y, z], C[x, z], C'[x, z]).
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    let limb = z / BITS_PER_LIMB;
                    let bit_in_limb = z % BITS_PER_LIMB;
                    let a_limb = cols.a[y][x][limb].as_canonical_u64() as u8;
                    let a_bit = F::from_bool(((a_limb >> bit_in_limb) & 1) != 0);
                    cols.a_prime[y][x][z] = xor([a_bit, cols.c[x][z], cols.c_prime[x][z]]);
                }
            }
        }

        // Populate A''.
        // A''[x, y] = xor(B[x, y], andn(B[x + 1, y], B[x + 2, y])).
        for y in 0..5 {
            for x in 0..5 {
                for limb in 0..U64_LIMBS {
                    cols.a_prime_prime[y][x][limb] = (limb * BITS_PER_LIMB
                        ..(limb + 1) * BITS_PER_LIMB)
                        .rev()
                        .fold(F::zero(), |acc, z| {
                            let bit = xor([
                                cols.b(x, y, z),
                                andn(cols.b((x + 1) % 5, y, z), cols.b((x + 2) % 5, y, z)),
                            ]);
                            acc.double() + bit
                        });
                }
            }
        }

        // For the XOR, we split A''[0, 0] to bits.
        let mut val = 0;
        for limb in 0..U64_LIMBS {
            let val_limb = cols.a_prime_prime[0][0][limb].as_canonical_u64();
            val |= val_limb << (limb * BITS_PER_LIMB);
        }
        let val_bits: Vec<bool> = (0..64)
            .scan(val, |acc, _| {
                let bit = (*acc & 1) != 0;
                *acc >>= 1;
                Some(bit)
            })
            .collect();
        for (i, bit) in cols.a_prime_prime_0_0_bits.iter_mut().enumerate() {
            *bit = F::from_bool(val_bits[i]);
        }

        // A''[0, 0] is additionally xor'd with RC.
        for limb in 0..U64_LIMBS {
            let rc_lo = rc_value_limb(round_24, limb);
            cols.a_prime_prime_prime_0_0_limbs[limb] = F::from_canonical_u8(
                cols.a_prime_prime[0][0][limb].as_canonical_u64() as u8 ^ rc_lo,
            );
        }

        let mut output = [[[F::zero(); U64_LIMBS]; 5]; 5];

        for y in 0..5 {
            for x in 0..5 {
                for limb in 0..U64_LIMBS {
                    output[y][x][limb] = cols.a_prime_prime_prime(x, y, limb);
                }
            }
        }

        cols.base_address = op.base_adddress.transform(F::from_canonical_u8);
        cols.clk = F::from_canonical_u32(op.clk);

        (row, output)
    }
}

pub trait MachineWithKeccakFChip<F: PrimeField>: MachineWithCpuChip<F> {
    fn keccak_f(&self) -> &KeccakFChip;
    fn keccak_f_mut(&mut self) -> &mut KeccakFChip;

    fn set_preimage(&mut self, preimage: [Word<u8>; 50]);
}

instructions!(KeccakFInstruction);

impl<M, F> Instruction<M, F> for KeccakFInstruction
where
    M: MachineWithCpuChip<F> + MachineWithKeccakFChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = KECCAKF;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode: u32 = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;
        let fp = state.machine.cpu().fp;

        let base_address_loc: u32 = (fp as i32 + ops.b()) as u32;
        let base_address = M::read(state, clk, base_address_loc);

        let mut preimage: [Word<u8>; 50] = [Word::default(); 50];
        for i in 0..50 {
            let current_address = base_address + (i as u32 * 4).into();
            preimage[i as usize] = M::read(state, clk, u32::from(current_address));
        }

        state.machine.set_preimage(preimage);

        let keccak = KeccakFExecute;

        let mut state_bytes: [u8; 200] = preimage
            .into_iter()
            .flat_map(Word::into_iter_le)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("Failed to convert preimage to state bytes"));

        keccak.permute_mut(&mut state_bytes);

        let postimage: [Word<u8>; 50] = state_bytes
            .chunks_exact(4)
            .map(|chunk| Word::from_components_le(chunk.try_into().unwrap()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| panic!("Failed to convert state bytes to postimage"));

        for i in 0..50 {
            let current_address = base_address + ((i as u32) * 4).into();
            M::write(state, clk, current_address.into(), postimage[i as usize]);
        }

        if state.machine.log_enabled() {
            state.machine.keccak_f_mut().operations.push(Operation {
                base_adddress: base_address,
                clk: clk,
            });
            state.machine.push_pointer_op(opcode, ops);
        }

        state.machine.step_pc();
    }
}
