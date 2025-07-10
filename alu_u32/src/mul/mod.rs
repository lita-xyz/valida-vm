extern crate alloc;

use alloc::{collections::BTreeMap, vec};
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use core::borrow::Borrow;
use core::cmp::min;
use core::num::Wrapping;
use core::ops::Sub;
use core::{fmt::Debug, ops::Add};

use itertools::iproduct;
use spin::Mutex;

use columns::{
    Mul32Cols, CARRY_LENGTH, CARRY_MAX, LIMB_SIZE, MUL_COL_MAP, NUM_MUL_COLS, PRODUCT_LENGTH,
    PRODUCT_LIMBS,
};
use valida_bus::{MachineWithBytesBus, MachineWithGeneralBus, MachineWithRangeBus8};
use valida_cpu::MachineWithCpuChip;
use valida_machine::{
    instructions, Chip, ChipTraceHeight, ChipWithPersistence, Instruction, Interaction, Mulhs,
    Mulhu, Operands, PublicTrace, RunningMachine, Word,
};
use valida_opcodes::{MUL32, MULHS32, MULHU32};

use core::{borrow::BorrowMut, iter::Sum, ops::Mul};
use p3_air::VirtualPairCol;
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use valida_machine::StarkConfig;

use valida_bytes::{
    byte_send_simple, range8_sends_word, ByteOperation, MachineWithBytesChip,
    MachineWithRangeCheckeru8,
};

use valida_memory_footprint::MemoryFootprint;

pub mod columns;
pub mod stark;

#[derive(Clone, Debug, Copy, Default)]
pub struct Long<F> {
    pub low: Word<F>,
    pub high: Word<F>,
}

impl<F> Long<F> {
    pub fn iter_le(&self) -> impl Iterator<Item = &F> {
        self.low.iter_le().chain(self.high.iter_le())
    }
}

impl From<Long<u8>> for u64 {
    fn from(long: Long<u8>) -> u64 {
        let low: u64 = Into::<u32>::into(long.low) as u64;
        let high: u64 = Into::<u32>::into(long.high) as u64;
        (high << 32) | low
    }
}

impl From<u64> for Long<u8> {
    fn from(val: u64) -> Self {
        let low = val as u32;
        let high = (val >> 32) as u32;
        Long {
            low: low.into(),
            high: high.into(),
        }
    }
}

impl Mul for Long<u8> {
    type Output = Long<u8>;

    fn mul(self, other: Self) -> Self::Output {
        let b: u64 = self.into();
        let c: u64 = other.into();
        let res = (Wrapping(b) * Wrapping(c)).0;
        res.into()
    }
}

impl Long<u8> {
    pub fn sign_extend_word(word: Word<u8>) -> Long<u8> {
        ((Into::<i32>::into(word) as i64) as u64).into()
    }
    pub fn zero_extend_word(word: Word<u8>) -> Long<u8> {
        Long {
            low: word,
            high: 0.into(),
        }
    }
}

/// This computes a partially-reduced sum of the "limbs" of the product in the range `[lower, upper)`, i.e. the partial sum
/// of all terms in the `base`-expanded product which get multiplied by `base[i]` for `i` in `[lower, upper)`.
/// Note that there are at most PRODUCT_LENGTH terms in the sum below which are multiplied by
/// `b[i]` for `i` in `[0, lower - upper)`.
/// ---=-------------------------------------------Taking `base` to be powers of 2^8: -------------------------
/// Assume that all entries of `input_1` and `input_2` are in the range [0, 2^8).
/// At most `PRODUCT_LENGTH = 2 * MEMORY_CELL_BYTES` terms of the sum are multiplied by each power of 2^8 from `0` to `U - L - 1`.
/// Each term in the product is at most 2^16, so the sum of the terms  a `2^(8*(U - L - 1))` is at most PRODUCT_LENGTH * 2^{8(U - L  + 1))`.
/// The sum of the other terms is at most `PRODUCT_LENGTH * 2^(8(U - L) + 1)`.
/// Thus, the full sum is at most `PRODUCT_LENGTH * 2^(8*(U - L - 1)) * (2^16 - 2^9 + 2) < PRODUCT_LENGTH * 2^(8(U-L+ 1)) < PI_MAX/2`.
fn pi_m<F, E>(base_le: &[E], input_1: &Long<F>, input_2: &Long<F>, lower: usize, upper: usize) -> E
where
    E: Mul<E, Output = E> + Clone + Sum,
    F: Clone + Into<E>,
{
    let (input_1_le, input_2_le) = (
        input_1
            .iter_le()
            .map(|i| (i.clone()).into())
            .collect::<Vec<E>>(),
        input_2
            .iter_le()
            .map(|i| (i.clone()).into())
            .collect::<Vec<E>>(),
    );
    let max = min(upper, PRODUCT_LENGTH);
    iproduct!(0..max, 0..max)
        .filter(|(i, j)| i + j < upper && i + j >= lower)
        .map(|(i, j)| {
            base_le[i + j - lower].clone() * input_1_le[i].clone() * input_2_le[j].clone()
        })
        .sum()
}

/// This computes the sum of the `lower`-th through `upper-1`-st` digits of the 'base' expansion of `input`.
/// -----------------------------------Taking `base` to be powers of 2^8: -------------------------
/// Assuming all entries of input are in the range [0, 2^8), the output will be in the range [0, 2^{8`upper`}).
fn sigma_m<F, E>(base_le: &[E], input: &Long<F>, lower: usize, upper: usize) -> E
where
    E: Mul<E, Output = E> + Clone + Sum,
    F: Into<E> + Clone,
{
    debug_assert!(lower < upper);
    input
        .iter_le()
        .skip(lower)
        .take(upper - lower)
        .zip(base_le.iter())
        .map(|(x, b)| b.clone() * x.clone().into())
        .sum()
}

/// This computes the partially reduced sums of products of the input limbs as well as the
/// (reduced) sums of the output limbs. Each limb consists of two entries of the input `Word`s,
/// i.e. it is 16 bits long.
/// Note, as computed in the comment above `pi_m`, that the size of the partially reduced sums
/// for a limb of length `L` is at most `PRODUCT_LENGTH * 2^(8(L+1))`, so having
/// limbs of size greater than `2` can overflow a 32-bit field.
fn get_partially_reduced_and_reduced_sums<const MIN_LENGTH: usize, F, E>(
    base_le: &[E],
    input_1: &Long<F>,
    input_2: &Long<F>,
    output: &Long<F>,
) -> ([E; PRODUCT_LIMBS], [E; PRODUCT_LIMBS])
where
    E: Add<E, Output = E> + Mul<Output = E> + Sub<Output = E> + Sum<E> + Debug + Clone,
    F: Into<E> + Clone,
{
    debug_assert!(base_le.len() >= LIMB_SIZE);
    let pis: [E; PRODUCT_LIMBS] = (0..PRODUCT_LENGTH)
        .step_by(LIMB_SIZE)
        .map(move |i| pi_m(base_le, input_1, input_2, i, i + LIMB_SIZE))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let sigmas: [E; PRODUCT_LIMBS] = (0..PRODUCT_LENGTH)
        .step_by(LIMB_SIZE)
        .map(|i| sigma_m(base_le, output, i, i + LIMB_SIZE))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    (pis, sigmas)
}

/// Compute the "carry" limbs for the multiplication result.
///
/// Recall that the `pi`'s hold
/// partial sums of the byte-expanded product. The `sigma`s hold the byte-expansion of each
/// set of LIMB_SIZE bytes of the output.
/// In particular, `pi[0]` is congruent to the actual product modulo `2^(8*LIMB_SIZE)`, as
/// it contains all terms with total degree less than `LIMB_SIZE`. Thus, constraining
/// `sigma[0]` to be congruent to `pi[0]` modulo `2^8*LIMB_SIZE` ensures that the first
/// `LIMB_SIZE` bytes of the output are correct.
///
/// We require `carry[0]` to witness this congruence, i.e. constrain
/// `pi[0] = sigma[0] + 2^(8*LIMB_SIZE) * carry[0]`.
/// We require `carry[0] < CARRY_MAX`, possible since the range bounds on the input bytes
/// imply that `pi[0] < PI_MAX/2`.
///
/// Next, we know that `pi[0] + 2^(8*LIMB_SIZE)*pi[1]`, which sums all terms of the product
/// of degree at most 2*LIMB_SIZE, is congruent to the product modulo `2^(8*LIMB_SIZE)*2`.
/// Substituting the previous congruence, we get that the actual product is congruent to
/// `sigma[0] + 2^(8*LIMB_SIZE)*(carry[0] + pi[1])` modulo 2^(8*LIMB_SIZE*2).
/// So to ensure that `sigma[1]` equals the correct reduced second limb of the output,
/// it suffices to require `(pi[1] + carry[0])` to be congruent to `sigma[1]` modulo `2^(8*LIMB_SIZE)`.
/// Thus we require `pi[1] + carry[0] - sigma[1] = carry[1] * 2^(8*LIMB_SIZE)`.
/// As `pi[1] < PI_MAX/2`, `carry[0] < CARRY_MAX < PI_MAX/4`, and `sigma[1] < 2^(8*LIMB_SIZE) < PI_MAX/4`,
/// we know that the sum does not overflow.
/// We then proceed as above to constrain the remaining limbs of the output.
pub fn get_carries(input_1: &Long<u8>, input_2: &Long<u8>) -> Word<u16> {
    let base = [1u64, 1 << 8, 1 << 16, 1 << 24];
    let (pis, sigmas) = get_partially_reduced_and_reduced_sums::<{ Mul32Chip::MIN_LENGTH }, _, _>(
        &base,
        input_1,
        input_2,
        &(*input_1 * *input_2),
    );
    Word::from_components_le(
        pis.into_iter()
            .zip(sigmas)
            .enumerate()
            .scan(0, |prior_carry, (index, (pi, sigma))| {
                if index > CARRY_LENGTH {
                    None
                } else {
                    let diff = pi + *prior_carry - sigma;
                    debug_assert!(diff % base[2] == 0);
                    let carry = diff / base[2];
                    debug_assert!(carry < Mul32Chip::MIN_LENGTH as u64);
                    debug_assert!(carry < 1 << 16);
                    *prior_carry = carry;
                    Some(carry as u16)
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
    )
}

#[derive(Clone)]
pub enum Operation {
    Mul32(Word<u8>, Word<u8>, Word<u8>),
    Mulhs32(Word<u8>, Word<u8>, Word<u8>),
    Mulhu32(Word<u8>, Word<u8>, Word<u8>),
}

impl MemoryFootprint for Operation {
    fn memory_footprint(&self) -> usize {
        match self {
            Operation::Mul32(a, b, c) => 3 * b.memory_footprint(),
            Operation::Mulhs32(a, b, c) => 3 * b.memory_footprint(),
            Operation::Mulhu32(a, b, c) => 3 * b.memory_footprint(),
        }
    }
}

#[derive(Default)]
pub struct Mul32Chip {
    pub operations: Vec<Operation>,
    pub range_check_counts: BTreeMap<u16, u32>,
}

impl MemoryFootprint for Mul32Chip {
    fn memory_footprint(&self) -> usize {
        self.operations.memory_footprint() + self.range_check_counts.memory_footprint()
    }
}

impl ChipTraceHeight for Mul32Chip {
    fn trace_height(&self) -> u32 {
        self.operations.len() as u32
    }
}

impl<M, SC> Chip<M, SC> for Mul32Chip
where
    M: MachineWithGeneralBus<SC::Val>
        + MachineWithRangeBus8<SC::Val>
        + MachineWithBytesBus<SC::Val>,
    SC: StarkConfig,
{
    type Public = PublicTrace<SC::Val>;

    fn name(&self) -> String {
        "Mul32".to_string()
    }

    fn generate_main_trace(
        &self,
        _machine: &M,
        verbose: bool,
    ) -> (Option<RowMajorMatrix<SC::Val>>, Option<Vec<String>>) {
        let num_ops = self.operations.len();
        let num_padded_ops = num_ops.next_power_of_two().max(Self::MIN_LENGTH);
        let values = Mutex::new(vec![SC::Val::zero(); num_padded_ops * NUM_MUL_COLS]);

        // Encode the real operations.
        self.operations.par_iter().enumerate().for_each(|(i, op)| {
            let mut values = values.lock();
            let row = &mut values[i * NUM_MUL_COLS..(i + 1) * NUM_MUL_COLS];
            let cols: &mut Mul32Cols<SC::Val> = row.borrow_mut();
            if i < Self::MIN_LENGTH {
                let mult = self.range_check_counts.get(&(i as u16)).unwrap_or(&0);
                cols.counter_mult = SC::Val::from_canonical_u32(*mult);
                cols.counter = SC::Val::from_canonical_usize(i);
            } else {
                cols.counter = SC::Val::from_canonical_usize(Self::MIN_LENGTH - 1);
            }
            self.op_to_row(op, cols);
        });
        let log = if verbose {
            let mut log_prints = Vec::with_capacity(num_ops);
            for (i, row) in values.lock().chunks(NUM_MUL_COLS).take(num_ops).enumerate() {
                let cols: &Mul32Cols<SC::Val> = row.borrow();
                log_prints.push(format!("Mul32 row {}: {:?}", i, cols));
            }
            Some(log_prints)
        } else {
            None
        };

        // Encode dummy operations as needed to pad the trace.
        (num_ops..num_padded_ops).into_par_iter().for_each(|i| {
            let mut values = values.lock();
            let row = &mut values[i * NUM_MUL_COLS..(i + 1) * NUM_MUL_COLS];
            let cols: &mut Mul32Cols<SC::Val> = row.borrow_mut();
            if i < Self::MIN_LENGTH {
                let mult = self.range_check_counts.get(&(i as u16)).unwrap_or(&0);
                cols.counter_mult = SC::Val::from_canonical_u32(*mult);
                cols.counter = SC::Val::from_canonical_usize(i);
            } else {
                cols.counter = SC::Val::from_canonical_usize(Self::MIN_LENGTH - 1);
            }
        });

        (
            Some(RowMajorMatrix {
                values: values.into_inner(),
                width: NUM_MUL_COLS,
            }),
            log,
        )
    }

    fn global_receives(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let opcode = VirtualPairCol::new_main(
            vec![
                (MUL_COL_MAP.is_mul, SC::Val::from_canonical_u32(MUL32)),
                (MUL_COL_MAP.is_mulhs, SC::Val::from_canonical_u32(MULHS32)),
                (MUL_COL_MAP.is_mulhu, SC::Val::from_canonical_u32(MULHU32)),
            ],
            SC::Val::zero(),
        );
        let input_1 = MUL_COL_MAP.input_1.transform(VirtualPairCol::single_main);
        let input_2 = MUL_COL_MAP.input_2.transform(VirtualPairCol::single_main);

        let mul_output = MUL_COL_MAP
            .lower_word
            .transform(VirtualPairCol::single_main);
        let mulhu_output = MUL_COL_MAP
            .upper_word
            .transform(VirtualPairCol::single_main);
        // TODO: implement output for mulhs
        let mulhs_output = mulhu_output.clone();

        let is_mul = VirtualPairCol::single_main(MUL_COL_MAP.is_mul);
        let is_mulhu = VirtualPairCol::single_main(MUL_COL_MAP.is_mulhu);
        let is_mulhs = VirtualPairCol::single_main(MUL_COL_MAP.is_mulhs);

        let mul_receive = {
            let mut fields = vec![opcode.clone()];
            fields.extend(input_1.clone().into_iter_le());
            fields.extend(input_2.clone().into_iter_le());
            fields.extend(mul_output.into_iter_le());

            Interaction {
                fields,
                count: is_mul,
                argument_index: machine.general_bus(),
            }
        };
        let mulhu_receive = {
            let mut fields = vec![opcode.clone()];
            fields.extend(input_1.clone().into_iter_le());
            fields.extend(input_2.clone().into_iter_le());
            fields.extend(mulhu_output.into_iter_le());

            Interaction {
                fields,
                count: is_mulhu,
                argument_index: machine.general_bus(),
            }
        };
        let mulhs_receive = {
            let mut fields = vec![opcode];
            fields.extend(input_1.into_iter_le());
            fields.extend(input_2.into_iter_le());
            fields.extend(mulhs_output.into_iter_le());

            Interaction {
                fields,
                count: is_mulhs,
                argument_index: machine.general_bus(),
            }
        };
        vec![mul_receive, mulhu_receive, mulhs_receive]
    }

    fn global_sends(&self, machine: &M) -> Vec<Interaction<SC::Val>> {
        let is_real = VirtualPairCol::sum_main(vec![
            MUL_COL_MAP.is_mul,
            MUL_COL_MAP.is_mulhs,
            MUL_COL_MAP.is_mulhu,
        ]);
        let output_range_sends = [MUL_COL_MAP.lower_word, MUL_COL_MAP.upper_word]
            .into_iter()
            .flat_map(|word| {
                range8_sends_word(
                    machine,
                    word.transform(VirtualPairCol::single_main),
                    &is_real,
                )
            })
            .collect::<Vec<_>>();

        let (sign_1_col, sign_2_col) = (
            VirtualPairCol::single_main(MUL_COL_MAP.sign_1),
            VirtualPairCol::single_main(MUL_COL_MAP.sign_2),
        );
        let (top_byte_1, top_byte_2) = (
            VirtualPairCol::single_main(*MUL_COL_MAP.input_1.index_be(0)),
            VirtualPairCol::single_main(*MUL_COL_MAP.input_2.index_be(0)),
        );

        let (sign_1_send, sign_2_send) = (
            byte_send_simple(
                machine,
                top_byte_1,
                Some(sign_1_col),
                VirtualPairCol::single_main(MUL_COL_MAP.is_mulhs),
                ByteOperation::MostSignificantBit,
            ),
            byte_send_simple(
                machine,
                top_byte_2,
                Some(sign_2_col),
                VirtualPairCol::single_main(MUL_COL_MAP.is_mulhs),
                ByteOperation::MostSignificantBit,
            ),
        );

        output_range_sends
            .into_iter()
            .chain(vec![sign_1_send, sign_2_send])
            .collect()
    }

    fn local_sends(&self) -> Vec<Interaction<SC::Val>> {
        MUL_COL_MAP
            .carry
            .iter_le()
            .map(|carry| {
                let carry = VirtualPairCol::single_main(*carry);
                let is_real = VirtualPairCol::sum_main(vec![
                    MUL_COL_MAP.is_mul,
                    MUL_COL_MAP.is_mulhs,
                    MUL_COL_MAP.is_mulhu,
                ]);
                Interaction {
                    fields: vec![carry],
                    count: is_real,
                    argument_index: valida_machine::BusArgument::Local(0),
                }
            })
            .collect()
    }

    fn local_receives(&self) -> Vec<Interaction<SC::Val>> {
        vec![Interaction {
            fields: vec![VirtualPairCol::single_main(MUL_COL_MAP.counter)],
            count: VirtualPairCol::single_main(MUL_COL_MAP.counter_mult),
            argument_index: valida_machine::BusArgument::Local(0),
        }]
    }
}
impl<M, SC> ChipWithPersistence<M, SC> for Mul32Chip
where
    M: MachineWithGeneralBus<SC::Val>
        + MachineWithRangeBus8<SC::Val>
        + MachineWithBytesBus<SC::Val>,
    SC: StarkConfig,
{
}
impl Mul32Chip {
    const MIN_LENGTH: usize = CARRY_MAX; // for the range check counter

    fn op_to_row<F>(&self, op: &Operation, cols: &mut Mul32Cols<F>)
    where
        F: PrimeField,
    {
        match op {
            Operation::Mul32(a, b, c) => {
                cols.is_mul = F::one();
                self.set_cols(a, b, c, cols);
            }
            Operation::Mulhs32(a, b, c) => {
                cols.is_mulhs = F::one();
                self.set_cols(a, b, c, cols);
            }
            Operation::Mulhu32(a, b, c) => {
                cols.is_mulhu = F::one();
                self.set_cols(a, b, c, cols);
            }
        }
    }

    fn set_cols<F>(&self, _a: &Word<u8>, b: &Word<u8>, c: &Word<u8>, cols: &mut Mul32Cols<F>)
    where
        F: PrimeField,
    {
        cols.input_1 = b.transform(F::from_canonical_u8);
        cols.input_2 = c.transform(F::from_canonical_u8);

        let (b_extended, c_extended) = if cols.is_mulhs == F::one() {
            (Long::sign_extend_word(*b), Long::sign_extend_word(*c))
        } else {
            (Long::zero_extend_word(*b), Long::zero_extend_word(*c))
        };

        let (sign_b, sign_c) = if cols.is_mulhs == F::one() {
            (Into::<i32>::into(*b) < 0, Into::<i32>::into(*c) < 0)
        } else {
            (false, false)
        };
        cols.sign_1 = F::from_bool(sign_b);
        cols.sign_2 = F::from_bool(sign_c);

        let Long {
            low: prod_lower,
            high: prod_upper,
        } = b_extended * c_extended;

        cols.carry = get_carries(&b_extended, &c_extended).transform(F::from_canonical_u16);

        cols.lower_word = Word::transform(prod_lower, F::from_canonical_u8);
        cols.upper_word = Word::transform(prod_upper, F::from_canonical_u8);
    }
}

pub trait MachineWithMul32Chip<F: PrimeField32>:
    MachineWithCpuChip<F> + MachineWithRangeCheckeru8<F>
{
    fn mul_32(&self) -> &Mul32Chip;
    fn mul_32_mut(&mut self) -> &mut Mul32Chip;

    // Checks that each entry of `cols.carry` is at most `MIN_LENGTH`.
    fn range_check_mul_chip_carries<T>(&mut self, carries: &Word<T>)
    where
        T: Into<u32> + Clone + Debug,
    {
        if self.log_enabled() {
            carries.iter_le().for_each(|carry| {
                let carry_32: u32 = ((*carry).clone()).into();
                debug_assert!((carry_32 as usize) < Mul32Chip::MIN_LENGTH);
                *(self
                    .mul_32_mut()
                    .range_check_counts
                    .entry(carry_32 as u16)
                    .or_insert(0)) += 1;
            })
        }
    }
}

instructions!(Mul32Instruction, Mulhs32Instruction, Mulhu32Instruction);

impl<M, F> Instruction<M, F> for Mul32Instruction
where
    M: MachineWithMul32Chip<F> + MachineWithRangeCheckeru8<F>,
    F: PrimeField32,
{
    const OPCODE: u32 = MUL32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;

        let mut imm: Option<Word<u8>> = None;
        let read_addr_1 = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        let b = M::read(state, clk, read_addr_1);
        let c: Word<u8> = if ops.is_imm() == 1 {
            let c = (ops.c() as u32).into();
            imm = Some(c);
            c
        } else {
            let read_addr_2 = (state.machine.cpu().fp as i32 + ops.c()) as u32;
            M::read(state, clk, read_addr_2)
        };
        let a = b * c;
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .mul_32_mut()
                .operations
                .push(Operation::Mul32(a, b, c));

            let b_u32: u32 = b.into();
            let c_u32: u32 = c.into();
            let prod: u64 = (b_u32 as u64) * (c_u32 as u64);
            let lower: u32 = prod as u32;
            let upper: u32 = (prod >> 32) as u32;

            debug_assert_eq!(Word::from(lower), a);

            state.machine.range_check_word(Word::from(lower));
            state.machine.range_check_word(Word::from(upper));

            let b_extended = Long::zero_extend_word(b);
            let c_extended = Long::zero_extend_word(c);

            let carries = get_carries(&b_extended, &c_extended);

            state.machine.range_check_mul_chip_carries(&carries);

            state.machine.push_bus_op(imm, opcode, ops);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for Mulhs32Instruction
where
    M: MachineWithMul32Chip<F> + MachineWithBytesChip<F>,
    F: PrimeField32,
{
    const OPCODE: u32 = MULHS32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;

        let mut imm: Option<Word<u8>> = None;
        let read_addr_1 = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        let b = M::read(state, clk, read_addr_1);
        let c: Word<u8> = if ops.is_imm() == 1 {
            let c = (ops.c() as u32).into();
            imm = Some(c);
            c
        } else {
            let read_addr_2 = (state.machine.cpu().fp as i32 + ops.c()) as u32;
            M::read(state, clk, read_addr_2)
        };

        let a = b.mulhs(c);
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .mul_32_mut()
                .operations
                .push(Operation::Mulhs32(a, b, c));

            let b_i32: i32 = b.into();
            let sign_b = b_i32 < 0;
            let c_i32: i32 = c.into();
            let sign_c = c_i32 < 0;
            let prod: i64 = (b_i32 as i64) * (c_i32 as i64);
            let lower: u32 = prod as i32 as u32;
            let upper: u32 = ((prod >> 32) as i32) as u32;

            let b_extended = Long::sign_extend_word(b);
            let c_extended = Long::sign_extend_word(c);

            debug_assert_eq!(Word::from(upper), a);

            state.machine.range_check_word(Word::from(lower));
            state.machine.range_check_word(Word::from(upper));

            let res = state
                .machine
                .check_byte_op(*b.index_be(0), ByteOperation::MostSignificantBit);
            debug_assert_eq!(res.len(), 1);
            let sign_b_expected = res[0];
            debug_assert_eq!(sign_b as u8, sign_b_expected);
            let res = state
                .machine
                .check_byte_op(*c.index_be(0), ByteOperation::MostSignificantBit);
            debug_assert_eq!(res.len(), 1);
            let sign_c_expected = res[0];
            debug_assert_eq!(sign_c as u8, sign_c_expected);

            let carries = get_carries(&b_extended, &c_extended);
            state.machine.range_check_mul_chip_carries(&carries);

            state.machine.push_bus_op(imm, opcode, ops);
        }

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for Mulhu32Instruction
where
    M: MachineWithMul32Chip<F> + MachineWithRangeCheckeru8<F>,
    F: PrimeField32,
{
    const OPCODE: u32 = MULHU32;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let opcode = <Self as Instruction<M, F>>::OPCODE;
        let clk = state.machine.cpu().clock;

        let mut imm: Option<Word<u8>> = None;
        let read_addr_1 = (state.machine.cpu().fp as i32 + ops.b()) as u32;
        let write_addr = (state.machine.cpu().fp as i32 + ops.a()) as u32;
        let b = M::read(state, clk, read_addr_1);
        let c: Word<u8> = if ops.is_imm() == 1 {
            let c = (ops.c() as u32).into();
            imm = Some(c);
            c
        } else {
            let read_addr_2 = (state.machine.cpu().fp as i32 + ops.c()) as u32;
            M::read(state, clk, read_addr_2)
        };

        let a = b.mulhu(c);
        M::write(state, clk, write_addr, a);

        if state.machine.log_enabled() {
            state
                .machine
                .mul_32_mut()
                .operations
                .push(Operation::Mulhu32(a, b, c));

            let b_u32: u32 = b.into();
            let c_u32: u32 = c.into();
            let prod: u64 = (b_u32 as u64) * (c_u32 as u64);
            let lower: u32 = prod as u32;
            let upper: u32 = (prod >> 32) as u32;

            debug_assert_eq!(Word::from(upper), a);

            state.machine.range_check_word(Word::from(lower));
            state.machine.range_check_word(Word::from(upper));

            let b_extended = Long::zero_extend_word(b);
            let c_extended = Long::zero_extend_word(c);

            let carries = get_carries(&b_extended, &c_extended);
            state.machine.range_check_mul_chip_carries(&carries);

            state.machine.push_bus_op(imm, opcode, ops);
        }

        state.machine.step_pc();
    }
}
