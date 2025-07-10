extern crate alloc;

use alloc::vec::Vec;
use valida_cpu::MachineWithCpuChip;
use valida_machine::{instructions, is_mul_4, Instruction, Operands, RunningMachine, Word};
use valida_opcodes::{COMBSECP256K1, MULSSECP256K1, SINVSECP256K1, SMULSECP256K1};

use p3_field::PrimeField;

use halo2curves::serde::endian::EndianRepr;
use halo2curves::{secp256k1::Secp256k1, secp256k1::Secp256k1Affine};
use halo2curves::{CurveAffine, CurveExt};

use halo2curves::ff::Field as Halo2Field;
use halo2curves::secp256k1::Fq as Scalar;

fn point_from_bytes<C: CurveAffine>(x_bytes: &[u8; 32], y_bytes: &[u8; 32]) -> C
where
    C::Base: EndianRepr,
{
    let x_b = C::Base::from_bytes(x_bytes);
    let x = x_b.unwrap();
    let y = C::Base::from_bytes(y_bytes).unwrap();
    C::from_xy(x, y).unwrap()
}

fn point_to_bytes<C: CurveExt>(point: &C) -> ([u8; 32], [u8; 32])
where
    <<C as CurveExt>::AffineExt as CurveAffine>::Base: EndianRepr,
{
    let p_aff: C::AffineExt = (*point).into();
    (
        p_aff
            .coordinates()
            .unwrap()
            .x()
            .to_bytes()
            .try_into()
            .unwrap(),
        p_aff
            .coordinates()
            .unwrap()
            .y()
            .to_bytes()
            .try_into()
            .unwrap(),
    )
}

/// Flatten the input array of 8 `Word<u8>` into a flat array of 32 bytes.
fn to_bytes(x: [Word<u8>; 8]) -> [u8; 32] {
    x.map(|w| w.into_iter_le().collect::<Vec<u8>>())
        .concat()
        .try_into()
        .unwrap()
}

fn read_secp256k1_point<M, F>(
    state: &mut RunningMachine<'_, F, M>,
    base_address: u32,
    clk: u32,
) -> Secp256k1Affine
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    let mut ba = base_address;
    let x_bytes = to_bytes(std::array::from_fn(|_| {
        let r = M::read(state, clk, ba);
        ba = ba + 4;
        r
    }));
    let y_bytes = to_bytes(std::array::from_fn(|_| {
        let r = M::read(state, clk, ba);
        ba = ba + 4;
        r
    }));

    point_from_bytes::<Secp256k1Affine>(&x_bytes, &y_bytes)
}

fn write_secp256k1_point<M, F>(
    state: &mut RunningMachine<'_, F, M>,
    base_address: u32,
    point: &Secp256k1Affine,
    clk: u32,
) where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    let (x, y) = point_to_bytes::<Secp256k1>(&point.into());
    let res: [u8; 64] = [x.clone(), y.clone()].concat().try_into().unwrap();

    for i in (0..64).step_by(4) {
        let w = Word::from_components_le([res[i], res[i + 1], res[i + 2], res[i + 3]]);
        let addr = base_address + i as u32;
        M::write(state, clk, addr, w);
    }
}

fn read_secp256k1_scalar<M, F>(
    state: &mut RunningMachine<'_, F, M>,
    base_address: u32,
    clk: u32,
) -> Scalar
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    let mut ba = base_address;
    let value = to_bytes(std::array::from_fn(|_| {
        let r = M::read(state, clk, ba);
        ba = ba + 4;
        r
    }));

    Scalar::from_bytes(&value).unwrap()
}

fn write_secp256k1_scalar<M, F>(
    state: &mut RunningMachine<'_, F, M>,
    base_address: u32,
    scalar: &Scalar,
    clk: u32,
) where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    let res = scalar.to_bytes();

    for i in (0..32).step_by(4) {
        let w = Word::from_components_le([res[i], res[i + 1], res[i + 2], res[i + 3]]);
        let addr = base_address + i as u32;
        M::write(state, clk, addr, w);
    }
}

instructions!(
    CombSecp256k1Instruction,
    MulsSecp256k1Instruction,
    SinvSecp256k1Instruction,
    SmulSecp256k1Instruction
);

impl<M, F> Instruction<M, F> for CombSecp256k1Instruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = COMBSECP256K1;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let clk = state.machine.cpu().clock;
        let fp = state.machine.cpu().fp;

        let base_address_loc_1: u32 = (fp as i32 + ops.a()) as u32;
        assert!(
            is_mul_4(base_address_loc_1),
            "COMBSECP256K1: base_address_loc_1 not a multiple of 4!"
        );
        let base_address_1: u32 = M::read(state, clk, base_address_loc_1).into();
        assert!(
            is_mul_4(base_address_1),
            "COMBSECP256K1: base_address_1 not a multiple of 4!"
        );

        let base_address_loc_2: u32 = (fp as i32 + ops.b()) as u32;
        assert!(
            is_mul_4(base_address_loc_2),
            "COMBSECP256K1: base_address_loc_2 not a multiple of 4!"
        );
        let base_address_2: u32 = M::read(state, clk, base_address_loc_2).into();
        assert!(
            is_mul_4(base_address_2),
            "COMBSECP256K1: base_address_2 not a multiple of 4!"
        );

        let point_1 = read_secp256k1_point(state, base_address_1, clk);
        let scalar_1 = read_secp256k1_scalar(state, base_address_1 + 64, clk);

        let point_2 = read_secp256k1_point(state, base_address_2, clk);
        let scalar_2 = read_secp256k1_scalar(state, base_address_2 + 64, clk);

        let res = Secp256k1::from(point_1) * scalar_1 + Secp256k1::from(point_2) * scalar_2;
        let res: Secp256k1Affine = Secp256k1Affine::from(res);

        write_secp256k1_point(state, base_address_2, &res, clk);

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for SmulSecp256k1Instruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = SMULSECP256K1;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let clk = state.machine.cpu().clock;
        let fp = state.machine.cpu().fp;

        let base_address_loc_1: u32 = (fp as i32 + ops.a()) as u32;
        assert!(
            is_mul_4(base_address_loc_1),
            "COMBSECP256K1: base_address_loc_1 not a multiple of 4!"
        );
        let base_address_1: u32 = M::read(state, clk, base_address_loc_1).into();
        assert!(
            is_mul_4(base_address_1),
            "COMBSECP256K1: base_address_1 not a multiple of 4!"
        );

        let base_address_loc_2: u32 = (fp as i32 + ops.b()) as u32;
        assert!(
            is_mul_4(base_address_loc_2),
            "COMBSECP256K1: base_address_loc_2 not a multiple of 4!"
        );
        let base_address_2: u32 = M::read(state, clk, base_address_loc_2).into();
        assert!(
            is_mul_4(base_address_2),
            "COMBSECP256K1: base_address_2 not a multiple of 4!"
        );

        let scalar = read_secp256k1_scalar(state, base_address_1, clk);
        let point = read_secp256k1_point(state, base_address_2, clk);

        let res = Secp256k1::from(point) * scalar;
        let res: Secp256k1Affine = res.into();

        write_secp256k1_point(state, base_address_2, &res, clk);

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for SinvSecp256k1Instruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = SINVSECP256K1;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let clk = state.machine.cpu().clock;

        let fp = state.machine.cpu().fp;

        let base_address_loc: u32 = (fp as i32 + ops.b()) as u32;
        assert!(
            is_mul_4(base_address_loc),
            "COMBSECP256K1: base_address_loc not a multiple of 4!"
        );
        let base_address: u32 = M::read(state, clk, base_address_loc).into();
        assert!(
            is_mul_4(base_address),
            "COMBSECP256K1: base_address not a multiple of 4!"
        );

        let scalar: Scalar = read_secp256k1_scalar(state, base_address, clk);

        let res = scalar.invert().unwrap();

        write_secp256k1_scalar(state, base_address, &res, clk);

        state.machine.step_pc();
    }
}

impl<M, F> Instruction<M, F> for MulsSecp256k1Instruction
where
    M: MachineWithCpuChip<F>,
    F: PrimeField,
{
    const OPCODE: u32 = MULSSECP256K1;

    fn execute(state: &mut RunningMachine<'_, F, M>, ops: Operands<i32>) {
        let clk = state.machine.cpu().clock;

        let fp = state.machine.cpu().fp;

        let base_address_loc_1: u32 = (fp as i32 + ops.a()) as u32;
        assert!(
            is_mul_4(base_address_loc_1),
            "COMBSECP256K1: base_address_loc_1 not a multiple of 4!"
        );
        let base_address_1: u32 = M::read(state, clk, base_address_loc_1).into();
        assert!(
            is_mul_4(base_address_1),
            "COMBSECP256K1: base_address_1 not a multiple of 4!"
        );

        let base_address_loc_2: u32 = (fp as i32 + ops.b()) as u32;
        assert!(
            is_mul_4(base_address_loc_2),
            "COMBSECP256K1: base_address_loc_2 not a multiple of 4!"
        );
        let base_address_2: u32 = M::read(state, clk, base_address_loc_2).into();
        assert!(
            is_mul_4(base_address_2),
            "COMBSECP256K1: base_address_2 not a multiple of 4!"
        );

        let scalar_1 = read_secp256k1_scalar(state, base_address_1, clk);
        let scalar_2 = read_secp256k1_scalar(state, base_address_2, clk);

        let res = scalar_1 * scalar_2;

        write_secp256k1_scalar(state, base_address_2, &res, clk);

        state.machine.step_pc();
    }
}
