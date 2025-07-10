use crate::columns::{NUM_STATIC_DATA_LOOKUP_COLS, NUM_STATIC_DATA_PRIVATE_COLS};
use crate::{StaticDataChip, StaticDataChipType};
use core::fmt::Debug;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

impl<F: PrimeField32> BaseAir<F> for StaticDataChip {
    fn width(&self) -> usize {
        NUM_STATIC_DATA_PRIVATE_COLS
    }

    fn preprocessed_width(&self) -> usize {
        match self.chip_type() {
            StaticDataChipType::Preprocessed => NUM_STATIC_DATA_LOOKUP_COLS,
            _ => 0,
        }
    }

    fn public_width(&self) -> usize {
        match self.chip_type() {
            StaticDataChipType::Public => NUM_STATIC_DATA_LOOKUP_COLS,
            _ => 0,
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        panic!("Use get_preprocessed_trace() method defined for the Chip intsead");
    }
}

impl<AB> Air<AB> for StaticDataChip
where
    AB: AirBuilder,
    AB::F: PrimeField32 + Debug,
    AB::Expr: Debug,
    AB::Var: Debug,
{
    // There are no constraints which need to be enforced here: the chip consists entirely of public/preprocessed values
    fn eval(&self, _builder: &mut AB) {}
}
