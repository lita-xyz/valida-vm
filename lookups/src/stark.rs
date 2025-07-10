use p3_air::{Air, BaseAir};
use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;
use valida_machine::ValidaAirBuilder;

use crate::{LookupChip, MultiLookupTable};

impl<L, F> BaseAir<F> for LookupChip<L, F>
where
    F: PrimeField + Sync,
    L: MultiLookupTable<F> + Sync,
{
    fn width(&self) -> usize {
        self.table.num_private_columns() + self.table.num_receives()
    }

    fn public_width(&self) -> usize {
        self.table.num_public_columns()
    }

    fn preprocessed_width(&self) -> usize {
        self.table.num_preprocessed_columns()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        panic!("Use get_preprocessed_trace() method defined for the Chip intsead");
    }
}

// in a pure lookup, there are no constraints to evaluate on the lookup table or the
// column of mutliplicities.
impl<L, AB> Air<AB> for LookupChip<L, AB::F>
where
    L: MultiLookupTable<AB::F> + Sync,
    AB: ValidaAirBuilder,
    AB::F: PrimeField,
{
    fn eval(&self, _builder: &mut AB) {}
}
