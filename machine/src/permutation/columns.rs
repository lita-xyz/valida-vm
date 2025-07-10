use core::cmp::min;

pub const MAX_PERMUTATION_CONSTRAINT_DEGREE: usize = 3;

#[derive(Debug)]
pub struct PermutationColsViewMut<'a, T> {
    pub ephemeral_cols: &'a mut [T],
    pub persistent_cols: &'a mut [T],
}

impl<'a, T> PermutationColsViewMut<'a, T> {
    pub fn as_view_mut<'b, const D: usize>(
        num_ephemeral: usize,
        num_persistent_sends: usize,
        num_persistent_receives: usize,
        row: &'b mut [T],
    ) -> Self
    where
        'b: 'a,
    {
        let (num_ephemeral_cols, num_persistent_cols) = sizes_from_interactions::<D>(
            num_ephemeral,
            num_persistent_sends,
            num_persistent_receives,
        );
        let (ephemeral_cols, persistent_cols) = row.split_at_mut(num_ephemeral_cols);
        debug_assert_eq!(ephemeral_cols.len(), num_ephemeral_cols);
        debug_assert_eq!(persistent_cols.len(), num_persistent_cols);

        Self {
            ephemeral_cols,
            persistent_cols,
        }
    }
}
#[derive(Debug)]
pub struct PermutationColsView<'a, T> {
    pub ephemeral_cols: &'a [T],
    pub persistent_cols: &'a [T],
}

impl<'a, T> PermutationColsView<'a, T> {
    pub fn as_view<'b, const D: usize>(
        num_ephemeral: usize,
        num_persistent_sends: usize,
        num_persistent_receives: usize,
        row: &'b [T],
    ) -> Self
    where
        'b: 'a,
    {
        let (num_ephemeral_cols, num_persistent_cols) = sizes_from_interactions::<D>(
            num_ephemeral,
            num_persistent_sends,
            num_persistent_receives,
        );
        let (ephemeral_cols, persistent_cols) = row.split_at(num_ephemeral_cols);
        debug_assert_eq!(ephemeral_cols.len(), num_ephemeral_cols);
        debug_assert_eq!(persistent_cols.len(), num_persistent_cols);

        Self {
            ephemeral_cols,
            persistent_cols,
        }
    }
}

pub fn sizes_from_interactions<const D: usize>(
    num_ephemeral: usize,
    num_persistent_sends: usize,
    num_persistent_receives: usize,
) -> (usize, usize) {
    let ephemeral_cols = if num_ephemeral == 0 {
        0
    } else {
        // each column equals the prior column plus the sum of at most D - 1 terms, a degree D constraint
        let chunk_size = D - 1;
        // the degree of the constraint for the first column is one higher than the constraints for the other columns,
        // due to the need to condition on whether the row is the initial row or not.
        let first_chunk_size = min(num_ephemeral, chunk_size - 1);
        let remaining_terms = num_ephemeral - first_chunk_size;
        1 + remaining_terms.div_ceil(chunk_size)
    };

    let persistent_cols = if num_persistent_sends + num_persistent_receives == 0 {
        0
    } else {
        // each column equals the prior column plus the sum of at most D - 1 terms, a degree D constraint
        let chunk_size = D - 1;
        // the degree of the constraint for the first column is one higher than the constraints for the other columns,
        // due to the need to condition on whether the row is the initial row or not.
        let first_chunk_size = min(
            num_persistent_sends + num_persistent_receives,
            chunk_size - 1,
        );
        let remaining_terms = num_persistent_sends + num_persistent_receives - first_chunk_size;
        1 + remaining_terms.div_ceil(chunk_size)
    };
    (ephemeral_cols, persistent_cols)
}
