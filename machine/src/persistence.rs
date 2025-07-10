use crate::{
    columns::{sizes_from_interactions, MAX_PERMUTATION_CONSTRAINT_DEGREE},
    Chip, Interaction, Machine, StarkConfig,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PersistentInteractionType {
    PersistentSend,
    PersistentReceive,
}

/// In the context of VM executions proved across multiple segments, we need
/// sends and receives across chip traces in different segments for e.g.
/// reading memory cells written in a prior segment
pub trait ChipWithPersistence<M, SC>: Chip<M, SC>
where
    SC: StarkConfig,
    M: Machine<SC::Val>,
{
    fn persistent_sends(&self, _machine: &M) -> Vec<Interaction<SC::Val>> {
        vec![]
    }

    fn persistent_receives(&self, _machine: &M) -> Vec<Interaction<SC::Val>> {
        vec![]
    }

    fn persistent_interactions(
        &self,
        machine: &M,
    ) -> Vec<(Interaction<SC::Val>, PersistentInteractionType)> {
        self.persistent_sends(machine)
            .into_iter()
            .map(|i| (i, PersistentInteractionType::PersistentSend))
            .chain(
                self.persistent_receives(machine)
                    .into_iter()
                    .map(|i| (i, PersistentInteractionType::PersistentReceive)),
            )
            .collect()
    }
    fn permutation_width(&self, machine: &M) -> usize {
        let (logup_width, product_width) =
            sizes_from_interactions::<MAX_PERMUTATION_CONSTRAINT_DEGREE>(
                self.ephemeral_interactions(machine).len(),
                self.persistent_sends(machine).len(),
                self.persistent_receives(machine).len(),
            );
        logup_width + product_width
    }
}
