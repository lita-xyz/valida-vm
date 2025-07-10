use valida_machine::{Chip, Interaction, InteractionType, Machine, StarkConfig};

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

    fn all_interactions(&self, machine: &M) -> Vec<(Interaction<SC::Val>, InteractionType)> {
        let interactions = <Self as Chip<M, SC>>::ephemeral_interactions(self, machine)
            .into_iter()
            .map(|(interaction, interaction_type)| (interaction, interaction_type));
        interactions
            .chain(
                self.persistent_sends(machine)
                    .into_iter()
                    .map(|i| (i, InteractionType::PersistentSend)),
            )
            .chain(
                self.persistent_receives(machine)
                    .into_iter()
                    .map(|i| (i, InteractionType::PersistentReceive)),
            )
            .collect()
    }
}
