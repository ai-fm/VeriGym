from verigym.policy.policy import PolicyClass
from verigym.abstraction.abstractionmapper import AbstractionMapper


class StormpyPolicy(PolicyClass):
    def __init__(self, policy, abstraction_mapper: AbstractionMapper):
        super().__init__(policy=policy, abstraction_mapper=abstraction_mapper)

    def _action_from_policy(self, obs):
        choice = self.policy.get_choice(obs)  # distribution over actions
        action_index = choice.get_deterministic_choice()
        # action = state.actions[action_index]
        return action_index
