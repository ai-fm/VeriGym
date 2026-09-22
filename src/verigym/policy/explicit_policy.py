
from typing import Optional
from verigym.abstraction.abstractionmapper import AbstractionMapper
from verigym.policy.policy import PolicyClass

class ExplicitDeterministicPolicy(PolicyClass):
    """
    A native MDP policy class that plays deterministically according to an explicit lookup table.
    """
    def __init__(
        self, lookup_table:list , abstraction_mapper: Optional[AbstractionMapper] = None
    ):
        return super().__init__(self, lookup_table, abstraction_mapper)

    def _action_from_policy(self, obs):
        return self.policy[obs]