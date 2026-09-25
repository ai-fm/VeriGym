from verigym.policy.policy import PolicyClass
from verigym.abstraction.abstractionmapper import AbstractionMapper
import stormpy
from verigym.frameworks.stormpy.stormpy_utils import _unwrap_scheduler

class StormpyPolicy(PolicyClass):
    def __init__(self, policy, abstraction_mapper: AbstractionMapper, mdp: stormpy.storage.SparseMdp):
        unwrapped_policy = _unwrap_scheduler(mdp, policy)

        super().__init__(policy=unwrapped_policy, abstraction_mapper=abstraction_mapper)


    def _action_from_policy(self, obs):
        action_index = self.policy[obs]
        return action_index
    
    def get_action(self, obs, info=None):
        o = self.abstraction_mapper.original_to_abstract_state_enum(
            obs
        ) 
        a = self._action_from_policy(o)
        action = self.abstraction_mapper.abstract_to_original_action(a)
        return action
