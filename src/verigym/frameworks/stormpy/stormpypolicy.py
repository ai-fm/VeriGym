from verigym.policy.policy import PolicyClass
from verigym.abstraction.abstractionmapper import AbstractionMapper
import stormpy
from verigym.frameworks.stormpy.stormpy_utils import _unwrap_scheduler

class StormpyPolicy(PolicyClass):
    """A native MDP policy class that picks actions according to an explicit mapping, such as imported from stormpy."""
    def __init__(self, policy, abstraction_mapper: AbstractionMapper, mdp: stormpy.storage.SparseMdp,
                 label_to_action: dict | None = None):
        """Initialize using a stormpy policy and mdp.

        Parameters
        ----------
        policy : stormpy.storage.Scheduler
        abstraction_mapper : AbstractionMapper
        mdp : stormpy.storage.SparseMdp
        label_to_action : dict | None
            Maps the choice labels of `mdp` to the actions of the env, e.g., `env.formatter.label_to_action` for
            envs loaded from a prism file. If None (default), it is derived from the choice labels, which are the
            action indices or the action names (see `_unwrap_scheduler`).
        """
        unwrapped_policy = _unwrap_scheduler(mdp, policy, label_to_action)

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
