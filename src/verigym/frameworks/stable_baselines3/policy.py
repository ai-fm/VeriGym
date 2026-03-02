from stable_baselines3.common import type_aliases

from verigym.policy.policy import PolicyClass
from verigym.abstraction.abstractionmapper import AbstractionMapper

SB3_original_Policy = type_aliases.PolicyPredictor

__all__ = ["SB3Policy"]


class SB3Policy(PolicyClass):
    def __init__(self, policy: SB3_original_Policy, abstraction_mapper=None):
        """
        Initializes a `verigym` compatible from a policy from the stable baselines 3 (`SB3`) framework.

        Implements an identity `AbstractionMapper`, meaning there is no abstraction to map from/to.

        Parameters
        ----------
        policy : sb3_Policy
            The `SB3` policy. This can be any object that implements a `predict` method such as `SB3`'s `BaseAlgorithm` or `BasePolicy` and child classes.

        Returns
        -------
        SB3Policy
            `verigym` compatible policy.
        """
        # default to identity abstraction mapper
        if abstraction_mapper is None:
            abstraction_mapper = AbstractionMapper()

        return super().__init__(policy, abstraction_mapper)

    def _action_from_policy(self, obs):
        action, _ = self.policy.predict(obs)
        return action
