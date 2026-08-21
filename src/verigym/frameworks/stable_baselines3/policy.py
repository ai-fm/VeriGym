from typing import Optional
from stable_baselines3.common import type_aliases
import gymnasium as gym

from verigym.policy.policy import PolicyClass
from verigym.abstraction.abstractionmapper import AbstractionMapper

SB3_original_Policy = type_aliases.PolicyPredictor

__all__ = ["SB3Policy"]


class SB3Policy(PolicyClass):
    def __init__(
        self,
        policy: SB3_original_Policy,
        env: gym.Env,
        abstraction_mapper: Optional[AbstractionMapper] = None,
    ):
        """
        Initializes a `verigym` compatible policy from a policy by the stable baselines 3 (SB3) framework.

        Defaults to an identity `AbstractionMapper`, meaning the policy's actions remain unchanged (as there is no abstraction to map from/to).

        Parameters
        ----------
        policy : SB3_original_Policy
            The `SB3` policy. This can be any object that implements a `predict` method such as the SB3 framework's `BaseAlgorithm` or `BasePolicy` and child classes.
        abstraction_mapper: AbstractionMapper, optional
            The abstraction mapper may map the action to another action space (e.g. from abstracted to original environment). Defaults to None, meaning the action remains unchanged.

        Returns
        -------
        SB3Policy
            A `verigym` compatible policy.
        """
        
        if abstraction_mapper is None:
            abstraction_mapper = AbstractionMapper.initialize_identity_mapper(env)
        
        return super().__init__(policy, abstraction_mapper)

    def _action_from_policy(self, obs):
        action, _ = self.policy.predict(obs)
        return action
