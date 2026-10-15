from typing import Optional
from stable_baselines3.common import type_aliases
from stable_baselines3.common.utils import set_random_seed
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
        deterministic: bool = False,
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
        deterministic: bool, optional
            Whether to take the most likely action instead of sampling from the policy's action distribution. Defaults to False.

        Returns
        -------
        SB3Policy
            A `verigym` compatible policy.
        """
        
        if abstraction_mapper is None:
            abstraction_mapper = AbstractionMapper.initialize_identity_mapper(env.observation_space, env.action_space)

        self.deterministic = deterministic
        return super().__init__(policy, abstraction_mapper)

    def seed(self, seed=None):
        """
        Seeds the generators SB3 samples actions from (`random`, `numpy` and `torch` are global). None leaves them as they are.
        """
        if seed is None:
            return
        if hasattr(self.policy, "set_random_seed"):
            self.policy.set_random_seed(seed)  # an SB3 algorithm: also seeds its action space
        else:
            set_random_seed(seed)

    def _action_from_policy(self, obs):
        action, _ = self.policy.predict(obs, deterministic=self.deterministic)
        return action
