from .policy import PolicyClass
from verigym.environments.verigymenv import VeriGymEnv
from verigym.abstraction.abstractionmapper import AbstractionMapper

class RandomizedPolicy(PolicyClass):
    """
    A policy that returns random actions, as sampled from the provided environment.
    Works for every class inheriting from `VeriGymEnv` (and therefore `gym.Env`).
    """

    def __init__(self, env:VeriGymEnv):
        def policy(obs):
            return env.action_space.sample()

        abstraction_mapper = AbstractionMapper.initialize_identity_mapper(env.observation_space, env.action_space)  # Identity mapping
        return super().__init__(policy, abstraction_mapper)

    def _action_from_policy(self, obs):
        return self.policy(obs)