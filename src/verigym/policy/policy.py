from verigym.abstraction.abstractionmapper import AbstractionMapper
from verigym import VeriGymEnv
import gymnasium as gym


class PolicyClass:
    """
    An abstract class providing an interface to any kind of model policy.
    Given an observation in the VeriGym environment,
    return an action for the VeriGym environment based on the abstract model's policy.
    """

    def __init__(self, policy, abstraction_mapper: AbstractionMapper):
        self.policy = policy
        self.abstraction_mapper = abstraction_mapper

    def _action_from_policy(self, obs):
        """
        Get an action from the model's policy. 
        This function needs to be implemented / adapted for every 

        Parameters
        ----------
        obs : object
            an observation in the abstract model's space

        Returns
        -------
        action : object
            an action in the abstract model's space according to the policy
        """
        # This should be implemented in specific child classes
        raise NotImplementedError

    def get_action(self, obs, info=None):
        """
        Gets an observation from `env.observation_space`.
        Queries the model's policy.
        Map the action from the model's policy to `env.action_space`

        Parameters
        ----------
        obs : object
            An observation in the environment's observation space

        Returns
        -------
        action : object
            An action in the environment's action space

        """
        o = self.abstraction_mapper.original_to_abstract_state(
            obs
        )  # self._obs_to_model(obs)
        a = self._action_from_policy(o)
        action = self.abstraction_mapper.abstract_to_original_action(a)
        return action


class RandomizedPolicy(PolicyClass):
    """
    A policy that returns random actions, as sampled from the provided environment.
    Works for every class inheriting from `VeriGymEnv` (and therefore `gym.Env`).
    """

    def __init__(self, env: gym.Env):
        def policy(obs): 
            return env.action_space.sample()
        abstraction_mapper = AbstractionMapper()  # Identity mapping
        return super().__init__(policy, abstraction_mapper)

    def _action_from_policy(self, obs):
        return self.policy(obs)
