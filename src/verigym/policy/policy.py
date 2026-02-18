from verigym.abstraction.abstractionmapper import AbstractionMapper
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
    """

    def __init__(self, env: gym.Env):
        self.policy = lambda obs: env.action_space.sample()
        self.abstraction_mapper = AbstractionMapper()  # Identity mapping

    def _action_from_policy(self, obs):
        return self.policy(obs)
