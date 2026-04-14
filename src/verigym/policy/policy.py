from typing import TYPE_CHECKING, Optional, Any
from abc import abstractmethod

from ..abstraction.abstractionmapper import AbstractionMapper
# import gymnasium as gym

if TYPE_CHECKING:
    from ..environments.verigymenv import VeriGymEnv


class PolicyClass:
    """
    An abstract class providing an interface to any kind of model policy.
    Given an observation in the VeriGym environment,
    return an action for the VeriGym environment based on the abstract model's policy.
    """

    def __init__(
        self, policy: Any, abstraction_mapper: Optional[AbstractionMapper] = None
    ):
        """
        Initializes a policy.

        Parameters
        ----------
        policy : Any
            The object that contains the policy (e.g. computed by an external framework). Can be used in the `self._action_from_policy
        abstraction_mapper : AbstractionMapper, optional
            An optional mapping that translates actions from e.g. abstracted environment to original environment. By default None, then an identity mapping is used, not changing the outputted action.
        """
        self.policy = policy

        if abstraction_mapper is None:
            abstraction_mapper = AbstractionMapper()

        self.abstraction_mapper = abstraction_mapper

    @abstractmethod
    def _action_from_policy(self, obs):
        """
        Get an action from the model's policy.

        Parameters
        ----------
        obs : object
            an observation

        Returns
        -------
        action : object
            an action according to the policy
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

    def update_for_abstraction_refinement(
        self, T_counts, P_tot, R_counts, S_init_counts
    ):
        """
        TODO

        Parameters
        ----------
        T_counts : _type_
            _description_
        P_tot : _type_
            _description_
        R_counts : _type_
            _description_
        S_init_counts : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return self


class RandomizedPolicy(PolicyClass):
    """
    A policy that returns random actions, as sampled from the provided environment.
    Works for every class inheriting from `VeriGymEnv` (and therefore `gym.Env`).
    """

    def __init__(self, env: "VeriGymEnv"):
        def policy(obs):
            return env.action_space.sample()

        abstraction_mapper = AbstractionMapper()  # Identity mapping
        return super().__init__(policy, abstraction_mapper)

    def _action_from_policy(self, obs):
        return self.policy(obs)
