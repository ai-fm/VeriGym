from typing import TYPE_CHECKING, Optional, Any
from abc import abstractmethod

from collections import defaultdict

from numpy.typing import NDArray

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
        self,
        dataset: list[tuple],
        T_counts: defaultdict[int, defaultdict[int, defaultdict[int, int]]],
        P_tot: dict,
        R_counts: defaultdict[int, defaultdict[int, list]],
        S_init_counts: NDArray,
    ) -> "PolicyClass":
        """
        Updates the policy during the exploration phase of the abstraction learning phase.
        - If this policy does not change during abstraction learning, it just returns itself, unchanged.
        - If the policy does update, this function needs to be redefined for that policy class.

        Examples for which policies this function may be useful:
        - state-based sampling (e.g. entropy-based sampling)
        - RL policy that is updated throughout the abstraction learning process.

        Parameters
        ----------
        dataset: list[tuple]
            A dataset with observations, list of tuples (state, action, reward, next_state).
        T_counts : defaultdict[int, defaultdict[int, defaultdict[int, int]]]
            A nested dict `T_count[state][action][next_state]` returns the integer count.
        P_tot : dict[tuple[int, int], int]
            A dict storing the occurrences of state-action pairs. Useful for iterating through `T_counts` and `R_counts`, see `learn_abstraction.py` for examples.
        R_counts : defaultdict[int, defaultdict[int, list]]
            A nested dict, where `R_counts[state][action]` returns a list of encountered rewards for the given state-action pair.
        S_init_counts : NDArray
            A list, where `S_init_counts[state]` returns the number of occurences that `state` was the intial state.

        Returns
        -------
        PolicyClass
            The policy that should be further used for exploring through the abstraction learning.
        """
        return self