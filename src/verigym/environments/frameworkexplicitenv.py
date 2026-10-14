import gymnasium as gym
import numpy as np
from typing import Any
from gymnasium import Env, Wrapper
from collections.abc import Callable, Sequence

import stormpy

from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.environments.formatter import ExplicitFormatter
from verigym.environments.interval_explicitenv import IntervalEnvMixin
from verigym.environments.reward_func import RewardFunction
from verigym.environments.transition_func import TransitionFunction
from verigym.frameworks.stormpy.formatter import StormpyFormatter


class FrameworkExplicitEnv(BaseExplicitEnv):
    formatter: ExplicitFormatter

    def __init__(self, model: Any, formatter: ExplicitFormatter, flat: bool = True, render_mode: str | None = None):
        """
        An explicit env of a model of a framework (e.g., a stormpy MDP), converted with the framework's formatter.

        Parameters
        ----------
        model : Any
            The model, e.g., a `stormpy.storage.SparseMdp`.
        formatter : ExplicitFormatter
            The formatter of the model, e.g., a `StormpyFormatter`. Also provides the action names, labels and
            valuations of the env.
        flat : bool, default=True
            Whether the env uses flat (integer) states, or factored states from the state valuations.
        render_mode : str | None, default=None
            Environment render mode. Currently can only be None.
        """
        self.formatter = formatter
        transition_function, reward_function = self._convert_model(model)

        super().__init__(
            nr_states=formatter.nr_states,
            nr_actions=formatter.nr_actions,
            initial_state_distr=formatter.initial_states,
            transition_function=transition_function,
            reward_function=reward_function,
            nr_rewards=formatter.n_rewards,
            observation_space=self._init_observation_space(flat),
            render_mode=render_mode,
        )

        self.state = self.sample_initial_state()

    def _convert_model(self, model) -> tuple[TransitionFunction, RewardFunction]:
        """
        Converts the model to the transition and reward function of the env, with the formatter.
        """
        transition_function = self.formatter._convert_transition_matrix(model)
        reward_function = self.formatter._convert_reward_matrix(model)
        return transition_function, reward_function

    def _init_observation_space(self, flat: bool) -> gym.spaces.Space:
        """
        `Discrete` for flat states, or `MultiDiscrete` for factored states (see `_init_md_observation_space`).
        """
        if flat:
            return gym.spaces.Discrete(self.formatter.nr_states)
        if not self.formatter.has_state_valuations:
            raise ValueError("Requested featured state representation, but state valuations are not available.")
        return self._init_md_observation_space()

    def _init_md_observation_space(self):
        """
        Uses the info on state valuations from `self.formatter.state_to_values` to initialise a `MultiDiscrete` observation space.
        Needs to ensure that the states from obs_space are matched to the correct transitions/formatter state indices.
        """
        # 0 is also a valid valuation
        nvec = [v+1 for v in self.formatter.max_valuations]

        obs_space = gym.spaces.MultiDiscrete(np.array(nvec))

        return obs_space

    def decode(self, state_idx):
        if isinstance(self.observation_space, gym.spaces.Discrete):
            return state_idx
        else:
            return self.formatter.get_full_state_from_idx(state_idx)

    def encode(self, state_values):
        if isinstance(self.observation_space, gym.spaces.Discrete):
            return state_values
        else:
            return self.formatter.values_to_state[state_values]

    @classmethod
    def from_stormpy(cls, mdp, flat: bool = True, render_mode: str | None = None):
        """
        Builds an env from a stormpy MDP, or an `IntervalFrameworkExplicitEnv` from a stormpy interval MDP
        (`stormpy.storage.SparseIntervalMdp`).
        """
        if isinstance(mdp, stormpy.storage.SparseIntervalMdp) and not issubclass(cls, IntervalFrameworkExplicitEnv):
            cls = IntervalFrameworkExplicitEnv
        return cls(model=mdp, formatter=StormpyFormatter(mdp), flat=flat, render_mode=render_mode)
    
    @classmethod
    def vec_from_stormpy(cls, mdp, render_mode: str | None = None,
                         num_envs: int = 1,
                         vectorization_mode: str = "sync",
                         vector_kwargs: dict[str, Any] | None = None,
                         wrappers: Sequence[Callable[[Env], Wrapper]] | None = None,
                         wrapper_kwargs: list | None = None
                         ):
        """
        Builds vectorized envs of this class from a stormpy MDP (see `from_stormpy`).
        Note that only the "sync" vectorization mode works here, since we cannot pickle and serialize stormpy MDPs, which are C++ objects.
        We could enable "async" by not storing the stormpy MDP.

        Parameters
        ----------
        mdp : stormpy.storage.SparseMdp
            The underlying MDP.
        num_envs : int
            How many envs to contain in the vectorized env.
        render_mode : str
            Environment render mode. Currently can only be None.
        vectorization_mode : str, default="sync"
            The vectorization mode. Here, we can only use "sync"
        vector_kwargs : dict
            Further arguments to apply to vectorization.
        wrappers : Sequence
            List of wrappers to be applied to the environments.

        Returns
        -------
        env : gym.SyncVectorEnv
            The vectorized envs, built with `cls.from_stormpy`.
        """

        if vectorization_mode == "async":
            raise ValueError("Cannot use async vectorization with built stormpy MDP (C++ object that cannot be serialized).")
        return cls.make_vec(num_envs, vectorization_mode, vector_kwargs, wrappers, wrapper_kwargs,
                            cls.from_stormpy,
                            mdp=mdp, render_mode=render_mode)

    def _get_info(self):
        """
        Accumulate additional information about the state/environment.

        Returns
        -------
        info : dict
        """
        s_idx = self.encode(self.state)
        info = super()._get_info()
        if self.formatter.has_state_valuations:
            info["state_valuations"] = self.formatter.state_to_values[s_idx]
        if self.formatter.has_state_labels:
            info["state_labels"] = self.formatter.state_to_labels[s_idx]
        if self.formatter.has_reward_labels:
            info["reward_labels"] = list(self.formatter.reward_labels.keys())
        if self.formatter.has_action_labels:
            info["action_labels"] = self.formatter.action_to_label
        return info


class IntervalFrameworkExplicitEnv(IntervalEnvMixin, FrameworkExplicitEnv):
    """
    A `FrameworkExplicitEnv` of an interval MDP (IMDP), e.g., loaded with stormpy: its formatter converts the
    transition probabilities and rewards to intervals. Apart from that, it is the same as for MDPs (action names,
    labels, valuations, factored states). 
    
    Depends on `IntervalEnvMixin` for shared functionality between this class and `IntervalExplicitEnv`.
    See `IntervalEnvMixin` for how the intervals and their point estimates are used; `step` reports the intervals of the transition in its info, as `IntervalExplicitEnv` does.
    """

    def _convert_model(self, model) -> tuple[TransitionFunction, RewardFunction]:
        # The formatter converts the interval model to interval functions: keep them as the intervals, and use their
        # point estimates as the transition and reward function
        return self._point_estimates(*super()._convert_model(model))
