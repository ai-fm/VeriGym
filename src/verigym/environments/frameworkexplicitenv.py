import gymnasium as gym
import numpy as np
from typing import Optional, Any
from gymnasium import Env, Wrapper
from collections.abc import Callable, Sequence

from verigym.environments.explicitenv import BaseExplicitEnv
from verigym.frameworks.stormpy.formatter import StormpyFormatter


class FrameworkExplicitEnv(BaseExplicitEnv):
    def __init__(self, model, formatter, flat: bool = True, render_mode: str | None = None):
        model: Any
        formatter: Any

        super().__init__(render_mode)
        self.model = model
        self.formatter = formatter

        self.nr_states = self.formatter.nr_states
        self.nr_actions = self.formatter.nr_actions

        self.transition_function = self.formatter.transition_function
        self.reward_function = self.formatter.reward_function

        if flat:
            self.observation_space = gym.spaces.Discrete(self.nr_states)
        else: 
            if not self.formatter.has_state_valuations:
                raise ValueError("Requested featured state representation, but state valuations are not available.")
            self.observation_space = self._init_md_observation_space()
        
        self.action_space = gym.spaces.Discrete(self.nr_actions)

        self.state = self.sample_initial_state()

        # Which actions are available in a state?
        self.action_mask = self.formatter.action_mask

    def _init_md_observation_space(self):
        """
        Uses the info on state valuations from `self.formatter.state_to_values` to initialise a `MultiDiscrete` observation space.
        Needs to ensure that the states from obs_space are matched to the correct transitions/formatter state indices.
        """
        # 0 is also a valid valuation
        nvec = [v+1 for v in self.formatter.max_valuations]

        obs_space = gym.spaces.MultiDiscrete(np.array(nvec))

        return obs_space

    def sample_initial_state(self):
        return self.decode(self.formatter.sample_initial_state())
    
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
        instance = cls.__new__(cls)

        formatter = StormpyFormatter(mdp)

        instance.__init__(model=mdp, formatter=formatter, flat=flat, render_mode=render_mode)

        return instance
    
    @classmethod
    def vec_from_stormpy(cls, mdp, render_mode: str | None = None,
                         num_envs: int = 1,
                         vectorization_mode: str | None = None,
                         vector_kwargs: dict[str, Any] | None = None,
                         wrappers: Sequence[Callable[[Env], Wrapper]] | None = None,
                         wrapper_kwargs: list | None = None
                         ):
        """
        Builds vectorized FrameworkExplicitEnvs from a stormpy MDP.
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
        vectorization_mode : str
            The vectorization mode. Here, we can only use "sync"
        vector_kwargs : dict
            Further arguments to apply to vectorization.
        wrappers : Sequence
            List of wrappers to be applied to the environments.

        Returns
        -------
        env : gym.SyncVectorEnv(FrameworkExplicitEnv)
            The vectorized FrameworkExplicitEnvs.
        """

        if vectorization_mode == "async":
            raise ValueError("Cannot use async vectorization with built stormpy MDP (C++ object that cannot be serialized).")
        return FrameworkExplicitEnv.make_vec(num_envs, vectorization_mode, vector_kwargs, wrappers, wrapper_kwargs,
                                             FrameworkExplicitEnv.from_stormpy,
                                             mdp=mdp, render_mode=render_mode)


    @classmethod
    def from_julia(cls, mdp, flat: bool = True, render_mode: str | None = None):
        instance = cls.__new__(cls)

        # TODO: implement this

        return instance

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset to an initial state
        """
        super().reset(seed=seed, options=options)

        self.state = self.formatter.sample_initial_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Take a step in the environment from the current state.

        Parameters
        ----------
        action : int
            Chosen action from self.action_space

        Returns :
            observation : dict if self.formatter.has_state_valuations else int
            reward : list(int)
            terminated : bool
            truncated : bool
            info : dict
        """
        s_idx = self.encode(self.state)
        if self.action_mask[s_idx][action] > 0:
            reward = self.reward_function[s_idx][action]
            self.state = self.decode(self._sample_transition(s_idx, action))
        else:
            reward = [0.0 for _ in range(self.formatter.n_rewards)]

        # terminal states are those that have no actions available
        terminated = True if sum(self.action_mask[s_idx]) == 0.0 else False
        truncated = False

        state = self.state
        info = self._get_info()

        if isinstance(reward, list):
            r = sum(reward)  # Note: gym requires to return an int/float, not a list
            info["reward"] = reward
        else:
            r = reward
            info["reward"] = [reward]

        return state, r, terminated, truncated, info

    def _get_info(self):
        """
        Accumulate additional information about the state/environment.

        Returns
        -------
        info : dict
        """
        s_idx = self.encode(self.state)
        info = {"action_mask": self.action_mask[s_idx]}
        if self.formatter.has_state_valuations:
            info["state_valuations"] = self.formatter.state_to_values[s_idx]
        if self.formatter.has_state_labels:
            info["state_labels"] = self.formatter.state_to_labels[s_idx]
        if self.formatter.has_reward_labels:
            info["reward_labels"] = list(self.formatter.reward_labels.keys())
        if self.formatter.has_action_labels:
            info["action_labels"] = self.formatter.action_to_label
        return info
