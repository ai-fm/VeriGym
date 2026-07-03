import gymnasium as gym
import numpy as np
from typing import Optional, Any

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
        self.state = self.sample_initial_state()

        if flat:
            self.transition_function = self.formatter.transition_function
            self.reward_function = self.formatter.reward_function
            self.observation_space = gym.spaces.Discrete(self.nr_states)
        else: 
            if not self.formatter.has_state_valuations:
                raise ValueError("Requested featured state representation, but state valuations are not available.")
            self.observation_space = self._init_md_obervation_space()
        
        self.action_space = gym.spaces.Discrete(self.nr_actions)

        # Which actions are available in a state?
        self.action_mask = self.formatter.action_mask

    def _init_md_observation_space(self):
        """
        Uses the info on state valuations from `self.formatter.state_to_values` to initialise a `MultiDiscrete` observation space.
        Needs to ensure that the states from obs_space are matched to the correct transitions/formatter state indices.
        """
        nvec = self.formatter.max_valuations

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
            return self.formatter.values_to_state(state_values)

    @classmethod
    def from_stormpy(cls, mdp, flat: bool = True, render_mode: str | None = None):
        instance = cls.__new__(cls)

        formatter = StormpyFormatter(mdp)

        instance.__init__(model=mdp, formatter=formatter, flat=flat, render_mode=render_mode)

        return instance

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
        if self.action_mask[self.state][action] > 0:
            reward = self.reward_function[self.state][action]
            self.state = self._sample_transition(self.state, action)
        else:
            reward = [0.0 for _ in range(self.formatter.n_rewards)]

        # terminal states are those that have no actions available
        terminated = True if sum(self.action_mask[self.state]) == 0.0 else False
        truncated = False

        state = self.state
        info = self._get_info()
        info["reward"] = reward

        if isinstance(reward, list):
            r = sum(reward)  # Note: gym requires to return an int/float, not a list
        else:
            r = reward

        return state, r, terminated, truncated, info

    def _get_info(self):
        """
        Accumulate additional information about the state/environment.

        Returns
        -------
        info : dict
        """
        info = {"action_mask": self.action_mask[self.state]}
        if self.formatter.has_state_valuations:
            info["state_valuations"] = self.formatter.state_to_values[self.state]
        if self.formatter.has_state_labels:
            info["state_labels"] = self.formatter.state_to_labels[self.state]
        if self.formatter.has_reward_labels:
            info["reward_labels"] = list(self.formatter.reward_labels.keys())
        if self.formatter.has_action_labels:
            info["action_labels"] = self.formatter.action_to_label
        return info
