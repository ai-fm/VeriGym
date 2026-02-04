import gymnasium as gym
from typing import Optional
import random
import numpy as np

from verigym.environments.base_explicitenv import BaseExplicitEnv


class ExplicitEnv(BaseExplicitEnv):
    
    def __init__(self,
                 nr_states, nr_actions, nr_rewards,
                 initial_state_distr,
                 transition_function,
                 reward_function,
                 abstraction_map: Optional["AbstractionMap"] = None,
                 original_env : Optional["AbstractionMap"] = None,
                 render_mode: Optional[str] = None
                 ):
        
        super().__init__(render_mode)

        self.transition_function = transition_function
        self.reward_function = reward_function

        self.abstraction_map = abstraction_map
        self.original = original_env

        self.state = 0
        self.nr_states = nr_states
        self.nr_actions = nr_actions
        self.nr_rewards = nr_rewards

        self.initial_states = initial_state_distr

        self.observation_space = gym.spaces.Discrete(self.nr_states)
        self.action_space = gym.spaces.Discrete(self.nr_actions)

        # Which actions are available in a state?
        self.action_mask = self._init_action_mask()
    
    def _init_action_mask(self):
        action_mask = np.zeros((self.nr_states, self.nr_actions))
        for s, vals in self.transition_function.items():
            for a, trs in vals.items():
                action_mask[s, a] = 1.0
        return action_mask

    def sample_initial_state(self):
        assert self.initial_states is not None
        
        idx = np.random.choice(len(self.initial_states), p=self.initial_states)
        return idx

    def step(self, action):
        """
        Take a step in the environment.
        """
        # Implement in child class.
        if self.action_mask[self.state][action] > 0:
            reward = self.reward_function[self.state][action]
            self.state = self._sample_transition(self.state, action)
        else:
            reward = [0.0 for _ in range(self.nr_rewards)]
        
        # terminal states are those that have no actions available
        terminated = True if sum(self.action_mask[self.state]) == 0.0 else False
        truncated = False

        state = self.state
        info = self._get_info()
        info["reward"] = reward

        r = sum(reward) # Note: gym requires to return an int/float, not a list

        return state, r, terminated, truncated, info

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None):
        """
        Reset to an initial state.
        """
        # Overwrite in child class.
        super().reset(seed=seed, options=options)

        self.state = self.sample_initial_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_info(self):
        """
        Accumulate additional information about the environment/state.
        """
        return {}