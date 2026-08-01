from typing import Optional, Any
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.environments.verigymenv import VeriGymEnv
from verigym.environments.transition_func import TransitionFunction


class ExplicitEnv(BaseExplicitEnv):
    abstractionmap: Any
    original_env: VeriGymEnv
    observation_space: gym.spaces.Discrete
    action_space: gym.spaces.Discrete
    action_mask: NDArray

    def __init__(
        self,
        nr_states: int,
        nr_actions: int,
        initial_state_distr: NDArray,
        transition_function: TransitionFunction,
        reward_function: dict,
        nr_rewards: int = 1,
        abstraction_map=None,
        original_env: VeriGymEnv = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__(render_mode)

        self.transition_function = transition_function
        self.reward_function = reward_function

        self.abstraction_map = abstraction_map
        self.original_env = original_env

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
        for s, vals in self.transition_function.T_dict.items():
            for a, trs in vals.items():
                action_mask[s, a] = 1.0
        return action_mask

    def sample_initial_state(self):
        assert self.initial_states is not None

        if isinstance(self.initial_states, np.ndarray):
            return np.random.choice(len(self.initial_states), p=self.initial_states)
        
        if isinstance(self.initial_states, dict):
            random_nmbr = np.random.rand()
            cum_p = 0
            for (idx, p) in self.initial_states.items():
                cum_p += p 
                if cum_p >=random_nmbr:
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

        if isinstance(reward, list):
            r = sum(reward)  # Note: gym requires to return an int/float, not a list
        else:
            r = reward

        return state, r, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
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
