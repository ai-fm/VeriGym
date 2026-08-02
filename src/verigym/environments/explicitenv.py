from typing import Optional, Any
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.environments.verigymenv import VeriGymEnv
from verigym.environments.transition_func import TransitionFunction
from verigym.environments.reward_func import RewardFunction


class ExplicitEnv(BaseExplicitEnv):
    abstraction_map: Any
    original_env: VeriGymEnv
    observation_space: gym.spaces.Discrete
    action_space: gym.spaces.Discrete
    action_mask: NDArray # TODO: should this be made sparse?
    terminal_states: NDArray # TODO: should this be made sparse?

    def __init__(
        self,
        nr_states: int,
        nr_actions: int,
        initial_state_distr: NDArray,
        transition_function: TransitionFunction,
        reward_function: RewardFunction,
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

        self.terminal_states = []
        self._init_terminal_states()

    def _init_terminal_states(self):
        self.terminal_states = []
        for s in range(self.nr_states):
            if (sum(self.action_mask[s]) == 0) or \
                all([self.transition_function[s][a][s] == 1.0 for a in range(self.nr_actions) if self.action_mask[s][a]]):
                self.terminal_states.append(s)

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

        state = self.state
        if self.action_mask[self.state][action] > 0:
            reward = self.reward_function[self.state][action]
            next_state = self._sample_transition(self.state, action)
            self.state = next_state
        else:
            reward = [0.0 for _ in range(self.nr_rewards)]
            next_state = self.state

        info = self._gather_transition_info(state, action, next_state)

        # terminal states are those that have no actions available
        terminated = self.state in self.terminal_states
        truncated = False

        info = self._get_info() | info
        info["reward"] = reward

        if isinstance(reward, list):
            r = sum(reward)  # Note: gym requires to return an int/float, not a list
        else:
            r = reward

        return next_state, r, terminated, truncated, info

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
    
    def get_abstraction_map(self):
        return self.abstraction_map

    def _get_info(self):
        """
        Accumulate additional information about the environment/state.
        """
        return {}
    
    def _gather_transition_info(self, state, action, next_state):
        """
        Accumulate additional information about the current transition.
        """
        return {}
