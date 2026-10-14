from typing import Optional, Any
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.environments.verigymenv import VeriGymEnv
from verigym.environments.transition_func import TransitionFunction
from verigym.environments.reward_func import RewardFunction
from verigym.environments.state_distributions import StateDistribution


class ExplicitEnv(BaseExplicitEnv):
    abstraction_map: Any
    original_env: VeriGymEnv
    observation_space: gym.spaces.Discrete
    action_space: gym.spaces.Discrete
    action_mask: dict[int, NDArray]
    terminal_states: dict[int, bool]

    def __init__(
        self,
        nr_states: int,
        nr_actions: int,
        initial_state_distr: StateDistribution | NDArray, # gets converted to StateDistr if it isn't one
        transition_function: TransitionFunction,
        reward_function: RewardFunction,
        nr_rewards: int = 1,
        abstraction_map=None,
        original_env: VeriGymEnv = None,
        render_mode: Optional[str] = None,
        dummy_terminal_state: int = None
    ):
        super().__init__(render_mode)

        self.transition_function = transition_function
        self.reward_function = reward_function

        self.abstraction_map = abstraction_map
        self.original_env = original_env
        
        self.dummy_terminal_state = dummy_terminal_state

        self.state = 0
        self.nr_states = nr_states
        self.nr_actions = nr_actions
        self.nr_rewards = nr_rewards

        if not isinstance(initial_state_distr, StateDistribution):
            initial_state_distr = StateDistribution.from_array(initial_state_distr)
        self.initial_states = initial_state_distr

        self.observation_space = gym.spaces.Discrete(self.nr_states)
        self.action_space = gym.spaces.Discrete(self.nr_actions)

        # Which actions are available in a state?
        self.action_mask = self._init_action_mask()

        self.terminal_states = {}
        for s, mask_s in self.action_mask.items():
            self.terminal_states[s] = bool((sum(mask_s) == 0) or all(
                [
                    self.transition_function[s][a].get(s, 0.0) == 1.0  # .get: do not add keys
                    for a in range(self.nr_actions)
                    if mask_s[a]
                ]
            ))
        if self.dummy_terminal_state is not None:
            self.terminal_states[dummy_terminal_state] = True

    def _init_action_mask(self) -> dict[int, NDArray]:
        action_mask = {}
        for s, vals in self.transition_function.T_dict.items():
            action_mask[s] = np.zeros(self.nr_actions)
            for a in vals.keys():
                action_mask[s][a] = 1.0
        return action_mask

    def sample_initial_state(self):
        assert self.initial_states is not None
        return self.initial_states.sample()

    def step(self, action):
        """
        Take a step in the environment.
        """
        # Implement in child class.

        state = self.state
        mask_s = self.action_mask.get(self.state)
        if mask_s is not None and mask_s[action] > 0:
            reward = self.reward_function[self.state][action]
            next_state = self._sample_transition(self.state, action)
            self.state = next_state
        else:
            reward = [0.0 for _ in range(self.nr_rewards)]
            next_state = self.state

        info = self._gather_transition_info(state, action, next_state)

        # terminal states are those that have no actions available
        terminated = self.terminal_states.get(self.state, True)
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
