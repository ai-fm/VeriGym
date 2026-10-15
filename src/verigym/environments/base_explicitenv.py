from abc import ABC
from typing import Optional
import random

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from verigym.environments.verigymenv import VeriGymEnv
from verigym.environments.transition_func import TransitionFunction
from verigym.environments.reward_func import RewardFunction
from verigym.environments.state_distributions import StateDistribution


class BaseExplicitEnv(VeriGymEnv, ABC):
    render_mode: Optional[str]
    transition_function: TransitionFunction
    initial_states: StateDistribution
    reward_function: RewardFunction
    state: int
    nr_states: int
    nr_actions: int
    nr_rewards: int
    dummy_terminal_state: int | None
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Discrete
    action_mask: dict[int, NDArray]
    terminal_states: dict[int, bool]

    def __init__(
        self,
        nr_states: int,
        nr_actions: int,
        initial_state_distr: StateDistribution | NDArray,
        transition_function: TransitionFunction,
        reward_function: RewardFunction,
        nr_rewards: int = 1,
        observation_space: gym.spaces.Space | None = None,
        render_mode: str | None = None,
        dummy_terminal_state: int | None = None,
    ):
        """
        Stores the explicit model and sets up the spaces. Shared by all explicit envs, which pass their model in this
        form (e.g., `FrameworkExplicitEnv` converts its model with its formatter).

        Parameters
        ----------
        nr_states : int
            Number of states.
        nr_actions : int
            Number of actions.
        initial_state_distr : StateDistribution | NDArray
            The initial state distribution. An array (with the probability of each state) is converted to a
            `StateDistribution`.
        transition_function : TransitionFunction
            The transition function.
        reward_function : RewardFunction
            The reward function.
        nr_rewards : int, default=1
            Number of reward functions/models.
        observation_space : gym.spaces.Space | None, default=None
            The observation space, `Discrete(nr_states)` by default.
        render_mode : str | None, default=None
            Environment render mode. Currently can only be None.
        dummy_terminal_state : int | None, default=None
            The terminal sink state, e.g., of a learned abstraction.
        """
        super().__init__()
        self.render_mode = render_mode

        self.nr_states = nr_states
        self.nr_actions = nr_actions
        self.nr_rewards = nr_rewards

        if not isinstance(initial_state_distr, StateDistribution):
            initial_state_distr = StateDistribution.from_array(initial_state_distr)
        self.initial_states = initial_state_distr

        self.transition_function = transition_function
        self.reward_function = reward_function
        self.dummy_terminal_state = dummy_terminal_state

        self.observation_space = gym.spaces.Discrete(nr_states) if observation_space is None else observation_space
        self.action_space = gym.spaces.Discrete(nr_actions)

        # Which actions are available in a state?
        self.action_mask = self._init_action_mask()
        # Which states end an episode?
        self.terminal_states = self._init_terminal_states()

        self.state = None

    def _init_action_mask(self) -> dict[int, NDArray]:
        """
        The available actions of the states of the transition function: a mask with 1 for each of their actions.
        States without an entry have no available actions, so the mask stays small for sparse models.
        The masks are int8, as `action_space.sample(mask)` requires.
        """
        action_mask = {}
        for s, vals in self.transition_function.T_dict.items():
            action_mask[s] = np.zeros(self.nr_actions, dtype=np.int8)
            for a in vals.keys():
                action_mask[s][a] = 1
        return action_mask

    def _init_terminal_states(self) -> dict[int, bool]:
        """
        Whether each state of the transition function is terminal.
        
        A state is terminal if 
        
        - it has no available actions or does not exist in the transition function dictionary (deadlock states), 
        or 
        - all its available actions are self-loops with probability 1.
        
        The `dummy_terminal_state` is always terminal as well. 
        """
        terminal_states = {}
        for s, actions in self.transition_function.T_dict.items():
            # .get: do not add keys
            terminal_states[s] = all(successors.get(s, 0.0) == 1.0 for successors in actions.values()) # all returns True for empty iterable
        if self.dummy_terminal_state is not None:
            terminal_states[self.dummy_terminal_state] = True
        return terminal_states

    def _is_terminal(self, state_idx) -> bool:
        """
        Whether the state with index `state_idx` is terminal (see `_init_terminal_states`).
        
        We assume that states that are not in the transition function dictionary have no available actions, so they are terminal too.
        """
        return self.terminal_states.get(state_idx, True)

    def _action_mask_at(self, state_idx) -> NDArray:
        """The action mask of a state (all zeros if it has no available actions)."""
        mask_s = self.action_mask.get(state_idx)
        return mask_s if mask_s is not None else np.zeros(self.nr_actions, dtype=np.int8)

    def _is_available(self, state_idx, action) -> bool:
        """Whether `action` is available in the state with index `state_idx`."""
        return bool(self._action_mask_at(state_idx)[action] > 0)

    # Explicit functionality
    def _sample_transition(self, state, action):
        """
        Sample a transition from the transition function according to the transition probabilities.

        Parameters
        ----------
        state : int
            The current state index
        action : int
            The chosen action index

        Returns
        -------
        next_state : int
            The sampled next state according to the transition probabilities.
        """
        assert self.transition_function is not None

        transitions = self.transition_function[state][action]
        next_states = list(transitions.keys())
        probs = [transitions[next_state] for next_state in next_states]

        next_state = random.choices(next_states, weights=probs, k=1)[0]
        return next_state

    def get_transition_function(self) -> TransitionFunction:
        """
        Provides access to the transition function.
        The format is defined by self.formatter.
        """
        return self.transition_function

    def get_reward_function(self) -> RewardFunction:
        """
        Provides access to the reward function.
        The format is defined by self.formatter.
        """
        return self.reward_function

    def set_state(self, state) -> None:
        """
        Set the environment to a given state.
        """
        assert state in self.observation_space, (
            "The state is not in the observation space."
        )
        self.state = state

    def get_reward(self, state, action) -> list:

        return self.reward_function[state][action]
    
    def get_abstraction_map(self):
        """
        Returns the environment's abstraction map or None if it does not have any.
        """
        return None

    def encode(self, state):
        """
        Returns the index of a state of the observation space. The identity, unless the env has other states (e.g.,
        factored states).
        """
        return state

    def decode(self, state_idx):
        """
        Returns the state of the observation space with index `state_idx`. The identity, unless the env has other
        states (e.g., factored states).
        """
        return state_idx

    def sample_initial_state(self):
        """
        Samples a state from the initial state distribution.
        """
        return self.decode(self.initial_states.sample())

    # Gymnasium functionality
    def step(self, action):
        """
        Take a step in the environment from the current state.

        Parameters
        ----------
        action : int
            Chosen action from self.action_space. Unavailable actions do not change the state and give no reward.

        Returns
        -------
        observation
            The new state (a factored state if the env has factored states).
        reward : float
            The sum of the rewards, since gymnasium requires a single reward. `info["reward"]` has the list of rewards.
        terminated : bool
            Whether the new state is terminal (see `_init_terminal_states`).
        truncated : bool
            Always False.
        info : dict
            The info of the new state (see `_get_info`) and of the transition (see `_gather_transition_info`).
        """
        action = int(action)
        s_idx = self.encode(self.state)
        if self._is_available(s_idx, action):
            reward = self.reward_function[s_idx][action]
            next_idx = self._sample_transition(s_idx, action)
            self.state = self.decode(next_idx)
        else:
            reward = [0.0 for _ in range(self.nr_rewards)]
            next_idx = s_idx

        terminated = self._is_terminal(next_idx)
        truncated = False

        info = self._get_info() | self._gather_transition_info(s_idx, action, next_idx)
        
        got_multiple_rewards = isinstance(reward, list)
        
        info["reward"] = reward if got_multiple_rewards else [reward]
        r = sum(reward) if got_multiple_rewards else reward

        return self.state, r, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset to an initial state.
        """
        super().reset(seed=seed, options=options)

        # decoded to the factored state if the env has factored states
        self.state = self.sample_initial_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_info(self):
        """
        Accumulate additional information about the environment/state: the action mask of the current state.
        """
        return {"action_mask": self._action_mask_at(self.encode(self.state))}

    def _gather_transition_info(self, state, action, next_state):
        """
        Accumulate additional information about the current transition.
        """
        return {}

    def _get_obs(self):
        """
        Returns the state.
        """
        return self.state
