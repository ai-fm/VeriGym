from abc import ABC, abstractmethod
from typing import Optional
import random

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

    def __init__(self, render_mode: str | None = None, dummy_terminal_state: int = None):
        super().__init__()
        self.render_mode = render_mode

        self.transition_function = None
        self.reward_function = None

        self.state = None
        self.nr_states = None
        self.nr_actions = None
        self.nr_rewards = None
        self.dummy_terminal_state = dummy_terminal_state

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

    # Gymnasium functionality
    @abstractmethod
    def step(self, action):
        """
        Take a step in the environment.
        """
        # Implement in child class.
        ...

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset to an initial state.
        """
        # Overwrite in child class.
        return super().reset(seed=seed, options=options)

    @abstractmethod
    def _get_info(self):
        """
        Accumulate additional information about the environment/state.
        """
        # Implement in child class.
        ...

    def _get_obs(self):
        """
        Returns the state.
        """
        return self.state

    def render(self, **kwargs): ...  # TODO

    def _render_rgb_array(self, **kwargs): ...  # TODO

    def _render_human(self, **kwargs): ...  # TODO
