from abc import ABC, abstractmethod
from typing import Optional
import random

from numpy.typing import NDArray

from verigym.environments.verigymenv import VeriGymEnv
from verigym.environments.transition_func import TransitionFunction

class BaseExplicitEnv(VeriGymEnv, ABC):
    render_mode: Optional[str]
    transition_function: TransitionFunction
    initial_states: NDArray
    reward_function: dict
    state: int
    nr_states: int
    nr_actions: int
    nr_rewards: int
    
    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        self.transition_function = None
        self.reward_function = None

        self.state = None
        self.nr_states = None
        self.nr_actions = None
        self.nr_rewards = None

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

    def get_transition_function(self) -> dict:
        """
        Provides access to the transition function.
        The format is defined by self.formatter.
        """
        return self.transition_function

    def get_reward_function(self) -> dict:
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
        assert state in self.reward_function.keys(), (
            f"Provided state {state} is not a valid state."
        )
        assert action in self.reward_function[state].keys(), (
            f"Provided action {action} is not available in state {state}."
        )

        return self.reward_function[state][action]

    def _get_obs_at(self, state):
        """
        (Private) returns the observation at a given state.
        """
        if self.has_state_valuations:
            return self.state_to_values[state]
        else:
            return state

    def get_observation(self, state):
        """
        Returns the observation at a given state.
        """
        assert state in self.transition_function.keys(), (
            f"Provided state {state} is not a valid state."
        )

        return self._get_obs_at(state)

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
