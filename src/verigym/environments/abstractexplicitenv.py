from abc import ABC
from abc import abstractmethod
from typing import Optional
import random

from .verigymenv import VeriGymEnv

"""
Abstract class for explicit environments, i.e., those environments with a fully specified (sparse) transition function.
"""
class AbstractExplicitEnv(VeriGymEnv, ABC):
    
    def __init__(self, 
                 transition_function : dict,
                 reward_function : dict,
                 render_mode: str | None = None
                 ):
        super().__init__()

        self.render_mode = render_mode

        self.transition_function = transition_function
        self.reward_function = reward_function

        self.state = None
        self.nr_states = len(self.transition_function.keys())
        self.nr_actions = len(self.transition_function[next(iter(transition_function.keys()))].keys()) # FIXME: get any action space; but need largest action space!

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
        assert state in self.observation_space, "The state is not in the observation space."
        self.state = state

    def get_reward(self, state, action) -> list:
        assert state in self.reward_function.keys(), f"Provided state {state} is not a valid state."
        assert action in self.reward_function[state].keys(), f"Provided action {action} is not available in state {state}."
        
        return self.reward_function[state][action]

    # Gymnasium functionality
    @abstractmethod
    def step(self, action):
        """
        Take a step in the environment.
        """
        # Implement in child class.
        ...
    
    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None):
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
