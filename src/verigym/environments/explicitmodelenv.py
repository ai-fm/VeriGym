from abc import ABC

from .verigymenv import VeriGymEnv
from .formatter import ExplicitFormatter

import gymnasium as gym

class ExplicitModelEnv(VeriGymEnv, ABC):
    """
    Abstract gymnasium/VeriGym wrapper for MDP/model-based frameworks.
    
    Notes
    -----
    This class cannot be instantiated.
    To instantiate a functioning gym-like child, you need to define
    `self.observation_space` and `self.action_space` using `gym.spaces`
    """
    def __init__(self, 
                 model: object,
                 formatter : ExplicitFormatter,
                 render_mode: str | None = None
                 ):

        self.render_mode = render_mode
        self.model = model

        self.formatter = formatter
        self.formatter.format(model)
        super().__init__(self.formatter.transition_function, self.formatter.reward_function, render_mode=render_mode)

        self.state = self.formatter.initial_states[0]
        self.nr_states = self.formatter.nr_states
        self.nr_actions = self.formatter.nr_actions

        self.observation_space = gym.spaces.Discrete(self.nr_states)
        self.action_space = gym.spaces.Discrete(self.nr_actions)

        # Which actions are available in a state?
        self.action_mask = self.formatter.action_mask
