from abc import ABC
from abc import abstractmethod
from typing import Optional
import random

from .verigymenv import VeriGymEnv
from .formatter import ExplicitFormatter

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
        self.formatter.format()
        super().__init__(self.formatter.transition_function, self.formatter.reward_function, render_mode=render_mode)
        
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

    def render(self, **kwargs):
        ... # TODO

    def _render_rgb_array(self, **kwargs):
        ... # TODO
    
    def _render_human(self, **kwargs):
        ... # TODO
    
