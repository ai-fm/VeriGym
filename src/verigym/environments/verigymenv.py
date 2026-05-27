import logging
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .labeling import StateLabeler, StateLabel

if TYPE_CHECKING:
    from ..policy.policy import PolicyClass

logger = logging.getLogger(__name__)


class VeriGymEnv(gym.Env):
    """
    Abstract wrapper around `gymnasium.Env` for a unified VeriGym interface.
    Parent to all types of VeriGym environments.
    """

    def __init__(self):
        self.state_labeler = StateLabeler({})

    def simulate(
        self, policy: "PolicyClass", n_steps: int = 1, verbose = True
    ) -> list[list[NDArray, NDArray, NDArray, NDArray]]:
        """
        Simulate the environment for `n_steps` using the provided `policy`.

        Args:
            policy: A function that takes an observation and returns an action.
            n_steps (int): Number of steps to simulate.

        Returns:
            A list of trajectories containing a list of tuples (state, action, reward, next_state) for each step.
        """
        dataset, trajectory = [], []
        state, info = self.reset()
        
        for _ in tqdm(range(n_steps), desc="Simulating", disable=not verbose):
            action = policy.get_action(state)
            next_state, reward, done, truncated, info = self.step(action)
            next_state, action, reward = (
                np.array(next_state),
                np.array(action),
                np.array(reward),
            )
            trajectory.append((state, action, reward, next_state))
            state = next_state
            if done or truncated:
                dataset.append(trajectory)
                state, info = self.reset()
                trajectory = []

        if len(trajectory) > 0:
            # Append remaining trajectories
            dataset.append(trajectory)

        return dataset
    
        
    def has_state_labels(self):
        return len(self.state_labeler.labels) > 0
    
    def add_state_label(self, label: StateLabel):
        self.state_labeler.add_state_label(label)

    def add_state_labels(self, labels: list[StateLabel]):
        for label in labels:
            self.state_labeler.add_state_label(label)
    
    def get_labels_of_state(self, state):
        return self.state_labeler.get_labels_of_state(state)
    
