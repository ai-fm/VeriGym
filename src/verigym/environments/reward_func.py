from collections import defaultdict

from tqdm.auto import tqdm

import numpy as np
from numpy.typing import NDArray

class RewardFunction:
    """
    Class for reward functions in Verigym environments.
    It is based on dicts in dicts.
    All states and actions are flattened to integers indices.
    ✅ Indexing
    ❌ Slicing

    Example:
    ```
    R_array  # numpy array of shape (n_states, n_actions)
    R = RewardFunction.from_array(R_array)
    s, a = 0, 0
    R[s, a] == R[s][a]  # reward for taking action a in state s
    R[s]
    ```
    """

    R_dict: defaultdict[int, dict[int, float]]
    n_states: int
    n_actions: int

    def __init__(self, n_states: int, n_actions: int, R_dict=None):
        if R_dict is not None:
            self.R_dict = R_dict
        else:
            self.R_dict = defaultdict(lambda: defaultdict(float))
        self.n_states = n_states
        self.n_actions = n_actions

    def __getitem__(self, idx):
        # Normalize to tuple
        if not isinstance(idx, tuple):
            idx = (idx,)

        if len(idx) == 1:
            (s,) = idx
            return self.R_dict[s]
        if len(idx) == 2:
            s, a = idx
            return self.R_dict[s][a]
        raise IndexError("Too many indices for RewardFunction")

    @classmethod
    def from_array(cls, array: NDArray) -> "RewardFunction":
        """
        Create a RewardFunction from a 2D numpy array.

        Parameters
        ----------
        array : NDArray
            A 2D numpy array of shape (n_states, n_actions) representing the reward function.
            
        Returns
        -------
        RewardFunction
            An instance of RewardFunction initialized with the values from the array.
        """
        n_states, n_actions = array.shape
        R_dict = defaultdict(dict)
        for s in range(n_states):
            for a in range(n_actions):
                R_dict[s][a] = array[s, a]
        return cls(n_states=n_states, n_actions=n_actions, R_dict=R_dict)