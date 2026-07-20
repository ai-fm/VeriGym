from collections import defaultdict

from numpy.typing import NDArray
import numpy as np


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
            # check if structure of R_dict is correct
            assert isinstance(R_dict, defaultdict), (
                f"R_dict must be a defaultdict, is {type(R_dict)}"
            )
            # check for each state, there is a dict of actions
            for s in R_dict:
                assert isinstance(s, int), (
                    f"State index must be an integer, got {type(s)}"
                )
                assert isinstance(R_dict[s], dict), (
                    f"R_dict[{s}] must be a dict, is {type(R_dict[s])}"
                )
                # check for each action, there is a float reward
                for a in R_dict[s]:
                    assert isinstance(a, int), (
                        f"Action index must be an integer, got {type(a)}"
                    )
                    assert isinstance(R_dict[s][a], float), (
                        f"R_dict[{s}][{a}] must be a float, is {type(R_dict[s][a])}"
                    )
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

    @classmethod
    def from_dict(cls, R_dict: dict, n_states: int, n_actions: int) -> "RewardFunction":
        for s in range(n_states):
            if s in R_dict.keys():
                for a in R_dict[s].keys():
                    if isinstance(R_dict[s][a], list):
                        if len(R_dict[s][a]) > 1:
                            raise NotImplementedError(
                                "currently cannot handle multiple reward functions"
                            )
                        R_dict[s][a] = R_dict[s][a][0]
        return cls(n_states=n_states, n_actions=n_actions, R_dict=R_dict)