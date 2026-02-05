from collections import defaultdict

from tqdm.auto import tqdm

import numpy as np
from numpy.typing import NDArray


class TransitionFunction:
    """
    Base class for transition functions in Verigym environments.
    It is based on dicts in dicts.
    All states and actions are flattened to integers indices.
    ✅ Indexing
    ❌ Slicing (coming soon)

    Example:
    ```
    T_array  # numpy array of shape (n_states, n_actions, n_states)
    T = TransitionFunction.from_array(T_array)
    s, a, s_next = 0, 0, 0
    T[s, a, s_next] == T[s][a][
        s_next
    ]  # probability of transitioning from state s to s_next given action a
    T[s][a]
    ```
    """

    T_dict: defaultdict[int, dict[int, defaultdict[int, float]]]
    n_states: int
    n_actions: int

    def __init__(self, n_states: int = 0, n_actions: int = 0):
        self.T_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.n_states = n_states
        self.n_actions = n_actions

    def __getitem__(self, idx):
        # Normalize to tuple
        if not isinstance(idx, tuple):
            idx = (idx,)

        if len(idx) == 1:
            (s,) = idx
            return self.T_dict[s]
        if len(idx) == 2:
            s, a = idx
            return self.T_dict[s][a]
        if len(idx) == 3:
            s, a, s_next = idx
            return self.T_dict[s][a][s_next]
        raise IndexError("Too many indices for TransitionFunction")

    @classmethod
    def from_array(cls, array: NDArray) -> "TransitionFunction":
        """
        Create a TransitionFunction from a 3D numpy array.

        Parameters
        ----------
        array : NDArray
            A 3D numpy array where array[s, a, s'] represents the probability
            of transitioning from state s to state s' given action a. Hence, s and a need to already be flattened into integer indices.

        Returns
        -------
        TransitionFunction
            An instance of TransitionFunction with the transition probabilities set.
        """
        T_dict: defaultdict[tuple, dict[int, defaultdict[tuple, float]]] = defaultdict(
            dict
        )
        n_states, n_actions, n_states_next = array.shape
        assert n_states == n_states_next, (
            "The first and third dimensions of the array must be the same (number of states)."
        )
        T = TransitionFunction(n_states, n_actions)

        for s in range(n_states):
            for a in range(n_actions):
                T_dict[s][a] = defaultdict(float)
                for s_next in range(n_states):
                    prob = array[s, a, s_next]
                    if prob > 0:
                        T_dict[s][a][s_next] = prob

        T.T_dict = T_dict
        return T

    def sanity_check(self) -> bool:
        """
        Check if the transition function is valid, i.e., if the probabilities sum to 1 for each (s,a) pair.
        """
        for s, actions in tqdm(self.T_dict.items(), desc="Sanity checking T"):
            for a, transitions in actions.items():
                total = 0
                for s_next, prob in transitions.items():
                    total += prob
                if not np.isclose(total, 1.0):
                    print(
                        f"Sanity check failed for state {s} and action {a}: total probability is {total}"
                    )
                    return False
        return True
