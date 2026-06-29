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
    ❌ Slicing

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

    T_dict: defaultdict[int, defaultdict[int, defaultdict[int, float]]]
    n_states: int
    n_actions: int

    def __init__(self, n_states: int, n_actions: int, T_dict=None):
        if T_dict is not None:
            # check if structure of T_dict is correct
            assert isinstance(T_dict, defaultdict), (
                f"T_dict must be a defaultdict, is {type(T_dict)}"
            )
            # check for each state, there is a dict of actions
            for s in T_dict:
                assert isinstance(T_dict[s], dict), (
                    f"T_dict[{s}] must be a dict, is {type(T_dict[s])}"
                )
                # check for each action, there is a dict of next states
                for a in T_dict[s]:
                    assert isinstance(T_dict[s][a], defaultdict), (
                        f"T_dict[{s}][{a}] must be a defaultdict, is {type(T_dict[s][a])}"
                    )
                    # check for each next state, there is a float probability
                    for s_next in T_dict[s][a]:
                        assert isinstance(T_dict[s][a][s_next], float), (
                            f"T_dict[{s}][{a}][{s_next}] must be a float"
                        )
            self.T_dict = T_dict
        else:
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

        for s in range(n_states):
            for a in range(n_actions):
                T_dict[s][a] = defaultdict(float)
                for s_next in range(n_states):
                    prob = array[s, a, s_next]
                    if prob > 0:
                        T_dict[s][a][s_next] = prob

        T = cls(n_states, n_actions, T_dict)
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

    def get_sparsity(self) -> float:
        """
        Returns the sparsity of the transition function.

        It is a fraction of all state-action-state transitions that have non-zero probability over all possible state-action-state transitions.

        Returns
        -------
        float
            Sparsity value of transition function.
        """

        total_entries = self.n_states * self.n_actions * self.n_states

        num_entries = 0

        # Iterate through all entries
        for s, actions in self.T_dict.items():
            for a, transitions in actions.items():
                for s_next, prob in transitions.items():
                    # Add all entries where transition prob is unequal zero
                    num_entries += prob != 0

        return num_entries / total_entries
