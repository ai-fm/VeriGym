import numpy as np
from numpy.typing import NDArray

class StateDistribution:
    """
    Sparse probability distribution over states.
    Only states with an explicitly written probability are stored; all other states have probability 0.0.
    """
    n_states: int
    distr_dict: dict[int, float]

    def __init__(self, n_states: int, distr_dict: dict[int, float] | None = None):
        self.n_states = n_states
        self.distr_dict = {} if distr_dict is None else distr_dict

    def __getitem__(self, s):
        """Probability of state `s`. Writes no new data if `s` is not present."""
        return self.distr_dict.get(s, 0.0)

    def __setitem__(self, s, prob):
        """Set the probability of state `s`."""
        self.distr_dict[s] = prob

    def sample(self) -> int:
        """Sample a state according to the distribution."""
        states = list(self.distr_dict.keys())
        probs = list(self.distr_dict.values())
        return int(np.random.choice(states, p=probs))

    @classmethod
    def from_array(cls, array: NDArray) -> "StateDistribution":
        """
        Create a StateDistribution from a 1D array, where `array[s]` is the probability of state `s`.
        Only states with non-zero probability are stored.
        """
        array = np.asarray(array, dtype=float)
        distr_dict = {int(s): float(array[s]) for s in np.flatnonzero(array)}
        return cls(n_states=len(array), distr_dict=distr_dict)
