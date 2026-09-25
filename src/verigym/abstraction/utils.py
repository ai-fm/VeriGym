import numpy as np
from numpy.typing import NDArray

from verigym.abstraction.discretization import BinEdges
from verigym.abstraction.gym_utils.mapping import sample_to_discrete


def factored_to_index(state: NDArray, bin_edges: BinEdges) -> int:
    """Converts a discrete factored state representation to an index representation.
    Indices start at 0.

    Parameters
    ----------
    bin_edges : BinEdges
        The bin edges used for discretization.
    state : NDArray
        The factored state representation.

    Returns
    -------
    int
        The index representation of the state.

    """
    d_state = sample_to_discrete(state, bin_edges, return_idx=True)
    newindex = 0
    for i, pos in enumerate(d_state[::-1]):
        i = i + 1
        newindex += pos * (np.prod(bin_edges.n_bins[-i + 1 :]) if i != 1 else 1)

    return int(newindex)


def index_to_factored(state_index: int, bin_edges: BinEdges) -> NDArray:
    """Converts an index representation of a state to a (discrete) factored state representation.
    Indices start at 0.

    Parameters
    ----------
    state_index : int
        The index representation of the state.
    bin_edges : BinEdges
        The bin edges used for discretization.

    Returns
    -------
    NDArray
        The factored state representation.

    """
    state = np.zeros((len(bin_edges.ranges),))
    index = state_index

    for i in range(len(bin_edges.ranges) - 1, -1, -1):
        dim_size = bin_edges.n_bins[i]
        pos = index % dim_size
        state[i] = bin_edges[i][pos]
        index = index // dim_size

    return state
