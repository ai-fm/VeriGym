import math
import time
import numpy as np
from numpy.typing import NDArray

from verigym.abstraction.discretization import BinEdges

import functools

def cumprod():
    pass

def factored_to_index(bin_edges: BinEdges, state: NDArray) -> int:
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
    lens = [len(dim) for dim in bin_edges]

    newindex = 0
    for i in range(1, len(bin_edges)+1):
        pos = (np.digitize(state[-i], bin_edges[-i]) - 1).item()
        newindex += pos * (math.prod(lens[-i + 1:]) if i != 1 else 1)

    return newindex


def index_to_factored(bin_edges: BinEdges, state_index: int) -> NDArray:
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
    state = np.zeros((len(bin_edges),))
    index = state_index

    for i in range(len(bin_edges) - 1, -1, -1):
        dim_size = len(bin_edges[i])
        pos = index % dim_size
        state[i] = bin_edges[i][pos]
        index = index // dim_size

    return state
