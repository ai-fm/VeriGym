from gymnasium.spaces import Box
import numpy as np

from verigym.abstraction.discretization import BinEdges
from verigym.abstraction.utils import (
    factored_to_index,
    index_to_factored,
)


def test_factored_to_index():
    """Test the factored_to_index function with a simple example with inhomogenous bins per dim."""
    bin_edges = BinEdges(
        space=Box(np.asarray([0, 0, 2, 1]), np.asarray([2, 1, 3, 9]), (4,)),
        edges=np.array([0, 1, 2, 0, 1, 2, 3, 1, 2, 5, 9]),
        ranges=np.asarray([[0, 3], [3, 5], [5, 7], [7, 11]]),
    )

    # create a list with all possible combinations of the bin edges
    states = []
    len_edges = tuple(bin_edges.n_bins)
    # iterate over all possible combinations of the bin edges
    for indices in np.ndindex(len_edges):
        state = np.array(
            [bin_edges[i][indices[i]] for i in range(len(bin_edges.ranges))]
        )
        states.append(state)
    # iterate over all states and check index
    for i, state in enumerate(states):
        index_true = i
        index = factored_to_index(state, bin_edges)
        assert index == index_true, f"Expected index {index_true} but found {index}"


def test_factored_to_index_random():
    """Test the index_to_factored functions with random states and inhomogenous bins per dim. Note this function relies on factored_to_index, so if that function is incorrect this test may fail even if index_to_factored is correct."""
    bin_edges = BinEdges(
        space=Box(np.asarray([0, 0, 2, 1]), np.asarray([4, 2, 5, 9]), (4,)),
        edges=np.array([0, 1, 2, 3, 4, 0, 1, 2, 2, 3, 4, 5, 1, 2, 5, 9]),
        ranges=np.asarray([[0, 5], [5, 8], [8, 12], [12, 16]]),
    )
    # test random states
    len_edges = tuple(bin_edges.n_bins)
    # iterate through all possible combinations of the bin edges
    for indices in np.ndindex(len_edges):
        state = np.array(
            [bin_edges[i][indices[i]] for i in range(len(bin_edges.ranges))]
        )
        # get index of state
        index = factored_to_index(state, bin_edges)
        # reconstruct state from index - testing this function !
        state_reconstructed = index_to_factored(index, bin_edges)
        assert np.array_equal(state, state_reconstructed), (
            f"Expected state {state} but found {state_reconstructed}"
        )
