import numpy as np

from verigym.abstraction.utils import (
    factored_to_index,
    index_to_factored,
)


def test_factored_to_index():
    """Test the factored_to_index function with a simple example with inhomogenous bins per dim."""
    bin_edges = [
        np.array([0, 1, 2]),
        np.array([0, 1]),
        np.array([2, 3]),
        np.array([1, 2, 5, 9]),
    ]

    # create a list with all possible combinations of the bin edges
    states = []
    len_edges = tuple(len(edges) for edges in bin_edges)
    # iterate over all possible combinations of the bin edges
    for indices in np.ndindex(len_edges):
        state = np.array([bin_edges[i][indices[i]] for i in range(len(bin_edges))])
        states.append(state)
    # iterate over all states and check index
    for i, state in enumerate(states):
        index_true = i + 1
        index = factored_to_index(bin_edges, state)
        assert index == index_true, f"Expected index {index_true} but found {index}"


def test_factored_to_index_random():
    """Test the index_to_factored functions with random states and inhomogenous bins per dim. Note this function relies on factored_to_index, so if that function is incorrect this test may fail even if index_to_factored is correct."""
    bin_edges = [
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 2]),
        np.array([2, 3, 4, 5]),
        np.array([1, 2, 5, 9]),
    ]
    # test random states
    len_edges = tuple(len(edges) for edges in bin_edges)
    # iterate through all possible combinations of the bin edges
    for indices in np.ndindex(len_edges):
        state = np.array([bin_edges[i][indices[i]] for i in range(len(bin_edges))])
        # get index of state
        index = factored_to_index(bin_edges, state)
        # reconstruct state from index - testing this function !
        state_reconstructed = index_to_factored(bin_edges, index)
        assert np.array_equal(state, state_reconstructed), (
            f"Expected state {state} but found {state_reconstructed}"
        )