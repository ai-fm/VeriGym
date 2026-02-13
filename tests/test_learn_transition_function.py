import numpy as np

from verigym.abstraction.learn_transitions import (
    create_abstraction,
    factored_to_index,
    index_to_factored,
    learn_transition_function,
)

from verigym.environments.generativeenv import GenerativeEnv

from utils import (
    generate_dataset,
    make_original_env,
    make_discretized_env,
    initialize_transition_array,
)


def test_random_exploration_strategy():
    env, NUM_STEPS = make_discretized_env()
    _dataset = env.simulate("dummy_policy", NUM_STEPS)


def test_create_abstraction():
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    _abstracted_env = create_abstraction(
        original_env=generative_env,
        exploration_strategy="random",
        num_steps=NUM_STEPS,
        bin_edges_per_dim=BIN_EDGES_PER_DIM,
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


def test_learn_transition_function():
    # create a simple trnasition function
    n_states, n_actions = 10, 5
    T_array = initialize_transition_array(n_states, n_actions)

    # generate a fake dataset of trajectories and state, action next_state tuples
    n_trajectories, trajectory_length = 10, 2000
    dataset = generate_dataset(
        n_states, n_actions, T_array, n_trajectories, trajectory_length
    )

    # use learn_transition_function to approximate T
    T = learn_transition_function(
        dataset=dataset, n_states=n_states, n_actions=n_actions
    )

    # check that T is close to T_array
    for s in range(n_states):
        for a in range(n_actions):
            for s_next in range(n_states):
                # print(T[s][a], T_array[s, a], flush=True)
                assert np.allclose(T[s, a, s_next], T_array[s, a, s_next], atol=0.1), (
                    f"Transition probabilities for state {s} and action {a} are not close enough to the true transition function."
                )
