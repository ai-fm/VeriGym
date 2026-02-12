import gymnasium as gym
import numpy as np

from verigym.abstraction.learning_transitions import (
    create_abstraction,
    generate_box_bins,
    factored_to_index,
    index_to_factored,
    learn_transition_function,
)
from verigym.abstraction.gym_utils.transform_observation import (
    ReplaceInfObservation,
    DiscretizeBoxObservation,
)
from verigym.environments.generativeenv import GenerativeEnv


def make_original_env() -> tuple[gym.Env, int, int]:
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env = ReplaceInfObservation(env, neg_inf=-10, pos_inf=10)
    NUM_STEPS = 1000
    BIN_EDGES_PER_DIM = 5

    return env, NUM_STEPS, BIN_EDGES_PER_DIM


def make_discretized_env():
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    bin_edges = generate_box_bins(env.observation_space, np.linspace, BIN_EDGES_PER_DIM)
    discretized_env = DiscretizeBoxObservation(
        env, bin_edges=bin_edges, use_box_space=False
    )
    generative_env = GenerativeEnv.from_gymnasium(discretized_env)
    return generative_env, NUM_STEPS


def initialize_transition_array(num_s: int, num_a: int) -> np.ndarray:
    # Create a sample transition function as a numpy array with (s,a,s') structure
    transition_array = np.random.random((num_s, num_a, num_s))
    # normalize to prob. distributions
    transition_array /= transition_array.sum(axis=2, keepdims=True)
    return transition_array


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
    num_s, num_a = 10, 5
    T_array = initialize_transition_array(num_s, num_a)

    # generate a fake dataset of trajectories and state, action next_state tuples
    dataset = []
    n_trajectories, trajectory_length = 10, 2000
    dummy_reward = 0
    for i in range(n_trajectories):
        trajectory = []
        for _ in range(trajectory_length):
            s = np.random.randint(0, num_s)
            a = np.random.randint(0, num_a)
            s_next = np.random.choice(num_s, p=T_array[s, a])
            s, a, s_next = np.array(s), np.array(a), np.array(s_next)
            trajectory.append((s, a, dummy_reward, s_next))
        dataset.append(trajectory)

    # use learn_transition_function to approximate T
    T = learn_transition_function(dataset=dataset, n_states=num_s, n_actions=num_a)

    # check that T is close to T_array
    for s in range(num_s):
        for a in range(num_a):
            for s_next in range(num_s):
                # print(T[s][a], T_array[s, a], flush=True)
                assert np.allclose(T[s, a, s_next], T_array[s, a, s_next], atol=0.1), (
                    f"Transition probabilities for state {s} and action {a} are not close enough to the true transition function."
                )
