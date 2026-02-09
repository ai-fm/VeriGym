import pytest
import gymnasium as gym
import numpy as np

from verigym.abstraction.learning_transitions import (
    create_abstraction,
    generate_samples,
    generate_box_bins,
    factored_to_index,
    index_to_factored,
)
from verigym.abstraction.gym_utils.transform_observation import (
    ReplaceInfObservation,
    DiscretizeBoxObservation,
)


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

    return discretized_env, NUM_STEPS


def test_create_abstraction():
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    _abstracted_env = create_abstraction(
        original_env=env,
        exploration_strategy="random",
        num_steps=NUM_STEPS,
        bin_edges_per_dim=BIN_EDGES_PER_DIM,
    )


def test_random_exploration_strategy():
    env, NUM_STEPS = make_discretized_env()
    strategy = "random"
    _dataset = generate_samples(env, NUM_STEPS, strategy)


def test_sb3_exploration_strategy():
    env, NUM_STEPS = make_discretized_env()
    strategy = "sb3 policy"
    with pytest.raises(NotImplementedError):
        _dataset = generate_samples(env, NUM_STEPS, strategy)


def test_bad_exploration_strategy():
    env, NUM_STEPS = make_discretized_env()
    strategy = "bad strategy"
    with pytest.raises(ValueError):
        _dataset = generate_samples(env, NUM_STEPS, strategy)


def test_factored_to_index():
    bin_edges = [
        np.array([0, 1, 2]),
        np.array([0, 1]),
        np.array([2, 3]),
        np.array([1, 2, 5, 9]),
    ]

    # create a list with all possible combinations of the bin edges
    states = []
    len_edges = tuple(len(edges) for edges in bin_edges)
    for indices in np.ndindex(len_edges):
        state = np.array([bin_edges[i][indices[i]] for i in range(len(bin_edges))])
        states.append(state)

    for i, state in enumerate(states):
        index_true = i + 1
        index = factored_to_index(bin_edges, state)
        assert index == index_true, f"Expected index {index_true} but found {index}"


def test_factored_to_index_random():
    bin_edges = [
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 2]),
        np.array([2, 3, 4, 5]),
        np.array([1, 2, 5, 9]),
    ]

    # test random states
    len_edges = tuple(len(edges) for edges in bin_edges)
    for indices in np.ndindex(len_edges):
        state = np.array([bin_edges[i][indices[i]] for i in range(len(bin_edges))])
        index = factored_to_index(bin_edges, state)

        state_reconstructed = index_to_factored(bin_edges, index)
        assert np.array_equal(state, state_reconstructed), (
            f"Expected state {state} but found {state_reconstructed}"
        )
