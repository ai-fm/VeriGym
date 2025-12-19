import pytest
import gymnasium as gym
import numpy as np

from verigym.abstraction.learning_transitions import (
    create_abstraction,
    generate_samples,
    generate_box_bins,
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
