"""Helpful functions for testing."""

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym

from verigym.abstraction.gym_utils.transform_observation import (
    ReplaceInfObservation,
    DiscretizeBoxObservation,
)
from verigym.abstraction.discretization import generate_box_bins
from verigym.environments import GenerativeEnv

def get_vector(precision=2):
    numpy_state = np.array([0.87, 0.24])
    numpy_state = numpy_state.round(decimals=precision)
    print(numpy_state)
    return numpy_state

def vector_to_int(vector, precision=2):
    """
    Map a single continuous numpy vector to a single integer.

    Args:
        vector: 1D numpy array
        precision: Number of decimal places to consider for discretization

    Returns:
        A single integer representing the vector
    """
    # Discretize each component
    discretized = np.round(vector * (10 ** precision)).astype(int)
    # Pack components into a single integer using bit shifting
    result = 0
    for i, val in enumerate(discretized):
        # Shift left and add the current value
        result = (result << 20) | (val + (1 << 19))  # 20 bits per component, offset to handle negatives
    return result

def int_to_vector(integer, length, precision=2):
    """
    Map a single integer back to the original numpy vector.

    Args:
        integer: Integer representing the vector
        length: Length of the original vector
        precision: Number of decimal places used for discretization

    Returns:
        1D numpy array representing the original vector
    """
    discretized = []
    for _ in range(length):
        # Extract 20 bits, subtract offset, and append
        val = (integer & ((1 << 20) - 1)) - (1 << 19)
        discretized.insert(0, val)
        integer >>= 20
    return np.array(discretized) / (10 ** precision)

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


def generate_dataset(
    n_states: int,
    n_actions: int,
    T_array: NDArray,
    n_trajectories: int,
    trajectory_length: int,
    rewards: NDArray = None,
) -> list[list[tuple[NDArray, NDArray, NDArray, NDArray]]]:
    """Generates a dataset of trajectories. No need for a VeriGymEnv. Requires a numpy array as transition function.
    See `initialize_transition_array()` for quick instantiation of array."""
    dataset = []
    rewards = rewards if rewards is not None else np.zeros(1)
    for i in range(n_trajectories):
        trajectory = []
        for _ in range(trajectory_length):
            s = np.random.randint(0, n_states)
            a = np.random.randint(0, n_actions)
            s_next = np.random.choice(n_states, p=T_array[s, a])
            reward = np.random.choice(rewards)
            s, a, s_next = np.array(s), np.array(a), np.array(s_next)
            trajectory.append((s, a, reward, s_next))
        dataset.append(trajectory)
    return dataset
