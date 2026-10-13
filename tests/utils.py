"""Helpful functions for testing."""

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
import stormpy

from verigym.abstraction.gym_utils.transform_observation import (
    ReplaceInfObservation,
    DiscretizeBoxObservation,
)
from verigym.abstraction.discretization import generate_box_bins
from verigym.abstraction.abstractionmapper import AbstractionMapper, linspace_mapper
from verigym.environments import GenerativeEnv


def get_vector(precision=2):
    numpy_state = np.array([0.87, 0.24])
    numpy_state = numpy_state.round(decimals=precision)
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
    discretized = np.round(vector * (10**precision)).astype(int)
    # Pack components into a single integer using bit shifting
    result = 0
    for i, val in enumerate(discretized):
        # Shift left and add the current value
        result = (result << 20) | (
            val + (1 << 19)
        )  # 20 bits per component, offset to handle negatives
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
    return np.array(discretized) / (10**precision)


def make_original_env() -> tuple[gym.Env, int, int]:
    """Returns CartPole environment, NUM_STEPS, BIN_EDGES_PER_DIM"""
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env = ReplaceInfObservation(env, neg_inf=-10, pos_inf=10)
    NUM_STEPS = 1000
    BIN_EDGES_PER_DIM = 5

    return env, NUM_STEPS, BIN_EDGES_PER_DIM

def get_abstraction_mapper_to_discrete(
    env: gym.Env, bin_edges_per_state_dim, bin_edges_per_action_dim
) -> AbstractionMapper:
    """
    Create a simple abstraction mapper that maps from an original space to a discrete space.
    """
    return linspace_mapper(env, bin_edges_per_state_dim, bin_edges_per_action_dim)


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
            trajectory.append((s, a, reward, s_next))
        dataset.append(trajectory)
    return dataset


def helper_to_create_stormpy_model_for_tests(model_class, rows, row_groups=None, reward_models=None, choice_labels=None,
                        state_labels=None, nr_states=2):
    """A small stormpy model with initial state 0 and without state valuations.
    `rows` lists the {column: value} of each row (choice), `row_groups` the first row of each state (MDPs only), and
    `reward_models` maps names to the (state rewards, state-action rewards), either of which can be None.
    `choice_labels` lists the label of each row (None for unlabelled rows); without it, the model has no choice
    labels. `state_labels` maps labels to their states, besides "init".
    For interval MDPs, the values are (lower, upper) tuples."""
    interval = model_class is stormpy.storage.SparseIntervalMdp
    to_value = (lambda bounds: stormpy.pycarl.Interval(*bounds)) if interval else (lambda value: value)
    builder_class = stormpy.IntervalSparseMatrixBuilder if interval else stormpy.SparseMatrixBuilder
    builder = builder_class(rows=0, columns=nr_states, entries=0, force_dimensions=True,
                            has_custom_row_grouping=row_groups is not None)
    for row, entries in enumerate(rows):
        if row_groups is not None and row in row_groups:
            builder.new_row_group(row)
        for column, value in entries.items():
            builder.add_next_value(row, column, to_value(value))
    labeling = stormpy.storage.StateLabeling(nr_states)
    for label, states in {"init": [0], **(state_labels or {})}.items():
        labeling.add_label(label)
        for state in states:
            labeling.add_label_to_state(label, state)
    components_class = stormpy.SparseIntervalModelComponents if interval else stormpy.SparseModelComponents
    components = components_class(transition_matrix=builder.build(), state_labeling=labeling)
    reward_class = stormpy.SparseIntervalRewardModel if interval else stormpy.SparseRewardModel
    components.reward_models = {
        name: reward_class(
            optional_state_reward_vector=None if state_rewards is None else [to_value(r) for r in state_rewards],
            optional_state_action_reward_vector=None if choice_rewards is None else [to_value(r) for r in choice_rewards],
        )
        for name, (state_rewards, choice_rewards) in (reward_models or {}).items()
    }
    if choice_labels is not None:
        choice_labeling = stormpy.storage.ChoiceLabeling(len(rows))
        for label in sorted({label for label in choice_labels if label is not None}):
            choice_labeling.add_label(label)
        for row, label in enumerate(choice_labels):
            if label is not None:
                choice_labeling.add_label_to_choice(label, row)
        components.choice_labeling = choice_labeling
    return model_class(components)
