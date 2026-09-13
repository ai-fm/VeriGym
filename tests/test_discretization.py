import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Text
import pytest

import numpy as np

import verigym
from verigym.abstraction.gym_utils.mapping import (
    box_to_discrete,
    _continuous_to_discrete,
    _discrete_to_continuous,
    _sample_to_discrete_values,
    _sample_to_discrete_idx,
)
from verigym.abstraction.discretization import (
    BinEdges,
    generate_box_bins,
    generate_box_linspace_bins,
    linspace_map,
    linspace_mapper
)
from verigym.policy.policy import RandomizedPolicy

from verigym.abstraction.gym_utils.transform_action import DiscretizeBoxAction

from utils import get_abstraction_mapper_to_discrete, make_original_env


@pytest.mark.parametrize(
    "low, high, shape, n_samples",
    [
        (-1, 1, (1,), 10),
        (-1, 1, (2,), 10),
        (-1, 1, (2, 2), 10),
        (-1, 1, (2, 3, 4), 10),
        (-1, 1, (2, 3, 4, 5), 10),
    ],
)
def test_bijectivity_transform(low, high, shape, n_samples):
    space = Box(low, high, shape, seed=42)
    bin_edges = generate_box_linspace_bins(space, n_samples)
    discrete_space, to_discrete, to_continuous = box_to_discrete(space, bin_edges)
    discrete_sample = discrete_space.sample()
    assert np.array_equal(
        np.array(to_discrete(to_continuous(discrete_sample))), discrete_sample
    )


@pytest.mark.parametrize(
    "sample, bin_edges, result",
    [
        (
            np.asarray([3.5]),
            BinEdges(
                space=Box(0, 5, (1,)),
                edges=np.asarray([0, 1.1, 2.2, 3.3, 4.4]),
                ranges=np.asarray([[0, 5]]),
            ),
            np.asarray([3]),
        ),
        (
            np.asarray([3.5, 2.5]),
            BinEdges(
                space=Box(0, 5, (2,)),
                edges=np.asarray([0, 1.1, 2.2, 3.3, 4.4, 0, 1.1, 2.2, 3.3, 4.4]),
                ranges=np.asarray([[0, 5], [5, 10]]),
            ),
            np.asarray([3, 2]),
        ),
        (
            np.asarray([0, 3.3]),
            BinEdges(
                space=Box(0, 5, (2,)),
                edges=np.asarray([0, 1.1, 2.2, 3.3, 4.4, 0, 1.1, 2.2, 3.3, 4.4]),
                ranges=np.asarray([[0, 5], [5, 10]]),
            ),
            np.asarray([0, 3]),
        ),
    ],
)
def test_continuous_to_discrete(sample, bin_edges, result):
    d_sample = _continuous_to_discrete(sample, bin_edges)
    np.array_equal(result, d_sample)


@pytest.mark.parametrize(
    "sample, bin_edges, result",
    [
        (
            np.asarray([3]),
            BinEdges(
                space=Box(0, 5, (1,)),
                edges=np.asarray([0, 1.1, 2.2, 3.3, 4.4]),
                ranges=np.asarray([[0, 5]]),
            ),
            np.asarray([3.3]),
        ),
        (
            np.asarray([0, 3]),
            BinEdges(
                space=Box(0, 5, (2,)),
                edges=np.asarray([0, 1.1, 2.2, 3.3, 4.4, 0, 1.1, 2.2, 3.3, 4.4]),
                ranges=np.asarray([[0, 5], [5, 10]]),
            ),
            np.asarray([0, 3.3]),
        ),
    ],
)
def test_discrete_to_continuous(sample, bin_edges, result):
    c_sample = _discrete_to_continuous(sample, bin_edges)
    np.array_equal(result, c_sample)


def test_generate_box_bins_unsupported_space():
    space = Text(min_length=0, max_length=10)
    with pytest.raises(TypeError):
        generate_box_bins(space, np.linspace, 5)


def test_abstracted_env():
    """Check whether it is possible to construct an abstract env from a
    discretized action space"""
    gym_env = gym.make("Pendulum-v1")
    gym_env = DiscretizeBoxAction(gym_env, 10, np.linspace, use_box_space=False)
    abstraction_mapper = get_abstraction_mapper_to_discrete(gym_env, 5, 5)
    generative_env = verigym.GenerativeEnv.from_gymnasium(gym_env)
    _abstracted_env = verigym.create_abstraction(
        original_env=generative_env,
        abstraction_mapper=abstraction_mapper,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=int(1e5),
    )


def test_njit_sample_to_discrete_values():
    sample = np.asarray([0.5])
    edges = np.asarray([-1, 0, 1, 2])
    ranges = np.asarray([[0, 5]])
    result = _sample_to_discrete_values(sample, edges, ranges)
    assert np.array_equal(result, np.asarray([0]))


def test_njit_sample_to_discrete_idx():
    sample = np.asarray([0.5])
    edges = np.asarray([-1, 0, 1, 2])
    ranges = np.asarray([[0, 5]])
    result = _sample_to_discrete_idx(sample, edges, ranges)
    assert np.array_equal(result, np.asarray([1]))


def test_linmap():
    """Checking functionality of linspace_map."""
    bins_per_dim = [10,1]
    space = Box(low=np.array([-1,0]), high=np.array([1,1]), seed=42)
    abstractionmap = linspace_map(space=space, n_bins=bins_per_dim)

    assert abstractionmap.abstract_n_elements == np.prod(bins_per_dim)
    assert abstractionmap.from_continuous_space
    assert abstractionmap.has_backward_map
    assert abstractionmap.original_space is space
    assert abstractionmap.original_n_elements == float("inf")
    assert isinstance(abstractionmap.abstract_space, MultiDiscrete)
    assert np.array_equal(abstractionmap.abstract_space.nvec, bins_per_dim)

    for i in range(10):
        # test the mapping to abstract space and back, should result in the same value
        abstract_sample = abstractionmap.abstract_space.sample()
        original_sample = abstractionmap.backward_map(abstract_sample)
        abstract_sample_returned = abstractionmap.forward_map(original_sample)
        assert np.isclose(abstract_sample, abstract_sample_returned).all()

    # the lower/upper bounds of the space should map to the first/last bin index
    assert np.array_equal(
        abstractionmap.forward_map(space.low), np.zeros_like(bins_per_dim)
    )
    assert np.array_equal(
        abstractionmap.forward_map(space.high), np.array(bins_per_dim) - 1
    )

    # n_bins with a shape mismatching the space should be rejected rather than
    # silently misinterpreted
    with pytest.raises(AssertionError):
        linspace_map(space=space, n_bins=[10, 1, 5])

def test_linmapper():
    """Checking functionality of linspace_mapper."""
    # make env and replace the spaces
    env, _, _ = make_original_env()
    env.observation_space = Box(low=np.array([-1,0]), high=np.array([1,1]), seed=42)
    env.action_space = Box(low=np.array([-0.5]), high=np.array([0.5]), seed=42)
    bins_per_dim_space = [10,1]
    bins_per_dim_action = 5
    abstractionmapper = linspace_mapper(env, bins_per_dim_space, bins_per_dim_action)

    assert abstractionmapper.from_continuous_states
    assert abstractionmapper.from_continuous_actions
    assert abstractionmapper.original_n_states == float("inf")
    assert abstractionmapper.original_n_actions == float("inf")
    assert abstractionmapper.abstract_n_states == np.prod(bins_per_dim_space)
    assert abstractionmapper.abstract_n_actions == bins_per_dim_action
    assert abstractionmapper._state_abstraction_map.original_space is env.observation_space
    assert abstractionmapper._action_abstraction_map.original_space is env.action_space

    for i in range(5):
        # test the mapping to abstract space and back, should result in the same value
        # state
        abstract_state = abstractionmapper._state_abstraction_map.abstract_space.sample()
        original_state = abstractionmapper.abstract_to_original_state(abstract_state)
        abstract_state_returned = abstractionmapper.original_to_abstract_state(original_state)
        assert np.isclose(abstract_state, abstract_state_returned).all()
        # action
        abstract_action = abstractionmapper._action_abstraction_map.abstract_space.sample()
        original_action = abstractionmapper.abstract_to_original_action(abstract_action)
        abstract_action_returned = abstractionmapper.original_to_abstract_action(original_action)
        assert np.isclose(abstract_action, abstract_action_returned).all()
    