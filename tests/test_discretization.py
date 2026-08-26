import gymnasium as gym
from gymnasium.spaces import Box, Text
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
)
from verigym.policy.policy import RandomizedPolicy

from verigym.abstraction.gym_utils.transform_action import DiscretizeBoxAction

from utils import get_abstraction_mapper_to_discrete


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
