import gymnasium as gym
from gymnasium.spaces import Box
import pytest

import numpy as np

import verigym
from verigym.abstraction.gym_utils.mapping import box_to_discrete
from verigym.abstraction.discretization import generate_box_linspace_bins
from verigym.policy.policy import RandomizedPolicy

from verigym.abstraction.gym_utils.transform_action import DiscretizeBoxAction


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
    assert np.array_equal(to_discrete(to_continuous(discrete_sample)), discrete_sample)


def test_abstracted_env():
    """Check whether it is possible to construct an abstract env from a
    discretized action space"""
    gym_env = gym.make("Pendulum-v1")
    gym_env = DiscretizeBoxAction(gym_env, 10, np.linspace, use_box_space=False)
    gen_env = verigym.GenerativeEnv.from_gymnasium(gym_env)
    abstracted_model = verigym.create_abstraction(
        original_env=gen_env,
        bin_edges_per_dim=5,
        exploration_policy=RandomizedPolicy(gen_env),
        num_steps=int(1e5),
    )
