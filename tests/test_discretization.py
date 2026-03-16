from gymnasium.spaces import Box
import pytest

import numpy as np

from verigym.abstraction.gym_utils.mapping import box_to_discrete
from verigym.abstraction.discretization import generate_box_linspace_bins


@pytest.mark.parametrize(
    "low, high, shape, n_samples",
    [
        (-1, 1, (1,), 10),  # TODO: This currently fails
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
    continuous_sample = to_continuous(discrete_sample)
    back_discrete_sample = to_discrete(continuous_sample)
    assert np.array_equal(back_discrete_sample, discrete_sample)
