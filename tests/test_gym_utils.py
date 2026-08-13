import math

import numpy as np
import pytest
import gymnasium.spaces as spaces

from verigym.abstraction.gym_utils.spaces import get_n_elements_of_space


def test_discrete():
    """Discrete(n) has n elements, returned as a plain int, regardless of `start`."""
    assert get_n_elements_of_space(spaces.Discrete(5)) == 5
    assert isinstance(get_n_elements_of_space(spaces.Discrete(5)), int)
    # a non-zero `start` shifts the value range but not the number of elements
    assert get_n_elements_of_space(spaces.Discrete(5, start=3)) == 5


def test_multi_discrete():
    """MultiDiscrete has the product of its per-dimension `nvec` sizes, for 1D and 2D nvec."""
    assert get_n_elements_of_space(spaces.MultiDiscrete([3, 4, 5])) == 3 * 4 * 5
    assert get_n_elements_of_space(spaces.MultiDiscrete([7])) == 7
    space = spaces.MultiDiscrete(np.array([[2, 3], [4, 5]]))
    assert get_n_elements_of_space(space) == 2 * 3 * 4 * 5


def test_multi_binary():
    """MultiBinary has 2**n elements, whether n is a scalar or a list of dims."""
    assert get_n_elements_of_space(spaces.MultiBinary(4)) == 16
    assert get_n_elements_of_space(spaces.MultiBinary(1)) == 2
    # n as a list gives a matrix of independent bits; total elements is 2**(sum of dims)
    assert get_n_elements_of_space(spaces.MultiBinary([2, 3])) == 2 ** 5


def test_box():
    """Box is a continuous space, so it is always infinite, regardless of bounds or dtype."""
    assert get_n_elements_of_space(spaces.Box(0, 10, shape=(2,), dtype=np.float32)) == math.inf
    assert get_n_elements_of_space(spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)) == math.inf
    assert get_n_elements_of_space(spaces.Box(0, 10, shape=(2,), dtype=np.int32)) == math.inf


@pytest.mark.parametrize("space", [
    spaces.Dict({"a": spaces.Discrete(3), "b": spaces.Discrete(2)}),
    spaces.Tuple((spaces.Discrete(3), spaces.Discrete(2))),
    spaces.OneOf((spaces.Discrete(3), spaces.Discrete(2))),
    spaces.Text(min_length=0, max_length=10),
    spaces.Graph(node_space=spaces.Box(0.0, 1.0, shape=(3,)), edge_space=spaces.Discrete(5)),
    spaces.Sequence(spaces.Discrete(3)),
], ids=["dict", "tuple", "oneof", "text", "graph", "sequence"])
def test_unsupported_space_types_raise(space):
    """Space types with no defined element count (Dict, Tuple, OneOf, Text, Graph, Sequence) raise ValueError."""
    with pytest.raises(ValueError):
        get_n_elements_of_space(space)
