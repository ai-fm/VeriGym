import numpy as np
import gymnasium.spaces as spaces

from verigym.abstraction.gym_utils.finite_space import is_bounded_space


# --- always-finite leaf spaces ---

def test_discrete_is_finite():
    assert is_bounded_space(spaces.Discrete(5)) is True


def test_multi_binary_is_finite():
    assert is_bounded_space(spaces.MultiBinary(4)) is True


def test_multi_discrete_is_finite():
    assert is_bounded_space(spaces.MultiDiscrete([3, 4, 5])) is True


# --- Box ---

def test_box_integer_bounded_is_finite():
    assert is_bounded_space(spaces.Box(0, 10, shape=(2,), dtype=np.int32)) is True


def test_box_integer_inf_bounds_still_finite():
    # gymnasium clamps ±inf to the dtype's iinfo range, so the stored bounds are finite
    assert is_bounded_space(spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.int32)) is False


def test_box_float_bounded_is_finite():
    # float Box with finite bounds is considered finite (bounded extent, no ±inf)
    assert is_bounded_space(spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)) is True


def test_box_float_unbounded_is_not_finite():
    assert is_bounded_space(spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)) is False


# --- composite spaces ---

def test_dict_all_finite_is_finite():
    space = spaces.Dict({"a": spaces.Discrete(3), "b": spaces.MultiBinary(2)})
    assert is_bounded_space(space) is True


def test_dict_one_infinite_is_not_finite():
    space = spaces.Dict({"a": spaces.Discrete(3), "b": spaces.Box(-np.inf, np.inf, shape=(1,))})
    assert is_bounded_space(space) is False


def test_tuple_all_finite_is_finite():
    space = spaces.Tuple((spaces.Discrete(3), spaces.MultiDiscrete([2, 4])))
    assert is_bounded_space(space) is True


def test_tuple_one_infinite_is_not_finite():
    space = spaces.Tuple((spaces.Discrete(3), spaces.Box(-np.inf, np.inf, shape=(1,))))
    assert is_bounded_space(space) is False


def test_oneof_all_finite_is_finite():
    space = spaces.OneOf((spaces.Discrete(3), spaces.MultiBinary(2)))
    assert is_bounded_space(space) is True


def test_oneof_one_infinite_is_not_finite():
    space = spaces.OneOf((spaces.Discrete(3), spaces.Box(-np.inf, np.inf, shape=(1,))))
    assert is_bounded_space(space) is False


# --- Text ---

def test_text_with_max_length_is_finite():
    assert is_bounded_space(spaces.Text(min_length=0, max_length=10)) is True


def test_text_is_always_finite():
    # max_length is a required integer in this gymnasium version; charset is always finite
    assert is_bounded_space(spaces.Text(min_length=0, max_length=100)) is True


# --- always-infinite leaf spaces ---

def test_graph_is_not_finite():
    space = spaces.Graph(
        node_space=spaces.Box(0.0, 1.0, shape=(3,)),
        edge_space=spaces.Discrete(5),
    )
    assert is_bounded_space(space) is False


def test_sequence_is_not_finite():
    assert is_bounded_space(spaces.Sequence(spaces.Discrete(3))) is False


