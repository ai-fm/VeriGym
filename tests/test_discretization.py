"""
Tests for `verigym.abstraction.discretization`.
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Text

import verigym
from verigym.abstraction.discretization import (
    _check_compatible,
    _orig_to_idx_njit,
    _orig_to_value_njit,
    _ravel,
    _unravel,
    centered_pow_bin,
    generate_box_bins,
)
from verigym.abstraction.gym_utils.transform_action import DiscretizeBoxAction
from verigym.policy.policy import RandomizedPolicy

from utils import get_abstraction_mapper_to_discrete


# --- round-trips --------------------------------------------------------------


@pytest.mark.parametrize(
    "space, n_bins",
    [
        (Box(low=-1.0, high=1.0, shape=(1,), seed=1), 5),
        (Box(low=-1.0, high=1.0, shape=(2,), seed=2), 7),
        (Box(low=-1.0, high=1.0, shape=(2, 2), seed=3), 4),
        (Box(low=-1.0, high=1.0, shape=(2, 3), seed=4), np.array([[3, 4, 5], [6, 7, 8]])),
    ],
)
def test_roundtrip_enum_idx(space, n_bins):
    """Converting a flat enum to bin indices and back returns the original enum."""
    be = generate_box_bins(space, np.linspace, n_bins)
    total = int(np.prod(be.lengths))
    for i in range(total):
        assert be.idx_to_enum(be.enum_to_idx(i)) == i


@pytest.mark.parametrize(
    "space, n_bins",
    [
        (Box(low=-1.0, high=1.0, shape=(1,), seed=1), 5),
        (Box(low=-1.0, high=1.0, shape=(2,), seed=2), 7),
        (Box(low=-1.0, high=1.0, shape=(2, 2), seed=3), 4),
    ],
)
def test_orig_to_value_matches_idx_to_value(space, n_bins):
    """Snapping a sample directly gives the same value as going through its bin index."""
    be = generate_box_bins(space, np.linspace, n_bins)
    for _ in range(20):
        x = space.sample()
        assert np.allclose(be.orig_to_value(x), be.idx_to_value(be.orig_to_idx(x)))


# --- ndim > 1 ------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(2, 2), (2, 3, 4)],
)
def test_ndim_greater_1(shape):
    """For a space with more than one dimension, the conversions keep the shape of the
    space and the two ways of reaching a representation agree: going straight from
    the original sample, or via its enum."""
    space = Box(low=-1.0, high=1.0, shape=shape, seed=7)
    be = generate_box_bins(space, np.linspace, 4)

    x = space.sample()

    e = be.orig_to_enum(x)
    assert isinstance(e, int)

    idx = be.orig_to_idx(x)
    assert idx.shape == shape  # the factored index keeps the shape of the space
    assert be.enum_to_idx(e).tolist() == idx.tolist()

    assert np.allclose(be.enum_to_value(e), be.orig_to_value(x))


# --- Discrete / MultiDiscrete spaces -------------------------------------------


def test_discrete_space_codecs_and_lengths_flat():
    """
    All conversions work for a `Discrete` space, where `nvec` is 0-d but `lengths` stays 1-D.
    
    Reminder:
        a = np.array(5)      # 0-d: shape (),   ndim 0, a scalar, a single element
        b = np.array([5])    # 1-d: shape (1,), ndim 1

    """
    space = Discrete(5)
    be = generate_box_bins(space, np.linspace, 5)

    # lengths is always 1-D, nvec is 0-d for Discrete
    assert be.lengths.shape == (1,)
    assert be.nvec.shape == ()

    for x in range(5):
        idx = be.orig_to_idx(x)
        assert idx.shape == ()  # == space.shape
        e = be.orig_to_enum(x)
        assert isinstance(e, int)
        assert be.enum_to_idx(e).shape == ()
        orig = be.enum_to_orig(e)
        assert isinstance(orig, int)
        assert space.contains(orig)

    # 0-d nvec must not leak into ravel/unravel: lengths (not nvec) is what
    # _ravel/_unravel must be called with for this to work at all.
    assert _ravel(np.array([2]), be.lengths) == 2
    assert np.array_equal(_unravel(2, be.lengths), np.array([2]))


def test_multidiscrete_space_codecs():
    """All conversions work for a `MultiDiscrete` space."""
    space = MultiDiscrete([4, 3])
    be = generate_box_bins(space, np.linspace, np.array([4, 3]))

    assert be.lengths.shape == (2,)
    assert be.nvec.shape == (2,)

    x = space.sample()
    idx = be.orig_to_idx(x)
    assert idx.shape == space.shape
    e = be.orig_to_enum(x)
    assert be.enum_to_idx(e).tolist() == idx.tolist()
    orig = be.enum_to_orig(e)
    assert space.contains(orig)


# --- idx_to_interval ------------------------------------------------------------


@pytest.mark.parametrize(
    "space, n_bins",
    [
        (Box(low=0.0, high=10.0, shape=(1,), seed=1), 5),
        (Box(low=-1.0, high=1.0, shape=(2,), seed=2), 4),
        (Box(low=-1.0, high=1.0, shape=(2, 2), seed=3), 3),
    ],
)
def test_idx_to_interval_shape_and_contiguity(space, n_bins):
    """
    Testing the interval mapping: Every bin index maps to an interval. 
    The intervals should touch each other (contiguous) and together cover the whole space.
    """
    be = generate_box_bins(space, np.linspace, n_bins)
    total = int(np.prod(be.lengths))

    intervals = []
    for e in range(total):
        idx = be.enum_to_idx(e)
        interval = be.idx_to_interval(idx)
        assert interval.shape == (2, *space.shape)
        assert np.all(interval[0] <= interval[1])  # lower <= upper everywhere
        intervals.append((idx.copy(), interval))

    # per dimension, sorted intervals must be contiguous and cover the space
    for dim in range(len(be.lengths)):
        # gather a set of (lower, upper) for this flattened dimension, sorted by idx
        pairs = sorted(
            {
                # (idx[dim], upper[dim], lower[dim])
                (int(np.atleast_1d(idx).ravel()[dim]), float(np.atleast_1d(interval[0]).ravel()[dim]), float(np.atleast_1d(interval[1]).ravel()[dim]))
                for idx, interval in intervals
            }
        )
        lows = [p[1] for p in pairs]
        highs = [p[2] for p in pairs]
        # contiguous: each interval's upper == next interval's lower
        for k in range(len(pairs) - 1):
            assert np.isclose(highs[k], lows[k + 1])
        # covers the outer edges of the space
        edges_dim = be[dim]
        assert np.isclose(lows[0], edges_dim[0])
        assert np.isclose(highs[-1], edges_dim[-1])


def test_idx_to_interval_degenerate_top_interval():
    """
    The top bin INTERVAL of a dimension is a single point `[high, high]`, because `n_bins`
    counts representative values, not cells. Here: edges `[0, 2.5, 5, 7.5, 10]`, so
    index 4 is reachable only by exactly 10.0.
    TODO: Double check this test with the changes in PR #189 and #193 
    """
    space = Box(low=np.array([0.0]), high=np.array([10.0]))
    be = generate_box_bins(space, np.linspace, 5)

    assert np.allclose(be.edges, [0.0, 2.5, 5.0, 7.5, 10.0])

    interval = be.idx_to_interval(np.array([4]))
    assert np.allclose(interval[0], [10.0])
    assert np.allclose(interval[1], [10.0])

    # non-degenerate intervals for the other indices
    interval0 = be.idx_to_interval(np.array([0]))
    assert np.allclose(interval0[0], [0.0])
    assert np.allclose(interval0[1], [2.5])


# --- value_to_orig --------------------------------------------------------------

@pytest.mark.parametrize(
    'space', [
        # Box(low=-1.0, high=1.0, shape=(2,), seed=5, dtype=np.float16),  # incompatible with numba JIT
        Box(low=-1.0, high=1.0, shape=(2,), seed=5, dtype=np.float32),
        Box(low=-1.0, high=1.0, shape=(2,), seed=5, dtype=np.float64),
        # Box(low=-1.0, high=1.0, shape=(2,), seed=5, dtype=np.half),  # incompatible with numba JIT
        Box(low=-1.0, high=1.0, shape=(2,), seed=5, dtype=np.single),
        Box(low=-1.0, high=1.0, shape=(2,), seed=5, dtype=np.double),
        Box(low=-1.0, high=1.0, shape=(2,), seed=5, dtype=np.longdouble),
        Box(low=-1.0, high=1.0, shape=(2,), seed=5, dtype=np.int8),
        Box(low=-1.0, high=1.0, shape=(2,), seed=5, dtype=np.int16),
        Box(low=-1.0, high=1.0, shape=(2,), seed=5, dtype=np.int32),
        Box(low=-1.0, high=1.0, shape=(2,), seed=5, dtype=np.int64),
        Box(low=0, high=1.0, shape=(2,), seed=5, dtype=np.uint8),
        Box(low=0, high=1.0, shape=(2,), seed=5, dtype=np.uint16),
        Box(low=0, high=1.0, shape=(2,), seed=5, dtype=np.uint32),
        Box(low=0, high=1.0, shape=(2,), seed=5, dtype=np.uint64),
        Box(low=0, high=1.0, shape=(2,), seed=5, dtype=np.longlong),
        Box(low=0, high=1.0, shape=(2,), seed=5, dtype=np.ulonglong),
    ]
)
def test_value_to_orig_box(space):
    """Converting a snapped value back to a `Box` sample keeps the space's shape and dtype.
    There are limitaions by numba's JIT compilation supported datatypes: e.g. np.half/np.float16 are not supported."""
    # space = Box(low=-1.0, high=1.0, shape=(2,), seed=5)
    be = generate_box_bins(space, np.linspace, 4)
    x = space.sample()
    v = be.orig_to_value(x)
    orig = be.value_to_orig(v)
    assert orig.shape == space.shape
    assert orig.dtype == space.dtype


def test_value_to_orig_discrete():
    """Converting a snapped value back to a `Discrete` sample gives a plain `int`."""
    space = Discrete(5)
    be = generate_box_bins(space, np.linspace, 5)
    orig = be.value_to_orig(np.array([3.0]))
    assert isinstance(orig, int)
    assert space.contains(orig)


def test_value_to_orig_multidiscrete():
    """Converting a snapped value back gives a valid `MultiDiscrete` sample."""
    space = MultiDiscrete([4, 3])
    be = generate_box_bins(space, np.linspace, np.array([4, 3]))
    orig = be.value_to_orig(np.array([2.0, 1.0]))
    assert space.contains(orig)


# --- bounds -----------------------------------------------------------------


@pytest.mark.parametrize(
    "space, n_bins",
    [
        (Box(low=-1.0, high=1.0, shape=(1,), seed=1), 5),
        (Box(low=-1.0, high=1.0, shape=(3,), seed=2), 6),
        (Box(low=-1.0, high=1.0, shape=(2, 2), seed=3), 4),
    ],
)
def test_bounds_map_to_first_and_last_index(space, n_bins):
    """The lowest and the highest sample of a space land in the first and the last bin."""
    be = generate_box_bins(space, np.linspace, n_bins)
    lengths = be.lengths

    idx_low = be.orig_to_idx(space.low)
    idx_high = be.orig_to_idx(space.high)

    assert np.array_equal(np.atleast_1d(idx_low).ravel(), np.zeros_like(lengths))
    assert np.array_equal(np.atleast_1d(idx_high).ravel(), lengths - 1)


# --- generate_box_bins / centered_pow_bin --------------------------------------


def test_generate_box_bins_unsupported_space():
    """An unsupported space type (here `Text`) is rejected with a `TypeError`."""
    space = Text(min_length=0, max_length=10)
    with pytest.raises(TypeError):
        generate_box_bins(space, np.linspace, 5)


def test_centered_pow_bin_endpoints_and_sorted():
    """`centered_pow_bin` returns sorted edges from `low` to `high`, packed more densely
    around the centre than `np.linspace`."""
    b = centered_pow_bin(0, 10, 9, power=2)
    assert b.shape == (9,)
    assert np.isclose(b[0], 0)
    assert np.isclose(b[-1], 10)
    assert np.all(np.diff(b) >= 0)
    # denser near the centre than linspace
    lin = np.linspace(0, 10, 9)
    mid = len(b) // 2
    assert (b[mid + 1] - b[mid]) < (lin[mid + 1] - lin[mid]) or np.isclose(
        b[mid + 1] - b[mid], lin[mid + 1] - lin[mid]
    )


# --- _check_compatible ----------------------------------------------------------


def test_check_compatible_raises_on_shape_mismatch():
    """Using bin edges together with a differently shaped space raises a `ValueError`."""
    space1 = Box(low=-1.0, high=1.0, shape=(2,))
    space2 = Box(low=-1.0, high=1.0, shape=(3,))
    be = generate_box_bins(space1, np.linspace, 4)
    with pytest.raises(ValueError):
        _check_compatible(space2, be)


def test_check_compatible_passes_on_shape_match():
    """Bin edges and a space of matching shape pass the compatibility check."""
    space1 = Box(low=-1.0, high=1.0, shape=(2,))
    space2 = Box(low=-2.0, high=2.0, shape=(2,))
    be = generate_box_bins(space1, np.linspace, 4)
    _check_compatible(space2, be)  # must not raise


# --- __getitem__ ------------------------------------------------------------


def test_getitem_returns_bin_edges_slice():
    """`bin_edges[i]` returns the edges belonging to dimension `i`."""
    space = Box(low=-1.0, high=1.0, shape=(2,))
    be = generate_box_bins(space, np.linspace, 4)
    
    # `edges` is ONE flat array holding every dimension's edges, and
    # `ranges` says where each dimension starts and ends in it. 
    # Here: 2 dimensions with 4 bins each, so edges has 8 entries and ranges == [[0, 4], [4, 8]].
    # `be[i]` is shorthand for that slice, so each assert compares the shorthand
    # against the slice written out by hand.
    for dim in range(2):
        start, end = be.ranges[dim]
        assert np.array_equal(be[dim], be.edges[start:end])



# --- round-trips over whole spaces, njit kernels, integration ------------------


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
def test_bijectivity_idx_value_roundtrip(low, high, shape, n_samples):
    """Turning a bin index into a value and back returns the same index, for spaces of
    any shape."""
    space = Box(low, high, shape, seed=42)
    be = generate_box_bins(space, np.linspace, n_samples)
    idx = gym.spaces.MultiDiscrete(be.lengths).sample().reshape(shape)
    assert np.array_equal(be.orig_to_idx(be.idx_to_value(idx)), idx)


def test_abstracted_env():
    """An abstract environment can be built from an environment with a discretized
    action space."""
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


def test_njit_orig_to_value():
    """The numba kernel snaps a sample onto the bin edge below it."""
    sample = np.asarray([0.5])
    edges = np.asarray([-1, 0, 1, 2])
    ranges = np.asarray([[0, 5]])
    result = _orig_to_value_njit(sample, edges, ranges)
    assert np.array_equal(result, np.asarray([0]))


def test_njit_orig_to_idx():
    """The numba kernel returns the index of the bin a sample falls into."""
    sample = np.asarray([0.5])
    edges = np.asarray([-1, 0, 1, 2])
    ranges = np.asarray([[0, 5]])
    result = _orig_to_idx_njit(sample, edges, ranges)
    assert np.array_equal(result, np.asarray([1]))
