"""Tests for `verigym.abstraction.abstractionmapper`.

Covers `AbstractionMap` and `AbstractionMapper` themselves: 
- the attributes they derive from their spaces, 
- enumeration, 
- `backward_kind`, 
- validation, 
- caching and
- pickling,

plus the `linspace_map` / `linspace_mapper` convenience functions and building an
abstraction with `create_abstraction` end to end.
"""

import copy
import functools
import pickle

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete

import verigym
from verigym.abstraction.abstractionmapper import (
    AbstractionMap,
    AbstractionMapper,
    BackwardKind,
    bin_edges_map,
    enumeration_of_space,
    linspace_map,
    linspace_mapper,
    validate_for_abstraction,
)
from verigym.abstraction.discretization import _ravel, generate_box_bins
from verigym.abstraction.gym_utils.spaces import DummySpace
from verigym.abstraction.learn_abstraction import create_abstraction
from verigym.environments.generativeenv import GenerativeEnv
from verigym.policy.policy import RandomizedPolicy
from verigym.utils.utils import identity_map

from utils import (
    get_abstraction_mapper_to_discrete,
    get_vector,
    int_to_vector,
    make_original_env,
    vector_to_int,
)


# --- basic mapping ------------------------------------------------------------


def test_default_identity_abstraction_mapping():
    """An identity map returns every input unchanged, both forwards and backwards."""
    map = AbstractionMap.initialize_identity_map(gym.spaces.Box(-1, 1))
    abstract_state = 5
    numpy_state = get_vector()
    assert map.original_to_abstract(abstract_state) == abstract_state
    assert map.abstract_to_original(abstract_state) == abstract_state
    assert np.array_equal(map.original_to_abstract(numpy_state), numpy_state)
    assert np.array_equal(map.abstract_to_original(numpy_state), numpy_state)


def test_abstraction_mapping():
    """A map built from arbitrary callables round-trips a sample through the
    abstract space and back."""
    array = get_vector()
    map = AbstractionMap(
        forward_map=vector_to_int, 
        backward_map=functools.partial(int_to_vector, length=array.size),
        original_space=DummySpace(),
        abstract_space=DummySpace(),
    )
    abstract = map.original_to_abstract(array)
    recoveredarray = map.abstract_to_original(abstract)
    assert map.original_to_abstract(recoveredarray) == abstract
    assert np.array_equal(array, recoveredarray)
    

# --- attributes derived from the spaces ---------------------------------------

# each entry holds a space and the attributes an `AbstractionMap` should derive from it: (space, n_elements, is_continuous)
SPACES = [
    (Box(np.array((-1,3)), np.array((5,6))), np.inf, True),
    (Discrete(5), 5, False),
    (MultiDiscrete([2,3,4,5]), 2*3*4*5, False),
    (MultiBinary(4), 2**4, False),
    (DummySpace(), 1, False),
]

@pytest.mark.parametrize(["space", "n_elements", "is_continuous"], SPACES)
def test_abstraction_map_attributes(space, n_elements, is_continuous):
    """`AbstractionMap` derives its element counts and its continuity flag from the two spaces it is given."""
    # the fixed abstract space has a size that none of the `SPACES` shares, so a mix-up of the two spaces would be caught
    map = AbstractionMap(
            forward_map=None,
            backward_map=None,
            original_space=space,
            abstract_space=Discrete(7),
    )
    assert map.from_continuous_space == is_continuous
    assert map.original_n_elements == n_elements
    assert map.abstract_n_elements == 7


@pytest.mark.parametrize(["space", "n_elements", "is_continuous"], SPACES)
def test_abstraction_mapper_attributes(space, n_elements, is_continuous):
    """`AbstractionMapper` exposes the attributes of its state and action map, without mixing the two up."""
    # the parametrized space is the original space of the state map and the abstract space of the action map, so that
    # it is covered in both roles, while the fixed spaces have sizes that none of the `SPACES` shares
    state_map = AbstractionMap(
            forward_map=None,
            backward_map=None,
            original_space=space,
            abstract_space=Discrete(3),
    )
    action_map = AbstractionMap(
            forward_map=None,
            backward_map=None,
            original_space=Discrete(7),
            abstract_space=space,
    )
    mapper = AbstractionMapper(state_abstraction_map=state_map, action_abstraction_map=action_map)
    assert mapper.original_n_states == n_elements
    assert mapper.abstract_n_states == 3
    assert mapper.original_n_actions == 7
    assert mapper.abstract_n_actions == n_elements
    assert mapper.from_continuous_states == is_continuous
    assert mapper.from_continuous_actions is False




# --- convenience functions: linspace_map / linspace_mapper ----------------------------------


def test_linspace_map():
    """`linspace_map` builds a usable map: correct sizes and spaces, a working
    forward/backward round-trip, and a rejected `n_bins` of the wrong shape."""
    bins_per_dim = [10, 1]
    space = Box(low=np.array([-1, 0]), high=np.array([1, 1]), seed=42)
    abstractionmap = linspace_map(space=space, n_bins=bins_per_dim)

    assert abstractionmap.abstract_n_elements == np.prod(bins_per_dim)
    assert abstractionmap.from_continuous_space
    assert abstractionmap.has_backward_map
    assert abstractionmap.original_space is space
    assert abstractionmap.original_n_elements == float("inf")
    assert isinstance(abstractionmap.abstract_space, MultiDiscrete)
    assert np.array_equal(abstractionmap.abstract_space.nvec, bins_per_dim)

    for _ in range(10):
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


def test_linspace_mapper():
    """`linspace_mapper` builds maps for both the state and the action space of an
    environment, without mixing the two up."""
    env, _, _ = make_original_env()
    env.observation_space = Box(low=np.array([-1, 0]), high=np.array([1, 1]), seed=42)
    env.action_space = Box(low=np.array([-0.5]), high=np.array([0.5]), seed=42)
    bins_per_dim_state = [10, 1]
    bins_per_dim_action = 5
    mapper = linspace_mapper(env, bins_per_dim_state, bins_per_dim_action)

    assert mapper.from_continuous_states
    assert mapper.from_continuous_actions
    assert mapper.original_n_states == float("inf")
    assert mapper.original_n_actions == float("inf")
    assert mapper.abstract_n_states == np.prod(bins_per_dim_state)
    assert mapper.abstract_n_actions == bins_per_dim_action
    assert mapper._state_abstraction_map.original_space is env.observation_space
    assert mapper._action_abstraction_map.original_space is env.action_space

    for _ in range(5):
        # state
        abstract_state = mapper._state_abstraction_map.abstract_space.sample()
        original_state = mapper.abstract_to_original_state(abstract_state)
        abstract_state_returned = mapper.original_to_abstract_state(original_state)
        assert np.isclose(abstract_state, abstract_state_returned).all()
        # action
        abstract_action = mapper._action_abstraction_map.abstract_space.sample()
        original_action = mapper.abstract_to_original_action(abstract_action)
        abstract_action_returned = mapper.original_to_abstract_action(original_action)
        assert np.isclose(abstract_action, abstract_action_returned).all()


# --- enumeration ---------------------------------------------------------------


def test_bin_edges_map_matches_orig_to_enum():
    """A map built from `BinEdges` enumerates samples exactly like `BinEdges.orig_to_enum`,
    rather than reimplementing the enumeration."""
    space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), seed=42)
    bin_edges = generate_box_bins(space, np.linspace, [10, 4])
    amap = bin_edges_map(space, bin_edges)
    # not using the linspace_map because we require access to the bin_edges
    # amap = linspace_map(space, [10,4])

    mismatches = 0
    n = 50
    for _ in range(n):
        x = space.sample()
        if bin_edges.orig_to_enum(x) != amap.original_to_enum(x):
            mismatches += 1
    assert mismatches == 0, f"{mismatches}/{n} mismatches between orig_to_enum and original_to_enum"


def test_abstract_to_enum_vs_original_to_enum_take_different_inputs():
    """The two enum functions take different inputs: `original_to_enum` an original
    sample, `abstract_to_enum` an already abstract one. The first is the second
    composed with `forward_map`."""
    space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), seed=1)
    bin_edges = generate_box_bins(space, np.linspace, [5, 3])
    amap = bin_edges_map(space, bin_edges)

    for _ in range(50):
        x = space.sample()  # an ORIGINAL sample, valid input to original_to_enum
        idx = amap.forward_map(x)  # an ABSTRACT sample, valid input to abstract_to_enum
        assert amap.original_to_enum(x) == amap.abstract_to_enum(idx)
        assert amap.original_to_enum(x) == amap.abstract_to_enum(amap.forward_map(x))

    # abstract_to_enum does not call forward_map: feeding it an already-abstract sample
    # (bin indices) must NOT raise, even though it is not a member of `space`.
    zero_idx = np.zeros_like(bin_edges.lengths)
    assert amap.abstract_to_enum(zero_idx) == 0


def test_injected_enum_hooks_are_used_instead_of_ravel():
    """A map can be given its own enumeration functions and uses them instead of the default flattening."""
    # Only 3 of the 9 cells of a nominal MultiDiscrete([3, 3]) are "reachable";
    # the compacted enum skips the rest.
    idx_to_compact = {(0, 0): 0, (0, 1): 1, (2, 2): 2}
    compact_to_idx = {v: np.array(k) for k, v in idx_to_compact.items()}

    def lookup_to_enum(idx):
        return idx_to_compact[tuple(np.atleast_1d(idx).ravel().tolist())]

    def lookup_from_enum(enum):
        return compact_to_idx[enum]

    amap = AbstractionMap(
        forward_map=identity_map,
        backward_map=None,
        original_space=Discrete(3),
        abstract_space=MultiDiscrete([3, 3]),
        abstract_to_enum=lookup_to_enum,
        enum_to_abstract=lookup_from_enum,
    )

    assert amap.is_enumerable
    for enum in range(3):
        idx = amap.enum_to_abstract(enum)
        assert amap.abstract_to_enum(idx) == enum

    # plain ravel_multi_index over [3, 3] would give 8 for (2, 2), not 2 -> so
    # this is proof the hook, not `_ravel`, is what `abstract_to_enum` actually used.
    assert _ravel(np.array([2, 2]), np.array([3, 3])) == 8
    assert amap.abstract_to_enum(np.array([2, 2])) == 2


def test_identity_map_enumeration():
    """An identity map over a `Discrete` space maps every element onto itself."""
    space = Discrete(5)
    amap = AbstractionMap.initialize_identity_map(space)
    assert amap.is_enumerable
    for i in range(5):
        assert amap.original_to_enum(i) == i
        assert amap.abstract_to_enum(i) == i
        assert amap.enum_to_original(i) == i


def test_identity_mapper_enumeration():
    """An identity mapper leaves both states and actions unchanged."""
    state_space = Discrete(4)
    action_space = Discrete(3)
    mapper = AbstractionMapper.initialize_identity_mapper(state_space, action_space)
    for s in range(4):
        assert mapper.abstract_to_original_state(s) == s
        assert mapper.original_to_abstract_state(s) == s
        assert mapper.original_to_abstract_state_enum(s) == s
        assert mapper.abstract_to_original_state_enum(s) == s
    for a in range(3):
        assert mapper.original_to_abstract_action(a) == a
        assert mapper.abstract_to_original_action(a) == a
        assert mapper.original_to_abstract_action_enum(a) == a
        assert mapper.abstract_to_original_action_enum(a) == a


# --- a map that is not a binning -----------------------------------------------


def _cluster_forward(x) -> int:
    """Module-level clustering forward map (stand-in for a real clustering
    abstraction): splits a 1-D Box sample into 2 clusters by sign. Deliberately
    NOT a lambda, and involves no `BinEdges` whatsoever."""
    return int(np.asarray(x).ravel()[0] >= 0)


def _cluster_backward(cluster: int) -> list:
    """Module-level clustering backward map returning a representative set."""
    if cluster == 0:
        return [np.array([-1.0]), np.array([-0.5])]
    return [np.array([0.0]), np.array([0.5])]


def test_non_binning_map_is_generic():
    """`AbstractionMap` also works for an abstraction that is not a binning at all,
    here a clustering by sign."""
    space = Box(low=np.array([-1.0]), high=np.array([1.0]), seed=11)
    to_enum, from_enum = enumeration_of_space(Discrete(2))
    amap = AbstractionMap(
        forward_map=_cluster_forward,
        backward_map=_cluster_backward,
        original_space=space,
        abstract_space=Discrete(2),
        backward_kind="set",
        abstract_to_enum=to_enum,
        enum_to_abstract=from_enum,
    )

    assert amap.backward_kind == BackwardKind.SET
    assert amap.is_enumerable

    for _ in range(20):
        x = space.sample()
        cluster = _cluster_forward(x)
        assert amap.original_to_enum(x) == cluster
        # for a Discrete(2) abstract space (1-dimensional), abstract_to_enum is the identity
        assert amap.abstract_to_enum(cluster) == cluster

    assert amap.enum_to_original(0) == _cluster_backward(0)
    assert amap.enum_to_original(1) == _cluster_backward(1)


# --- backward_kind --------------------------------------------------------------


def test_backward_kind_defaults_and_coercion():
    """`backward_kind` defaults to `point`, falls back to `unknown` without a backward
    map, and rejects unknown values."""
    space = Discrete(3)

    with_backward = AbstractionMap(
        forward_map=identity_map, backward_map=identity_map,
        original_space=space, abstract_space=space,
    )
    assert with_backward.backward_kind == BackwardKind.POINT

    without_backward = AbstractionMap(
        forward_map=identity_map, backward_map=None,
        original_space=space, abstract_space=space,
    )
    assert without_backward.backward_kind == BackwardKind.UNKNOWN

    with pytest.raises(ValueError):
        # incorrect BackwardKind should lead to ValueError
        BackwardKind("bogus")

    assert BackwardKind.INTERVAL == "interval"


def test_bin_edges_map_interval_backward_kind():
    """With `backward_kind="interval"` the backward map returns lower and upper bounds
    instead of a single point."""
    space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), seed=3)
    bin_edges = generate_box_bins(space, np.linspace, [4, 5])
    amap = bin_edges_map(space, bin_edges, backward_kind="interval")

    assert amap.backward_kind == BackwardKind.INTERVAL
    idx = amap.abstract_space.sample()
    interval = amap.backward_map(idx)
    assert interval.shape == (2, *space.shape)


def test_abstract_to_original_raises_without_a_backward_map():
    """Without a backward map, every backward call raises a `ValueError` that names it."""
    space = Discrete(4)
    to_enum, from_enum = enumeration_of_space(space)
    amap = AbstractionMap(
        forward_map=identity_map,
        backward_map=None,
        original_space=space,
        abstract_space=space,
        abstract_to_enum=to_enum,
        enum_to_abstract=from_enum,
    )
    with pytest.raises(ValueError, match="backward_map"):
        amap.abstract_to_original(0)
    with pytest.raises(ValueError, match="backward_map"):
        amap.enum_to_original(0)

    mapper = AbstractionMapper(amap, AbstractionMap.initialize_identity_map(Discrete(2)))
    with pytest.raises(ValueError, match="backward_map"):
        mapper.abstract_to_original_state(0)
        
    with pytest.raises(ValueError, match="backward_map"):
        mapper.abstract_to_original_state_enum(0)


# --- validate_for_abstraction ---------------------------------------------------


def test_validate_for_abstraction_passes_for_good_mapper():
    """A usable mapper passes validation, with and without multithreading."""
    mapper = AbstractionMapper.initialize_identity_mapper(Discrete(5), Discrete(3))
    validate_for_abstraction(mapper)  # must not raise
    validate_for_abstraction(mapper, multithreading=True)  # must not raise


def test_validate_for_abstraction_flags_box_abstract_state_space():
    """A continuous (`Box`) abstract state space cannot be enumerated and is rejected,
    and the error names the state map as the culprit."""
    bad_state_map = AbstractionMap(
        forward_map=identity_map,
        backward_map=None,
        original_space=Box(low=-1.0, high=1.0, shape=(1,), seed=1),
        abstract_space=Box(low=-1.0, high=1.0, shape=(1,), seed=2),
    )
    good_action_map = AbstractionMap.initialize_identity_map(Discrete(3))
    mapper = AbstractionMapper(bad_state_map, good_action_map)

    with pytest.raises(ValueError, match="state"):
        validate_for_abstraction(mapper)


def test_validate_for_abstraction_flags_unpicklable_mapper():
    """A mapper built from lambdas cannot be pickled, so it is rejected only when
    multithreading is requested."""
    space = Discrete(4)
    # A lambda-built map is the exact failure mode the pickling probe exists
    # to catch (mirrors the latent bug in feature_selection.py:81-88).
    to_enum, from_enum = enumeration_of_space(space)
    lambda_state_map = AbstractionMap(
        forward_map=lambda x: x,
        backward_map=lambda x: x,
        original_space=space,
        abstract_space=space,
        abstract_to_enum=to_enum,
        enum_to_abstract=from_enum,
    )
    action_map = AbstractionMap.initialize_identity_map(Discrete(2))
    mapper = AbstractionMapper(lambda_state_map, action_map)

    # single-threaded: passes, since a lambda map is otherwise perfectly usable
    validate_for_abstraction(mapper, multithreading=False)

    with pytest.raises(ValueError, match=r"(?i)pickl"):
        validate_for_abstraction(mapper, multithreading=True)


# --- pickling -------------------------------------------------------------------


def _cache_env():
    """An environment with small, seeded Box spaces, for the caching tests."""
    env, _, _ = make_original_env()
    env.observation_space = Box(low=np.array([-1, 0]), high=np.array([1, 1]), seed=42)
    env.action_space = Box(low=np.array([-0.5]), high=np.array([0.5]), seed=42)
    return env


def test_linspace_mapper_pickles_and_unpickled_cache_is_empty():
    """A mapper survives pickling, still maps correctly afterwards, and comes back with
    empty caches."""
    env = _cache_env()
    mapper = linspace_mapper(env, [10, 1], 5, cache=True)

    state_map = mapper._state_abstraction_map
    action_map = mapper._action_abstraction_map

    # populate the caches
    for _ in range(5):
        mapper.original_to_abstract_state_enum(env.observation_space.sample())
        mapper.original_to_abstract_action_enum(env.action_space.sample())
    assert len(state_map._enum_cache) > 0
    assert len(action_map._enum_cache) > 0

    reloaded = pickle.loads(pickle.dumps(mapper))

    assert reloaded._state_abstraction_map._enum_cache == {}
    assert reloaded._state_abstraction_map._abstract_cache == {}
    assert reloaded._action_abstraction_map._enum_cache == {}
    assert reloaded._action_abstraction_map._abstract_cache == {}
    assert reloaded.abstract_n_states == mapper.abstract_n_states
    assert reloaded.abstract_n_actions == mapper.abstract_n_actions

    # the unpickled mapper must still work correctly
    for _ in range(5):
        x = env.observation_space.sample()
        assert reloaded.original_to_abstract_state_enum(x) == mapper.original_to_abstract_state_enum(x)


def test_deepcopied_mapper_starts_with_a_cold_cache():
    """A deepcopied mapper starts with an empty cache instead of copying a possibly
    large one. `learn_abstraction` deepcopies the mapper once per worker chunk."""
    env = _cache_env()
    mapper = linspace_mapper(env, [10, 1], 5, cache=True)

    for _ in range(5):
        mapper.original_to_abstract_state_enum(env.observation_space.sample())
    assert len(mapper._state_abstraction_map._enum_cache) > 0

    assert copy.deepcopy(mapper)._state_abstraction_map._enum_cache == {}


# --- caching ---------------------------------------------------------------------


def test_cache_disabled_by_default():
    """Without `cache=True` no cache is created at all, not even an empty one."""
    space = Box(low=np.array([-1, 0]), high=np.array([1, 1]), seed=42)
    amap = linspace_map(space, [10, 1])
    assert amap._abstract_cache is None and amap._enum_cache is None
    identity = AbstractionMapper.initialize_identity_mapper(
        Discrete(5), Discrete(3)
    )._state_abstraction_map
    assert identity._abstract_cache is None and identity._enum_cache is None


def test_mapper_cache_flag_forwards_to_both_maps():
    """`cache=True` on an `AbstractionMapper` switches caching on in both of its maps."""
    mapper = linspace_mapper(_cache_env(), [10, 1], 5, cache=True)
    for amap in (mapper._state_abstraction_map, mapper._action_abstraction_map):
        assert amap._abstract_cache == {} and amap._enum_cache == {}

    # the flag on AbstractionMapper itself is a forwarder onto existing maps
    state_map = linspace_map(Box(low=np.array([-1, 0]), high=np.array([1, 1]), seed=1), [4, 4])
    action_map = AbstractionMap.initialize_identity_map(Discrete(3))
    assert state_map._abstract_cache is None
    AbstractionMapper(state_map, action_map, cache=True)
    for amap in (state_map, action_map):
        assert amap._abstract_cache == {} and amap._enum_cache == {}


def test_cached_and_uncached_maps_agree():
    """Caching does not change the mapping itself."""
    space = Box(low=np.array([-1, 0]), high=np.array([1, 1]), seed=7)
    plain = linspace_map(space, [10, 4])
    cached = linspace_map(space, [10, 4], cache=True)

    for _ in range(50):
        x = space.sample()
        assert np.array_equal(cached.original_to_abstract(x), plain.original_to_abstract(x))
        assert cached.original_to_enum(x) == plain.original_to_enum(x)


def test_repeated_input_hits_the_cache():
    """Repeating an input reuses the cached result. The forward and the enum direction
    have separate caches."""
    space = Box(low=np.array([-1, 0]), high=np.array([1, 1]), seed=7)
    cached = linspace_map(space, [10, 4], cache=True)
    x = space.sample()

    first = cached.original_to_abstract(x)
    assert len(cached._abstract_cache) == 1
    for _ in range(10):
        # a hit returns the very same object, and adds no entry
        assert cached.original_to_abstract(x) is first
    assert len(cached._abstract_cache) == 1

    # the enum path has its own cache, filled independently
    assert len(cached._enum_cache) == 0
    enum = cached.original_to_enum(x)
    for _ in range(10):
        assert cached.original_to_enum(x) == enum
    assert len(cached._enum_cache) == 1

    # a distinct sample is a miss
    cached.original_to_abstract(space.sample())
    assert len(cached._abstract_cache) == 2


def test_standalone_map_caches_without_a_mapper():
    """A single `AbstractionMap` can cache on its own, without an `AbstractionMapper`
    around it."""
    amap = AbstractionMap.initialize_identity_map(Discrete(5), cache=True)
    assert amap.original_to_enum(3) == 3
    assert len(amap._enum_cache) == 1
    assert amap.original_to_enum(3) == 3
    assert len(amap._enum_cache) == 1


def test_identity_cache_leaves_the_callers_array_writeable():
    """Caching must not make the caller's own array read-only, which matters when
    `forward_map` returns the array it was given."""
    space = Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), seed=3)
    amap = AbstractionMap(
        forward_map=identity_map,
        backward_map=identity_map,
        original_space=space,
        abstract_space=space,
        cache=True,
    )
    x = space.sample()
    amap.original_to_abstract(x)
    x[0] = 0.5  # must not raise ValueError: assignment destination is read-only


# --- end to end with create_abstraction ----------------------------------------


def test_abstraction_mapping_from_abstraction():
    """
    The mapper used to build an abstraction is stored on the abstracted
    environment and still maps consistently afterwards.
    """
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    abstraction_mapper = get_abstraction_mapper_to_discrete(env, BIN_EDGES_PER_DIM, BIN_EDGES_PER_DIM)
    generative_env = GenerativeEnv.from_gymnasium(env)
    _abstracted_env = create_abstraction(
                original_env=generative_env,
                abstraction_mapper=abstraction_mapper,
                exploration_policy=RandomizedPolicy(generative_env),
                num_steps=NUM_STEPS,
            )
    abstraction_map: AbstractionMapper = _abstracted_env.abstraction_map
    assert abstraction_map is not None
    assert abstraction_map._state_abstraction_map is not None
    init_state, *_ = env.reset()

    init_abstract = abstraction_map.original_to_abstract_state_enum(init_state)
    assert abstraction_map.original_to_abstract_state_enum(init_state) == init_abstract


def test_create_abstraction_end_to_end():
    """A mapper from `linspace_mapper` can be handed straight to `create_abstraction`,
    and every combination of abstract state bins is counted as a state."""
    env = gym.make("MountainCarContinuous-v0")
    mapper = linspace_mapper(env, [10, 4], 3)
    gen = verigym.GenerativeEnv.from_gymnasium(env)

    abstracted = verigym.create_abstraction(
        gen, mapper, RandomizedPolicy(gen), num_steps=1000
    )
    assert abstracted.nr_states == 40
