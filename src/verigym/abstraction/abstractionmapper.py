"""`AbstractionMap`, `AbstractionMapper`, space
enumeration, and the convenience functions that build maps and mappers via `BinEdge`.
"""

import functools
import pickle
import math

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from collections.abc import Callable
from enum import StrEnum
from numpy.typing import NDArray

from verigym.abstraction.gym_utils.spaces import get_n_elements_of_space
from verigym.abstraction.discretization import (
    BinEdges,
    _check_compatible,
    _ravel,
    _unravel,
    centered_pow_bin,
    generate_box_bins,
)
from verigym.utils.utils import identity_map
from verigym.abstraction.types import Point, Interval, StateSet, BinEdgeGenFunc


__all__ = [
    "BackwardKind",
    "Point",
    "Interval",
    "StateSet",
    "AbstractionMap",
    "AbstractionMapper",
    "nvec_of_space",
    "enumeration_of_space",
    "validate_for_abstraction",
    "bin_edges_map",
    "binned_map",
    "binned_mapper",
    "linspace_map",
    "linspace_mapper",
    "pow_map",
]


# --- Backwardfunction StrEnum type -------------------------------------------------------------

class BackwardKind(StrEnum): # TODO eventually rename into BackwardType
    """
    Defining the types of objects a backward function can return.

    The shape may differ fundamentally by abstraction and cannot be inferred at
    runtime: a point, an interval and a two-element set can all be `(2, 2)`
    float arrays.

    Overview:

    =============== ================================================ ==========================
    kind            payload                                          example (2-D Box)
    =============== ================================================ ==========================
    `"point"`       one sample of `original_space`                   `array([0.5, 2.5])`
    `"interval"`    `ndarray` shape `(2, *space.shape)`;              `array([[0.5, 2.5],
                    `[0]` = lower, `[1]` = upper                       [1.0, 3.0]])`
    `"set"`         iterable of samples of `original_space`           `[array([0, 1]),
                                                                        array([0, 2])]`
    `"unknown"`     -- no backward map, or undeclared semantics       consumers raise
                                                                       rather than guess
    =============== ================================================ ==========================

    Using a `StrEnum` rather than a string so `BackwardKind.INTERVAL == "interval"`
    stays `True`, while a
    typo raises an error at import time instead of leading to downstream error at runtime.
    
    The corresponding data types corresponding to `BackwardKind.POINT`, `BackwardKind.INTERVAL` and `BackwardKind.SET` are defined in `discretization`:
        type Point    = NDArray                # shape (*space.shape,)
        type Interval = NDArray                # shape (2, *space.shape); [0]=lower, [1]=upper
        type StateSet = Sequence[NDArray]      # iterable of samples of original_space
    """

    POINT = "point"  # one sample of original_space
    INTERVAL = "interval"  # ndarray (2, *space.shape): [0]=lower, [1]=upper
    SET = "set"  # iterable of samples of original_space
    UNKNOWN = "unknown"  # no backward map, or undeclared semantics





def nvec_of_space(space: gym.spaces.Space) -> npt.NDArray:
    """Return the per-dimension cardinality vector of a finite, rectangular space.

    Parameters
    ----------
    space : gym.spaces.Space
        A finite space. `Discrete` and `MultiDiscrete` are supported.

    Returns
    -------
    npt.NDArray
        A flat, 1-D integer array of per-dimension sizes::

            Discrete(n)          -> array([n])
            MultiDiscrete(nvec)  -> np.asarray(nvec).ravel()

        Its product is the number of abstract elements, and it is the array that
        `np.ravel_multi_index` / `np.unravel_index` are taken over. Always 1-D,
        including for `Discrete` -- unlike `BinEdges.nvec`, which is 0-d for a
        `Discrete` original space. Use `BinEdges.lengths` there instead.

    Raises
    ------
    ValueError
        If `space` has no finite rectangular cardinality (e.g. `Box`).
        
    Example
    -------
    TODO
    """
    if isinstance(space, gym.spaces.Discrete):
        return np.array([space.n])
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.asarray(space.nvec).ravel()
    raise ValueError(
        f"nvec_of_space has no finite rectangular cardinality for {type(space) = }; "
        "only Discrete and MultiDiscrete are currently supported."
    )


def enumeration_of_space(
    space: gym.spaces.Discrete | gym.spaces.MultiDiscrete,
) -> tuple[Callable[[NDArray], int], Callable[[int], NDArray]]:
    """Builds enumeration functions (C-order ravel/unravel) for a (Multi-)Discrete space;
    discrete space -> enumeration and back.

    This function is used for creating identity maps. When the constructor notices that the space is discrete, 
    it will automatically create the abstract_to_enum and enum_to_abstract functions.

    Parameters
    ----------
    space : gym.spaces.MultiDiscrete
        A finite space, as accepted by `nvec_of_space`.

    Returns
    -------
    tuple[Callable, Callable]
        `(abstract_to_enum, enum_to_abstract)`, ready to pass to `AbstractionMap`.
        Both are picklable, as required for `multithreading=True`.

    Examples
    --------
    >>> to_enum, from_enum = enumeration_of_space(gym.spaces.MultiDiscrete([3, 4]))
    >>> to_enum(np.array([2, 1]))
    9
    """
    nvec = nvec_of_space(space)
    return functools.partial(_ravel, nvec=nvec), functools.partial(_unravel, nvec=nvec)


# ==============================================================================
# ==============================================================================
# 
# 
# --- ABSTRACTION MAP ----------------------------------------------------------
#
# 
# ==============================================================================
# ==============================================================================


class AbstractionMap:
    """Functions mapping between one original space and one abstract space.

    Usable for either a state space or an action space. Do not confuse with
    `AbstractionMapper`, which holds two maps: one for states and one for
    actions.

    Attributes
    ----------
    original_space : gym.spaces.Space
        The original environment's space.
    abstract_space : gym.spaces.Space
        The abstract environment's space. For a binned map this is
        `MultiDiscrete(bin_edges.lengths)`.
    forward_map : Callable
        `original_space` sample -> `abstract_space` sample.
    backward_map : Callable | None
        `abstract_space` sample -> `original_space` sample. Type is described by `backward_kind` paremeter. 
    backward_kind : BackwardKind
        The type of what `backward_map` returns. See `BackwardKind` class for details (POINT, INTERVAL, SET).
    original_n_elements : int | float | None
        Element count of `original_space`; `float('inf')` for a `Box`.
    abstract_n_elements : int | float | None
        Element count of `abstract_space`; `float('inf')` for a `Box`.
    from_continuous_space : bool | None
        Whether `original_space` is a `gym.spaces.Box`.
    abstract_to_enum : Callable | None
        `abstract_space` sample -> single flat index. `None` if the map was
        built without one, which makes it not `is_enumerable`.
    enum_to_abstract : Callable | None
        Single flat index -> `abstract_space` sample. Inverse of `abstract_to_enum`.

    Notes
    -----
    With `cache=True` the forward direction is memoised in two dicts:
    `_abstract_cache` for `original_to_abstract` and `_enum_cache` for
    `original_to_enum`.
    """

    original_space: gym.spaces.Space
    abstract_space: gym.spaces.Space
    forward_map: Callable
    backward_map: Callable | None
    backward_kind: BackwardKind
    abstract_to_enum: Callable | None
    enum_to_abstract: Callable | None
    original_n_elements: int | float | None
    abstract_n_elements: int | float | None
    from_continuous_space: bool | None
    _abstract_cache: dict | None
    _enum_cache: dict | None

    def __init__(
        self,
        forward_map: Callable[[NDArray], NDArray],
        backward_map: Callable[[NDArray], Point | Interval | StateSet] | None = None,
        original_space: gym.spaces.Space = None,
        abstract_space: gym.spaces.Space = None,
        backward_kind: BackwardKind | str = BackwardKind.POINT,
        abstract_to_enum: Callable[[NDArray], int] | None = None,
        enum_to_abstract: Callable[[int], NDArray] | None = None,
        cache: bool = False,
    ):
        """Build a map between an original and an abstract space.

        Parameters
        ----------
        forward_map : Callable[[NDArray], NDArray]
            Maps a sample of `original_space` to a sample of `abstract_space`.
        backward_map : Callable, optional
            Maps a sample of `abstract_space` to `original_space` in the type determined by
            `backward_kind`. Default `None`, which forces
            `backward_kind = BackwardKind.UNKNOWN`.
        original_space : gym.spaces.Space
            The original gym space.
        abstract_space : gym.spaces.Space
            The abstract gym space.
        backward_kind : BackwardKind | str, default `BackwardKind.POINT`
            "point", "interval", "set" among possible options. Declared type of the `backward_map` function output. See `BackwardKind` for details.
        abstract_to_enum : Callable[[NDArray], int], optional
            Maps a sample from abstract to space to enumerated index. Default `None`.
        enum_to_abstract : Callable[[int], NDArray], optional
            Maps an enumerated index to a sample in the abstract space. Default `None`.
        cache : bool, default False
            Memoise the original -> abstract direction: both
            `original_to_abstract` and `original_to_enum`, each in its own dict.
            See `original_to_abstract` for the contract this places on callers.

        Notes
        -----
        Not supported: a stochastic forward map (returning a distribution over
        abstract states), or a history-dependent abstraction (over the last k
        observations).
        """
        self.forward_map = forward_map
        self.backward_map = backward_map
        self.has_backward_map = self.backward_map is not None

        assert original_space is not None, "original_space should not be None."
        assert abstract_space is not None, "abstract_space should not be None."

        self.original_space = original_space
        self.original_n_elements = get_n_elements_of_space(original_space)

        self.abstract_space = abstract_space
        self.abstract_n_elements = get_n_elements_of_space(abstract_space)

        if isinstance(self.original_space, gym.spaces.Box):
            self.from_continuous_space = True
        else:
            self.from_continuous_space = False

        if backward_map is None:
            backward_kind = BackwardKind.UNKNOWN
        self.backward_kind = BackwardKind(backward_kind)

        self.abstract_to_enum = abstract_to_enum
        self.enum_to_abstract = enum_to_abstract

        # `None` means caching is off; a dict is both the flag and the store.
        # Two of them: `original_to_enum` is not just `original_to_abstract`
        # plus a cheap step, so it needs a cache of its own. See both methods.
        self._abstract_cache = {} if cache else None
        self._enum_cache = {} if cache else None

    def __getstate__(self) -> dict:
        """Return picklable state, with the forward caches emptied. (Also relevant for deepcopy)

        Returns
        -------
        dict
            `self.__dict__` with `_abstract_cache` / `_enum_cache` replaced by
            empty dicts where caching is enabled, so each `multiprocessing.Pool`
            chunk starts cold rather than pickling potentially large dicts.
            `None` is left as is, so a `cache=False` map stays uncached.

        Notes
        -----
        `learn_abstraction` deepcopies the mapper once per worker chunk, and
        `deepcopy` goes through this same protocol. Carrying the caches over
        would not help: a warm cache only serves a worker if the *same* inputs
        recur across the chunk boundary. For a `Box` space they never do (every
        sample is a fresh float vector), and for a discrete space the worker
        rebuilds the whole cache in the first few hundred calls anyway. So the
        copy costs memory per worker and buys close to nothing.
        """
        state = self.__dict__.copy()
        for name in ("_abstract_cache", "_enum_cache"):
            # `None` (caching off) must survive as `None`, not become `{}`.
            if state[name] is not None:
                state[name] = {}
        return state

    @property
    def is_enumerable(self) -> bool:
        """Whether this map can produce a single `int` abstract index.

        Returns
        -------
        bool
            `True` if an `abstract_to_enum` function was given and
            `abstract_space` is finite, i.e. `abstract_n_elements` is not
            `inf`. `False` otherwise (for example for a `Box` abstract space).
        """
        return (self.abstract_to_enum is not None) and (math.isfinite(self.abstract_n_elements))

    def original_to_abstract(self, x: NDArray) -> NDArray:
        """Original sample -> abstract sample. Memoised when `cache=True`.

        The single entry point to `self.forward_map`: every forward mapping goes
        through here, so the cache is defined in exactly one place.

        Parameters
        ----------
        x : NDArray
            A sample of `original_space`.

        Returns
        -------
        NDArray
            A sample of `abstract_space`.

        Notes
        -----
        The key of the cache is the raw to_byte `x`, which ignores shape and dtype.

        Examples
        --------
        >>> space = Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]))
        >>> amap = linspace_map(space, [4, 2], cache=True)
        >>> amap.original_to_abstract(np.array([0.5, 0.25], dtype=np.float32))
        array([2, 0])
        """
        cache = self._abstract_cache
        # `None` rather than an empty dict means caching is off
        if cache is None:
            return self.forward_map(x)
        # `tobytes()` is ~11x cheaper than `tuple(np.atleast_1d(x))`
        key = x.tobytes() if isinstance(x, np.ndarray) else x
        # `try` is faster than both `key in cache` + `cache[key]` and `cache.get(key, sentinel)`
        try:
            return cache[key]
        except KeyError:
            pass

        value = cache[key] = self.forward_map(x)
        return value

    def abstract_to_original(self, a: int | NDArray) -> Point | Interval | StateSet:
        """Abstract sample -> the original sample(s) it stands for.

        The mirror of `original_to_abstract` and the single entry point to
        `self.backward_map`. Not cached: only the forward direction is, since
        the backward direction is not on any hot path.

        Parameters
        ----------
        a : int | NDArray
            A sample of `abstract_space`.

        Returns
        -------
        Point | Interval | StateSet
            Shape determined by `backward_kind`; see `BackwardKind`.

        Raises
        ------
        ValueError
            If the map was initialized without a `backward_map`.

        Examples
        --------
        >>> abstraction_map = linspace_map(Box(low=np.array([-1.0, 0.0]),
        ...                         high=np.array([1.0, 1.0])), [4, 2])
        >>> abstraction_map.abstract_to_original(np.array([2, 0]))
        array([0.33333337, 0.        ], dtype=float32)
        """
        if self.backward_map is None:
            raise ValueError(
                "Cannot map an abstract sample to the original space: this "
                "AbstractionMap was initialized without a backward_map."
            )
        return self.backward_map(a)

    def original_to_enum(self, x: NDArray) -> int:
        """Original sample -> enumeration (single flat abstract index).

        This uses a composition of the functions `self.original_to_abstract()`
        and `self.abstract_to_enum()`, and is memoised when `cache=True`.

        The cache here is separate from the one in `original_to_abstract`.
        Parameters
        ----------
        x : NDArray
            A sample of `original_space`.

        Returns
        -------
        int
            A flat index in `[0, abstract_n_elements)`.

        Raises
        ------
        ValueError
            If the map was initialized without an `abstract_to_enum` function.

        Notes
        -----
        Keyed exactly like `original_to_abstract` (see its Notes). 
        Unlike there, the cached value is an immutable `int`.

        Examples
        --------
        >>> space = Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]))
        >>> abstraction_map = linspace_map(space, [4, 2], cache=True)
        >>> abstraction_map.original_to_enum(np.array([0.5, 0.25], dtype=np.float32))
        4
        """
        if self.abstract_to_enum is None:
            raise ValueError(
                "Cannot map an original sample to an enumeration: this "
                "AbstractionMap was initialized without an abstract_to_enum function."
            )
        cache = self._enum_cache
        if cache is None:
            return self.abstract_to_enum(self.original_to_abstract(x))
        # Same key scheme as `original_to_abstract`
        key = x.tobytes() if isinstance(x, np.ndarray) else x
        try:
            return cache[key]
        except KeyError:
            pass
        # On a miss this fills the forward cache too, via original_to_abstract.
        value = cache[key] = self.abstract_to_enum(self.original_to_abstract(x))
        return value

    def enum_to_original(self, e: int) -> Point | Interval | StateSet:
        """Enumeration index (single flat abstract index) -> original space (according to `self.backward_kind`).

        Parameters
        ----------
        e : int
            A flat index in `[0, abstract_n_elements)`.

        Returns
        -------
        Point | Interval | StateSet
            Shape determined by `backward_kind`; see `BackwardKind`.

        Raises
        ------
        ValueError
            If no backward map is available, or if the map was built without an
            `enum_to_abstract` function.
        """
        if self.enum_to_abstract is None:
            raise ValueError(
                "Cannot map an enumeration to the abstract space: this "
                "AbstractionMap was initialized without an enum_to_abstract function."
            )
        return self.abstract_to_original(self.enum_to_abstract(e))

    @classmethod
    def initialize_identity_map(cls, space: gym.Space, cache: bool = False) -> "AbstractionMap":
        """Creates an identity map. Any input will be returned unchanged.

        Parameters
        ----------
        space : gym.Space
            The space according to which samples will be input and output.
        cache : bool, default False
            Memoise the forward direction; see `AbstractionMap.__init__`.

        Returns
        -------
        AbstractionMap

        Examples
        --------
        >>> env = gym.make('Taxi-v3')
        >>> state_map = AbstractionMap.initialize_identity_map(env.observation_space)
        >>> state_map.forward_map(3)
        3
        """
        try:
            # For discrete spaces we can create enumeration mappings
            abstract_to_enum, enum_to_abstract = enumeration_of_space(space)
        except ValueError:
            # `space` is not discrete, so there
            # is nothing to enumerate: the identity map stays non-enumerable.
            abstract_to_enum, enum_to_abstract = None, None
        return cls(
            forward_map=identity_map,
            backward_map=identity_map,
            original_space=space,
            abstract_space=space,
            backward_kind=BackwardKind.POINT,
            abstract_to_enum=abstract_to_enum,
            enum_to_abstract=enum_to_abstract,
            cache=cache,
        )


# ==============================================================================
# ==============================================================================
# 
# 
# --- ABSTRACTION MAPPER  ------------------------------------------------------
#
# 
# ==============================================================================
# ==============================================================================


class AbstractionMapper:
    """The full abstraction mapping between two environments: states and actions.

    Holds one `AbstractionMap` for the state space and one for the action
    space. Do not confuse with `AbstractionMap`, which only covers a single space.

    Attributes
    ----------
    _state_abstraction_map : AbstractionMap
        The state abstraction map between original and abstract environment.
    _action_abstraction_map : AbstractionMap
        The action abstraction map between original and abstract environment.
    from_continuous_states : bool | None
        Whether the original state space is continuous (`gym.spaces.Box`).
    from_continuous_actions : bool | None
        Whether the original action space is continuous (`gym.spaces.Box`).
    original_n_states, original_n_actions : int | float | None
        Element counts of the original spaces; `float('inf')` if continuous.
    abstract_n_states, abstract_n_actions : int | float | None
        Element counts of the abstract spaces; `float('inf')` if continuous.
    """

    _state_abstraction_map: AbstractionMap
    _action_abstraction_map: AbstractionMap
    from_continuous_states: bool | None
    from_continuous_actions: bool | None
    original_n_states: int | float | None
    original_n_actions: int | float | None
    abstract_n_states: int | float | None
    abstract_n_actions: int | float | None

    def __init__(
        self,
        state_abstraction_map: AbstractionMap,
        action_abstraction_map: AbstractionMap,
        cache: bool = False,
    ):
        """Provides a mapping between abstract and original spaces for an environment (state and action space).

        Parameters
        ----------
        state_abstraction_map : AbstractionMap
            Mapping between original and abstract states.
        action_abstraction_map : AbstractionMap
            Mapping between original and abstract actions.
        cache : bool, default False
            Convenience forwarder: enables memoisation of the original ->
            abstract direction on both maps. The cache itself lives on
            `AbstractionMap` -- see `AbstractionMap.original_to_abstract`.

            Note that this *mutates* the two maps, so a map shared with another
            mapper becomes cached as well. That is sound, since the maps are
            required to be pure.
        """
        self._state_abstraction_map = state_abstraction_map
        self._action_abstraction_map = action_abstraction_map

        self.from_continuous_states = self._state_abstraction_map.from_continuous_space
        self.from_continuous_actions = self._action_abstraction_map.from_continuous_space

        self.original_n_states = state_abstraction_map.original_n_elements
        self.original_n_actions = action_abstraction_map.original_n_elements
        self.abstract_n_states = state_abstraction_map.abstract_n_elements
        self.abstract_n_actions = action_abstraction_map.abstract_n_elements

        # A convenience forwarder only: the caches themselves live on the two
        # maps. Enabling here mutates them, so a map shared with another mapper becomes cached too
        if cache:
            for amap in (self._state_abstraction_map, self._action_abstraction_map):
                amap._abstract_cache = {}
                amap._enum_cache = {}

    @property
    def state_backward_kind(self) -> BackwardKind:
        """The declared payload shape of the state backward map.

        Returns
        -------
        BackwardKind
        """
        return self._state_abstraction_map.backward_kind

    @property
    def action_backward_kind(self) -> BackwardKind:
        """The declared payload shape of the action backward map.

        Returns
        -------
        BackwardKind
        """
        return self._action_abstraction_map.backward_kind

    def original_to_abstract_state(self, orig_state: NDArray) -> NDArray:
        """Maps an original state to its abstract state.

        Parameters
        ----------
        orig_state : NDArray
            A state in the original environment.

        Returns
        -------
        NDArray
            A sample of the abstract state space. Use
            `original_to_abstract_state_enum` when a single `int` is required.
            Memoised when the state map was built with `cache=True`, in which
            case the result must be treated as read-only.
        """
        return self._state_abstraction_map.original_to_abstract(orig_state)

    def original_to_abstract_state_enum(self, orig_state: NDArray) -> int:
        """Map an original state to a single flat abstract state index.

        Parameters
        ----------
        orig_state : NDArray
            A state in the original environment.

        Returns
        -------
        int
            A flat index in `[0, abstract_n_states)`, usable as a `dict` key and
            an array index. Memoised when the state map was built with `cache=True`.
        """
        return self._state_abstraction_map.original_to_enum(orig_state)

    def abstract_to_original_state(self, abs_state: int | NDArray) -> Point | Interval | StateSet:
        """Maps an abstract state to the original state(s) it stands for.

        Parameters
        ----------
        abs_state : int | NDArray
            A state in the abstracted environment.

        Returns
        -------
        Point | Interval | StateSet
            Shape determined by `state_backward_kind`.

        Raises
        ------
        ValueError
            If the state map has no backward map.
        """
        return self._state_abstraction_map.abstract_to_original(abs_state)

    def abstract_to_original_state_enum(self, abs_state: int) -> Point | Interval | StateSet:
        """Map a flat abstract state index to the original state(s) it stands for.

        Parameters
        ----------
        abs_state : int
            A flat index in `[0, abstract_n_states)`.

        Returns
        -------
        Point | Interval | StateSet
            Shape determined by `state_backward_kind`.
        """
        return self._state_abstraction_map.enum_to_original(abs_state)

    def original_to_abstract_action(self, orig_action: NDArray) -> NDArray:
        """Maps an original action to its factored abstract action.

        Parameters
        ----------
        orig_action : NDArray
            An action in the original environment.

        Returns
        -------
        NDArray
            A sample of the abstract action space. Memoised when the action map
            was built with `cache=True`, in which case the result must be
            treated as read-only.
        """
        return self._action_abstraction_map.original_to_abstract(orig_action)

    def original_to_abstract_action_enum(self, orig_action: NDArray) -> int:
        """Map an original action to a single flat abstract action index.

        Parameters
        ----------
        orig_action : NDArray
            An action in the original environment.

        Returns
        -------
        int
            A flat index in `[0, abstract_n_actions)`. Memoised when the action
            map was built with `cache=True`.
        """
        return self._action_abstraction_map.original_to_enum(orig_action)

    def abstract_to_original_action(self, abs_action: int | NDArray) -> Point | Interval | StateSet:
        """Maps an abstract action to a(n) (set/range of) original action(s).

        Parameters
        ----------
        abs_action : int | NDArray
            An action in the abstract environment.

        Returns
        -------
        Point | Interval | StateSet
            Shape determined by `action_backward_kind`.

        Raises
        ------
        ValueError
            If the action map has no backward map.
        """
        return self._action_abstraction_map.abstract_to_original(abs_action)

    def abstract_to_original_action_enum(self, abs_action: int) -> Point | Interval | StateSet:
        """Map a flat abstract action index to the original action(s) it stands for.

        Parameters
        ----------
        abs_action : int
            A flat index in `[0, abstract_n_actions)`.

        Returns
        -------
        Point | Interval | StateSet
            Shape determined by `action_backward_kind`.
        """
        return self._action_abstraction_map.enum_to_original(abs_action)

    @classmethod
    def initialize_identity_mapper(
        cls, state_space: gym.Space, action_space: gym.Space, cache: bool = False
    ) -> "AbstractionMapper":
        """Initialize an `AbstractionMapper` instance that has an "identity map",
        meaning that both the original space and abstract space are the same.

        Parameters
        ----------
        state_space : gym.Space
            The gym space the state space corresponds to.
        action_space : gym.Space
            The gym space the action space corresponds to.
        cache : bool, default False
            Memoise the original -> abstract direction on both maps.

        Returns
        -------
        AbstractionMapper
            The initialized identity `AbstractionMapper`.
        """
        state_abstraction_map = AbstractionMap.initialize_identity_map(state_space, cache=cache)
        action_abstraction_map = AbstractionMap.initialize_identity_map(action_space, cache=cache)

        return cls(state_abstraction_map, action_abstraction_map)


def validate_for_abstraction(
    mapper: AbstractionMapper, *, multithreading: bool = False
) -> None:
    """Checking AbstractionMapper: fail fast, before simulation, naming which map is at fault.

    Meant to run once up front, not per sample.

    Parameters
    ----------
    mapper : AbstractionMapper
        The mapper about to be handed to `create_abstraction`.
    multithreading : bool, keyword-only, default False
        Whether the caller will run `create_abstraction` with a
        `multiprocessing.Pool`. Enables the pickling probe.

    Raises
    ------
    ValueError
        With a message naming the offending map (`"state"` / `"action"`) and
        what is wrong with it.

    Notes
    -----
    Three checks: (1) both maps are enumerable and their abstract element
    counts are finite -- catches a continuous abstract space before it dies far
    downstream; (2) a best-effort smoke test of ~10 samples through
    `original_to_enum`, skipped if the space cannot be sampled; (3) a pickling
    probe, only when `multithreading=True`.

    A `backward_map` is deliberately not required -- only labeling and policy
    deployment need one.
    """
    maps = (
        ("state", mapper._state_abstraction_map, mapper.abstract_n_states),
        ("action", mapper._action_abstraction_map, mapper.abstract_n_actions),
    )

    # 1. Enumerability: a finite abstract space AND a function to enumerate it.
    for name, amap, n in maps:
        if not math.isfinite(n):
            raise ValueError(
                f"The {name} abstraction map is not enumerable: its abstract_space "
                f"({amap.abstract_space!r}) has {n!r} elements, not a finite number "
                "-- a Box abstract space has infinite elements."
            )
        if amap.abstract_to_enum is None:
            raise ValueError(
                f"The {name} abstraction map is not enumerable: it was built without "
                "an abstract_to_enum function. Pass one, e.g. from "
                "enumeration_of_space(abstract_space)."
            )

    # 2. Best-effort smoke test.
    for name, amap, n in maps:
        try:
            samples = [amap.original_space.sample() for _ in range(10)]
        except Exception:
            samples = []
        for sample in samples:
            try:
                e = amap.original_to_enum(sample)
            except Exception as exc:
                raise ValueError(
                    f"The {name} abstraction map's original_to_enum raised on a sample "
                    f"from original_space.sample(): {exc}"
                ) from exc
            if not isinstance(e, (int, np.integer)) or not (0 <= int(e) < n):
                raise ValueError(
                    f"The {name} abstraction map's original_to_enum returned {e!r}, "
                    f"expected an int in [0, {n})."
                )

    # 3. Pickling probe.
    if multithreading:
        try:
            pickle.dumps(mapper)
        except Exception as exc:
            raise ValueError(
                "The mapper failed to pickle, which is required for "
                f"multithreading=True: {exc}"
            ) from exc


# ==============================================================================
# ==============================================================================
# 
# 
# --- CONVENIENCE FUNCTIONS  ---------------------------------------------------
#
# Every convenience function returns a `AbstractionMap` / `AbstractionMapper` 
# object. 
# Forward/backward maps are always (bound) methods of `BinEdges`, never lambdas 
# (a lambda-built mapper will raise `PicklingError` under `multithreading=True`).
# 
# ==============================================================================
# ==============================================================================


def bin_edges_map(
    space: gym.spaces.Space,
    bin_edges: BinEdges,
    *,
    backward_kind: BackwardKind | str = BackwardKind.POINT,
    cache: bool = False,
) -> AbstractionMap:
    """Build an `AbstractionMap` from an existing `BinEdges`.

    Parameters
    ----------
    space : gym.spaces.Space
        The original space. Validated against `bin_edges`.
    bin_edges : BinEdges
        The discretization structure supplying the codecs.
    backward_kind : BackwardKind | str, keyword-only, default `BackwardKind.POINT`
    cache : bool, keyword-only, default False
        Memoise the original -> abstract direction; see `AbstractionMap`.

    Returns
    -------
    AbstractionMap
        `forward_map = bin_edges.orig_to_idx` (factored),
        `abstract_space = MultiDiscrete(bin_edges.lengths)`.

    Notes
    -----
    `abstract_space` is built from `bin_edges.lengths`
    """
    _check_compatible(space, bin_edges)
    backward_kind = BackwardKind(backward_kind)
    if backward_kind is BackwardKind.INTERVAL:
        backward_map = bin_edges.idx_to_interval
    else:
        backward_map = bin_edges.idx_to_orig

    abstract_space = gym.spaces.MultiDiscrete(bin_edges.lengths)

    return AbstractionMap(
        forward_map=bin_edges.orig_to_idx,
        backward_map=backward_map,
        original_space=space,
        abstract_space=abstract_space,
        backward_kind=backward_kind,
        abstract_to_enum=bin_edges.idx_to_enum,
        enum_to_abstract=bin_edges.enum_to_idx,
        cache=cache,
    )


def binned_map(
    space: gym.spaces.Space,
    bin_func: BinEdgeGenFunc,
    n_bins: int | npt.NDArray,
    *,
    backward_kind: BackwardKind | str = BackwardKind.POINT,
    cache: bool = False,
    **bin_kwargs,
) -> AbstractionMap:
    """Build an `AbstractionMap` by binning `space` with `bin_func`.

    Parameters
    ----------
    space : gym.spaces.Space
        The space to discretize.
    bin_func : BinEdgeGenFunc
        `(start, end, n) -> NDArray` of ascending bin boundaries, e.g.
        `np.linspace` or `centered_pow_bin`.
    n_bins : int | array_like
        Bins per dimension; an array must match `space.shape`.
    backward_kind : BackwardKind | str, keyword-only, default `BackwardKind.POINT`
        Passed through to `bin_edges_map`.
    cache : bool, keyword-only, default False
        Passed through to `bin_edges_map`.
    **bin_kwargs
        Extra keyword arguments forwarded to `bin_func` (e.g. `power=3` for
        `centered_pow_bin`).

    Returns
    -------
    AbstractionMap

    Notes
    -----
    `bin_kwargs` is bound to `bin_func` with `functools.partial`, not a lambda
    -- the resulting partial is only used during bin generation and never
    stored on the map, so picklability of the map is unaffected.
    """
    if bin_kwargs:
        bin_func = functools.partial(bin_func, **bin_kwargs)
    bin_edges = generate_box_bins(space, bin_func, n_bins)
    return bin_edges_map(space, bin_edges, backward_kind=backward_kind, cache=cache)


def binned_mapper(
    env: gym.Env,
    bin_func: BinEdgeGenFunc,
    n_bins_states: int | npt.NDArray,
    n_bins_actions: int | npt.NDArray,
    *,
    backward_kind: BackwardKind | str = BackwardKind.POINT,
    cache: bool = False,
    **bin_kwargs,
) -> AbstractionMapper:
    """Build an `AbstractionMapper` by binning both spaces of `env`.

    Parameters
    ----------
    env : gym.Env
        The environment whose `observation_space` and `action_space` are binned.
    bin_func : BinEdgeGenFunc
        `(start, end, n) -> NDArray` of ascending bin boundaries.
    n_bins_states : int | array_like
        Bins per dimension for the state space.
    n_bins_actions : int | array_like
        Bins per dimension for the action space.
    backward_kind : BackwardKind | str, keyword-only, default `BackwardKind.POINT`
        Applied to both maps.
    cache : bool, keyword-only, default False
        Memoise the hot original -> abstract path on both maps.
    **bin_kwargs
        Extra keyword arguments forwarded to `bin_func`.

    Returns
    -------
    AbstractionMapper
    """
    state_map = binned_map(
        env.observation_space,
        bin_func,
        n_bins_states,
        backward_kind=backward_kind,
        cache=cache,
        **bin_kwargs,
    )
    action_map = binned_map(
        env.action_space,
        bin_func,
        n_bins_actions,
        backward_kind=backward_kind,
        cache=cache,
        **bin_kwargs,
    )
    return AbstractionMapper(state_map, action_map)


def linspace_map(
    space: gym.spaces.Space,
    n_bins: int | npt.NDArray,
    *,
    backward_kind: BackwardKind | str = BackwardKind.POINT,
    cache: bool = False,
) -> AbstractionMap:
    """Conveniently create an `AbstractionMap` with equidistant bins per dimension.
    
    The backward kind is point (see `BackwardKind.POINT`).

    Parameters
    ----------
    space : gym.spaces.Space
        The space to be discretized.
    n_bins : int | array_like
        Bins per dimension. If int, each dimension has the same amount of bins.
    cache : bool, keyword-only, default False
        Memoise the original -> abstract direction; see `AbstractionMap`.

    Returns
    -------
    AbstractionMap
        The map from the original space to the abstract space.
    """
    return binned_map(space, np.linspace, n_bins, backward_kind=BackwardKind.POINT, cache=cache)


def linspace_mapper(
    env: gym.Env,
    n_bins_states: int | npt.NDArray,
    n_bins_actions: int | npt.NDArray,
    *,
    cache: bool = False,
) -> AbstractionMapper:
    """Conveniently create an `AbstractionMapper` for a `gym.Env`, providing
    mappings from state and action space to their respective abstract spaces.
    Both spaces are discretized via equidistant binning.

    Note: see `linspace_map` for a mapping of a single space.
    The backward kind is point (see `BackwardKind.POINT`).
    

    Parameters
    ----------
    env : gym.Env
        The environment with original state and action spaces for which to
        create the abstract space and the mappings.
    n_bins_states : int | array_like
        Bins per dimension for the state space.
    n_bins_actions : int | array_like
        Bins per dimension for the action space.
    cache : bool, keyword-only, default False

    Returns
    -------
    AbstractionMapper
        The map from the original environment spaces to the abstract spaces.
    """
    return binned_mapper(
        env,
        np.linspace,
        n_bins_states,
        n_bins_actions,
        backward_kind=BackwardKind.POINT,
        cache=cache,
    )


def pow_map(
    space: gym.spaces.Space,
    n_bins: int | npt.NDArray,
    *,
    power: int = 2,
    cache: bool = False,
) -> AbstractionMap:
    """Build an `AbstractionMap` with polynomially spaced bins.

    Bins are denser near the centre of each dimension's range; see
    `centered_pow_bin`.
    The backward kind is point (see `BackwardKind.POINT`).
    

    Parameters
    ----------
    space : gym.spaces.Space
        The space to discretize.
    n_bins : int | array_like
        Bins per dimension.
    power : int, keyword-only, default 2
        Forwarded to `centered_pow_bin`.
    cache : bool, keyword-only, default False
        Memoise the original -> abstract direction; see `AbstractionMap`.

    Returns
    -------
    AbstractionMap
    """
    return binned_map(
        space,
        centered_pow_bin,
        n_bins,
        backward_kind=BackwardKind.POINT,
        cache=cache,
        power=power,
    )
