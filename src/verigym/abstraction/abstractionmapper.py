"""`AbstractionMap`, `AbstractionMapper`), space
enumeration, and the convenience functuions that build maps and mappers from a
`BinEdges`.
"""

import functools
import pickle
import math

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from collections.abc import Callable, Sequence
from enum import StrEnum
from numpy.typing import NDArray

from verigym.abstraction.gym_utils.spaces import get_n_elements_of_space
from verigym.abstraction.discretization import (
    BinEdgeGenFunc,
    BinEdges,
    Interval,
    Point,
    StateSet,
    _check_compatible,
    _ravel,
    _unravel,
    centered_pow_bin,
    generate_box_bins,
)
from verigym.utils.utils import identity_map

__all__ = [
    "BackwardKind",
    "Point",
    "Interval",
    "StateSet",
    "AbstractionMap",
    "AbstractionMapper",
    "nvec_of_space",
    "validate_for_abstraction",
    "bin_edges_map",
    "binned_map",
    "binned_mapper",
    "linspace_map",
    "linspace_mapper",
    "pow_map",
]


class BackwardKind(StrEnum):
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
        The shape of what `backward_map` returns. See `BackwardKind` class for details (POINT, INTERVAL, SET).
    original_n_elements : int | float | None
        Element count of `original_space`; `float('inf')` for a `Box`.
    abstract_n_elements : int | float | None
        Element count of `abstract_space`; `float('inf')` for a `Box`.
    from_continuous_space : bool | None
        Whether `original_space` is a `gym.spaces.Box`.
    _to_enum TODO
    _from_enum TODO
    is_enumerable TODO
    """

    original_space: gym.spaces.Space
    abstract_space: gym.spaces.Space
    forward_map: Callable
    backward_map: Callable | None
    backward_kind: BackwardKind
    original_n_elements: int | float | None
    abstract_n_elements: int | float | None
    from_continuous_space: bool | None

    def __init__(
        self,
        forward_map: Callable[[NDArray], NDArray],
        backward_map: Callable[[NDArray], Point | Interval | StateSet] | None = None,
        original_space: gym.spaces.Space = None,
        abstract_space: gym.spaces.Space = None,
        backward_kind: BackwardKind | str = BackwardKind.POINT,
        to_enum: Callable[[NDArray], int] | None = None,
        from_enum: Callable[[int], NDArray] | None = None,
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
        to_enum : Callable[[NDArray], int], optional TODO
            Injected enumeration hook. When `None`, `to_enum` falls back to
            `np.ravel_multi_index` over `nvec_of_space(abstract_space)`. Supply
            a lookup table for an irregular or compacted index.
        from_enum : Callable[[int], NDArray], optional TODO
            Inverse of `to_enum`. When `None`, `from_enum` falls back to
            `np.unravel_index`.

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
        self._to_enum_hook = to_enum
        self._from_enum_hook = from_enum

    @property
    def is_enumerable(self) -> bool:
        """Whether this map can produce a single `int` abstract index.

        Returns
        -------
        bool
            `True` if `abstract_space` is finite, i.e. `abstract_n_elements` is
            not `inf`. `False` otherwise (for example for a `Box`
            abstract space).
        """
        return bool(math.isfinite(self.abstract_n_elements))

    def to_enum(self, idx: NDArray) -> int:
        """Factored abstract sample -> single flat index.

        Parameters
        ----------
        idx : NDArray
            A sample of `abstract_space`.

        Returns
        -------
        int
            A flat index in `[0, abstract_n_elements)`.

        Notes
        -----
        Uses the `to_enum` callable injected at construction if one was given;
        otherwise `np.ravel_multi_index` over `nvec_of_space(self.abstract_space)`.
        No `BinEdges` coupling, so this also works for identity and clustering maps.
        """
        if self._to_enum_hook is not None:
            return self._to_enum_hook(idx)
        return _ravel(idx, nvec_of_space(self.abstract_space))

    def from_enum(self, enum: int) -> NDArray:
        """Single flat index -> factored abstract sample.

        Parameters
        ----------
        enum : int
            A flat index in `[0, abstract_n_elements)`.

        Returns
        -------
        NDArray
            A sample of `abstract_space`. Exact inverse of `to_enum`.
        """
        if self._from_enum_hook is not None:
            return self._from_enum_hook(enum)
        return _unravel(enum, nvec_of_space(self.abstract_space))

    def forward_enum(self, x: NDArray) -> int:
        """Original sample -> single flat abstract index. What the pipeline calls.

        Unlike `forward_map`/`to_enum`, this takes an *original* sample directly:
        `self.to_enum(self.forward_map(x))`. The abstract state is used as a
        `dict` key and an array index (`T_counts[s][a][s_next]`), which a
        factored ndarray cannot serve.

        Parameters
        ----------
        x : NDArray
            A sample of `original_space`.

        Returns
        -------
        int
            A flat index in `[0, abstract_n_elements)`.
        """
        return self.to_enum(self.forward_map(x))

    def backward_enum(self, e: int) -> Point | Interval | StateSet:
        """Single flat abstract index -> backward payload in the original space.

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
            If no backward map is available (`backward_kind is UNKNOWN`).
        """
        if self.backward_kind is BackwardKind.UNKNOWN or self.backward_map is None:
            raise ValueError(
                "Cannot map abstract enum to original space: no backward map is "
                "available for this AbstractionMap (backward_kind is UNKNOWN)."
            )
        return self.backward_map(self.from_enum(e))

    @classmethod
    def initialize_identity_map(cls, space: gym.Space) -> "AbstractionMap":
        """Creates an identity map. Any input will be returned unchanged.

        Parameters
        ----------
        space : gym.Space
            The space according to which samples will be input and output.

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
        return cls(
            forward_map=identity_map,
            backward_map=identity_map,
            original_space=space,
            abstract_space=space,
            backward_kind=BackwardKind.POINT,
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
    _state_enum_cache: dict
    _action_enum_cache: dict

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
            Memoise the original -> enum path in a plain `dict`, keyed on
            `tuple(np.atleast_1d(x))`. Only sound because `forward_map` is
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

        self._cache = cache
        self._state_enum_cache = {}
        self._action_enum_cache = {}

    def __getstate__(self) -> dict:
        """Return picklable state, with the enum caches dropped.

        Returns
        -------
        dict
            `self.__dict__` with `_state_enum_cache` / `_action_enum_cache`
            replaced by empty dicts, so each `multiprocessing.Pool` chunk starts
            cold rather than pickling a potentially large dict.
        """
        state = self.__dict__.copy()
        state["_state_enum_cache"] = {}
        state["_action_enum_cache"] = {}
        return state

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
        """
        return self._state_abstraction_map.forward_map(orig_state)

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
            an array index. Memoised when the mapper was built with `cache=True`.
        """
        if not self._cache:
            return self._state_abstraction_map.forward_enum(orig_state)
        key = tuple(np.atleast_1d(orig_state))
        if key not in self._state_enum_cache:
            self._state_enum_cache[key] = self._state_abstraction_map.forward_enum(orig_state)
        return self._state_enum_cache[key]

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
        if not self._state_abstraction_map.has_backward_map:
            raise ValueError(
                "Cannot map abstract state to original state without a backward "
                "map in the state abstraction."
            )
        return self._state_abstraction_map.backward_map(abs_state)

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
        return self._state_abstraction_map.backward_enum(abs_state)

    def original_to_abstract_action(self, orig_action: NDArray) -> NDArray:
        """Maps an original action to its factored abstract action.

        Parameters
        ----------
        orig_action : NDArray
            An action in the original environment.

        Returns
        -------
        NDArray
            A sample of the abstract action space.
        """
        return self._action_abstraction_map.forward_map(orig_action)

    def original_to_abstract_action_enum(self, orig_action: NDArray) -> int:
        """Map an original action to a single flat abstract action index.

        Parameters
        ----------
        orig_action : NDArray
            An action in the original environment.

        Returns
        -------
        int
            A flat index in `[0, abstract_n_actions)`. Memoised when the mapper
            was built with `cache=True`.
        """
        if not self._cache:
            return self._action_abstraction_map.forward_enum(orig_action)
        key = tuple(np.atleast_1d(orig_action))
        if key not in self._action_enum_cache:
            self._action_enum_cache[key] = self._action_abstraction_map.forward_enum(orig_action)
        return self._action_enum_cache[key]

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
        if not self._action_abstraction_map.has_backward_map:
            raise ValueError(
                "Cannot map abstract action to original action without a backward "
                "map in the action abstraction."
            )
        return self._action_abstraction_map.backward_map(abs_action)

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
        return self._action_abstraction_map.backward_enum(abs_action)

    @classmethod
    def initialize_identity_mapper(
        cls, state_space: gym.Space, action_space: gym.Space
    ) -> "AbstractionMapper":
        """Initialize an `AbstractionMapper` instance that has an "identity map",
        meaning that both the original space and abstract space are the same.

        Parameters
        ----------
        state_space : gym.Space
            The gym space the state space corresponds to.
        action_space : gym.Space
            The gym space the action space corresponds to.

        Returns
        -------
        AbstractionMapper
            The initialized identity `AbstractionMapper`.
        """
        state_abstraction_map = AbstractionMap.initialize_identity_map(state_space)
        action_abstraction_map = AbstractionMap.initialize_identity_map(action_space)

        return cls(state_abstraction_map, action_abstraction_map)


def validate_for_abstraction(
    mapper: AbstractionMapper, *, multithreading: bool = False
) -> None:
    """Fail fast, before simulation, naming which map is at fault.

    Meant to run once up front, never per sample.

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
    `forward_enum`, skipped if the space cannot be sampled; (3) a pickling
    probe, only when `multithreading=True`.

    A `backward_map` is deliberately not required -- only labeling and policy
    deployment need one.
    """
    maps = (
        ("state", mapper._state_abstraction_map, mapper.abstract_n_states),
        ("action", mapper._action_abstraction_map, mapper.abstract_n_actions),
    )

    # 1. Enumerability, i.e. a finite abstract space.
    for name, amap, n in maps:
        if not amap.is_enumerable:
            raise ValueError(
                f"The {name} abstraction map is not enumerable: its abstract_space "
                f"({amap.abstract_space!r}) has {n!r} elements, not a finite number "
                "-- a Box abstract space has infinite elements."
            )

    # 2. Best-effort smoke test.
    for name, amap, n in maps:
        try:
            samples = [amap.original_space.sample() for _ in range(10)]
        except Exception:
            samples = []
        for sample in samples:
            try:
                e = amap.forward_enum(sample)
            except Exception as exc:
                raise ValueError(
                    f"The {name} abstraction map's forward_enum raised on a sample "
                    f"from original_space.sample(): {exc}"
                ) from exc
            if not isinstance(e, (int, np.integer)) or not (0 <= int(e) < n):
                raise ValueError(
                    f"The {name} abstraction map's forward_enum returned {e!r}, "
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


# --- factories ---------------------------------------------------------------
#
# Every factory returns a plain `AbstractionMap` / `AbstractionMapper`, never a
# binning-specific subclass. Forward/backward maps are always bound methods of
# `BinEdges`, never lambdas -- a lambda-built mapper raises `PicklingError`
# under `multithreading=True`.


def bin_edges_map(
    space: gym.spaces.Space,
    bin_edges: BinEdges,
    *,
    backward_kind: BackwardKind | str = BackwardKind.POINT,
) -> AbstractionMap:
    """Build an `AbstractionMap` from an existing `BinEdges`.

    Parameters
    ----------
    space : gym.spaces.Space
        The original space. Validated against `bin_edges`.
    bin_edges : BinEdges
        The discretization structure supplying the codecs.
    backward_kind : BackwardKind | str, keyword-only, default `BackwardKind.POINT`
        Selects the backward codec and tags the map, from one parameter::

            POINT    -> bin_edges.idx_to_orig
            INTERVAL -> bin_edges.idx_to_interval

    Returns
    -------
    AbstractionMap
        `forward_map = bin_edges.orig_to_idx` (factored),
        `abstract_space = MultiDiscrete(bin_edges.lengths)`.

    Notes
    -----
    `abstract_space` is built from `bin_edges.lengths`, never from
    `bin_edges.nvec`: for a `Discrete` original space `nvec` is 0-d.

    No `cache` parameter here on purpose: a single `AbstractionMap` has nowhere
    to put a cache (the memoising dicts live on `AbstractionMapper`, which
    pairs a state and an action map). Build with `binned_mapper` /
    `linspace_mapper` / `AbstractionMapper(..., cache=True)` instead.
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
    )


def binned_map(
    space: gym.spaces.Space,
    bin_func: BinEdgeGenFunc,
    n_bins: int | npt.NDArray,
    *,
    backward_kind: BackwardKind | str = BackwardKind.POINT,
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
    stored on the map, so picklability of the map is unaffected. No `cache`
    parameter, for the same reason as `bin_edges_map`.
    """
    if bin_kwargs:
        bin_func = functools.partial(bin_func, **bin_kwargs)
    bin_edges = generate_box_bins(space, bin_func, n_bins)
    return bin_edges_map(space, bin_edges, backward_kind=backward_kind)


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
        Memoise the hot original -> enum path on the returned mapper.
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
        **bin_kwargs,
    )
    action_map = binned_map(
        env.action_space,
        bin_func,
        n_bins_actions,
        backward_kind=backward_kind,
        **bin_kwargs,
    )
    return AbstractionMapper(state_map, action_map, cache=cache)


def linspace_map(
    space: gym.spaces.Space,
    n_bins: int | npt.NDArray,
    *,
    backward_kind: BackwardKind | str = BackwardKind.POINT,
) -> AbstractionMap:
    """Conveniently create an `AbstractionMap` with equidistant bins per dimension.

    Parameters
    ----------
    space : gym.spaces.Space
        The space to be discretized.
    n_bins : int | array_like
        Bins per dimension. If int, each dimension has the same amount of bins.
    backward_kind : BackwardKind | str, keyword-only, default `BackwardKind.POINT`

    Returns
    -------
    AbstractionMap
        The map from the original space to the abstract space.
    """
    return binned_map(space, np.linspace, n_bins, backward_kind=backward_kind)


def linspace_mapper(
    env: gym.Env,
    n_bins_states: int | npt.NDArray,
    n_bins_actions: int | npt.NDArray,
    *,
    backward_kind: BackwardKind | str = BackwardKind.POINT,
    cache: bool = False,
) -> AbstractionMapper:
    """Conveniently create an `AbstractionMapper` for a `gym.Env`, providing
    mappings from state and action space to their respective abstract spaces.
    Both spaces are discretized via equidistant binning.

    Note: see `linspace_map` for a mapping of a single space.

    Parameters
    ----------
    env : gym.Env
        The environment with original state and action spaces for which to
        create the abstract space and the mappings.
    n_bins_states : int | array_like
        Bins per dimension for the state space.
    n_bins_actions : int | array_like
        Bins per dimension for the action space.
    backward_kind : BackwardKind | str, keyword-only, default `BackwardKind.POINT`
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
        backward_kind=backward_kind,
        cache=cache,
    )


def pow_map(
    space: gym.spaces.Space,
    n_bins: int | npt.NDArray,
    *,
    power: int = 2,
    backward_kind: BackwardKind | str = BackwardKind.POINT,
) -> AbstractionMap:
    """Build an `AbstractionMap` with polynomially spaced bins.

    Bins are denser near the centre of each dimension's range; see
    `centered_pow_bin`.

    Parameters
    ----------
    space : gym.spaces.Space
        The space to discretize.
    n_bins : int | array_like
        Bins per dimension.
    power : int, keyword-only, default 2
        Forwarded to `centered_pow_bin`.
    backward_kind : BackwardKind | str, keyword-only, default `BackwardKind.POINT`

    Returns
    -------
    AbstractionMap
    """
    return binned_map(
        space,
        centered_pow_bin,
        n_bins,
        backward_kind=backward_kind,
        power=power,
    )
