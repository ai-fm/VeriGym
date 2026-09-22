"""Frozen reference implementations of the pre-refactor abstraction arithmetic.

`verigym.abstraction.utils`, `verigym.abstraction.gym_utils.mapping`, the old
(pre-Step-6) `verigym.abstraction.abstractionmapper.{AbstractionMap,
AbstractionMapper}`, and `CachedDiscretizer` / `forward_mapping` /
`backward_mapping` from `verigym.abstraction.learn_abstraction` were all
deleted in Step 6 of the abstraction-layer refactor, once the new `BinEdges`
codecs (`orig_to_enum`, `enum_to_value`, ...) and the new
`AbstractionMap`/`AbstractionMapper` were proven bit-identical to them.

Everything below is copied *verbatim* (module-level logic unchanged) from
those deleted modules, so that the equivalence guarantee `tests/test_discretization.py`
and `tests/test_legacy_parity.py` provide survives the deletion: the whole
point is to compare the new arithmetic against a frozen snapshot of the OLD
arithmetic that will never change again -- not against the new arithmetic
re-imported under a different name. Do not "clean up" or refactor this file
to match the new API; its only value is being frozen.
"""

from functools import partial

import gymnasium as gym
import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = [
    "legacy_sample_to_discrete",
    "legacy_get_discrete_box_tf",
    "legacy_factored_to_index",
    "legacy_index_to_factored",
    "LegacyAbstractionMap",
    "LegacyAbstractionMapper",
    "LegacyCachedDiscretizer",
    "legacy_forward_mapping",
    "legacy_backward_mapping",
]


# --- from verigym/abstraction/gym_utils/mapping.py (deleted in Step 6) --------


@njit
def _legacy_sample_to_discrete_values(flat_sample, edges, ranges):
    discrete_sample = np.empty(flat_sample.shape)
    for idx, (value, range_) in enumerate(zip(flat_sample, ranges)):
        start, end = range_
        bin_edges = edges[start:end]
        bin_edge_idx = np.searchsorted(bin_edges, value)
        if bin_edges[bin_edge_idx] != value:  # round to the left bin if not exact
            bin_edge_idx -= 1
        discrete_value = bin_edges[bin_edge_idx]
        discrete_sample[idx] = discrete_value
    return discrete_sample


@njit
def _legacy_sample_to_discrete_idx(flat_sample, edges, ranges):
    discrete_sample = np.empty(flat_sample.shape, dtype=np.int64)
    for idx, (value, range_) in enumerate(zip(flat_sample, ranges)):
        start, end = range_
        bin_edges = edges[start:end]
        bin_edge_idx = np.searchsorted(bin_edges, value)
        if bin_edges[bin_edge_idx] != value:  # round to the left bin if not exact
            bin_edge_idx -= 1
        discrete_sample[idx] = bin_edge_idx
    return discrete_sample


def legacy_sample_to_discrete(sample, bin_edges, return_idx: bool = False):
    """Verbatim copy of the deleted `gym_utils.mapping.sample_to_discrete`."""
    if return_idx:
        flat_discrete_sample = _legacy_sample_to_discrete_idx(
            sample.ravel(), bin_edges.edges, bin_edges.ranges
        )
    else:
        flat_discrete_sample = _legacy_sample_to_discrete_values(
            sample.ravel(), bin_edges.edges, bin_edges.ranges
        )
    return flat_discrete_sample.reshape(sample.shape)


def legacy_get_discrete_box_tf(space, bin_edges):
    """Verbatim copy of the deleted `gym_utils.mapping.get_discrete_box_tf`."""
    if space.shape != bin_edges.space.shape:
        raise ValueError(
            "The provided BinEdges are incompatible with the space "
            f"The provided space has shape: {space.shape} and the BinEdges "
            f"were create with a space of shape {bin_edges.space.shape}."
        )
    return partial(legacy_sample_to_discrete, bin_edges=bin_edges, return_idx=False)


# --- from verigym/abstraction/utils.py (deleted in Step 6) --------------------


def legacy_factored_to_index(state: NDArray, bin_edges) -> int:
    """Verbatim copy of the deleted `verigym.abstraction.utils.factored_to_index`.

    Reverses only axis 0 of `state`, which is why it crashes for `ndim > 1`
    (see `test_ndim_gt_1_regression` in `tests/test_discretization.py`).
    """
    d_state = legacy_sample_to_discrete(state, bin_edges, return_idx=True)
    newindex = 0
    for i, pos in enumerate(d_state[::-1]):
        i = i + 1
        newindex += pos * (np.prod(bin_edges.lengths[-i + 1:]) if i != 1 else 1)
    return int(newindex)


def legacy_index_to_factored(state_index: int, bin_edges) -> NDArray:
    """Verbatim copy of the deleted `verigym.abstraction.utils.index_to_factored`."""
    state = np.zeros((len(bin_edges.ranges),))
    index = state_index

    for i in range(len(bin_edges.ranges) - 1, -1, -1):
        dim_size = bin_edges.lengths[i]
        pos = index % dim_size
        state[i] = bin_edges[i][pos]
        index = index // dim_size

    return state


# --- from verigym/abstraction/abstractionmapper.py (pre-Step-6 version) -------


class LegacyAbstractionMap:
    """Verbatim copy of the pre-Step-6 `AbstractionMap`: no `backward_kind`, no
    injectable `abstract_to_enum`/`enum_to_abstract`, no `is_enumerable`."""

    def __init__(
        self,
        forward_map,
        backward_map=None,
        original_space: gym.spaces.Space = None,
        abstract_space: gym.spaces.Space = None,
    ):
        from verigym.abstraction.gym_utils.spaces import get_n_elements_of_space

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


class LegacyAbstractionMapper:
    """Verbatim copy of the pre-Step-6 `AbstractionMapper`: no `*_enum`
    accessors, and `original_to_abstract_state`/`_action` squeeze a size-1
    ndarray to a scalar with `.item()`."""

    def __init__(self, state_abstraction_map, action_abstraction_map):
        self._state_abstraction_map = state_abstraction_map
        self._action_abstraction_map = action_abstraction_map

        self.from_continuous_states = self._state_abstraction_map.from_continuous_space
        self.from_continuous_actions = self._action_abstraction_map.from_continuous_space

        self.original_n_states = state_abstraction_map.original_n_elements
        self.original_n_actions = action_abstraction_map.original_n_elements
        self.abstract_n_states = state_abstraction_map.abstract_n_elements
        self.abstract_n_actions = action_abstraction_map.abstract_n_elements

    def abstract_to_original_state(self, abs_state):
        if self._state_abstraction_map.has_backward_map:
            return self._state_abstraction_map.backward_map(abs_state)
        raise ValueError(
            "Cannot map abstract state to original state without a backward map in the state abstraction."
        )

    def original_to_abstract_state(self, orig_state):
        abs_state = self._state_abstraction_map.forward_map(orig_state)
        if isinstance(abs_state, np.ndarray):
            if abs_state.size == 1:
                abs_state = abs_state.item()
        return abs_state

    def original_to_abstract_action(self, orig_action):
        abs_action = self._action_abstraction_map.forward_map(orig_action)
        if isinstance(abs_action, np.ndarray):
            if abs_action.size == 1:
                abs_action = abs_action.item()
        return abs_action

    def abstract_to_original_action(self, abs_action):
        if self._action_abstraction_map.has_backward_map:
            return self._action_abstraction_map.backward_map(abs_action)
        raise ValueError(
            "Cannot map abstract action to original action without a backward map in the action abstraction."
        )


# --- from verigym/abstraction/learn_abstraction.py (deleted in Step 6) --------


class LegacyCachedDiscretizer:
    """Verbatim copy of the deleted `learn_abstraction.CachedDiscretizer`."""

    def __init__(self, discretizer):
        self.cache = {}
        self.discretizer = discretizer

    def discretize(self, input) -> int:
        arr = np.atleast_1d(input)
        key = tuple(arr)
        if key not in self.cache:
            self.cache[key] = self.discretizer(arr)
        return self.cache[key]


def legacy_forward_mapping(x, to_bins, to_int):
    """Verbatim copy of the deleted `learn_abstraction.forward_mapping`."""
    x = np.atleast_1d(x)
    return to_int(to_bins(x))


def legacy_backward_mapping(x: int, backward_map, space: gym.Space):
    """Verbatim copy of the deleted `learn_abstraction.backward_mapping`."""
    value = np.atleast_1d(backward_map(x))
    if isinstance(space, gym.spaces.Discrete):
        return int(np.rint(value.reshape(-1)[0]))
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.rint(value).astype(space.dtype).reshape(space.shape)
    return value.astype(space.dtype).reshape(space.shape)
