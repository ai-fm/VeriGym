"""Parity between the old (pre-refactor) and new abstraction stacks.

Builds the same CartPole abstraction two ways -- once via a frozen copy of the
OLD stack (`tests.legacy_reference`: `LegacyAbstractionMap`/`LegacyAbstractionMapper`,
`legacy_get_discrete_box_tf`, `legacy_factored_to_index`/`legacy_index_to_factored`,
`LegacyCachedDiscretizer`, `legacy_forward_mapping`/`legacy_backward_mapping` --
assembled the same way `tests.utils.get_abstraction_mapper_to_discrete` did before
it collapsed onto `binned_mapper`), once via the new
`verigym.abstraction.abstractionmapper.linspace_mapper` -- and checks they agree.

The "old" construction uses `tests.legacy_reference` rather than
`tests.utils.get_abstraction_mapper_to_discrete` because that helper now calls
`binned_mapper` directly -- importing it here would make this test compare the
new stack against itself. `legacy_reference` is a frozen, independent snapshot
of the deleted old modules (see its module docstring) kept alive purely so this
parity guarantee survives their deletion.

Binning semantics were deliberately kept identical between the two stacks (the
"keep current semantics" decision from the discretization refactor), so this
parity must be **exact**, not approximate: every sampled original state/action
must discretize to the identical abstract index under both stacks. An inexact
result here means a real bug in the new discretization/enumeration arithmetic,
not an expected divergence -- if that happens, this test should fail loudly
rather than have its assertions loosened.
"""

import functools
from math import prod

import gymnasium as gym
import numpy as np

from legacy_reference import (
    LegacyAbstractionMap,
    LegacyAbstractionMapper,
    LegacyCachedDiscretizer,
    legacy_backward_mapping,
    legacy_factored_to_index,
    legacy_forward_mapping,
    legacy_get_discrete_box_tf,
    legacy_index_to_factored,
)
from utils import make_original_env
from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.abstraction.discretization import generate_box_bins

SEED = 12345
N_SAMPLES = 500


def _old_style_mapper(env: gym.Env, bin_edges_per_state_dim, bin_edges_per_action_dim) -> LegacyAbstractionMapper:
    """The pre-refactor body of `tests.utils.get_abstraction_mapper_to_discrete`
    (`use_box_space=True` branch, its default), rebuilt from the frozen
    `legacy_reference` helpers so this test keeps exercising the genuinely OLD
    arithmetic regardless of how the live test helper evolves.
    """
    bin_edges_observations = generate_box_bins(
        env.observation_space, np.linspace, bin_edges_per_state_dim
    )
    bin_edges_actions = generate_box_bins(
        env.action_space, np.linspace, bin_edges_per_action_dim
    )
    n_states = int(prod(len(dimension) for dimension in bin_edges_observations))
    n_actions = int(prod(len(dimension) for dimension in bin_edges_actions))

    forward_state_map = legacy_get_discrete_box_tf(env.observation_space, bin_edges_observations)
    forward_action_map = legacy_get_discrete_box_tf(env.action_space, bin_edges_actions)
    backward_state_map = functools.partial(legacy_index_to_factored, bin_edges=bin_edges_observations)
    backward_action_map = functools.partial(legacy_index_to_factored, bin_edges=bin_edges_actions)
    abstract_state_space = gym.spaces.MultiDiscrete([n_states])
    abstract_action_space = gym.spaces.MultiDiscrete([n_actions])

    discretizer_state = LegacyCachedDiscretizer(
        functools.partial(legacy_factored_to_index, bin_edges=bin_edges_observations)
    )
    discretizer_action = LegacyCachedDiscretizer(
        functools.partial(legacy_factored_to_index, bin_edges=bin_edges_actions)
    )

    abstraction_map_state = LegacyAbstractionMap(
        forward_map=functools.partial(legacy_forward_mapping, to_int=discretizer_state.discretize, to_bins=forward_state_map),
        backward_map=functools.partial(legacy_backward_mapping, backward_map=backward_state_map, space=env.observation_space),
        original_space=env.observation_space,
        abstract_space=abstract_state_space,
    )
    abstraction_map_action = LegacyAbstractionMap(
        forward_map=functools.partial(legacy_forward_mapping, to_int=discretizer_action.discretize, to_bins=forward_action_map),
        backward_map=functools.partial(legacy_backward_mapping, backward_map=backward_action_map, space=env.action_space),
        original_space=env.action_space,
        abstract_space=abstract_action_space,
    )

    return LegacyAbstractionMapper(
        state_abstraction_map=abstraction_map_state,
        action_abstraction_map=abstraction_map_action,
    )


def _build_mappers():
    """One shared env, both mappers built from it, seeded for reproducible sampling."""
    env, _num_steps, bin_edges_per_dim = make_original_env()
    env.observation_space.seed(SEED)
    env.action_space.seed(SEED)

    old_mapper = _old_style_mapper(env, bin_edges_per_dim, bin_edges_per_dim)
    new_mapper = linspace_mapper(env, bin_edges_per_dim, bin_edges_per_dim)
    return env, old_mapper, new_mapper


def test_parity_state_and_action_counts():
    """Both stacks must discretize CartPole into the same number of abstract states/actions."""
    _env, old_mapper, new_mapper = _build_mappers()
    assert int(old_mapper.abstract_n_states) == int(new_mapper.abstract_n_states)
    assert int(old_mapper.abstract_n_actions) == int(new_mapper.abstract_n_actions)


def test_parity_forward_state_indices_exact():
    """Every sampled original state must discretize to the identical abstract state index."""
    env, old_mapper, new_mapper = _build_mappers()

    mismatches = []
    for _ in range(N_SAMPLES):
        state = env.observation_space.sample()
        old_idx = old_mapper.original_to_abstract_state(state)
        new_idx = new_mapper.original_to_abstract_state_enum(state)
        if old_idx != new_idx:
            mismatches.append((state, old_idx, new_idx))

    assert not mismatches, (
        f"{len(mismatches)}/{N_SAMPLES} state index mismatches "
        f"(first: {mismatches[0] if mismatches else None})"
    )


def test_parity_forward_action_indices_exact():
    """Every sampled original action must discretize to the identical abstract action index."""
    env, old_mapper, new_mapper = _build_mappers()

    mismatches = []
    for _ in range(N_SAMPLES):
        action = env.action_space.sample()
        old_idx = old_mapper.original_to_abstract_action(action)
        new_idx = new_mapper.original_to_abstract_action_enum(action)
        if old_idx != new_idx:
            mismatches.append((action, old_idx, new_idx))

    assert not mismatches, (
        f"{len(mismatches)}/{N_SAMPLES} action index mismatches "
        f"(first: {mismatches[0] if mismatches else None})"
    )


def test_parity_boundary_samples():
    """Boundary samples (space.low / space.high) exercise the degenerate top bin
    explicitly, rather than relying on random sampling to hit it."""
    env, old_mapper, new_mapper = _build_mappers()

    for state in (env.observation_space.low, env.observation_space.high):
        old_idx = old_mapper.original_to_abstract_state(state)
        new_idx = new_mapper.original_to_abstract_state_enum(state)
        assert old_idx == new_idx, f"boundary state {state}: old={old_idx} new={new_idx}"
