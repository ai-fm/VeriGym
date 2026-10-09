"""
Tests for exporting explicit environments to stormpy models:
    export_to_stormpy_mdp
    export_to_stormpy_imdp

Both exports must produce well-formed models for sparse environments (e.g., learned abstractions with and
without intervals), handle deadlocks the same way, and only differ in point values vs. intervals.
"""

import copy
import os
from collections import defaultdict

import numpy as np
import pytest
import stormpy

from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.abstraction.learn_abstraction import create_abstraction
from verigym.environments.explicitenv import ExplicitEnv
from verigym.environments.exporter import export_to_stormpy_imdp, export_to_stormpy_mdp
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv
from verigym.environments.generativeenv import GenerativeEnv
from verigym.environments.interval_explicitenv import IntervalExplicitEnv
from verigym.environments.reward_func import IntervalRewardFunction, RewardFunction
from verigym.environments.transition_func import IntervalTransitionFunction, TransitionFunction
from verigym.frameworks.stormpy.formatter import StormpyFormatter
from verigym.frameworks.stormpy.stormpy_utils import _unwrap_scheduler, load_stormpy_model
from verigym.policy.randomized import RandomizedPolicy

from utils import make_original_env

PRISM_TEST = os.path.join(os.path.dirname(__file__), "test_2d.prism")
EXPORTS = {"mdp": export_to_stormpy_mdp, "imdp": export_to_stormpy_imdp}


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


def _make_deadlock_env(empty_action=False, intervals=False):
    """3 states, 2 actions: 0 -a0-> {1: 0.5, 2: 0.5} and 1 -a1-> {1: 1.0}. State 2 is a deadlock (no actions).
    If `empty_action`, state 2 has an action without successors, which is a deadlock as well."""
    T_dict = defaultdict(dict)
    T_dict[0][0] = defaultdict(float, {1: 0.5, 2: 0.5})
    T_dict[1][1] = defaultdict(float, {1: 1.0})
    if empty_action:
        T_dict[2][0] = defaultdict(float)
    R_dict = defaultdict(dict)
    R_dict[0][0] = 2.0
    R_dict[1][1] = 3.0
    env_args = dict(
        nr_states=3,
        nr_actions=2,
        initial_state_distr=np.array([1.0, 0.0, 0.0]),
        transition_function=TransitionFunction(3, 2, T_dict),
        reward_function=RewardFunction(3, 2, R_dict),
    )
    if not intervals:
        return ExplicitEnv(**env_args)

    T_i_dict = defaultdict(dict)
    T_i_dict[0][0] = defaultdict(lambda: (0.0, 0.0), {1: (0.4, 0.6), 2: (0.4, 0.6)})
    T_i_dict[1][1] = defaultdict(lambda: (0.0, 0.0), {1: (1.0, 1.0)})
    R_i_dict = defaultdict(dict)
    R_i_dict[0][0] = (1.5, 2.5)
    R_i_dict[1][1] = (3.0, 3.0)
    return IntervalExplicitEnv(
        **env_args,
        interval_transition_function=IntervalTransitionFunction(3, 2, T_i_dict),
        interval_reward_function=IntervalRewardFunction(3, 2, R_i_dict),
    )


@pytest.fixture
def deadlock_env():
    return _make_deadlock_env()


@pytest.fixture
def deadlock_interval_env():
    return _make_deadlock_env(intervals=True)


@pytest.fixture
def empty_action_env():
    return _make_deadlock_env(empty_action=True)


@pytest.fixture
def prism_env():
    """Loaded from a prism file, with deadlock states."""
    mdp = load_stormpy_model(PRISM_TEST)
    return FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))


@pytest.fixture(
    scope="module",
    params=[(False, False), (True, False), (True, True)],
    ids=["point", "interval-hoeffding", "interval-iid"],
)
def abstraction(request):
    """Learned abstraction of CartPole, without (default) and with intervals."""
    intervals, assume_iid = request.param
    env, num_steps, bin_edges_per_dim = make_original_env()
    abstraction_mapper = linspace_mapper(env, bin_edges_per_dim, bin_edges_per_dim)
    generative_env = GenerativeEnv.from_gymnasium(env)
    return create_abstraction(
        original_env=generative_env,
        abstraction_mapper=abstraction_mapper,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=num_steps,
        intervals=intervals,
        assume_iid=assume_iid,
        multithreading=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _available_actions(env, T_dict, s):
    """Actions of `s` with at least one successor. A state without any is a deadlock."""
    return [a for a in range(env.nr_actions) if len(T_dict.get(s, {}).get(a, {})) > 0]


def _exported_T_dict(env, model_kind):
    """The transitions the export of `env` is built from."""
    if model_kind == "imdp" and isinstance(env, IntervalExplicitEnv):
        return env.get_interval_transition_function().T_dict
    return env.get_transition_function().T_dict


def _bounds(value):
    """(lower, upper) of a stormpy value, which is an interval in iMDPs and a number in MDPs."""
    if hasattr(value, "lower"):
        return value.lower(), value.upper()
    return value, value


def _as_bounds(value):
    """(lower, upper) of a value of the env, which is a (lower, upper) tuple for intervals and a number otherwise."""
    if isinstance(value, tuple):
        return value
    return value, value


def _env_dicts(env):
    """The dicts that define `env`, to check that exporting does not change it."""
    dicts = [env.get_transition_function().T_dict, env.get_reward_function().R_dict]
    if isinstance(env, IntervalExplicitEnv):
        dicts += [env.get_interval_transition_function().T_dict, env.get_interval_reward_function().R_dict]
    return dicts


def _export(env, model_kind, **kwargs):
    """Exports `env` to an MDP or iMDP and checks that exporting did not change `env`."""
    dicts_before = copy.deepcopy(_env_dicts(env))
    model = EXPORTS[model_kind](env, **kwargs)
    assert _env_dicts(env) == dicts_before, "exporting must not change the environment"
    return model


def _check(model, formula):
    """Model checks `formula` on all states, robustly for iMDPs, and produces a scheduler."""
    prop = stormpy.parse_properties(formula)[0]
    if isinstance(model, stormpy.storage.SparseIntervalMdp):
        task = stormpy.CheckTask(prop.raw_formula, only_initial_states=False)
        task.set_produce_schedulers()
        task.set_uncertainty_resolution_mode(stormpy.UncertaintyResolutionMode.ROBUST)
        return stormpy.check_interval_mdp(model, task, stormpy.Environment())
    return stormpy.check_model_sparse(model, prop, only_initial_states=False, extract_scheduler=True)


def assert_well_formed(model, env, T_dict):
    """Checks the exported `model` against the transitions `T_dict` of `env`:
    one row per available action (or a single self-loop for deadlocks), one reward per row, the action index as
    choice label, and deadlocks as unlabelled self-loops with probability 1 and zero reward, labelled "deadlock"."""
    available = {s: _available_actions(env, T_dict, s) for s in range(env.nr_states)}
    deadlocks = {s for s, actions in available.items() if len(actions) == 0}

    assert model.nr_states == env.nr_states
    assert model.nr_choices == sum(max(1, len(actions)) for actions in available.values())
    for reward_model in model.reward_models.values():
        assert len(reward_model.state_action_rewards) == model.nr_choices
    assert model.has_choice_labeling()
    assert set(model.labeling.get_states("deadlock")) == deadlocks

    row_groups = model.nondeterministic_choice_indices
    for s in range(env.nr_states):
        rows = list(range(row_groups[s], row_groups[s + 1]))
        labels = [model.choice_labeling.get_labels_of_choice(row) for row in rows]
        if s in deadlocks:
            assert len(rows) == 1
            entries = [(e.column, _bounds(e.value())) for e in model.transition_matrix.get_row(rows[0])]
            assert entries == [(s, (1.0, 1.0))]
            assert labels == [set()]
            for reward_model in model.reward_models.values():
                assert _bounds(reward_model.state_action_rewards[rows[0]]) == (0.0, 0.0)
        else:
            assert labels == [{str(a)} for a in available[s]]


def _assert_same_structure(env):
    """The MDP and iMDP exports of `env` have the same states, choices, successors and labels, and only differ in
    point values vs. intervals."""
    is_interval = isinstance(env, IntervalExplicitEnv)
    mdp, imdp = _export(env, "mdp"), _export(env, "imdp")

    assert mdp.nr_states == imdp.nr_states
    assert list(mdp.nondeterministic_choice_indices) == list(imdp.nondeterministic_choice_indices)
    for row in range(mdp.nr_choices):
        point = {e.column: e.value() for e in mdp.transition_matrix.get_row(row)}
        interval = {e.column: _bounds(e.value()) for e in imdp.transition_matrix.get_row(row)}
        assert point.keys() == interval.keys()
        for s_next, p in point.items():
            lb, ub = interval[s_next]
            if is_interval:
                assert lb <= p <= ub
            else:
                assert (lb, ub) == pytest.approx((p, p))
        assert mdp.choice_labeling.get_labels_of_choice(row) == imdp.choice_labeling.get_labels_of_choice(row)

    # the iMDP is exported with point rewards (use_reward_uncertainty=False)
    assert mdp.reward_models.keys() == imdp.reward_models.keys()
    for label in mdp.reward_models:
        rewards = mdp.reward_models[label].state_action_rewards
        interval_rewards = imdp.reward_models[label].state_action_rewards
        assert len(rewards) == len(interval_rewards)
        for r, r_i in zip(rewards, interval_rewards):
            assert _bounds(r_i) == pytest.approx((r, r))

    assert set(mdp.labeling.get_labels()) == set(imdp.labeling.get_labels())
    for label in mdp.labeling.get_labels():
        assert set(mdp.labeling.get_states(label)) == set(imdp.labeling.get_states(label))


# ---------------------------------------------------------------------------
# Learned abstractions (with and without intervals)
# ---------------------------------------------------------------------------


def test_export_abstraction_mdp(abstraction):
    """Learned abstractions export to a well-formed MDP."""
    mdp = _export(abstraction, "mdp")
    assert_well_formed(mdp, abstraction, abstraction.get_transition_function().T_dict)


@pytest.mark.parametrize("use_reward_uncertainty", [False, True])
def test_export_abstraction_imdp(abstraction, use_reward_uncertainty):
    """Learned abstractions export to a well-formed iMDP whose transitions and rewards are the env's intervals
    (or [p, p] for point estimates)."""
    if use_reward_uncertainty and not isinstance(abstraction, IntervalExplicitEnv):
        with pytest.raises(AssertionError):
            export_to_stormpy_imdp(abstraction, use_reward_uncertainty=True)
        return

    imdp = _export(abstraction, "imdp", use_reward_uncertainty=use_reward_uncertainty)
    T_dict = _exported_T_dict(abstraction, "imdp")
    assert_well_formed(imdp, abstraction, T_dict)

    if use_reward_uncertainty:
        R_dict = abstraction.get_interval_reward_function().R_dict
    else:
        R_dict = abstraction.get_reward_function().R_dict
    row_groups = imdp.nondeterministic_choice_indices
    rewards = imdp.reward_models["reward0"].state_action_rewards
    for s in range(abstraction.nr_states):
        rows = range(row_groups[s], row_groups[s + 1])
        for row, a in zip(rows, _available_actions(abstraction, T_dict, s)):
            exported = {e.column: _bounds(e.value()) for e in imdp.transition_matrix.get_row(row)}
            assert exported.keys() == T_dict[s][a].keys()
            for s_next, bounds in exported.items():
                assert bounds == pytest.approx(_as_bounds(T_dict[s][a][s_next]))
            assert _bounds(rewards[row]) == pytest.approx(_as_bounds(R_dict[s][a]))


def test_mdp_and_imdp_share_structure_abstraction(abstraction):
    """The MDP and iMDP exports of learned abstractions only differ in point values vs. intervals."""
    _assert_same_structure(abstraction)


@pytest.mark.parametrize("model_kind", ["mdp", "imdp"])
def test_deadlock_in_abstraction(abstraction, model_kind):
    """In learned abstractions, the deadlocks are exactly the states without available actions (e.g., where
    episodes ended), and reaching them can be model checked."""
    model = _export(abstraction, model_kind)
    no_actions = {s for s in range(abstraction.nr_states) if abstraction.action_mask[s].sum() == 0}
    assert set(model.labeling.get_states("deadlock")) == no_actions

    result = _check(model, 'Pmax=? [F "deadlock"]')
    for s in range(abstraction.nr_states):
        assert -1e-6 <= result.at(s) <= 1 + 1e-6


def test_robust_scheduler_maps_to_actions(abstraction):
    """A robust scheduler of the exported iMDP maps back to available actions of the env (see `StormpyPolicy`)."""
    imdp = _export(abstraction, "imdp")
    result = _check(imdp, 'Pmax=? [F "deadlock"]')
    policy = _unwrap_scheduler(imdp, result.scheduler)
    for s in range(abstraction.nr_states):
        if abstraction.action_mask[s].sum() > 0:
            assert abstraction.action_mask[s][policy[s]] == 1


# ---------------------------------------------------------------------------
# Hand-made and prism environments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_name", ["deadlock_env", "deadlock_interval_env", "empty_action_env", "prism_env"])
def test_mdp_and_imdp_share_structure(env_name, request):
    """The MDP and iMDP exports only differ in point values vs. intervals."""
    _assert_same_structure(request.getfixturevalue(env_name))


@pytest.mark.parametrize("model_kind", ["mdp", "imdp"])
@pytest.mark.parametrize("env_name", ["deadlock_env", "deadlock_interval_env", "empty_action_env", "prism_env"])
def test_deadlock_export(env_name, model_kind, request):
    """Deadlocks (states without an action that has successors) become a single unlabelled self-loop with
    probability 1 and zero reward, labelled "deadlock", and reaching them can be model checked."""
    env = request.getfixturevalue(env_name)
    model = _export(env, model_kind)
    assert_well_formed(model, env, _exported_T_dict(env, model_kind))
    _check(model, 'Pmax=? [F "deadlock"]')


@pytest.mark.parametrize(
    "env_name, model_kind, expected",
    [
        ("deadlock_env", "mdp", 0.5),
        ("deadlock_env", "imdp", 0.5),
        ("empty_action_env", "mdp", 0.5),
        ("empty_action_env", "imdp", 0.5),
        ("deadlock_interval_env", "mdp", 0.5),
        # robust: nature moves the least probability (0.4) to the deadlock
        ("deadlock_interval_env", "imdp", 0.4),
    ],
)
def test_deadlock_reachability(env_name, model_kind, expected, request):
    """From state 0, the deadlock 2 is reached with probability 0.5 (state 1 loops forever)."""
    model = _export(request.getfixturevalue(env_name), model_kind)
    result = _check(model, 'Pmax=? [F "deadlock"]')
    assert result.at(0) == pytest.approx(expected)


def test_unwrap_scheduler_requires_choice_labels():
    """Without choice labels, the choices of a scheduler cannot be mapped back to the actions of the env."""
    builder = stormpy.SparseMatrixBuilder(
        rows=0, columns=2, entries=0, force_dimensions=True, has_custom_row_grouping=True
    )
    builder.new_row_group(0)
    builder.add_next_value(0, 1, 1.0)
    builder.add_next_value(1, 0, 1.0)
    builder.new_row_group(2)
    builder.add_next_value(2, 1, 1.0)
    labeling = stormpy.storage.StateLabeling(2)
    labeling.add_label("init")
    labeling.add_label_to_state("init", 0)
    labeling.add_label("goal")
    labeling.add_label_to_state("goal", 1)
    mdp = stormpy.storage.SparseMdp(
        stormpy.SparseModelComponents(transition_matrix=builder.build(), state_labeling=labeling)
    )
    result = _check(mdp, 'Pmax=? [F "goal"]')

    with pytest.raises(AssertionError):
        _unwrap_scheduler(mdp, result.scheduler)
