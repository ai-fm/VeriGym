"""
Tests for `StormpyFormatter`, which converts stormpy models for `FrameworkExplicitEnv`. Specifically:
    state valuations, including boolean variables (stormpy prints true booleans as an empty token)
    choices of a state with the same label, which are separate actions ("label", "label_2", ...)
"""

import os

import numpy as np
import pytest
import stormpy

from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv
from verigym.frameworks.stormpy.stormpy_utils import (
    _unwrap_scheduler,
    _valuation_variables,
    build_stormpy_mdp,
    load_stormpy_model,
)

from utils import build_stormpy_model

TESTS_DIR = os.path.dirname(__file__)
PRISM_FILES = os.path.join(TESTS_DIR, "..", "examples", "prism_files")
BOOL_TEST = os.path.join(TESTS_DIR, "test_bool.prism")
RESOURCE_GATHERING = (os.path.join(PRISM_FILES, "resource-gathering.v2.pm"), "B=20,GOLD_TO_COLLECT=2,GEM_TO_COLLECT=2")
ZEROCONF = (os.path.join(PRISM_FILES, "zeroconf.v1.prism"), "N=20,K=2,reset=true")

# Models with boolean variables and the order of their variables
VALUATION_MODELS = {
    # booleans that are true in the initial state, declared after an integer (stormpy prints booleans first)
    "test_bool": (BOOL_TEST, None, ["b", "c", "x"]),
    "resource_gathering": (*RESOURCE_GATHERING, ["gold", "gem", "attacked", "x", "y", "required_gold", "required_gem"]),
}


# ---------------------------------------------------------------------------
# State valuations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", list(VALUATION_MODELS))
def test_state_valuations(model_name):
    """The valuations of every state are stormpy's values, with booleans as 0/1, also for true booleans, and the
    variables are in the order in which stormpy prints them."""
    path, constants, var_order = VALUATION_MODELS[model_name]
    mdp = load_stormpy_model(path, constants=constants)
    formatter = FrameworkExplicitEnv.from_stormpy(mdp).formatter

    assert formatter.var_order == var_order
    variables = {variable.name: variable for variable in mdp.state_valuations.get_all_variables()}
    for s in range(mdp.nr_states):
        expected = {name: int(mdp.state_valuations.get_value(s, variables[name])) for name in var_order}
        assert formatter.state_to_values[s] == expected
        assert list(formatter.state_to_values[s]) == var_order


def test_true_booleans_in_initial_state():
    """Regression: true booleans were missing from the valuations (with an "" key instead), and also from
    `var_order` if they were true in state 0."""
    formatter = FrameworkExplicitEnv.from_stormpy(load_stormpy_model(BOOL_TEST)).formatter
    assert formatter.state_to_values[0] == {"b": 1, "c": 1, "x": 0}
    assert all("" not in values for values in formatter.state_to_values.values())


@pytest.mark.parametrize("model_name", list(VALUATION_MODELS))
def test_featured_states(model_name):
    """Factored states are the valuations in `var_order`, lie in the observation space and map back to the states."""
    path, constants, _ = VALUATION_MODELS[model_name]
    mdp = load_stormpy_model(path, constants=constants)
    env = FrameworkExplicitEnv.from_stormpy(mdp, flat=False)
    formatter = env.formatter

    max_values = [max(formatter.state_to_values[s][name] for s in range(mdp.nr_states)) for name in formatter.var_order]
    np.testing.assert_array_equal(env.observation_space.nvec, np.array(max_values) + 1)
    for s in range(mdp.nr_states):
        state = env.decode(s)
        assert state == tuple(formatter.state_to_values[s][name] for name in formatter.var_order)
        assert env.observation_space.contains(np.array(state))
        assert env.encode(state) == s


class _StubValuations:
    """The parts of stormpy's `Valuations` that `_valuation_variables` uses, with a given printed valuation."""

    def __init__(self, variables, printed):
        self.variables = variables
        self.printed = printed

    def get_all_variables(self):
        return set(self.variables)

    def get_string(self, state, pretty=True):
        return self.printed


def test_valuation_variables():
    """The order of the printed valuation, with the true booleans (empty tokens) in the order of their index; the
    order of the indices if the printed valuation does not match the variables."""
    manager = stormpy.ExpressionManager()
    x = manager.create_integer_variable(name="x")
    b = manager.create_boolean_variable(name="b")
    c = manager.create_boolean_variable(name="c")

    assert _valuation_variables(_StubValuations([x, b, c], "[\t& !c\t& x=0]")) == [b, c, x]
    assert _valuation_variables(_StubValuations([x, b, c], "[\t& \t& x=0]")) == [b, c, x]
    assert _valuation_variables(_StubValuations([x, b, c], "[x=0]")) == sorted([x, b, c], key=lambda v: v.index)
    assert _valuation_variables(_StubValuations([], "[]")) == []


def _mdp_with_rational_variable(values):
    """An MDP with 2 states and a rational state variable `r` with the given values."""
    manager = stormpy.ExpressionManager()
    r = manager.create_rational_variable(name="r")
    builder = stormpy.StateValuationsBuilder()
    builder.add_variable(r)
    for s, value in enumerate(values):
        builder.add_state(state=s, rational_values=[value])
    mdp = build_stormpy_model(stormpy.storage.SparseMdp, rows=[{1: 1.0}, {1: 1.0}], row_groups=[0, 1],
                              choice_labels=["go", "go"])
    components = stormpy.SparseModelComponents(transition_matrix=mdp.transition_matrix, state_labeling=mdp.labeling)
    components.choice_labeling = mdp.choice_labeling
    components.state_valuations = builder.build()
    return stormpy.storage.SparseMdp(components)


def test_rational_variables():
    """Rational state variables are only supported if all their values are whole numbers."""
    formatter = FrameworkExplicitEnv.from_stormpy(_mdp_with_rational_variable([stormpy.Rational(0), stormpy.Rational(3)])).formatter
    assert formatter.state_to_values == {0: {"r": 0}, 1: {"r": 3}}

    half = stormpy.Rational(1) / stormpy.Rational(2)
    with pytest.raises(NotImplementedError, match="State variable r"):
        FrameworkExplicitEnv.from_stormpy(_mdp_with_rational_variable([stormpy.Rational(0), half]))


# ---------------------------------------------------------------------------
# Choices of a state with the same label
# ---------------------------------------------------------------------------


def _repeated_labels_mdp():
    """3 states. State 0 has two choices labelled "go" (to state 1 with reward 1, to state 2 with reward 5) and one
    labelled "stay". State 1 has a choice labelled "go_2", so the second "go" choice of state 0 is named "go_3".
    State 2, labelled "target", loops with "go"."""
    return build_stormpy_model(
        stormpy.storage.SparseMdp,
        rows=[{1: 1.0}, {2: 1.0}, {0: 1.0}, {1: 1.0}, {2: 1.0}],
        row_groups=[0, 3, 4],
        reward_models={"r": (None, [1.0, 5.0, 0.0, 0.0, 0.0])},
        choice_labels=["go", "go", "stay", "go_2", "go"],
        state_labels={"target": [2]},
        nr_states=3,
    )


def test_repeated_labels_are_separate_actions():
    """Several choices of a state with the same label are separate actions, with their own transitions and rewards."""
    env = FrameworkExplicitEnv.from_stormpy(_repeated_labels_mdp())

    assert env.formatter.label_to_action == {"go": 0, "go_2": 1, "go_3": 2, "stay": 3}
    assert env.nr_actions == 4
    T = env.get_transition_function().T_dict
    assert [{a: dict(successors) for a, successors in T[s].items()} for s in range(3)] == [
        {0: {1: 1.0}, 2: {2: 1.0}, 3: {0: 1.0}}, {1: {1: 1.0}}, {0: {2: 1.0}}
    ]
    R = env.get_reward_function().R_dict
    assert [R[0][0], R[0][2], R[0][3], R[1][1], R[2][0]] == [1.0, 5.0, 0.0, 0.0, 0.0]
    np.testing.assert_array_equal(env.action_mask, [[1, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]])

    # the export labels the choices with the action names, so it loads as the same env
    exported = FrameworkExplicitEnv.from_stormpy(build_stormpy_mdp(env))
    assert exported.formatter.label_to_action == env.formatter.label_to_action
    assert exported.get_transition_function().T_dict == T


@pytest.mark.parametrize("pass_label_to_action", [False, True])
def test_unwrap_scheduler_with_repeated_labels(pass_label_to_action):
    """A scheduler that picks the second "go" choice of state 0 is mapped to its action, "go_3"."""
    mdp = _repeated_labels_mdp()
    env = FrameworkExplicitEnv.from_stormpy(mdp)
    prop = stormpy.parse_properties('Pmax=? [F "target"]')[0]
    result = stormpy.check_model_sparse(mdp, prop, only_initial_states=False, extract_scheduler=True)
    assert result.scheduler.get_choice(0).get_deterministic_choice() == 1  # the second "go" choice

    label_to_action = env.formatter.label_to_action if pass_label_to_action else None
    policy = _unwrap_scheduler(mdp, result.scheduler, label_to_action)

    assert policy[0] == env.formatter.label_to_action["go_3"]


def test_repeated_labels_zeroconf():
    """In zeroconf, two `rec` commands can be enabled in the same state. Their choices are the actions "rec" and
    "rec_2", so every choice of the model is an action of the env with its own distribution."""
    path, constants = ZEROCONF
    mdp = load_stormpy_model(path, constants=constants)
    env = FrameworkExplicitEnv.from_stormpy(mdp)
    label_to_action = env.formatter.label_to_action
    assert "rec_2" in label_to_action

    T = env.get_transition_function().T_dict
    matrix = mdp.transition_matrix
    for s in range(mdp.nr_states):
        # the k-th choice of a state with a label is the action "label" (k = 1) or "label_k"
        expected, occurrences = {}, {}
        for row in matrix.get_rows_for_group(s):
            labels = mdp.choice_labeling.get_labels_of_choice(row)
            if len(labels) == 0:
                continue
            label = labels.pop()
            occurrences[label] = occurrences.get(label, 0) + 1
            name = label if occurrences[label] == 1 else f"{label}_{occurrences[label]}"
            expected[label_to_action[name]] = {entry.column: entry.value() for entry in matrix.get_row(row)}
        assert {a: dict(successors) for a, successors in T[s].items()} == expected
        assert set(np.flatnonzero(env.action_mask[s])) == set(expected)
        for successors in T[s].values():
            assert sum(successors.values()) == pytest.approx(1.0)
