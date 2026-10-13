"""
Tests for importing and exporting explicit environments from and to different formats.
Specifically:
    export_to_stormpy_mdp
    export_to_drn, load_from_drn
    export_to_umb, load_from_umb
for MDPs and interval MDPs, from existing models (PRISM, DRN and UMB files) and from learned abstractions.
WIP:
    export_to_julia_pomdp (not implemented)
    export_to_prism (not implemented)

Test stormpy_utils.build_stormpy_mdp
"""

import os
from collections import defaultdict

import gymnasium as gym
import numpy as np
import pytest
import stormpy

from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.abstraction.learn_abstraction import create_abstraction
from verigym.environments.explicitenv import ExplicitEnv
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv
from verigym.environments.generativeenv import GenerativeEnv
from verigym.environments.interval_explicitenv import IntervalExplicitEnv
from verigym.environments.reward_func import IntervalRewardFunction, RewardFunction
from verigym.environments.transition_func import IntervalTransitionFunction, TransitionFunction
from verigym.frameworks.stormpy.formatter import StormpyFormatter
from verigym.frameworks.stormpy.stormpy_utils import (
    _build_state_label_map,
    build_stormpy_mdp,
    load_stormpy_model,
)
from verigym.io.exporter import export_to_drn, export_to_stormpy_mdp, export_to_umb
from verigym.io.importer import load_from_drn, load_from_umb
from verigym.policy.randomized import RandomizedPolicy

from utils import helper_to_create_stormpy_model_for_tests, make_original_env

TESTS_DIR = os.path.dirname(__file__)
PRISM_FILES = os.path.join(TESTS_DIR, "..", "examples", "prism_files")
PRISM_TEST = os.path.join(TESTS_DIR, "test_2d.prism")
SEED = 12929582

# Standard models: prism file and constants (as in examples/tech_eval_mdp_memory.ipynb, but zeroconf without reset
# with K=1 instead of K=2 to keep the tests fast)
STANDARD_MODELS = {
    "test_2d": (PRISM_TEST, None),
    "test_2d_underground": (os.path.join(TESTS_DIR, "test_2d_underground.prism"), None),
    "consensus": (os.path.join(PRISM_FILES, "consensus.4.v1.prism"), "K=2"),
    "pacman": (os.path.join(PRISM_FILES, "pacman.v2.nm"), "MAXSTEPS=5"),
    "resource_gathering": (
        os.path.join(PRISM_FILES, "resource-gathering.v2.pm"),
        "B=20,GOLD_TO_COLLECT=2,GEM_TO_COLLECT=2",
    ),
    "zeroconf_reset": (os.path.join(PRISM_FILES, "zeroconf.v1.prism"), "N=20,K=2,reset=true"),
    "zeroconf_noreset": (os.path.join(PRISM_FILES, "zeroconf.v1.prism"), "N=20,K=1,reset=false"),
}

# Export and load functions per file format
FORMATS = {
    "umb": (export_to_umb, load_from_umb),
    "drn": (export_to_drn, load_from_drn),
}

# Writing a model to a file with stormpy directly, as if it came from another tool
STORM_EXPORTS = {
    "umb": stormpy.export_to_umb,
    "drn": lambda model, path: stormpy.export_to_drn(model, path),
}


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=list(STANDARD_MODELS))
def prism_model(request):
    """A standard model loaded from its prism file: the stormpy MDP and the env."""
    path, constants = STANDARD_MODELS[request.param]
    mdp = load_stormpy_model(path, constants=constants)
    return mdp, FrameworkExplicitEnv.from_stormpy(mdp)


@pytest.fixture(scope="module")
def abstraction():
    """Learned abstraction of CartPole (seeded). It uses only 2 of its 5 abstract actions."""
    env, num_steps, bin_edges_per_dim = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    return create_abstraction(
        original_env=generative_env,
        abstraction_mapper=linspace_mapper(env, bin_edges_per_dim, bin_edges_per_dim),
        exploration_policy=RandomizedPolicy(generative_env, seed=SEED),
        num_steps=num_steps,
        multithreading=False,
        seed=SEED,
    )


@pytest.fixture(scope="module", params=[False, True], ids=["noniid", "iid"])
def interval_abstraction(request):
    """Learned abstraction of CartPole with intervals (seeded)."""
    env, num_steps, bin_edges_per_dim = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    return create_abstraction(
        original_env=generative_env,
        abstraction_mapper=linspace_mapper(env, bin_edges_per_dim, bin_edges_per_dim),
        exploration_policy=RandomizedPolicy(generative_env, seed=SEED),
        num_steps=num_steps,
        intervals=True,
        assume_iid=request.param,
        multithreading=False,
        seed=SEED,
    )


def _make_explicit_env(nr_actions=2, initial_state_distr=(1.0, 0.0, 0.0)):
    """3 states: every action of state 0 leads to {1: 0.5, 2: 0.5} with reward a, the last action loops on state 1
    with reward 3, and state 2 is a deadlock."""
    T_dict, R_dict = defaultdict(dict), defaultdict(dict)
    for a in range(nr_actions):
        T_dict[0][a] = defaultdict(float, {1: 0.5, 2: 0.5})
        R_dict[0][a] = float(a)
    T_dict[1][nr_actions - 1] = defaultdict(float, {1: 1.0})
    R_dict[1][nr_actions - 1] = 3.0
    return ExplicitEnv(
        nr_states=3,
        nr_actions=nr_actions,
        initial_state_distr=np.array(initial_state_distr),
        transition_function=TransitionFunction(3, nr_actions, T_dict),
        reward_function=RewardFunction(3, nr_actions, R_dict),
    )


def _make_interval_env():
    """The env of `_make_explicit_env` with intervals: state 0 leads to {1: [0.4, 0.6], 2: [0.3, 0.6]} with reward
    [1.5, 2.5] or [0, 1], state 1 loops with reward 3, and state 2 is a deadlock."""
    T_dict = defaultdict(dict)
    T_dict[0][0] = defaultdict(lambda: (0.0, 0.0), {1: (0.4, 0.6), 2: (0.3, 0.6)})
    T_dict[0][1] = defaultdict(lambda: (0.0, 0.0), {1: (0.4, 0.6), 2: (0.3, 0.6)})
    T_dict[1][1] = defaultdict(lambda: (0.0, 0.0), {1: (1.0, 1.0)})
    R_dict = defaultdict(dict)
    R_dict[0][0] = (1.5, 2.5)
    R_dict[0][1] = (0.0, 1.0)
    R_dict[1][1] = (3.0, 3.0)
    return IntervalExplicitEnv(
        nr_states=3,
        nr_actions=2,
        initial_state_distr=np.array([1.0, 0.0, 0.0]),
        interval_transition_function=IntervalTransitionFunction(3, 2, T_dict),
        interval_reward_function=IntervalRewardFunction(3, 2, R_dict),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_trip(env, file_format, tmp_path, name="model"):
    """Exports `env` to a file and loads it again."""
    export, load = FORMATS[file_format]
    path = str(tmp_path / f"{name}.{file_format}")
    export(env, path)
    return load(path)


def _transitions(env):
    """Mapping (s, a) -> {s': p} for the actions of `env` with successors, without zero probabilities."""
    transitions = {}
    for s, actions in env.get_transition_function().T_dict.items():
        for a, successors in actions.items():
            successors = {s_next: p for s_next, p in successors.items() if p != 0}
            if len(successors) > 0:
                transitions[(s, a)] = successors
    return transitions


def _reward(env, s, a):
    """The reward of (s, a), or 0 if it has none."""
    return env.get_reward_function().R_dict.get(s, {}).get(a, 0.0)


def _available_actions(env):
    """The (s, a) pairs of the available actions of `env`. The action mask is an array for envs with a formatter, and a
    dict s -> mask (only for states with actions) for `ExplicitEnv`s."""
    mask = env.action_mask
    rows = mask.items() if isinstance(mask, dict) else enumerate(mask)
    return {(s, int(a)) for s, mask_s in rows for a in np.flatnonzero(mask_s)}


def _initial_distribution(env):
    """Mapping s -> probability for the initial states of `env` (with positive probability)."""
    return {s: prob for s, prob in env.initial_states.distr_dict.items() if prob > 0}


def _state_labels(env):
    """Mapping label -> states for the labels with states: of the loaded model for envs with a formatter, else as
    exported (incl. "init" and "deadlock")."""
    if hasattr(env, "formatter"):
        labels = env.formatter.labels_to_states
    else:
        labels = _build_state_label_map(env, env.get_transition_function().T_dict, overapproximate=True)
    labels = {label: set(states) for label, states in labels.items()}
    return {label: states for label, states in labels.items() if len(states) > 0}


def assert_same_env(env, loaded, valuations=True):
    """`loaded` is `env` after a round trip through a file: the same states, actions, transitions, rewards, initial
    states and state labels, and, if both have them, the same action names and (if `valuations`) state valuations."""
    assert loaded.nr_states == env.nr_states
    assert loaded.nr_actions == env.nr_actions

    transitions, loaded_transitions = _transitions(env), _transitions(loaded)
    assert loaded_transitions.keys() == transitions.keys()
    for (s, a), successors in transitions.items():
        assert loaded_transitions[(s, a)] == pytest.approx(successors)
        assert _reward(loaded, s, a) == pytest.approx(_reward(env, s, a))
    assert _available_actions(loaded) == transitions.keys()

    assert _initial_distribution(loaded) == pytest.approx(_initial_distribution(env))
    assert _state_labels(loaded) == _state_labels(env)

    if hasattr(env, "formatter") and hasattr(loaded, "formatter"):
        assert loaded.formatter.action_to_label == env.formatter.action_to_label
        assert loaded.formatter.reward_labels == env.formatter.reward_labels
        if valuations:
            assert loaded.formatter.has_state_valuations == env.formatter.has_state_valuations
            assert loaded.formatter.var_order == env.formatter.var_order
            assert loaded.formatter.state_to_values == env.formatter.state_to_values


def _interval_transitions(env):
    """Mapping (s, a) -> {s': (lower, upper)} for the actions of `env` with successors. Point envs have point
    intervals [p, p]."""
    if not isinstance(env, IntervalExplicitEnv):
        return {sa: {s_next: (p, p) for s_next, p in successors.items()} for sa, successors in _transitions(env).items()}
    transitions = {}
    for s, actions in env.get_interval_transition_function().T_dict.items():
        for a, successors in actions.items():
            successors = {s_next: bounds for s_next, bounds in successors.items() if bounds[1] != 0}
            if len(successors) > 0:
                transitions[(s, a)] = successors
    return transitions


def _interval_reward(env, s, a):
    """The reward interval of (s, a), [r, r] for point envs."""
    if not isinstance(env, IntervalExplicitEnv):
        return _reward(env, s, a), _reward(env, s, a)
    return env.get_interval_reward_function().R_dict.get(s, {}).get(a, (0.0, 0.0))


def assert_same_interval_env(env, loaded):
    """`loaded` is `env` after a round trip through a file as interval MDP: an `IntervalExplicitEnv` with the same
    states, actions, transition and reward intervals (point intervals if `env` has no intervals), initial states
    and state labels."""
    assert isinstance(loaded, IntervalExplicitEnv)
    assert loaded.nr_states == env.nr_states
    assert loaded.nr_actions == env.nr_actions

    transitions, loaded_transitions = _interval_transitions(env), _interval_transitions(loaded)
    assert loaded_transitions.keys() == transitions.keys()
    for (s, a), successors in transitions.items():
        assert loaded_transitions[(s, a)].keys() == successors.keys()
        for s_next, bounds in successors.items():
            assert loaded_transitions[(s, a)][s_next] == pytest.approx(bounds)
        assert _interval_reward(loaded, s, a) == pytest.approx(_interval_reward(env, s, a))
    assert _available_actions(loaded) == transitions.keys()

    assert _initial_distribution(loaded) == pytest.approx(_initial_distribution(env))
    assert _state_labels(loaded) == _state_labels(env)


def compare_mdps(mdp1, mdp2, from_drn=False):
    """
    Helper function to compare two stormpy MDPs
    """
    # compare the MDPs, should have rebuilt the same thing
    # Transitions/shape
    assert mdp1.nr_states == mdp2.nr_states
    assert mdp1.transition_matrix.nr_columns == mdp2.transition_matrix.nr_columns
    assert mdp1.transition_matrix.nr_rows == mdp2.transition_matrix.nr_rows
    assert mdp1.transition_matrix.nr_entries == mdp2.transition_matrix.nr_entries
    assert mdp1.nr_choices == mdp2.nr_choices
    assert mdp1.nr_states == mdp2.nr_states

    # Choice labeling
    if mdp1.has_choice_labeling():
        assert mdp2.has_choice_labeling()
        # the original action labels are kept
        assert mdp1.choice_labeling.get_labels() == mdp2.choice_labeling.get_labels()
    else:
        assert not mdp2.has_choice_labeling()

    # State labeling
    assert mdp1.labeling.get_labels() == mdp2.labeling.get_labels()

    if not from_drn:
        # State valuations
        if mdp1.has_state_valuations():
            assert mdp2.has_state_valuations()
            for s in range(mdp1.nr_states):
                assert mdp1.states[s].valuations == mdp2.states[s].valuations
        else:
            assert not mdp2.has_state_valuations()

    # Rewards
    assert len(mdp1.reward_models) == len(mdp2.reward_models)


# ---------------------------------------------------------------------------
# Export to stormpy and files
# ---------------------------------------------------------------------------


def test_build_stormpy_mdp():
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))
    mdp_2 = build_stormpy_mdp(env)

    compare_mdps(mdp, mdp_2)


def test_export_to_stormpy():
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))

    mdp_2 = export_to_stormpy_mdp(env)

    assert isinstance(mdp_2, stormpy.storage.SparseMdp)
    compare_mdps(mdp, mdp_2)


def test_export_to_drn(tmp_path):
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))

    out_path = str(tmp_path / "out_mdp.drn")
    export_to_drn(env, out_path)

    # check whether the exported drn can be re-used for a new sp env
    options_2 = stormpy._core.DirectEncodingParserOptions()
    if mdp.has_choice_labeling():
        options_2.build_choice_labels = True
    mdp_2 = stormpy.build_model_from_drn(out_path, options_2)

    compare_mdps(mdp, mdp_2, from_drn=True)


def test_export_to_umb(tmp_path):
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))

    out_path = str(tmp_path / "out_mdp.umb")
    export_to_umb(env, out_path)
    mdp_2 = stormpy.build_from_umb(out_path)

    assert isinstance(mdp_2, stormpy.storage.SparseMdp)
    compare_mdps(mdp, mdp_2)


@pytest.mark.parametrize("file_format", list(FORMATS))
def test_export_default_path(file_format, tmp_path, monkeypatch):
    """Without a path, the model is written to cwd/mdp.<format>."""
    monkeypatch.chdir(tmp_path)
    export, load = FORMATS[file_format]
    env = _make_explicit_env()
    export(env)
    assert_same_env(env, load(str(tmp_path / f"mdp.{file_format}")))


@pytest.mark.parametrize("file_format", list(FORMATS))
def test_export_wrapped_env(file_format, tmp_path):
    """Envs wrapped in gymnasium wrappers are exported as the env they wrap."""
    env = _make_explicit_env()
    assert_same_env(env, _round_trip(gym.Wrapper(env), file_format, tmp_path))


@pytest.mark.parametrize("file_format", list(FORMATS))
def test_export_warns_non_uniform_initial_distribution(file_format, tmp_path):
    """Files only store the set of initial states, so loading makes the initial distribution uniform."""
    env = _make_explicit_env(initial_state_distr=(0.25, 0.75, 0.0))
    with pytest.warns(UserWarning, match="initial state distribution is not uniform"):
        loaded = _round_trip(env, file_format, tmp_path)
    assert _initial_distribution(loaded) == pytest.approx({0: 0.5, 1: 0.5})


# ---------------------------------------------------------------------------
# Round trips of existing models (PRISM, DRN and UMB files)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("file_format", list(FORMATS))
def test_io_prism_input(prism_model, file_format, tmp_path):
    """Standard models loaded from prism files survive a round trip through a file."""
    _, env = prism_model
    loaded = _round_trip(env, file_format, tmp_path)
    assert isinstance(loaded, FrameworkExplicitEnv)
    assert_same_env(env, loaded, valuations=file_format == "umb")


@pytest.mark.parametrize("file_format", list(FORMATS))
def test_io_storm_file_input(prism_model, file_format, tmp_path):
    """Standard models written to a file by stormpy (i.e., by another tool) load as the same env as from the prism
    file, and survive another round trip through both file formats."""
    mdp, env = prism_model
    path = str(tmp_path / f"storm.{file_format}")
    STORM_EXPORTS[file_format](mdp, path)
    loaded = FORMATS[file_format][1](path)
    assert_same_env(env, loaded, valuations=file_format == "umb")

    for other_format in FORMATS:
        reloaded = _round_trip(loaded, other_format, tmp_path, name="reloaded")
        assert_same_env(loaded, reloaded, valuations=file_format == other_format == "umb")


@pytest.mark.parametrize("file_format", list(FORMATS))
def test_load_committed_files(file_format):
    """tests/test_2d.umb and tests/test_2d.drn (written by stormpy from tests/test_2d.prism) still load."""
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv.from_stormpy(mdp)
    loaded = FORMATS[file_format][1](os.path.join(TESTS_DIR, f"test_2d.{file_format}"))
    assert_same_env(env, loaded, valuations=file_format == "umb")


@pytest.mark.parametrize("model_name", ["test_2d", "resource_gathering"])
def test_load_umb_featured_states(model_name, tmp_path):
    """UMB keeps the state valuations (also of boolean variables), so the loaded env supports factored states."""
    path, constants = STANDARD_MODELS[model_name]
    mdp = load_stormpy_model(path, constants=constants)
    env = FrameworkExplicitEnv.from_stormpy(mdp, flat=False)
    path = str(tmp_path / "model.umb")
    export_to_umb(env, path)
    loaded = load_from_umb(path, flat=False)
    np.testing.assert_array_equal(loaded.observation_space.nvec, env.observation_space.nvec)
    for s in range(env.nr_states):
        assert loaded.decode(s) == env.decode(s)


# ---------------------------------------------------------------------------
# Round trips of learned abstractions and explicit envs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("file_format", list(FORMATS))
def test_io_abstraction(abstraction, file_format, tmp_path):
    """Learned abstractions survive a round trip through a file, including the indices of their actions."""
    loaded = _round_trip(abstraction, file_format, tmp_path)
    assert_same_env(abstraction, loaded)


@pytest.mark.parametrize("file_format", list(FORMATS))
def test_io_many_actions(file_format, tmp_path):
    """Envs with more than 10 actions keep their action indices: the choice labels "0", ..., "11" are numbers,
    not names (which would order "10" before "2")."""
    env = _make_explicit_env(nr_actions=12)
    assert_same_env(env, _round_trip(env, file_format, tmp_path))


def test_io_many_actions_imdp(tmp_path):
    """Interval MDPs with more than 10 actions keep their action indices as well."""
    env = _make_explicit_env(nr_actions=12)
    path = str(tmp_path / "model.umb")
    export_to_umb(env, path, intervals=True)
    assert_same_interval_env(env, load_from_umb(path))


def test_load_without_valuations_featured_raises(abstraction, tmp_path):
    """Envs without state valuations only have flat states."""
    path = str(tmp_path / "model.umb")
    export_to_umb(abstraction, path)
    with pytest.raises(ValueError, match="featured state representation"):
        load_from_umb(path, flat=False)


def test_load_umb_without_choice_labels(tmp_path):
    """Files without choice labels (e.g., from other tools): the i-th choice of a state is action i."""
    mdp = helper_to_create_stormpy_model_for_tests(stormpy.storage.SparseMdp, rows=[{1: 1.0}, {0: 0.5, 1: 0.5}, {1: 1.0}],
                               row_groups=[0, 2], reward_models={"r": (None, [1.0, 2.0, 3.0])})
    path = str(tmp_path / "unlabelled.umb")
    stormpy.export_to_umb(mdp, path)

    env = load_from_umb(path)

    assert env.nr_actions == 2
    assert _transitions(env) == {(0, 0): {1: 1.0}, (0, 1): {0: 0.5, 1: 0.5}, (1, 0): {1: 1.0}}
    assert [_reward(env, s, a) for s, a in [(0, 0), (0, 1), (1, 0)]] == [1.0, 2.0, 3.0]
    assert _available_actions(env) == {(0, 0), (0, 1), (1, 0)}


def test_load_rejects_unsupported_models(tmp_path):
    """Only MDPs and interval MDPs can be loaded as envs."""
    dtmc = helper_to_create_stormpy_model_for_tests(stormpy.storage.SparseDtmc, rows=[{1: 1.0}, {1: 1.0}])
    path = str(tmp_path / "dtmc.umb")
    stormpy.export_to_umb(dtmc, path)
    with pytest.raises(NotImplementedError, match="SparseDtmc"):
        load_from_umb(path)


# ---------------------------------------------------------------------------
# Round trips of interval MDPs
# ---------------------------------------------------------------------------


def test_io_imdp_prism_input(prism_model, tmp_path):
    """Standard models exported as interval MDP (with point intervals) load as `IntervalExplicitEnv`, and survive
    another round trip."""
    mdp, env = prism_model
    path = str(tmp_path / "model.umb")
    export_to_umb(env, path, intervals=True)
    loaded = load_from_umb(path)
    assert_same_interval_env(env, loaded)
    assert loaded.nr_rewards == len(mdp.reward_models)

    path = str(tmp_path / "reloaded.umb")
    export_to_umb(loaded, path)
    assert_same_interval_env(loaded, load_from_umb(path))


def test_io_imdp_abstraction(interval_abstraction, tmp_path):
    """Learned abstractions with intervals are exported as interval MDP (with the reward intervals) by default, and
    survive a round trip through a UMB file, including the indices of their actions."""
    loaded = _round_trip(interval_abstraction, "umb", tmp_path)
    assert_same_interval_env(interval_abstraction, loaded)


def test_io_imdp_hand_made(tmp_path):
    """An interval MDP with a deadlock and reward intervals survives a round trip through a UMB file; its point
    estimates are derived from the intervals again."""
    env = _make_interval_env()
    loaded = _round_trip(env, "umb", tmp_path)
    assert_same_interval_env(env, loaded)
    assert _transitions(loaded) == _transitions(env)
    assert all(s != 2 for s, _ in _available_actions(loaded))


@pytest.mark.parametrize("file_format", list(FORMATS))
def test_export_interval_env_as_mdp(file_format, tmp_path):
    """Exporting an interval env as MDP uses its point estimates."""
    env = _make_interval_env()
    export, load = FORMATS[file_format]
    path = str(tmp_path / f"model.{file_format}")
    if file_format == "umb":
        export(env, path, intervals=False)
    else:
        export(env, path)
    loaded = load(path)
    assert isinstance(loaded, FrameworkExplicitEnv)
    assert_same_env(env, loaded)


def test_load_imdp_featured_raises(tmp_path):
    """Interval MDPs only have flat states."""
    path = str(tmp_path / "model.umb")
    export_to_umb(_make_interval_env(), path)
    with pytest.raises(ValueError, match="interval MDPs only have flat states"):
        load_from_umb(path, flat=False)


def test_load_imdp_without_choice_labels(tmp_path):
    """Interval MDPs without choice labels (e.g., from other tools): the i-th choice of a state is action i.
    State rewards are added to the rewards of all actions of the state."""
    imdp = helper_to_create_stormpy_model_for_tests(
        stormpy.storage.SparseIntervalMdp,
        rows=[{1: (1.0, 1.0)}, {0: (0.4, 0.6), 1: (0.4, 0.6)}, {1: (1.0, 1.0)}],
        row_groups=[0, 2],
        reward_models={"r": ([(0.5, 1.0), (0.0, 0.0)], [(1.0, 1.0), (2.0, 3.0), (3.0, 3.0)])},
    )
    path = str(tmp_path / "unlabelled.umb")
    stormpy.export_to_umb(imdp, path)

    env = load_from_umb(path)

    assert isinstance(env, IntervalExplicitEnv)
    assert env.nr_actions == 2
    assert _interval_transitions(env) == {
        (0, 0): {1: (1.0, 1.0)}, (0, 1): {0: (0.4, 0.6), 1: (0.4, 0.6)}, (1, 0): {1: (1.0, 1.0)}
    }
    assert [_interval_reward(env, s, a) for s, a in [(0, 0), (0, 1), (1, 0)]] == [(1.5, 2.0), (2.5, 4.0), (3.0, 3.0)]
    assert _available_actions(env) == {(0, 0), (0, 1), (1, 0)}


def test_load_imdp_with_several_reward_models_raises(tmp_path):
    """`IntervalExplicitEnv` has a single reward function."""
    imdp = helper_to_create_stormpy_model_for_tests(
        stormpy.storage.SparseIntervalMdp,
        rows=[{1: (1.0, 1.0)}, {1: (1.0, 1.0)}],
        row_groups=[0, 1],
        reward_models={"a": (None, [(1.0, 1.0), (0.0, 0.0)]), "b": (None, [(2.0, 2.0), (0.0, 0.0)])},
    )
    path = str(tmp_path / "rewards.umb")
    stormpy.export_to_umb(imdp, path)
    with pytest.raises(NotImplementedError, match="more than one reward model"):
        load_from_umb(path)


def test_load_imdp_with_repeated_labels(tmp_path):
    """Several choices of a state with the same label are separate actions ("go", "go_2"), as in `StormpyFormatter`."""
    imdp = helper_to_create_stormpy_model_for_tests(
        stormpy.storage.SparseIntervalMdp,
        rows=[{1: (1.0, 1.0)}, {0: (0.4, 0.6), 1: (0.4, 0.6)}, {1: (1.0, 1.0)}],
        row_groups=[0, 2],
        reward_models={"r": (None, [(1.0, 1.0), (2.0, 3.0), (0.0, 0.0)])},
        choice_labels=["go", "go", "go"],
    )
    path = str(tmp_path / "repeated.umb")
    stormpy.export_to_umb(imdp, path)

    env = load_from_umb(path)

    # actions: go (0) and go_2 (1)
    assert env.nr_actions == 2
    assert _interval_transitions(env) == {
        (0, 0): {1: (1.0, 1.0)}, (0, 1): {0: (0.4, 0.6), 1: (0.4, 0.6)}, (1, 0): {1: (1.0, 1.0)}
    }
    assert [_interval_reward(env, s, a) for s, a in [(0, 0), (0, 1), (1, 0)]] == [(1.0, 1.0), (2.0, 3.0), (0.0, 0.0)]


def test_export_to_julia_pomdp(): ...  # TODO functionality not implemented


def test_export_to_prism(): ...  # TODO functionality not implemented
