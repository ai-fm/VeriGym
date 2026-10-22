"""
Tests that interval envs behave exactly like the point envs of their point estimates, apart from the intervals:
    IntervalFrameworkExplicitEnv vs FrameworkExplicitEnv (loaded from the MDP of its point estimates)
    IntervalExplicitEnv vs ExplicitEnv (with its point estimates)
Both are stepped with the same seeds and actions; every reset and step must give the same results, except that the
info of the interval env also has the intervals of the transition.
"""

import os

import numpy as np
import pytest
import stormpy

from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.abstraction.learn_abstraction import create_abstraction
from verigym.environments.explicitenv import AbstractedExplicitEnv
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv, IntervalFrameworkExplicitEnv
from verigym.environments.generativeenv import GenerativeEnv
from verigym.environments.interval_explicitenv import IntervalAbstractedExplicitEnv
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_imdp, build_stormpy_mdp, load_stormpy_model
from verigym.policy.randomized import RandomizedPolicy

from utils import helper_to_create_stormpy_model_for_tests, make_interval_mdp, make_original_env

TESTS_DIR = os.path.dirname(__file__)
PRISM_FILES = os.path.join(TESTS_DIR, "..", "docs", "prism_files")
SEED = 12929582
INTERVAL_KEYS = {"reward_interval", "transition_interval"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_index(env, state):
    """The index of a state of `env`, which is a valuation tuple for factored states."""
    return env.encode(state) if hasattr(env, "encode") else state


def _available_actions(env, state_idx):
    """The available actions of a state (states without an entry in the action mask have none)."""
    return np.flatnonzero(env.action_mask.get(state_idx, np.zeros(env.nr_actions)))


def _rollout(env, seed, n_steps):
    """Steps `env` with a random available action (from a seeded RNG) for `n_steps` steps, resetting after
    termination. `env.reset(seed=seed)` seeds the env's generator for sampling transitions and initial states.
    Returns the result of every reset and step, with the indices of the states before and after the step."""
    action_rng = np.random.default_rng(seed)

    records = [("reset", *env.reset(seed=seed))]
    for _ in range(n_steps):
        state_idx = _state_index(env, env.state)
        available = _available_actions(env, state_idx)
        action = int(action_rng.choice(available)) if len(available) > 0 else 0
        obs, reward, terminated, truncated, info = env.step(action)
        records.append(("step", state_idx, action, _state_index(env, obs), obs, reward, terminated, truncated, info))
        if terminated or truncated:
            records.append(("reset", *env.reset()))
    return records


def _assert_same_info(interval_info, point_info):
    assert interval_info.keys() - INTERVAL_KEYS == point_info.keys()
    for key, value in point_info.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(interval_info[key], value)
        else:
            assert interval_info[key] == value, key


def assert_same_steps(interval_env, point_env, seed=0, n_steps=300):
    """Stepping both envs the same way gives the same results; the interval env's info also has the intervals of
    each transition (or [0, 0] for unavailable actions)."""
    interval_records, point_records = _rollout(interval_env, seed, n_steps), _rollout(point_env, seed, n_steps)
    assert len(interval_records) == len(point_records)
    for interval_record, point_record in zip(interval_records, point_records):
        assert interval_record[:-1] == point_record[:-1]
        _assert_same_info(interval_record[-1], point_record[-1])

        if interval_record[0] == "step":
            _, state, action, next_state, *_, info = interval_record
            if action in interval_env.get_interval_transition_function().T_dict.get(state, {}):
                assert info["reward_interval"] == interval_env.get_interval_reward(state, action)
                assert info["transition_interval"] == interval_env.get_interval_transition_function()[state][action][next_state]
            else:
                assert info["reward_interval"] == info["transition_interval"] == [0.0, 0.0]


def _interval_abstraction(assume_iid):
    """Learned abstraction of CartPole with intervals (seeded)."""
    env, num_steps, bin_edges_per_dim = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    return create_abstraction(
        original_env=generative_env,
        abstraction_mapper=linspace_mapper(env, bin_edges_per_dim, bin_edges_per_dim),
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=num_steps,
        intervals=True,
        assume_iid=assume_iid,
        multithreading=False,
        seed=SEED,
    )


def _uncertain_self_loop_imdp():
    """2 states. State 0 moves to state 1 with "go"; state 1 has a single choice "stay", a self-loop with interval
    [0.8, 1], whose probability is 1 in every distribution within the interval."""
    return helper_to_create_stormpy_model_for_tests(
        stormpy.storage.SparseIntervalMdp,
        rows=[{1: (1.0, 1.0)}, {1: (0.8, 1.0)}],
        row_groups=[0, 1],
        choice_labels=["go", "stay"],
    )


def _prism_imdp(path, constants=None):
    """The model of a prism file as interval MDP with point intervals (and its valuations)."""
    return build_stormpy_imdp(FrameworkExplicitEnv.from_stormpy(load_stormpy_model(path, constants=constants)))


IMDPS = {
    "hand_made": make_interval_mdp,
    "uncertain_self_loop": _uncertain_self_loop_imdp,
    "abstraction": lambda: build_stormpy_imdp(_interval_abstraction(False), use_reward_uncertainty=True),
    "test_2d": lambda: _prism_imdp(os.path.join(TESTS_DIR, "test_2d.prism")),
    "pacman": lambda: _prism_imdp(os.path.join(PRISM_FILES, "pacman.v2.nm"), "MAXSTEPS=5"),
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("imdp_name", list(IMDPS))
@pytest.mark.parametrize("seed", [0, 1])
def test_interval_framework_env_steps_like_point_env(imdp_name, seed):
    """An `IntervalFrameworkExplicitEnv` steps exactly like the `FrameworkExplicitEnv` of its point estimates (the MDP
    of its point estimates, with the same action names, labels and valuations), apart from the interval info."""
    interval_env = FrameworkExplicitEnv.from_stormpy(IMDPS[imdp_name]())
    assert isinstance(interval_env, IntervalFrameworkExplicitEnv)
    point_env = FrameworkExplicitEnv.from_stormpy(build_stormpy_mdp(interval_env))
    assert not isinstance(point_env, IntervalFrameworkExplicitEnv)

    assert_same_steps(interval_env, point_env, seed=seed)


def test_interval_framework_env_steps_like_point_env_factored():
    """The same for factored states (`flat=False`)."""
    imdp = IMDPS["test_2d"]()
    interval_env = FrameworkExplicitEnv.from_stormpy(imdp, flat=False)
    point_env = FrameworkExplicitEnv.from_stormpy(build_stormpy_mdp(interval_env), flat=False)

    assert_same_steps(interval_env, point_env)


@pytest.mark.parametrize("assume_iid", [False, True], ids=["noniid", "iid"])
@pytest.mark.parametrize("seed", [0, 1])
def test_interval_explicit_env_steps_like_point_env(assume_iid, seed):
    """An `IntervalExplicitEnv` steps exactly like the `ExplicitEnv` of its point estimates, apart from the interval
    info."""
    interval_env = _interval_abstraction(assume_iid)
    assert isinstance(interval_env, IntervalAbstractedExplicitEnv)
    point_env = AbstractedExplicitEnv(
        nr_states=interval_env.nr_states,
        nr_actions=interval_env.nr_actions,
        initial_state_distr=interval_env.initial_states,
        transition_function=interval_env.get_transition_function(),
        reward_function=interval_env.get_reward_function(),
        nr_rewards=interval_env.nr_rewards,
    )

    assert_same_steps(interval_env, point_env, seed=seed)
