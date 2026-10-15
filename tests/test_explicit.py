"""
Tests for:
    FrameworkExplicitEnv.from_stormpy + functionality
    AbstractedExplicitEnv
"""

import os
from collections import defaultdict

import gymnasium as gym
import numpy as np
import pytest
import stormpy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv
from verigym.environments.explicitenv import AbstractedExplicitEnv
from verigym.environments.reward_func import RewardFunction
from verigym.environments.state_distributions import StateDistribution
from verigym.environments.transition_func import TransitionFunction
from verigym.frameworks.stormpy.stormpy_utils import load_stormpy_model
from verigym.frameworks.stormpy.formatter import StormpyFormatter

from utils import helper_to_create_stormpy_model_for_tests

PRISM_TEST = os.path.join(os.getcwd(), "tests/test_2d.prism")


def test_stormpy_env_1():
    # Test simulation
    mdp = load_stormpy_model(PRISM_TEST)
    env = gym.make(
        "FrameworkExplicitEnv-v0", model=mdp, formatter=StormpyFormatter(mdp)
    )
    for _ in range(10):
        obs, info = env.reset()
        while True:
            action = env.action_space.sample(info["action_mask"])
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break


def test_stormpy_env_2():
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv.from_stormpy(mdp)
    for _ in range(10):
        obs, info = env.reset()
        while True:
            action = env.action_space.sample(info["action_mask"])
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

def test_factored_stormpy_env():
    """
    Test initializing envs from stormpy with factored representation, using `gym.spaces.MultiDiscrete` as observation space.
    """
    mdp = load_stormpy_model(PRISM_TEST)
    # flat env for comparison:
    env = FrameworkExplicitEnv.from_stormpy(mdp, flat=True)
    # factored env:
    env2 = FrameworkExplicitEnv.from_stormpy(mdp, flat=False)

    assert isinstance(env2.observation_space, gym.spaces.MultiDiscrete)
    assert env.nr_states == env2.nr_states
    assert env.nr_actions == env2.nr_actions

    T1 = env.get_transition_function()
    T2 = env2.get_transition_function()

    R1 = env.get_reward_function()
    R2 = env2.get_reward_function()
    for s in range(env2.nr_states):
        if s not in T1.T_dict.keys():
            assert s not in T2.T_dict.keys()
            continue
        for a in range(env2.nr_actions):
            if a not in T1[s].keys():
                assert a not in T2[s].keys()
                continue
            for s_prime in range(env2.nr_states):
                assert T1[s][a][s_prime] == T2[s][a][s_prime]
            s_factored = env2.decode(s)
            s_encode = env2.encode(s_factored)

            env2.set_state(s_factored)
            info2 = env2._get_info()
            obs2, _, _, _, _ = env2.step(a)

            env.set_state(s)
            info = env._get_info()

            assert R1[s][a] == R2[s][a]
            assert s_factored in env2.observation_space
            assert s == s_encode
            assert obs2 in env2.observation_space
            assert info["state_valuations"] == info2["state_valuations"]


def test_factored_stormpy_env_reset():
    """`reset` of an env with factored states returns a factored state of the observation space, like the steps do.
    Regression: it returned the index of the initial state, so getting the info failed."""
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv.from_stormpy(mdp, flat=False)

    obs, info = env.reset(seed=0)

    assert obs == env.state
    assert obs in env.observation_space
    assert env.encode(obs) in set(mdp.initial_states)
    assert info["state_valuations"] == env.formatter.state_to_values[env.encode(obs)]


@pytest.mark.parametrize("prism_file", [
    "test_2d.prism",
    "test_bool.prism",
    pytest.param("test_2d_underground.prism", marks=pytest.mark.xfail(
        strict=True,
        reason="The MultiDiscrete observation space of factored states starts at 0 for every variable, but the "
               "variables of this model (x, y in [-2..2]) have negative values.",
    )),
])
def test_factored_states_in_observation_space(prism_file):
    """Every factored state of an env (e.g., its initial state) is in its observation space, so it can be set."""
    mdp = load_stormpy_model(os.path.join(os.getcwd(), "tests", prism_file))
    env = FrameworkExplicitEnv.from_stormpy(mdp, flat=False)

    obs, _ = env.reset(seed=0)
    assert obs in env.observation_space
    for s in range(env.nr_states):
        state = env.decode(s)
        assert state in env.observation_space, f"state {s} = {state} is not in {env.observation_space}"
        env.set_state(state)
        assert env.encode(env.state) == s




def test_stormpy_env_vectorized():
    # Test vectorization
    mdp = load_stormpy_model(PRISM_TEST)
    n_envs = 3
    envs = gym.make_vec(
        "FrameworkExplicitEnv-v0",
        model=mdp,
        formatter=StormpyFormatter(mdp),
        num_envs=n_envs,
        vectorization_mode="sync",
    )

    assert envs.num_envs == n_envs


def test_explicit_env_1():
    mdp = load_stormpy_model(PRISM_TEST)
    formatter = StormpyFormatter(mdp)

    env = gym.make(
        "ExplicitEnv-v0",
        nr_states=formatter.nr_states,
        nr_actions=formatter.nr_actions,
        nr_rewards=formatter.n_rewards,
        initial_state_distr=formatter.initial_states,
        transition_function=formatter._convert_transition_matrix(mdp),
        reward_function=formatter._convert_reward_matrix(mdp),
    )
    for _ in range(10):
        obs, info = env.reset()
        while True:
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break


def test_explicit_env_2():
    mdp = load_stormpy_model(PRISM_TEST)
    formatter = StormpyFormatter(mdp)

    env = AbstractedExplicitEnv(
        nr_states=formatter.nr_states,
        nr_actions=formatter.nr_actions,
        nr_rewards=formatter.n_rewards,
        initial_state_distr=formatter.initial_states,
        transition_function=formatter._convert_transition_matrix(mdp),
        reward_function=formatter._convert_reward_matrix(mdp),
    )
    for _ in range(10):
        obs, info = env.reset()
        while True:
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break


@pytest.mark.parametrize("vectorization_mode", ["sync", "async"])
def test_explicit_env_vectorized(vectorization_mode):
    mdp = load_stormpy_model(PRISM_TEST)
    formatter = StormpyFormatter(mdp)
    n_envs = 3

    envs = gym.make_vec(
        "ExplicitEnv-v0",
        nr_states=formatter.nr_states,
        nr_actions=formatter.nr_actions,
        nr_rewards=formatter.n_rewards,
        initial_state_distr=formatter.initial_states,
        transition_function=formatter._convert_transition_matrix(mdp),
        reward_function=formatter._convert_reward_matrix(mdp),
        num_envs=n_envs,
        vectorization_mode=vectorization_mode,
    )

    try:
        assert envs.num_envs == n_envs

        obs, infos = envs.reset(seed=0)
        assert obs.shape[0] == n_envs

        actions = envs.action_space.sample()
        obs, rewards, terminated, truncated, infos = envs.step(actions)
        assert obs.shape[0] == n_envs
        assert len(rewards) == n_envs
        assert len(terminated) == n_envs
        assert len(truncated) == n_envs

        # Take a few more steps, exercising the auto-reset path on
        # termination/truncation, to make sure the vec env keeps running.
        for _ in range(10):
            actions = envs.action_space.sample()
            obs, rewards, terminated, truncated, infos = envs.step(actions)
            assert obs.shape[0] == n_envs
            assert len(rewards) == n_envs
    finally:
        # prevents subprocesses from leaking in the async vector classes
        envs.close()


@pytest.mark.parametrize("vec_env_cls", [DummyVecEnv, SubprocVecEnv])
def test_explicit_env_vectorized_sb3(vec_env_cls):
    """
    Vectorize `ExplicitEnv-v0` using Stable-Baselines3's own vectorization
    utilities instead of `gym.make_vec`. SB3's `VecEnv`
    API is not compatible with `gymnasium.vector.VectorEnv` (different
    reset/step signatures and auto-reset conventions), so SB3 users would
    build their vec envs this way rather than wrapping a gymnasium vector env.
    """
    mdp = load_stormpy_model(PRISM_TEST)
    formatter = StormpyFormatter(mdp)
    n_envs = 3

    env_kwargs = dict(
        nr_states=formatter.nr_states,
        nr_actions=formatter.nr_actions,
        nr_rewards=formatter.n_rewards,
        initial_state_distr=formatter.initial_states,
        transition_function=formatter._convert_transition_matrix(mdp),
        reward_function=formatter._convert_reward_matrix(mdp),
    )

    envs = make_vec_env(
        "ExplicitEnv-v0",
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
    )

    try:
        assert envs.num_envs == n_envs

        obs = envs.reset()
        assert obs.shape[0] == n_envs

        actions = [envs.action_space.sample() for _ in range(n_envs)]
        obs, rewards, dones, infos = envs.step(actions)
        assert obs.shape[0] == n_envs
        assert len(rewards) == n_envs
        assert len(dones) == n_envs
        assert len(infos) == n_envs

        for _ in range(10):
            actions = [envs.action_space.sample() for _ in range(n_envs)]
            obs, rewards, dones, infos = envs.step(actions)
            assert obs.shape[0] == n_envs
    finally:
        # prevents subprocesses from leaking in the async vector classes
        envs.close()


def test_explicit_env_sparse_action_mask():
    """
    The action mask and terminal states only store the states of the transition function, so a large
    but sparse ExplicitEnv stays small. States that are not stored have no actions and are terminal.
    """
    nr_states, nr_actions = 10**8, 2
    T_dict = defaultdict(dict)
    T_dict[0][0] = defaultdict(float, {1: 0.5, 99_999_999: 0.5})
    T_dict[1][1] = defaultdict(float, {1: 1.0})  # self-loop only: terminal
    R_dict = defaultdict(dict)
    R_dict[0][0] = 2.0
    R_dict[1][1] = 3.0
    env = AbstractedExplicitEnv(
        nr_states=nr_states,
        nr_actions=nr_actions,
        initial_state_distr=StateDistribution(nr_states, {0: 1.0}),
        transition_function=TransitionFunction(nr_states, nr_actions, T_dict),
        reward_function=RewardFunction(nr_states, nr_actions, R_dict),
    )

    assert env.action_mask.keys() == {0, 1}
    np.testing.assert_array_equal(env.action_mask[0], [1, 0])
    np.testing.assert_array_equal(env.action_mask[1], [0, 1])
    assert all(mask_s.dtype == np.int8 for mask_s in env.action_mask.values()), "action_space.sample requires int8"
    assert env.terminal_states == {0: False, 1: True}

    # unavailable action: stay in the state without reward
    env.reset(seed=0)
    obs, rew, terminated, truncated, info = env.step(1)
    assert (obs, rew, terminated) == (0, 0.0, False)

    # a state that is not stored has no actions and is terminal
    env.set_state(99_999_999)
    obs, rew, terminated, truncated, info = env.step(0)
    assert (obs, rew, terminated) == (99_999_999, 0.0, True)
    assert env.action_mask.keys() == {0, 1}, "stepping must not add states to the action mask"
    assert env.terminal_states.keys() == {0, 1}, "stepping must not add states to the terminal states"


def _self_loops_mdp():
    """2 states. State 0 moves to state 1 with "go"; state 1 has two actions, "go" and "stay", which both loop."""
    return helper_to_create_stormpy_model_for_tests(
        stormpy.storage.SparseMdp,
        rows=[{1: 1.0}, {1: 1.0}, {1: 1.0}],
        row_groups=[0, 1],
        choice_labels=["go", "go", "stay"],
    )


def test_framework_env_terminal_with_several_self_loops():
    """A state whose available actions all loop with probability 1 is terminal, also if it has several actions, as
    for `ExplicitEnv`s. Regression: only states with a single choice that loops were terminal."""
    env = FrameworkExplicitEnv.from_stormpy(_self_loops_mdp())
    assert env.terminal_states == {0: False, 1: True}

    env.set_state(1)
    _, _, terminated, _, _ = env.step(env.formatter.label_to_action["stay"])
    assert terminated


def test_framework_env_terminates_on_entering_terminal_state():
    """`terminated` is whether the state after the step is terminal, as in gymnasium and for `ExplicitEnv`s.
    Regression: it was whether the state before the step was terminal, so an episode took one more step after
    entering a terminal state."""
    env = FrameworkExplicitEnv.from_stormpy(_self_loops_mdp())
    env.reset(seed=0)

    next_state, _, terminated, _, _ = env.step(env.formatter.label_to_action["go"])
    assert next_state == 1
    assert terminated


def test_explicit_env_info():
    """Like a `FrameworkExplicitEnv`, an `ExplicitEnv` reports the int8 action mask of the new state in its info (so
    actions can be sampled with it), and the rewards as a list."""
    mdp = load_stormpy_model(PRISM_TEST)
    formatter = StormpyFormatter(mdp)
    env = AbstractedExplicitEnv(
        nr_states=formatter.nr_states,
        nr_actions=formatter.nr_actions,
        nr_rewards=formatter.n_rewards,
        initial_state_distr=formatter.initial_states,
        transition_function=formatter._convert_transition_matrix(mdp),
        reward_function=formatter._convert_reward_matrix(mdp),
    )
    obs, info = env.reset(seed=0)
    np.testing.assert_array_equal(info["action_mask"], env.action_mask[obs])
    for _ in range(20):
        action = env.action_space.sample(info["action_mask"])
        obs, rew, terminated, truncated, info = env.step(action)
        assert info["action_mask"].dtype == np.int8
        np.testing.assert_array_equal(info["action_mask"], env.action_mask.get(obs, np.zeros(env.nr_actions)))
        assert info["reward"] == [rew]
        if terminated:
            break


@pytest.mark.parametrize("flat", [True, False])
def test_explicit_env_steps_like_framework_env(flat):
    """An `ExplicitEnv` with the components of a `FrameworkExplicitEnv` steps exactly like it: the same next states,
    rewards and termination, and the same info apart from the formatter's labels and valuations."""
    mdp = load_stormpy_model(PRISM_TEST)
    fw_env = FrameworkExplicitEnv.from_stormpy(mdp, flat=flat)
    ex_env = AbstractedExplicitEnv(
        nr_states=fw_env.nr_states,
        nr_actions=fw_env.nr_actions,
        nr_rewards=fw_env.nr_rewards,
        initial_state_distr=fw_env.initial_states,
        transition_function=fw_env.transition_function,
        reward_function=fw_env.reward_function,
    )
    formatter_keys = {"state_valuations", "state_labels", "reward_labels", "action_labels"}
    rng = np.random.default_rng(0)
    for s in range(fw_env.nr_states):
        for a in range(fw_env.nr_actions):
            seed = int(rng.integers(2**31))
            results = []
            for env in (fw_env, ex_env):
                env.state = env.decode(s)
                env.rng.seed(seed)
                next_state, rew, terminated, truncated, info = env.step(a)
                results.append((env.encode(next_state), rew, terminated, truncated, info))
            (fw_next, fw_rew, fw_term, fw_trunc, fw_info), (ex_next, ex_rew, ex_term, ex_trunc, ex_info) = results
            assert (fw_next, fw_rew, fw_term, fw_trunc) == (ex_next, ex_rew, ex_term, ex_trunc)
            assert fw_info.keys() - formatter_keys == ex_info.keys()
            np.testing.assert_array_equal(fw_info["action_mask"], ex_info["action_mask"])
            assert fw_info["reward"] == ex_info["reward"]
