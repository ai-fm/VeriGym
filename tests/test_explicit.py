"""
Tests for:
    FrameworkExplicitEnv.from_stormpy + functionality
    ExplicitEnv
"""

import os
import gymnasium as gym

from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv
from verigym.environments.explicitenv import ExplicitEnv
from verigym.frameworks.stormpy.stormpy_utils import load_stormpy_model
from verigym.frameworks.stormpy.formatter import StormpyFormatter

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
        transition_function=formatter.transition_function,
        reward_function=formatter.reward_function,
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

    env = ExplicitEnv(
        nr_states=formatter.nr_states,
        nr_actions=formatter.nr_actions,
        nr_rewards=formatter.n_rewards,
        initial_state_distr=formatter.initial_states,
        transition_function=formatter.transition_function,
        reward_function=formatter.reward_function,
    )
    for _ in range(10):
        obs, info = env.reset()
        while True:
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break


def test_explicit_env_vectorized():
    mdp = load_stormpy_model(PRISM_TEST)
    formatter = StormpyFormatter(mdp)
    n_envs = 3

    gym.make_vec(
        "ExplicitEnv-v0",
        nr_states=formatter.nr_states,
        nr_actions=formatter.nr_actions,
        nr_rewards=formatter.n_rewards,
        initial_state_distr=formatter.initial_states,
        transition_function=formatter.transition_function,
        reward_function=formatter.reward_function,
        num_envs=n_envs,
        vectorization_mode="sync",
    )
