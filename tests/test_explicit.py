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
