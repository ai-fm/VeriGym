"""
Tests for:
    FrameworkExplicitEnv.from_stormpy + functionality
    ExplicitEnv
"""

import os
import gymnasium as gym
import pytest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

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
        transition_function=formatter.transition_function,
        reward_function=formatter.reward_function,
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
        transition_function=formatter.transition_function,
        reward_function=formatter.reward_function,
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
