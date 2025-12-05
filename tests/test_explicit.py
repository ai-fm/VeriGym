"""
Tests for classes:
    ExplicitEnv/StormpyEnv
    ExplicitFormatter/StormpyExplicitFormatter
"""

import os
import gymnasium as gym

from verigym.frameworks.stormpy.stormpy_utils import load_stormpy_model

PRISM_TEST = os.path.join(os.getcwd(), "tests/test_2d.prism")

def test_stormpy_env():
    # Test simulation
    mdp = load_stormpy_model(PRISM_TEST)
    env = gym.make("StormpyEnv-v0", model=mdp)
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
    envs = gym.make_vec("StormpyEnv-v0", model=mdp, 
                       num_envs=n_envs, 
                       vectorization_mode="sync")

    assert envs.num_envs == n_envs