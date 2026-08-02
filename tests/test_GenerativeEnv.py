from verigym.environments.generativeenv import GenerativeEnv
from verigym.policy.implemented_policies import RandomizedPolicy
import gymnasium as gym
import numpy as np


def test_gymnasium_env():
    """Sanity-check that a gymnasium env can be merged and behaves equivalently."""
    env = gym.make("FrozenLake-v1", is_slippery=False)
    action = 0
    obs1, info = env.reset()
    obs2, reward, teminated, trunc, info = env.step(action)
    print("Before instantiation: ", obs1)

    # instantiate a GenerativeEnv from the gym environment
    verienv = GenerativeEnv.from_gymnasium(env)
    veriobs1, info = verienv.reset()
    veriobs2, verireward, teminated, trunc, info = env.step(action)
    print("After instantiation: ", veriobs1)

    assert obs1 == veriobs1, "Observations do not match after wrapping!"
    assert obs2 == veriobs2, "Observations after step do not match after wrapping!"
    assert reward == verireward, "Rewards do not match after wrapping!"

    print(f"{type(verienv) = }")
    print(f"{isinstance(verienv, GenerativeEnv) = }")


def test_from_gym_mutation_isolated():
    """Ensure simple attribute mutations do not affect the original env (shallow-copy only)."""
    env = gym.make("FrozenLake-v1", is_slippery=False)
    # Note: mutable internals (e.g., RNG/state arrays) are shared without a deep copy.
    env.custom_value = 1

    verienv = GenerativeEnv.from_gymnasium(env)
    verienv.custom_value = 2

    assert env.custom_value == 1, (
        "Mutating copied attribute should not affect original env."
    )
    assert verienv.custom_value == 2, "Copied env should reflect its own mutations."


def test_from_gym_reset_step_parity_multiple_actions():
    """Verify reset/step parity across multiple actions for merged vs original envs."""
    env_to_copy = gym.make("FrozenLake-v1", is_slippery=False)
    verienv = GenerativeEnv.from_gymnasium(env_to_copy)

    env = gym.make("FrozenLake-v1", is_slippery=False)
    obs1, info1 = env.reset(seed=123)
    obs2, info2 = verienv.reset(seed=123)

    assert obs1 == obs2, "Initial observations should match after reset."

    actions = [0, 1, 2, 3, 0, 1]
    for action in actions:
        obs1, reward1, terminated1, truncated1, _ = env.step(action)
        obs2, reward2, terminated2, truncated2, _ = verienv.step(action)

        assert obs1 == obs2, "Observations do not match after step."
        assert reward1 == reward2, "Rewards do not match after step."
        assert terminated1 == terminated2, "Termination flags do not match after step."
        assert truncated1 == truncated2, "Truncation flags do not match after step."

        if terminated1 or truncated1:
            break


def test_simulate():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    venv = GenerativeEnv.from_gymnasium(env)

    # TODO: Replace with a non-random policy once supported.
    dataset = venv.simulate(RandomizedPolicy(venv), n_steps=100)
    len_dataset = np.sum([len(trajectory) for trajectory in dataset])

    assert len_dataset == 100, (
        f"Dataset should have 100 steps has {len_dataset} steps instead."
    )
