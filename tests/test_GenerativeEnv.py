from verigym.environments.generativeenv import GenerativeEnv, SymbolicGenerativeEnv
from verigym.policy.policy import RandomizedPolicy
import gymnasium as gym
import numpy as np

from test_vectorized import run_vec_env_eval


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
    dataset = venv.simulate(RandomizedPolicy(env), n_steps=100)
    len_dataset = np.sum([len(trajectory) for trajectory in dataset])

    assert len_dataset == 100, (
        f"Dataset should have 100 steps has {len_dataset} steps instead."
    )

def test_from_prism_program():
    prism_path = "tests/test_2d.prism"

    s_env = SymbolicGenerativeEnv.from_prism(prism_path)

    g_env = GenerativeEnv.from_prism(prism_path)

    # The function .from_prism should return an instance of SymbolicGenerativeEnv,
    # no matter from which class it is called.
    assert isinstance(g_env, SymbolicGenerativeEnv)
    assert isinstance(s_env, SymbolicGenerativeEnv)

    for _ in range(5):
        max_steps=100
        step = 0

        obs, _ = s_env.reset()
        while True:
            step += 1
            action = s_env.action_space.sample()
            obs, rew, term, trunc, info = s_env.step(action)

            if term or trunc:
                break

            assert step < max_steps, "Reached max steps, something is wrong with terminal states."

def test_symbolic_equivalent_to_generative_from_gym():
    # Tests that SymbolicGenerativeEnv.from_gymnasium also returns a GenerativeEnv
    # and does not change the logic.
    env = SymbolicGenerativeEnv.from_gymnasium(
        gym.make("CartPole-v1", render_mode=None)
    )

    assert isinstance(env, GenerativeEnv)
    assert not isinstance(env, SymbolicGenerativeEnv)

    env.reset()
    while True:
        action = env.action_space.sample()
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

def test_symbolic_simulate():
    """
    Make sure that the overwritten step and reset functions do not affect VeriGymEnv.simulate()
    """
    prism_path = "tests/test_2d.prism"
    env = SymbolicGenerativeEnv.from_prism(prism_path)
    policy = RandomizedPolicy(env)
    env.simulate(policy, n_steps = 100)

def test_vec_from_prism_program():
    prism_path = "tests/test_2d.prism"

    vec_envs_sync = SymbolicGenerativeEnv.vec_from_prism(prism_path,
                                                    num_envs=4,
                                                    vectorization_mode="sync")

    gen_vec_envs_sync = GenerativeEnv.vec_from_prism(prism_path,
                                                     num_envs=4,
                                                     vectorization_mode="sync")

    assert type(vec_envs_sync.envs[0]) is type(gen_vec_envs_sync.envs[0])
    assert isinstance(vec_envs_sync.envs[0], SymbolicGenerativeEnv)

    run_vec_env_eval(vec_envs_sync, 4)

    caught_error = False
    try:
        SymbolicGenerativeEnv.vec_from_prism(prism_path,
                                             num_envs=4,
                                             vectorization_mode="async")
    except ValueError:
        caught_error = True

    assert caught_error

