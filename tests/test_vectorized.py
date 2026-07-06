import os
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np

from verigym.frameworks.stormpy.stormpy_utils import load_stormpy_model
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv
from verigym.environments.generativeenv import GenerativeEnv
from verigym.environments.explicitenv import ExplicitEnv
from verigym.environments.transition_func import TransitionFunction
from verigym.environments.reward_func import RewardFunction

PRISM_TEST = os.path.join(os.getcwd(), "tests/test_2d.prism")

def run_vec_env_eval(vec_env, num_envs):
    """
    I am testing all of them the same way for most function, this is to avoid duplicate code.
    """

    assert vec_env.num_envs == num_envs
    
    obs, _ = vec_env.reset()
    assert obs.shape[0] == num_envs

    # test step
    actions = [vec_env.single_action_space.sample() for _ in range(num_envs)]
    obs, reward, term, trunc, info = vec_env.step(actions)
    assert len(obs) == num_envs or obs.shape[0] == num_envs
    assert len(reward) == num_envs
    assert len(term) == num_envs
    assert len(trunc) == num_envs

    # test independence
    if isinstance(vec_env, SyncVectorEnv):
        # async does not expose envs
        obs1, _ = vec_env.reset()
        vec_env.envs[0].step(vec_env.single_action_space.sample())
        obs2, _ = vec_env.reset()
        assert obs1 is not obs2

    # test several steps
    vec_env.reset()
    for _ in range(10):
        actions = [vec_env.single_action_space.sample() for _ in range(num_envs)]
        obs, reward, term, trunc, info = vec_env.step(actions)
        if any(term) or any(trunc):
            vec_env.reset()

    assert obs.shape[0] == num_envs

    

def test_make_vec_explicit():
    num_envs = 4
    
    T = TransitionFunction.from_array(
        np.array(
            [[[0.0, 0.5, 0.5], # s0, a
            [0.2, 0.8, 0.0]], # s0, b
            [[0.5, 0.5, 0.0], # s1, a
             [0.0, 0.2, 0.8]],# s1, b
             [[0.0, 0.0, 1.0],
              [0.0, 0.0, 1.0]]]
        )
    )
    R = RewardFunction.from_array(
        np.array(
            [[1.0, 0.0],
            [0.2, 1.0],
            [0.0, 0.0]]
        )
    )
    kwargs = {
        "nr_states": 3,
        "nr_actions": 2,
        "initial_state_distr": np.array([1.0, 0.0, 0.0]),
        "transition_function": T,
        "reward_function": R
    }
    vec_env_sync = ExplicitEnv.make_vec(num_envs, vectorization_mode="sync",
                                        **kwargs)
    vec_env_async = ExplicitEnv.make_vec(num_envs=num_envs, vectorization_mode="async",
                                         **kwargs)
    
    run_vec_env_eval(vec_env_sync, num_envs)
    run_vec_env_eval(vec_env_async, num_envs)


def test_vec_from_gymnasium():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    num_envs = 4

    vec_env_sync = GenerativeEnv.vec_from_gymnasium(env, num_envs=num_envs, vectorization_mode="sync")
    vec_envs_async = GenerativeEnv.vec_from_gymnasium(env, num_envs=num_envs, vectorization_mode="async")

    run_vec_env_eval(vec_env_sync, num_envs)
    run_vec_env_eval(vec_envs_async, num_envs)

def test_vec_from_stormpy():
    mdp = load_stormpy_model(PRISM_TEST)
    num_envs = 4

    vec_env_sync = FrameworkExplicitEnv.vec_from_stormpy(mdp, num_envs = num_envs, vectorization_mode="sync")
    run_vec_env_eval(vec_env_sync, num_envs)

    caught_error = False
    try:    
        FrameworkExplicitEnv.vec_from_stormpy(mdp, num_envs = num_envs, vectorization_mode="async")
    except ValueError:
        caught_error = True

    assert caught_error
