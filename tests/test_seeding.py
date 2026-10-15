"""
Tests that a seed fully determines the random behaviour of:
    explicit envs (reset(seed))
    policies (PolicyClass.seed)
    create_abstraction(seed)
    abstraction_refinement(seed)
    PRISM envs (reset(seed))
    the post-shield wrapper (seed)
"""

import os
import random

import numpy as np
import stable_baselines3 as sb3

from verigym.abstraction.abstraction_refinement import abstraction_refinement
from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.abstraction.learn_abstraction import create_abstraction
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv
from verigym.environments.generativeenv import GenerativeEnv, SymbolicGenerativeEnv
from verigym.frameworks.stable_baselines3.policy import SB3Policy
from verigym.frameworks.stormpy.stormpy_utils import load_stormpy_model
from verigym.policy.qvalue import EntropyLearningPolicy, QValuePolicy
from verigym.policy.randomized import RandomizedPolicy
from verigym.supplements.shield_wrapper import AbstractPostShieldWrapper

from utils import make_original_env

TESTS_DIR = os.path.dirname(__file__)
ZEROCONF = os.path.join(TESTS_DIR, "..", "examples", "prism_files", "zeroconf.v1.prism")


def _explicit_rollout(env, seed, n_steps=200):
    """The states visited after `env.reset(seed=seed)`, with the same available actions for every seed."""
    action_rng = np.random.default_rng(0)
    env.reset(seed=seed)
    states = [env.encode(env.state)]
    for _ in range(n_steps):
        available = np.flatnonzero(env.action_mask.get(env.encode(env.state), np.zeros(env.nr_actions)))
        _, _, terminated, truncated, _ = env.step(int(action_rng.choice(available)) if len(available) > 0 else 0)
        states.append(env.encode(env.state))
        if terminated or truncated:
            env.reset()
            states.append(env.encode(env.state))
    return states


def test_explicit_env_reset_seed():
    """Same seed, same rollout; the env samples from its own generator, not from the global `random`/`np.random`."""
    env = FrameworkExplicitEnv.from_stormpy(load_stormpy_model(ZEROCONF, constants="N=20,K=2,reset=true"))
    random.seed(0)
    np.random.seed(0)
    py_state, np_state = random.getstate(), np.random.get_state()

    rollout = _explicit_rollout(env, seed=3)
    assert _explicit_rollout(env, seed=3) == rollout
    assert _explicit_rollout(env, seed=4) != rollout

    assert random.getstate() == py_state
    np_state_after = np.random.get_state()
    assert np.array_equal(np_state[1], np_state_after[1]) and np_state[2:] == np_state_after[2:]


def _generative_cartpole():
    env = GenerativeEnv.from_gymnasium(make_original_env()[0])
    return env, linspace_mapper(env, 5, 2)


def test_policy_seed():
    """After `seed(s)`, a policy that samples its actions repeats them."""
    env, mapper = _generative_cartpole()
    env.observation_space.seed(0)
    observations = [env.observation_space.sample() for _ in range(50)]
    policies = [
        RandomizedPolicy(env),
        QValuePolicy(env, mapper, epsilon=0.5),
        EntropyLearningPolicy(env, mapper),
        SB3Policy(sb3.PPO("MlpPolicy", env), env),
    ]
    for policy in policies:
        policy.seed(3)
        actions = [np.asarray(policy.get_action(obs)).tolist() for obs in observations]
        policy.seed(3)
        assert [np.asarray(policy.get_action(obs)).tolist() for obs in observations] == actions, type(policy).__name__


def _abstract_model(model):
    """The learned transition, reward and initial state functions of an abstract model, as plain dicts."""
    T = {s: {a: dict(nexts) for a, nexts in actions.items()} for s, actions in model.transition_function.T_dict.items()}
    R = {s: dict(actions) for s, actions in model.reward_function.R_dict.items()}
    return T, R, dict(model.initial_states.distr_dict)


def test_create_abstraction_seed():
    """`seed` determines the abstraction, also for unseeded policies; with seed=None, a seeded env and policy do."""
    env, mapper = _generative_cartpole()

    def learn(policy, seed):
        model = create_abstraction(env, mapper, policy, num_steps=500, n_iterations=2, multithreading=False, seed=seed)
        return _abstract_model(model)

    for make_policy in (lambda: RandomizedPolicy(env), lambda: EntropyLearningPolicy(env, mapper)):
        assert learn(make_policy(), seed=5) == learn(make_policy(), seed=5)

        results = []
        for _ in range(2):
            env.reset(seed=6)
            policy = make_policy()
            policy.seed(7)
            results.append(learn(policy, seed=None))
        assert results[0] == results[1]


def test_abstraction_refinement_seed():
    """`seed` determines the refinement, with a reused dataset and with exploration per mapper."""
    env, _ = _generative_cartpole()
    schedule = [linspace_mapper(env, n_bins, 2) for n_bins in (2, 3)]

    def callback(original_env, abstracted_model):
        # also depends on the random state the env is left in
        trajectories = original_env.simulate(policy=RandomizedPolicy(original_env, seed=0), n_steps=20)
        return False, (_abstract_model(abstracted_model), repr(trajectories))

    def refine(**kwargs):
        return [metric for metric, _ in abstraction_refinement(env, schedule, 400, callback, seed=9, **kwargs)]

    for kwargs in (
        dict(make_exploration_policy=lambda mapper: RandomizedPolicy(env)),
        dict(make_exploration_policy=lambda mapper: EntropyLearningPolicy(env, mapper), n_iterations=2, reuse_dataset=False),
    ):
        assert refine(**kwargs) == refine(**kwargs)


def test_prism_env_reset_seed():
    """Same seed, same rollout of the stormpy simulator."""
    env = SymbolicGenerativeEnv.from_prism(os.path.join(TESTS_DIR, "test_2d.prism"))

    def rollout(seed, n_steps=100):
        action_rng = np.random.default_rng(0)
        obs, _ = env.reset(seed=seed)
        observations = [np.asarray(obs).tolist()]
        for _ in range(n_steps):
            obs, _, terminated, truncated, _ = env.step(int(action_rng.integers(env.action_space.n)))
            observations.append(np.asarray(obs).tolist())
            if terminated or truncated:
                obs, _ = env.reset()
                observations.append(np.asarray(obs).tolist())
        return observations

    assert rollout(seed=3) == rollout(seed=3)
    assert rollout(seed=4) != rollout(seed=3)


def test_shield_wrapper_seed():
    """The shield's generator is seeded by the constructor (also with numpy integers) and by `reset(seed)`."""
    env, mapper = _generative_cartpole()
    shield = np.ones((mapper.abstract_n_states, mapper.abstract_n_actions), dtype=bool)
    wrapper = AbstractPostShieldWrapper(env, shield, mapper, seed=np.int64(3))
    assert wrapper.rng.random() == random.Random(3).random()
    wrapper.reset(seed=4)
    assert wrapper.rng.random() == random.Random(4).random()
