import gymnasium as gym 
import numpy as np
import pytest

import verigym
from verigym.abstraction.learn_abstraction import create_abstraction

from verigym.environments.generativeenv import GenerativeEnv

from verigym.policy.policy import RandomizedPolicy

from utils import (
    make_original_env,
)

@pytest.mark.parametrize("use_box_space", [True, False])
def test_create_abstraction(use_box_space):
    """Just a test that the create_abstraction function runs through."""
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    abstracted_env = create_abstraction(
        original_env=generative_env,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=NUM_STEPS,
        bin_edges_per_state_dim=BIN_EDGES_PER_DIM,
        bin_edges_per_action_dim=BIN_EDGES_PER_DIM,
        use_box_space=use_box_space
    )
    
    assert isinstance(abstracted_env, verigym.ExplicitEnv)
    


# Test the interleaving abstraction learning
class RandomizedPolicyTest(RandomizedPolicy):
    """This policy class bahves just like `RandomizedPolicy` but it logs
    how many interleaving calls were made during the abstraction refinement 
    process in the `self.iterations` variable."""
    iterations: int = 0

    def __init__(self, env):
        super().__init__(env)

    def update_for_abstraction_refinement(
        self, dataset, T_counts, P_tot, R_counts, S_init_counts
    ) -> "RandomizedPolicyTest":
        self.iterations += 1
        return self


def test_policy_call():
    """Check that the desired number of interleaving abstraction
    refinement steps were performed."""
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    N_ITERATIONS = 5

    policy = RandomizedPolicyTest(generative_env)
    _abstracted_env = create_abstraction(
        original_env=generative_env,
        exploration_policy=policy,
        num_steps=NUM_STEPS,
        bin_edges_per_state_dim=BIN_EDGES_PER_DIM,
        bin_edges_per_action_dim=BIN_EDGES_PER_DIM,
        n_iterations=N_ITERATIONS,
    )

    assert policy.iterations == N_ITERATIONS


# ---------------------------------------------------------------------------
# Behavioral / contract tests for the resulting abstracted `ExplicitEnv`.
#
# The abstraction of `CartPole-v1` (4 observation dims, `Discrete(2)`
# actions, 5 bins per dimension) is expected to be a finite MDP with:
#   * n_states  = 5 ** 4 = 625
#   * n_actions = 5      (the Discrete(2) action space is discretized into 5
#                         bins; only abstract actions 0 and 4 are reachable)
# These tests pin down the *contract* of the abstracted env. Whether the
# learned dynamics faithfully approximate the true (discretized) CartPole
# dynamics is a separate, semantic question tested elsewhere.
# ---------------------------------------------------------------------------

EXPECTED_N_STATES = 5**4  # 625
EXPECTED_N_ACTIONS = 5


@pytest.fixture(scope="module")
def abstracted_env():
    """Build the CartPole abstraction once for the whole module.

    `create_abstraction` simulates 1000 steps and learns the model, so it is
    comparatively expensive; a module-scoped fixture keeps the testing fast,
    as we only need to compute the fixture once.
    """
    env, num_steps, bin_edges_per_dim = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    return create_abstraction(
        original_env=generative_env,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=num_steps,
        bin_edges_per_state_dim=bin_edges_per_dim,
        bin_edges_per_action_dim=bin_edges_per_dim,
    )


def _visited_state_action_pairs(abstracted_env):
    """Helper func: Yield every `(s, a)` pair present in the transition function."""
    for s, actions in abstracted_env.transition_function.T_dict.items():
        for a in actions:
            yield s, a


def test_space_sizes(abstracted_env):
    """Space / size contract."""
    assert abstracted_env.nr_states == EXPECTED_N_STATES
    assert abstracted_env.nr_actions == EXPECTED_N_ACTIONS
    assert abstracted_env.observation_space.n == EXPECTED_N_STATES
    assert abstracted_env.action_space.n == EXPECTED_N_ACTIONS
    assert abstracted_env.nr_rewards == 1


def test_transition_function_is_valid_distribution(abstracted_env):
    """Transition function is a valid distribution."""
    assert abstracted_env.transition_function.sanity_check()


def test_transition_probabilities_and_indices_in_range(abstracted_env:verigym.ExplicitEnv):
    T = abstracted_env.transition_function
    for s, actions in T.T_dict.items():
        assert 0 <= s < EXPECTED_N_STATES
        for a, transitions in actions.items():
            assert 0 <= a < EXPECTED_N_ACTIONS
            for s_next, prob in transitions.items():
                assert 0 <= s_next < EXPECTED_N_STATES
                assert 0.0 <= prob <= 1.0
    
    # we should also make sure that probabilites are correct
    assert T.sanity_check()


def test_reward_and_transition_share_keys(abstracted_env):
    """Reward function: R and T are built from the same (s, a) observations, so their keys match."""
    t_pairs = set(_visited_state_action_pairs(abstracted_env))
    r_pairs = {
        (s, a)
        for s, actions in abstracted_env.reward_function.R_dict.items()
        for a in actions
    }
    assert t_pairs == r_pairs


def test_reward_is_constant_one_for_cartpole(abstracted_env):
    """CartPole yields +1 on every (non-terminal) step, so every learned reward
    is exactly 1.0 -- a direct check of the reward-averaging pipeline."""
    R = abstracted_env.reward_function
    for s, actions in R.R_dict.items():
        for a, reward in actions.items():
            assert reward == pytest.approx(1.0)


def test_initial_state_distribution(abstracted_env):
    """Initial-state distribution."""
    s_init = abstracted_env.initial_states
    assert s_init.shape == (EXPECTED_N_STATES,)
    assert np.all(s_init >= 0.0)
    assert s_init.sum() == pytest.approx(1.0)


def test_state_abstraction_map_roundtrip(abstracted_env):
    """Abstraction-map consistency (forward / backward roundtrip): forward(backward(idx)) == idx for every abstract state index."""
    mapper = abstracted_env.abstraction_map
    for idx in range(EXPECTED_N_STATES):
        original = mapper.abstract_to_original_state(idx)
        assert mapper.original_to_abstract_state(original) == idx


def test_action_abstraction_map_roundtrip(abstracted_env):
    """forward(backward(idx)) == idx for every abstract action index."""
    mapper = abstracted_env.abstraction_map
    for idx in range(EXPECTED_N_ACTIONS):
        original = mapper.abstract_to_original_action(idx)
        assert mapper.original_to_abstract_action(original) == idx


def test_action_mask_matches_transition_keys(abstracted_env):
    """Runtime dynamics: action_mask[s, a] == 1 exactly for the visited (s, a) pairs."""
    mask = abstracted_env.action_mask
    visited = set(_visited_state_action_pairs(abstracted_env))
    for s in range(EXPECTED_N_STATES):
        for a in range(EXPECTED_N_ACTIONS):
            expected = 1.0 if (s, a) in visited else 0.0
            assert mask[s, a] == expected


def test_reset_returns_supported_state(abstracted_env):
    """Make sure resetting env leads to an initial state."""
    s_init = abstracted_env.initial_states
    for _ in range(20):
        state, _info = abstracted_env.reset()
        assert 0 <= state < EXPECTED_N_STATES
        assert s_init[state] > 0.0


def test_rollout_stays_valid(abstracted_env):
    """A rollout using only available actions stays in-range, is rewarded with
    1.0, and only terminates in states with no available actions."""
    state, _info = abstracted_env.reset()
    for _ in range(200):
        available = np.flatnonzero(abstracted_env.action_mask[state])
        if len(available) == 0:
            # Terminal state: no actions to take.
            break
        action = int(np.random.choice(available))
        state, reward, terminated, truncated, _info = abstracted_env.step(action)
        assert 0 <= state < EXPECTED_N_STATES
        assert reward == pytest.approx(1.0)
        assert not truncated
        if terminated:
            assert abstracted_env.action_mask[state].sum() == 0.0
            break


# ---------------------------------------------------------------------------
# Testing ExplicitEnv -> ExplicitEnv 
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_box_space", [True, False])
def test_abstracting_ExplicitEnv(use_box_space):
    """Just a test that the create_abstraction function runs through."""
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    abstracted_env = create_abstraction(
        original_env=generative_env,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=NUM_STEPS,
        bin_edges_per_state_dim=BIN_EDGES_PER_DIM,
        bin_edges_per_action_dim=BIN_EDGES_PER_DIM,
        use_box_space=use_box_space
    )
    
    assert isinstance(abstracted_env, verigym.ExplicitEnv)
    
    # Now we abstract again
    abstracted_env_v2 = create_abstraction(
        original_env=abstracted_env,
        exploration_policy=RandomizedPolicy(abstracted_env),
        num_steps=NUM_STEPS,
        bin_edges_per_state_dim=BIN_EDGES_PER_DIM,
        bin_edges_per_action_dim=BIN_EDGES_PER_DIM,
        use_box_space=use_box_space
    )
    assert isinstance(abstracted_env_v2, verigym.ExplicitEnv)
    
    
    
# ---------------------------------------------------------------------------
# Testing Gym (Spaces obs: Discrete; actions: Discrete) -> ExplicitEnv 
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_box_space", [True, False])
def test_gym_space_Discrete_Discrete(use_box_space):
    env_name = "Taxi-v4"
    env = gym.make(env_name)
    # env = ReplaceInfObservation(env, neg_inf=-10, pos_inf=10)
    NUM_STEPS = 100
    BIN_EDGES_PER_DIM = 2
    
    generative_env = GenerativeEnv.from_gymnasium(env)
    abstracted_env = create_abstraction(
        original_env=generative_env,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=NUM_STEPS,
        bin_edges_per_state_dim=BIN_EDGES_PER_DIM,
        bin_edges_per_action_dim=BIN_EDGES_PER_DIM,
        use_box_space=use_box_space
    )
    
# ---------------------------------------------------------------------------
# Testing Gym (Spaces obs: Box; actions: Box) -> ExplicitEnv 
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_box_space", [True, False])
def test_gym_space_Box_Box(use_box_space):
    env_name = "Taxi-v4"
    env = gym.make(env_name)
    # env = ReplaceInfObservation(env, neg_inf=-10, pos_inf=10)
    NUM_STEPS = 100
    BIN_EDGES_PER_DIM = 2
    
    generative_env = GenerativeEnv.from_gymnasium(env)
    abstracted_env = create_abstraction(
        original_env=generative_env,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=NUM_STEPS,
        bin_edges_per_state_dim=BIN_EDGES_PER_DIM,
        bin_edges_per_action_dim=BIN_EDGES_PER_DIM,
        use_box_space=use_box_space
    )
