import copy

import gymnasium as gym
import numpy as np
import pytest

import verigym
from verigym.abstraction.learn_abstraction import (
    _create_count_databases,
    create_abstraction,
    get_interval_transition_reward,
    normalize_aggregated_counts,
)
from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.environments.generativeenv import GenerativeEnv
from verigym.environments.interval_explicitenv import IntervalExplicitEnv
from verigym.policy.randomized import RandomizedPolicy

from utils import make_original_env


def test_new_create_abstraction():
    """Check that `create_abstraction` runs through and returns an `ExplicitEnv`."""
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    abstraction_mapper = linspace_mapper(env, BIN_EDGES_PER_DIM, BIN_EDGES_PER_DIM)
    generative_env = GenerativeEnv.from_gymnasium(env)
    abstracted_env = create_abstraction(
        original_env=generative_env,
        abstraction_mapper=abstraction_mapper,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=NUM_STEPS,
    )

    assert isinstance(abstracted_env, verigym.ExplicitEnv)


# Test the interleaving abstraction learning
class RandomizedPolicyTest(RandomizedPolicy):
    """This policy class behaves just like `RandomizedPolicy` but it logs
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
    abstraction_mapper = linspace_mapper(env, BIN_EDGES_PER_DIM, BIN_EDGES_PER_DIM)
    generative_env = GenerativeEnv.from_gymnasium(env)
    N_ITERATIONS = 5

    policy = RandomizedPolicyTest(generative_env)
    _abstracted_env = create_abstraction(
        original_env=generative_env,
        abstraction_mapper=abstraction_mapper,
        exploration_policy=policy,
        num_steps=NUM_STEPS,
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
# These tests check the sanity of the abstracted env.
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
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    abstraction_mapper = linspace_mapper(env, BIN_EDGES_PER_DIM, BIN_EDGES_PER_DIM)
    generative_env = GenerativeEnv.from_gymnasium(env)
    return create_abstraction(
        original_env=generative_env,
        abstraction_mapper=abstraction_mapper,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=NUM_STEPS,
    )


def _visited_state_action_pairs(abstracted_env):
    """Helper func: Yield every `(s, a)` pair present in the transition function."""
    for s, actions in abstracted_env.transition_function.T_dict.items():
        for a in actions:
            yield s, a


def test_space_sizes(abstracted_env):
    """The abstracted env reports the expected number of states, actions and
    rewards, both as attributes and as gym spaces."""
    assert abstracted_env.nr_states == EXPECTED_N_STATES
    assert abstracted_env.nr_actions == EXPECTED_N_ACTIONS
    assert abstracted_env.observation_space.n == EXPECTED_N_STATES
    assert abstracted_env.action_space.n == EXPECTED_N_ACTIONS
    assert abstracted_env.nr_rewards == 1


def test_transition_function(abstracted_env: verigym.ExplicitEnv):
    """Transition function is a valid distribution and state-action indices are in range."""
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
    """Checking the consistency of the reward. CartPole yields +1 on every (non-terminal) step, so every learned reward
    is exactly 1.0."""
    R = abstracted_env.reward_function
    for s, actions in R.R_dict.items():
        for a, reward in actions.items():
            assert reward == pytest.approx(1.0)


def test_initial_state_distribution(abstracted_env):
    """The initial-state distribution has one entry per abstract state, is
    non-negative everywhere, and sums to 1."""
    s_init = abstracted_env.initial_states
    assert s_init.shape == (EXPECTED_N_STATES,)
    assert np.all(s_init >= 0.0)
    assert s_init.sum() == pytest.approx(1.0)


def test_state_abstraction_map_roundtrip(abstracted_env):
    """Mapping an abstract state index back to an original state and forward
    again returns the same index.

    The `_enum` accessors are needed here: the plain `original_to_abstract_state`
    returns the factored index (an ndarray), so `==` would compare element-wise
    rather than test the roundtrip. `original_to_abstract_state_enum` returns the
    flat `int` that can be compared against `idx`.
    """
    mapper = abstracted_env.abstraction_map
    for idx in range(EXPECTED_N_STATES):
        original = mapper.abstract_to_original_state_enum(idx)
        assert mapper.original_to_abstract_state_enum(original) == idx


def test_action_abstraction_map_roundtrip(abstracted_env):
    """backward(idx) is a valid original action, and forward(backward(idx)) is idempotent."""
    mapper = abstracted_env.abstraction_map
    action_space = abstracted_env.original_env.action_space
    for idx in range(EXPECTED_N_ACTIONS):
        original = mapper.abstract_to_original_action_enum(idx)
        assert action_space.contains(original)
        roundtrip_idx = mapper.original_to_abstract_action_enum(original)
        roundtrip_original = mapper.abstract_to_original_action_enum(roundtrip_idx)
        assert mapper.original_to_abstract_action_enum(roundtrip_original) == roundtrip_idx


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
            assert abstracted_env.action_mask[state].sum() == pytest.approx(0.0)
            break


# ---------------------------------------------------------------------------
# Testing ExplicitEnv -> ExplicitEnv
# ---------------------------------------------------------------------------


def test_abstracting_ExplicitEnv():
    """An already abstracted `ExplicitEnv` can be abstracted a second time,
    yielding another `ExplicitEnv`."""
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    abstraction_mapper = linspace_mapper(env, BIN_EDGES_PER_DIM, BIN_EDGES_PER_DIM)
    generative_env = GenerativeEnv.from_gymnasium(env)
    abstracted_env = create_abstraction(
        original_env=generative_env,
        abstraction_mapper=abstraction_mapper,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=NUM_STEPS,
    )

    assert isinstance(abstracted_env, verigym.ExplicitEnv)

    # Now we abstract again. The mapper must be built from `abstracted_env`, whose
    # observation/action spaces are the *abstract* (discrete) ones.
    abstraction_mapper = linspace_mapper(abstracted_env, BIN_EDGES_PER_DIM-1, BIN_EDGES_PER_DIM-1)
    abstracted_env_v2 = create_abstraction(
        original_env=abstracted_env,
        abstraction_mapper=abstraction_mapper,
        exploration_policy=RandomizedPolicy(abstracted_env),
        num_steps=10,
    )

    assert isinstance(abstracted_env_v2, verigym.ExplicitEnv)


# ---------------------------------------------------------------------------
# Testing Gym (Spaces obs: Discrete; actions: Discrete) -> ExplicitEnv
# ---------------------------------------------------------------------------


def test_gym_space_Discrete_Discrete():
    """An environment with a `Discrete` observation space and a `Discrete` action
    space can be abstracted."""
    env_name = "Taxi-v4"
    env = gym.make(env_name)
    NUM_STEPS = 100
    BIN_EDGES_PER_DIM = 2
    abstraction_mapper = linspace_mapper(env, BIN_EDGES_PER_DIM, BIN_EDGES_PER_DIM)

    generative_env = GenerativeEnv.from_gymnasium(env)
    _abstracted_env = create_abstraction(
            original_env=generative_env,
            abstraction_mapper=abstraction_mapper,
            exploration_policy=RandomizedPolicy(generative_env),
            num_steps=NUM_STEPS,
        )


# ---------------------------------------------------------------------------
# Testing Gym (Spaces obs: Box; actions: Box) -> ExplicitEnv
# ---------------------------------------------------------------------------

def test_gym_space_Box_Box():
    """An environment with a `Box` observation space and a `Box` action space can
    be abstracted."""
    env_name = "MountainCarContinuous-v0"
    env = gym.make(env_name)
    NUM_STEPS = 100
    BIN_EDGES_PER_DIM = 2
    abstraction_mapper = linspace_mapper(env, BIN_EDGES_PER_DIM, BIN_EDGES_PER_DIM)
    
    generative_env = GenerativeEnv.from_gymnasium(env)
    _abstracted_env = create_abstraction(
                original_env=generative_env,
                abstraction_mapper=abstraction_mapper,
                exploration_policy=RandomizedPolicy(generative_env),
                num_steps=NUM_STEPS,
            )


# ---------------------------------------------------------------------------
# Test interval learning
# ---------------------------------------------------------------------------



def test_run_interval_tests():
    """Check that `create_abstraction` with intervals==True and assume_iid in [True, False] runs through and 
    returns an `IntervalExplicitEnv`.
    
    Then, calls further test functions.
    All in one function, so we don't have to recompute the abstraction per test.
    """
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    abstraction_mapper = linspace_mapper(env, BIN_EDGES_PER_DIM, BIN_EDGES_PER_DIM)
    generative_env = GenerativeEnv.from_gymnasium(env)

    interval_env_1 = create_abstraction(
        original_env=generative_env,
        abstraction_mapper=abstraction_mapper,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=NUM_STEPS,
        intervals=True,
        assume_iid=False,
        multithreading=False
    )

    assert isinstance(interval_env_1, IntervalExplicitEnv)

    interval_env_2 = create_abstraction(
        original_env=generative_env,
        abstraction_mapper=abstraction_mapper,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=NUM_STEPS,
        intervals=True,
        assume_iid=True,
        multithreading=False
    )

    assert isinstance(interval_env_2, IntervalExplicitEnv)

    interval_test_reward_and_transition_share_keys(interval_env_1)
    interval_test_reward_and_transition_share_keys(interval_env_2)

    test_space_sizes(interval_env_1)
    test_space_sizes(interval_env_2)

    interval_test_transition_function(interval_env_1)
    interval_test_transition_function(interval_env_2)

    test_initial_state_distribution(interval_env_1)
    test_initial_state_distribution(interval_env_2)

    test_action_mask_matches_transition_keys(interval_env_1)
    test_action_mask_matches_transition_keys(interval_env_2)

def interval_test_transition_function(interval_env: IntervalExplicitEnv):
    """Transition function is a valid distribution and state-action indices are in range.
    Interval transition function is valid w.r.t. the point estimates."""
    T = interval_env.transition_function
    T_i = interval_env.get_interval_transition_function()

    for s, actions in T.T_dict.items():
        assert 0 <= s < EXPECTED_N_STATES
        for a, transtitions in actions.items():
            assert 0 <= a < EXPECTED_N_ACTIONS
            for s_next, prob in transtitions.items():
                assert 0 <= s_next < EXPECTED_N_STATES
                assert 0.0 <= prob <= 1.0

    assert T.sanity_check()
    assert T_i.sanity_check()
    # raises a ValueError if a point estimate lies outside its interval
    interval_env._check_interval_transitions(T, T_i)

def interval_test_reward_and_transition_share_keys(interval_env: IntervalExplicitEnv):
    """
    Reward function, transition function, and their interval variants, 
    are built from the same (s, a) observations, so their keys match.
    """
    t_pairs = {
        (s, a)
        for s, actions in interval_env.transition_function.T_dict.items()
        for a in actions
    }
    r_pairs = {
        (s, a)
        for s, actions in interval_env.reward_function.R_dict.items()
        for a in actions
    }
    i_t_pairs = {
        (s, a) 
        for s, actions in interval_env.get_interval_transition_function().T_dict.items()
        for a in actions
    }
    i_r_pairs = {
        (s, a)
        for s, actions in interval_env.get_interval_reward_function().R_dict.items()
        for a in actions
    }

    assert t_pairs == r_pairs
    assert t_pairs == i_t_pairs
    assert t_pairs == i_r_pairs

def _make_interval_counts():
    """Hand-made counts for 3 states and 2 actions. (0, 0) is a self-loop observed 100 times, the
    case that broke when intervals were computed from normalized probabilities instead of counts."""
    T_counts, R_counts, P_tot, S_init_counts = _create_count_databases(n_states=3)
    for s, a, s_next, count in [(0, 0, 0, 100), (0, 1, 1, 30), (0, 1, 2, 70), (1, 0, 0, 1), (1, 0, 2, 5)]:
        T_counts[s][a][s_next] += count
        P_tot[(s, a)] += count
        R_counts[s][a].extend([1.0] * count)
    S_init_counts[0] = 10
    return T_counts, R_counts, P_tot, S_init_counts


def test_normalize_aggregated_counts_does_not_mutate_inputs():
    """`create_abstraction` builds the point estimates and the intervals from the same counts, so
    normalizing must leave the counts untouched."""
    counts = _make_interval_counts()
    counts_before = copy.deepcopy(counts)
    normalize_aggregated_counts(*counts, n_states=3, n_actions=2)

    T_counts, R_counts, P_tot, S_init_counts = counts
    assert T_counts == counts_before[0]
    assert R_counts == counts_before[1]
    assert P_tot == counts_before[2]
    np.testing.assert_array_equal(S_init_counts, counts_before[3])


@pytest.mark.parametrize("iid", [False, True])
def test_intervals_contain_point_estimates(iid):
    """Every point estimate lies within its interval when both are computed from the same counts."""
    T_counts, R_counts, P_tot, S_init_counts = _make_interval_counts()
    T, R, _ = normalize_aggregated_counts(T_counts, R_counts, P_tot, S_init_counts, n_states=3, n_actions=2)
    T_i, R_i = get_interval_transition_reward(T_counts, R_counts, P_tot, n_states=3, n_actions=2, iid=iid)

    for s, actions in T.T_dict.items():
        for a, transitions in actions.items():
            for s_next, prob in transitions.items():
                lb, ub = T_i[s, a, s_next]
                assert lb <= prob <= ub
            lb, ub = R_i[s, a]
            assert lb <= R[s, a] <= ub

    # (0, 0) always leads to 0: the point estimate is 1.0, so the interval has to reach 1.0
    assert T[0, 0, 0] == 1.0
    assert T_i[0, 0, 0][1] == 1.0
    assert T_i.sanity_check()


def test_intervals_reject_normalized_counts():
    """Computing intervals from probabilities instead of counts fails loudly."""
    T_counts, R_counts, P_tot, S_init_counts = _make_interval_counts()
    T, _, _ = normalize_aggregated_counts(T_counts, R_counts, P_tot, S_init_counts, n_states=3, n_actions=2)
    with pytest.raises(TypeError):
        get_interval_transition_reward(T.T_dict, R_counts, P_tot, n_states=3, n_actions=2)


@pytest.mark.parametrize("iid", [False, True])
def test_interval_env_construction_has_no_side_effects(iid):
    """Building an `IntervalExplicitEnv` checks the point estimates against the intervals; this must not
    add entries to the (default)dicts of the transition and reward functions."""
    T_counts, R_counts, P_tot, S_init_counts = _make_interval_counts()
    T, R, S_init = normalize_aggregated_counts(T_counts, R_counts, P_tot, S_init_counts, n_states=3, n_actions=2)
    T_i, R_i = get_interval_transition_reward(T_counts, R_counts, P_tot, n_states=3, n_actions=2, iid=iid)
    dicts_before = copy.deepcopy((T.T_dict, R.R_dict, T_i.T_dict, R_i.R_dict))

    IntervalExplicitEnv(
        nr_states=3,
        nr_actions=2,
        initial_state_distr=S_init,
        transition_function=T,
        reward_function=R,
        interval_transition_function=T_i,
        interval_reward_function=R_i,
    )

    assert (T.T_dict, R.R_dict, T_i.T_dict, R_i.R_dict) == dicts_before
    assert T.sanity_check()
