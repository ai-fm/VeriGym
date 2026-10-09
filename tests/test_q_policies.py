import gymnasium as gym
import numpy as np
import pytest

from verigym.abstraction.abstractionmapper import AbstractionMapper, linspace_mapper
from verigym.abstraction.learn_abstraction import (
    collect_data_from_trajectories,
    create_abstraction,
)
from verigym.environments.generativeenv import GenerativeEnv
from verigym.policy.qvalue import (
    QTable,
    QValuePolicy,
    ActiveLearningPolicy,
    EntropyLearningPolicy,
)

from utils import make_original_env

# ---------------------------------------------------------------------------
# Toy MDP used throughout these tests (identity abstraction, 3 states, 2 actions):
#   * state 0 --a0--> state 1 with reward 1
#   * state 1 --a0--> state 1 with reward 0 (absorbing)
#   * action 1 is never taken, state 2 is never visited.
# Each trajectory starts in state 0, so (0, 0) is visited 5 times and (1, 0) 10 times.
# ---------------------------------------------------------------------------
N_STATES = 3
N_ACTIONS = 2
TRAJECTORIES = [[(0, 0, 1.0, 1), (1, 0, 0.0, 1), (1, 0, 0.0, 1)]] * 5


@pytest.fixture
def identity_mapper():
    state_space = gym.spaces.Discrete(N_STATES)
    action_space = gym.spaces.Discrete(N_ACTIONS)
    return AbstractionMapper.initialize_identity_mapper(state_space, action_space)


@pytest.fixture
def counts():
    """Count databases (T_counts, R_counts, P_tot, S_init_counts) for `TRAJECTORIES`."""
    return collect_data_from_trajectories(TRAJECTORIES, N_STATES)


def _refine(policy, counts):
    """Helper func: Run `update_for_abstraction_refinement` on the toy MDP."""
    T_counts, R_counts, P_tot, S_init_counts = counts
    return policy.update_for_abstraction_refinement(
        TRAJECTORIES, T_counts, P_tot, R_counts, S_init_counts
    )


# ---------------------------------------------------------------------------
# QTable
# ---------------------------------------------------------------------------


def test_qtable_read_is_sparse_and_readonly():
    """Reading unset entries returns the default value without storing anything,
    and the returned default row cannot be written to."""
    Q = QTable(nr_states=4, nr_actions=3, default_value=0.5)

    assert Q[2, 1] == 0.5
    row = Q[2]
    np.testing.assert_array_equal(row, np.full(3, 0.5))
    assert Q.Q_dict == {}

    with pytest.raises(ValueError):
        row[0] = 1.0


def test_qtable_write():
    """Writing a single (s, a) entry fills the rest of the row with defaults;
    writing a full row stores a copy and checks its shape."""
    Q = QTable(nr_states=4, nr_actions=3, default_value=-1.0)

    Q[1, 2] = 7.0
    np.testing.assert_array_equal(Q[1], [-1.0, -1.0, 7.0])
    assert set(Q.Q_dict) == {1}

    new_row = np.array([1.0, 2.0, 3.0])
    Q[3] = new_row
    new_row[0] = 100.0  # must not leak into the table
    np.testing.assert_array_equal(Q[3], [1.0, 2.0, 3.0])

    with pytest.raises(AssertionError):
        Q[0] = [1.0, 2.0]


# ---------------------------------------------------------------------------
# QValuePolicy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ["strategy", "expected_default"], [("zero", 0.0), ("uniform", 1 / N_ACTIONS)]
)
def test_q_init_strategy(identity_mapper, strategy, expected_default):
    """The Q-table default value depends on the initialization strategy."""
    policy = QValuePolicy(None, identity_mapper, Q_init_strategy=strategy)

    assert policy.Q_table.default_value == pytest.approx(expected_default)
    assert policy.Q_table.nr_states == N_STATES
    assert policy.Q_table.nr_actions == N_ACTIONS


def test_action_selection_epsilon(identity_mapper):
    """With epsilon=0 a dominant Q-value is (practically) always chosen; with
    epsilon=1 actions are uniform regardless of the Q-values."""
    np.random.seed(0)
    n_samples = 500

    greedy = QValuePolicy(None, identity_mapper, epsilon=0.0)
    greedy.Q_table[0] = [0.0, 50.0]
    greedy_actions = [greedy.get_action(0) for _ in range(n_samples)]
    assert all(a == 1 for a in greedy_actions)

    random = QValuePolicy(None, identity_mapper, epsilon=1.0)
    random.Q_table[0] = [0.0, 50.0]
    random_actions = np.array([random.get_action(0) for _ in range(n_samples)])
    assert set(random_actions) == {0, 1}
    assert 0.4 < np.mean(random_actions == 0) < 0.6


def test_q_update_matches_bellman(identity_mapper, counts):
    """After refinement the Q-table equals the hand-computed Bellman solution
    of the toy MDP, input counts are not mutated, and unvisited states stay sparse."""
    T_counts, R_counts, P_tot, _ = counts
    T_before = {s: {a: dict(d) for a, d in Ts.items()} for s, Ts in T_counts.items()}
    R_before = {s: {a: list(r) for a, r in Rs.items()} for s, Rs in R_counts.items()}

    policy = QValuePolicy(None, identity_mapper, discount=0.9, update_iterations=50)
    returned = _refine(policy, counts)

    assert returned is policy
    # Q(1, 0) = 0 + 0.9 * max Q(1) = 0, Q(0, 0) = 1 + 0.9 * max Q(1) = 1,
    # unvisited actions get R_unvisited / (1 - discount) = 0.
    np.testing.assert_allclose(policy.Q_table[0], [1.0, 0.0])
    np.testing.assert_allclose(policy.Q_table[1], [0.0, 0.0])
    assert 2 not in policy.Q_table.Q_dict

    assert {s: {a: dict(d) for a, d in Ts.items()} for s, Ts in T_counts.items()} == T_before
    assert {s: {a: list(r) for a, r in Rs.items()} for s, Rs in R_counts.items()} == R_before


# ---------------------------------------------------------------------------
# ActiveLearningPolicy
# ---------------------------------------------------------------------------


def test_active_learning_q_values(identity_mapper, counts):
    """Count-based rewards: visited pairs get 1/count, unvisited ones Rmax/(1 - discount)."""
    discount = 0.95
    policy = ActiveLearningPolicy(None, identity_mapper, discount=discount, update_iterations=200)
    _refine(policy, counts)

    q_unvisited = 1 / (1 - discount)  # Rmax = 1
    np.testing.assert_allclose(policy.Q_table[0, 1], q_unvisited)
    np.testing.assert_allclose(policy.Q_table[1, 1], q_unvisited)
    np.testing.assert_allclose(policy.Q_table[0, 0], 1 / 5 + discount * q_unvisited)
    np.testing.assert_allclose(policy.Q_table[1, 0], 1 / 10 + discount * q_unvisited)


def test_active_learning_prefers_unvisited_actions(identity_mapper, counts):
    """After refinement the policy should select the never-tried action more often."""
    np.random.seed(0)
    policy = ActiveLearningPolicy(None, identity_mapper)
    _refine(policy, counts)

    actions = np.array([policy.get_action(0) for _ in range(500)])
    assert np.mean(actions == 1) > 0.5


# ---------------------------------------------------------------------------
# EntropyLearningPolicy
# ---------------------------------------------------------------------------


def test_entropy_policy_samples_from_tabular_policy(identity_mapper):
    """The policy starts uniform (without storing any states) and samples actions from `tabular_policy`."""
    np.random.seed(0)
    policy = EntropyLearningPolicy(None, identity_mapper)

    for s in range(N_STATES):
        np.testing.assert_allclose(policy.tabular_policy[s], 1 / N_ACTIONS)
    assert policy.tabular_policy.Q_dict == {}

    policy.tabular_policy[2] = [0.0, 1.0]
    assert all(policy.get_action(2) == 1 for _ in range(100))


def test_entropy_policy_update(identity_mapper, counts):
    """After refinement, every row of `tabular_policy` is still a distribution,
    and it is a mixture of the old policy and softmax(Q) with the learning rate.
    The unvisited state is not stored and stays uniform."""
    learning_rate = 0.2
    policy = EntropyLearningPolicy(None, identity_mapper, learning_rate=learning_rate)
    old_policy = {s: policy.tabular_policy[s].copy() for s in range(N_STATES)}
    _refine(policy, counts)

    for s in range(N_STATES):
        np.testing.assert_allclose(policy.tabular_policy[s].sum(), 1.0)
        assert np.all(policy.tabular_policy[s] >= 0)
        q = policy.Q_table[s]
        softmax_q = np.exp(q - q.max()) / np.exp(q - q.max()).sum()
        expected = (1 - learning_rate) * old_policy[s] + learning_rate * softmax_q
        np.testing.assert_allclose(policy.tabular_policy[s], expected)
    # The unvisited action 1 should have become more likely in the visited states.
    assert policy.tabular_policy[0, 1] > 0.5
    assert policy.tabular_policy[1, 1] > 0.5
    assert 2 not in policy.tabular_policy.Q_dict
    np.testing.assert_allclose(policy.tabular_policy[2], 1 / N_ACTIONS)


# ---------------------------------------------------------------------------
# Non-identity abstraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy_class", [QValuePolicy, ActiveLearningPolicy, EntropyLearningPolicy]
)
def test_non_identity_mapper(policy_class):
    """With a binning abstraction of CartPole (625 abstract states, 2 abstract actions):
    * the policy runs inside the interleaved `create_abstraction` loop, and its
      Q-table only holds valid abstract state indices that were actually explored;
    * `get_action` looks up the row of the *abstract* state of an original
      observation and maps the chosen abstract action back to the original action."""
    np.random.seed(0)
    env, _, bin_edges_per_dim = make_original_env()
    # 2 action bins, so abstract and original actions correspond one-to-one
    mapper = linspace_mapper(env, bin_edges_per_dim, 2)
    generative_env = GenerativeEnv.from_gymnasium(env)
    policy = policy_class(generative_env, mapper)

    abstracted_env = create_abstraction(
        original_env=generative_env,
        abstraction_mapper=mapper,
        exploration_policy=policy,
        num_steps=300,
        n_iterations=3,
        multithreading=False,
    )

    # The Q-table is refined on (a subset of) the explored abstract states only
    explored_states = set(abstracted_env.transition_function.T_dict)
    assert len(policy.Q_table.Q_dict) > 0
    assert set(policy.Q_table.Q_dict) <= explored_states
    for s, row in policy.Q_table.Q_dict.items():
        assert 0 <= s < mapper.abstract_n_states
        assert row.shape == (mapper.abstract_n_actions,)
        assert np.all(np.isfinite(row))
    if isinstance(policy, EntropyLearningPolicy):
        assert set(policy.tabular_policy.Q_dict) <= explored_states

    # Two original observations falling into different abstract states
    obs_a = np.array([0.0, 0.0, 0.0, 0.0])
    obs_b = np.array([2.0, 0.0, 0.0, 0.0])
    s_a = mapper.original_to_abstract_state_enum(obs_a)
    s_b = mapper.original_to_abstract_state_enum(obs_b)
    assert s_a != s_b

    # Force each abstract state to (practically) always pick a different abstract action
    for s, preferred in [(s_a, 0), (s_b, 1)]:
        if isinstance(policy, EntropyLearningPolicy):
            policy.tabular_policy[s] = np.eye(mapper.abstract_n_actions)[preferred]
        else:
            policy.epsilon_random = 0.0
            policy.Q_table[s] = 50.0 * np.eye(mapper.abstract_n_actions)[preferred]

    for obs, preferred in [(obs_a, 0), (obs_b, 1)]:
        expected = mapper.abstract_to_original_action(
            mapper._action_abstraction_map.enum_to_abstract(preferred)
        )
        actions = [policy.get_action(obs) for _ in range(50)]
        assert all(env.action_space.contains(a) for a in actions)
        assert all(a == expected for a in actions)
