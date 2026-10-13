import copy
from collections import defaultdict

import numpy as np
import pytest


from verigym.environments.transition_func import IntervalTransitionFunction, TransitionFunction
from verigym.environments.reward_func import IntervalRewardFunction, RewardFunction
from verigym.environments.interval_explicitenv import IntervalExplicitEnv
from verigym.environments.explicitenv import ExplicitEnv
from verigym.io.exporter import export_to_stormpy_mdp, export_to_stormpy_imdp

# Tests for IntervalRewardFunction and IntervalTransitionFunction
def get_invalid_interval_transition_arrays():
    """
    Build two wrong interval transition arrays to make sure the sanity check fails.
    t1:
    s0 -a-[0.3, 0.2]->s1
    s0 -a-[0.1, 0.9]->s2

    t2:
    s0 -a-[0.2, 0.4]->s1
    """
    n_s = 3
    n_a = 1
    t0 = np.zeros((n_s, n_a, n_s, 2))
    t0[0, 0, 1] = [0.3, 0.2]
    t0[0, 0, 2] = [0.1, 0.9]

    t1 = np.zeros((n_s, n_a, n_s, 2))
    t1[0, 0, 1] = [0.2, 0.4]

    return t0, t1

def get_interval_transition_array():
    """
    Build the following IMDP:
    s0 -a-[0.2, 0.5]-> s1
    s0 -a-[0.4, 0.9]-> s2
    """
    n_s = 3
    n_a = 1
    t = np.zeros((n_s, n_a, n_s, 2))
    t[0, 0, 1] = [0.2, 0.5]
    t[0, 0, 2] = [0.4, 0.9]

    return t

def get_interval_reward_array():
    """
    s0, a: [0.0, 1.0]
    s0, b: [2.0, 3.0]
    s1, a: [4.0, 4.0]
    """
    n_s = 2
    n_a = 2

    r = np.zeros((n_s, n_a, 2))
    r[0, 0] = [0.0, 1.0]
    r[0, 1] = [2.0, 3.0]
    r[1, 0] = [4.0, 4.0]
    return r

def get_interval_reward_dict():
    """
    s0, a: [0.0, 1.0]
    s0, b: [2.0, 3.0]
    s1, a: [4.0, 4.0]
    """
    r = defaultdict()
    r[0] = {
            0: (0.0, 1.0),
            1: (2.0, 3.0)
        }
    r[1] = { 0: (4.0, 4.0) }
    return r

def test_interval_transition_function():
    t_array = get_interval_transition_array()
    i_transitions = IntervalTransitionFunction.from_array(t_array)
    sp = i_transitions.get_sparsity()
    assert sp >= 0 and sp <= 1
    assert i_transitions.sanity_check()

    t0, t1 = get_invalid_interval_transition_arrays()
    t0_tr = IntervalTransitionFunction.from_array(t0)
    assert not t0_tr.sanity_check()
    t1_tr = IntervalTransitionFunction.from_array(t1)
    assert not t1_tr.sanity_check()

def test_interval_reward_function():
    r0 = get_interval_reward_array()
    r0_func = IntervalRewardFunction.from_array(r0)
    assert r0_func is not None
    r1 = get_interval_reward_dict()
    r1_func = IntervalRewardFunction.from_dict(r1, 2, 2)
    assert r1_func is not None

    for s in range(2):
        for a in range(2):
            assert r0_func[s, a] == r1_func[s, a]

# Tests for IntervalExplicitEnv
# + helper functions
def get_point_transitions():
    T_array = np.array(
        [
            [
                [0.0, 0.9, 0.0, 0.1],   # (s0, a)
                [0.5, 0.0, 0.5, 0.0]    # (s0, b)
            ], [
                [0.9, 0.0, 0.0, 0.1],   # (s1, a)
                [0.0, 0.7, 0.3, 0.0]    # (s1, b)
            ], [
                [0.0, 0.4, 0.6, 0.0],   # (s2, a)
                [0.8, 0.0, 0.0, 0.2]    # (s2, b)
            ], [
                [0.0, 0.0, 0.0, 1.0],   # (s3, a), terminal
                [0.0, 0.0, 0.0, 1.0]    # (s3, b), terminal 
            ]
        ]
    )
    T = TransitionFunction.from_array(T_array)
    return T

def get_interval_transitions():
    T_array = np.array(
        [
            [
                [[0.0, 0.0], [0.8, 0.95], [0.0, 0.01], [0.05, 0.3]],    # (s0, a)
                [[0.5, 0.5], [0.0, 0.2], [0.4, 0.6], [0.0, 0.0]]        # (s0, b)
            ], [   
                [[0.8, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.2]],       # (s1, a)
                [[0.0, 0.01], [0.7, 0.9], [0.25, 0.4], [0.0, 0.1]]      # (s1, b)
            ], [
                [[0.0, 0.0], [0.35, 0.4], [0.57, 0.7], [0.0, 0.0]],     # (s2, a)
                [[0.5, 0.9], [0.0, 0.05], [0.0, 0.05], [0.1, 0.25]]     # (s2, b)
            ], [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],       # (s3, a), terminal
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]        # (s3, b), terminal
            ]
        ]
    )
    T = IntervalTransitionFunction.from_array(T_array)
    return T

def get_point_rewards():
    R_array = np.array(
        [
            [3, 1], # s0 a/b
            [2, 4], # s1 a/b
            [1, 2], # s2 a/b
            [0, 0], # s3 a/b
        ], dtype=float
    )
    R = RewardFunction.from_array(R_array)
    return R

def get_interval_rewards():
    R_array = np.array(
        [
            [[2, 3], [1, 2]], # s0 a/b
            [[2, 2], [3, 5]], # s1 a/b
            [[0, 5], [1, 2]], # s2 a/b
            [[0, 0], [0, 0]]  # s3 a/b
        ], dtype=float
    )
    R = IntervalRewardFunction.from_array(R_array)
    return R

def get_interval_env():
    nr_states, nr_actions = 4, 2
    i_env = IntervalExplicitEnv(
        nr_states=nr_states,
        nr_actions=nr_actions,
        initial_state_distr=[1.0, 0.0, 0.0, 0.0],
        interval_transition_function=get_interval_transitions(),
        interval_reward_function=get_interval_rewards()
    )
    return i_env

def get_point_env():
    nr_states, nr_actions = 4, 2
    env = ExplicitEnv(
        nr_states=nr_states,
        nr_actions=nr_actions,
        initial_state_distr=[1.0, 0.0, 0.0, 0.0],
        transition_function=get_point_transitions(),
        reward_function=get_point_rewards()
    )
    return env

# actual tests
def test_interval_env():
    """
    Tests that the IntervalExplicitEnv works: it steps with the point estimates of the intervals and reports the
    intervals of the transitions.
    """
    i_env = get_interval_env()
    T_i = get_interval_transitions()
    R_i = get_interval_rewards()
    obs, _ = i_env.reset()
    for s in range(i_env.nr_states):
        for a in range(i_env.nr_actions):
            i_env.set_state(s)
            obs, rew, term, trunc, info = i_env.step(a)
            assert obs in i_env.observation_space
            assert rew == (R_i[s][a][0] + R_i[s][a][1]) / 2
            assert info["reward_interval"] == R_i[s][a]
            assert info["transition_interval"] == T_i[s][a][obs]
            assert info["transition_interval"][1] > 0
            assert term == (obs == 3)

def test_interval_env_point_estimates():
    """
    The point estimates of the IntervalExplicitEnv are valid distributions within the intervals, with the same
    state-action pairs and successors.
    """
    i_env = get_interval_env()
    T = i_env.get_transition_function()
    T_i = i_env.get_interval_transition_function()
    assert T.sanity_check()
    assert T.T_dict.keys() == T_i.T_dict.keys()
    for s, actions in T_i.T_dict.items():
        assert T.T_dict[s].keys() == actions.keys()
        for a, transitions in actions.items():
            assert T.T_dict[s][a].keys() == transitions.keys()
            for s_next, (lb, ub) in transitions.items():
                assert lb <= T[s, a, s_next] <= ub

def test_interval_env_from_point():
    """
    Tests that the IntervalExplicitEnv works when initialized with the point intervals of point transitions and
    rewards: its point estimates are the point transitions and rewards.
    """
    nr_states, nr_actions = 4, 2
    T = get_point_transitions()
    R = get_point_rewards()
    i_env = IntervalExplicitEnv(
        nr_states, nr_actions,
        initial_state_distr=[1.0, 0.0, 0.0, 0.0],
        interval_transition_function=IntervalTransitionFunction.from_transition_function(T),
        interval_reward_function=IntervalRewardFunction.from_reward_function(R)
    )
    assert i_env.get_transition_function().T_dict == T.T_dict
    assert i_env.get_reward_function().R_dict == R.R_dict
    obs, _ = i_env.reset()
    for s in range(i_env.nr_states):
        for a in range(i_env.nr_actions):
            i_env.set_state(s)
            obs, rew, term, trunc, info = i_env.step(a)
            assert obs in i_env.observation_space
            assert rew == R[s][a]
            assert info["reward_interval"] == (R[s][a], R[s][a])
            assert info["transition_interval"] == (T[s][a][obs], T[s][a][obs])

def test_interval_env_invalid_transitions():
    """
    Tests that interval transitions without a valid distribution are detected.
    """
    for t in get_invalid_interval_transition_arrays():
        with pytest.raises(ValueError, match="Sanity check"):
            IntervalExplicitEnv(
                3, 1,
                initial_state_distr=[1.0, 0.0, 0.0],
                interval_transition_function=IntervalTransitionFunction.from_array(t),
                interval_reward_function=IntervalRewardFunction.from_array(np.zeros((3, 1, 2)))
            )

def test_interval_env_invalid_rewards():
    """
    Tests that reward intervals with a lower bound above the upper bound are detected.
    """
    R_array = np.zeros((4, 2, 2))
    R_array[2, 1] = [2.0, 1.0]
    with pytest.raises(ValueError, match=r"reward interval for \(2,1\)"):
        IntervalExplicitEnv(
            4, 2,
            initial_state_distr=[1.0, 0.0, 0.0, 0.0],
            interval_transition_function=get_interval_transitions(),
            interval_reward_function=IntervalRewardFunction.from_array(R_array)
        )

def test_interval_env_getters():
    i_env = get_interval_env()
    R_env = i_env.get_reward_function()
    R_i = get_interval_rewards()

    for s in range(i_env.nr_states):
        for a in range(i_env.nr_actions):
            rew = i_env.get_reward(s, a)
            assert rew == R_env[s][a] and rew == (R_i[s][a][0] + R_i[s][a][1]) / 2
            assert i_env.get_interval_reward(s, a) == R_i[s][a]

# Tests for the point estimates of intervals
def test_transition_point_estimate():
    """
    The point estimate takes the lower bounds and distributes the remaining probability mass proportionally to the
    widths of the intervals. It keeps the state-action pairs and successors, and does not change the intervals.
    """
    T_dict = defaultdict(dict)
    T_dict[0][0] = defaultdict(lambda: (0.0, 0.0), {1: (0.2, 0.5), 2: (0.4, 0.9)})  # 0.4 left, widths 0.3 and 0.5
    T_dict[0][1] = defaultdict(lambda: (0.0, 0.0), {0: (0.3, 0.3), 1: (0.7, 0.7)})  # point intervals
    T_dict[1][0] = defaultdict(lambda: (0.0, 0.0), {1: (0.5, 0.6), 2: (0.2, 0.4)})  # upper bounds sum to 1
    T_dict[1][1] = defaultdict(lambda: (0.0, 0.0), {0: (0.6, 1.0), 2: (0.4, 0.5)})  # lower bounds sum to 1
    T_dict[2][0] = defaultdict(lambda: (0.0, 0.0), {2: (0.8, 1.0)})  # a single successor
    T_dict[2][1] = defaultdict(lambda: (0.0, 0.0))  # no successors
    T_i = IntervalTransitionFunction(3, 2, T_dict)
    intervals_before = copy.deepcopy(T_i.T_dict)

    T = T_i.point_estimate()

    assert T_i.T_dict == intervals_before
    assert T[0, 0] == pytest.approx({1: 0.2 + 0.4 * 0.3 / 0.8, 2: 0.4 + 0.4 * 0.5 / 0.8})
    assert T[0, 1] == {0: 0.3, 1: 0.7}
    assert T[1, 0] == pytest.approx({1: 0.6, 2: 0.4})
    assert T[1, 1] == {0: 0.6, 2: 0.4}
    assert T[2, 0] == {2: 1.0}  # exactly 1, e.g., for terminal self-loops
    assert T.T_dict[2][1] == {}
    assert T.T_dict.keys() == T_dict.keys()
    assert all(T.T_dict[s].keys() == T_dict[s].keys() for s in T_dict)

def test_reward_point_estimate():
    """
    The point estimate of a reward interval is its midpoint.
    """
    R_i = IntervalRewardFunction.from_array(get_interval_reward_array())
    R = R_i.point_estimate()
    assert [R[0, 0], R[0, 1], R[1, 0], R[1, 1]] == [0.5, 2.5, 4.0, 0.0]

# Tests for constructing stormpy IMDPs from input IMDPs and MDPs
def test_build_imdp_from_intervals():
    """
    Tests that building a stormpy IMDP from an IntervalExplicitEnv works.
    """
    i_env = get_interval_env()

    export_to_stormpy_imdp(i_env, use_reward_uncertainty=False)
    export_to_stormpy_imdp(i_env, use_reward_uncertainty=True)

def test_mdp_export_compatibility():
    """
    Tests that IntervalExplicitEnvs can be exported as stormpy mdps. 
    This should just consider the point estimates of the intervals (the TransitionFunction and RewardFunction
    of the env) and ignore the IntervalTransitionFunction and IntervalRewardFunction.
    """
    i_env = get_interval_env()

    export_to_stormpy_mdp(i_env)

def test_build_imdp_from_point():
    """
    Tests that standard ExplicitEnvs can be exported to stormpy IMDPs. 
    This should return an IMDP where for all transitions and rewards, upper bound = lower bound.
    """
    env = get_point_env()
    export_to_stormpy_imdp(env, use_reward_uncertainty=False)

    # This should throw an error because the env does not have an uncertain reward function
    caught_invalid = False
    try:
        export_to_stormpy_imdp(env, use_reward_uncertainty=True)
    except AssertionError:
        caught_invalid = True

    assert caught_invalid