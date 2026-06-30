import numpy as np
from collections import defaultdict


from verigym.environments.transition_func import IntervalTransitionFunction
from verigym.environments.reward_func import IntervalRewardFunction

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

