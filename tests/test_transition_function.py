import numpy as np
from verigym.environments.transition_func import TransitionFunction


def initialize_transition_array(num_s: int, num_a: int) -> np.ndarray:
    # Create a sample transition function as a numpy array with (s,a,s') structure
    transition_array = np.random.random((num_s, num_a, num_s))
    # normalize to prob. distributions
    transition_array /= transition_array.sum(axis=2, keepdims=True)
    return transition_array


def test_empty_initialization():
    T = TransitionFunction(n_states=0, n_actions=0)
    assert isinstance(T, TransitionFunction), (
        "Initialization failed: object is not an instance of TransitionFunction."
    )
    assert len(T.T_dict) == 0, "Initialization failed: T_dict should be empty."
    assert T.T_dict[0][0][0] == 0.0, (
        "Initialization failed: default value for T_dict should be 0.0."
    )
    assert not T.sanity_check(), (
        "Sanity check failed: probabilities do not sum to 1 for some (s,a) pairs."
    )
    assert T.n_states == 0, "Initialization failed: n_states should be 0."
    assert T.n_actions == 0, "Initialization failed: n_actions should be 0"


def test_initialization_from_numpy_array():
    num_s, num_a = 10, 5
    transition_array = initialize_transition_array(num_s, num_a)

    # Initialize the TransitionFunction with the numpy array
    T = TransitionFunction.from_array(transition_array)
    assert T.sanity_check(), (
        "Sanity check failed: probabilities do not sum to 1 for some (s,a) pairs."
    )
    assert T.n_states == num_s
    assert T.n_actions == num_a

    for s in range(num_s):
        for a in range(num_a):
            for s_next in range(num_s):
                expected_prob = transition_array[s, a, s_next]
                actual_prob = T.T_dict[s][a][s_next]
                assert np.isclose(expected_prob, actual_prob), (
                    f"Initialization from array failed: expected probability {expected_prob} for (s={s}, a={a}, s'={s_next}), got {actual_prob}."
                )

    # test with incorrect value in dict
    T.T_dict[0][0][0] += 0.1  # to avoid any zero probs
    assert not T.sanity_check(), (
        "Sanity check failed: probabilities do not sum to 1 for some (s,a) pairs."
    )

    # test with incorrect np.ndarray
    transition_array[0, 0, 0] += 0.1  # to avoid any zero probs
    T = TransitionFunction.from_array(transition_array)
    assert not T.sanity_check(), (
        "Sanity check failed: probabilities do not sum to 1 for some (s,a) pairs."
    )


def test__get_item__():
    num_s, num_a = 10, 5
    transition_array = initialize_transition_array(num_s, num_a)
    T = TransitionFunction.from_array(transition_array)

    # Test __getitem__ for a valid state
    s = 0
    actions = T[s]
    assert isinstance(actions, dict), "__getitem__ failed: expected a dict of actions."
    assert len(actions) == num_a, (
        f"__getitem__ failed: expected {num_a} actions, got {len(actions)}."
    )

    # Test __getitem__ for an invalid state (not in T_dict)
    s_invalid = 999
    actions_invalid = T[s_invalid]
    assert isinstance(actions_invalid, dict), (
        "__getitem__ failed: expected a dict of actions for invalid state."
    )
    assert len(actions_invalid) == 0, (
        "__getitem__ failed: expected no actions for invalid state."
    )

    # compare different ways of accessing the probabilities
    # tf.T_dict[s][a][s_next]
    # tf[s][a][s_next]
    # tf[s, a, s_next]
    for s in range(num_s):
        for a in range(num_a):
            for s_next in range(num_s):
                assert T[s][a][s_next] == transition_array[s, a, s_next], (
                    f"__getitem__ failed: expected probability {transition_array[s, a, s_next]} for (s={s}, a={a}, s'={s_next}), got {T.T_dict[s][a][s_next]}."
                )
                assert T.T_dict[s][a][s_next] == T[s, a, s_next]

    assert T.sanity_check(), (
        "Sanity check failed: probabilities do not sum to 1 for some (s,a) pairs."
    )
