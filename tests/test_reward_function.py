from collections import defaultdict

import numpy as np

from verigym.environments.reward_func import RewardFunction


def initialize_array(num_s: int, num_a: int) -> RewardFunction:
    # Create a sample reward function as a numpy array with (s,a) structure
    reward_array = np.random.random((num_s, num_a))
    return reward_array


def test_incorrect_initialization():
    """Testing whether the RewardFunction raises an error when initialized with incorrect types."""
    # wrong type for R_dict (dict instead of defaultdict)
    try:
        R = RewardFunction(n_states=10, n_actions=5, R_dict={"not": "xxx"})
        assert False, "Initialization should have failed with incorrect R_dict type."
    except AssertionError as e:
        assert "R_dict must be a defaultdict" in str(e), (
            f"Unexpected error message: {str(e)}"
        )
    # wrong type for state index
    try:
        R = RewardFunction(
            n_states=10,
            n_actions=5,
            R_dict=defaultdict(lambda: defaultdict(float), {"not": defaultdict(float)}),
        )
        assert False, (
            "Initialization should have failed with incorrect R_dict structure."
        )
    except AssertionError as e:
        assert "State index must be an integer" in str(e), (
            f"Unexpected error message: {str(e)}"
        )
    # wrong type for action index
    try:
        R = RewardFunction(
            n_states=10,
            n_actions=5,
            R_dict=defaultdict(lambda: defaultdict(float), {0: {"not": 0.0}}),
        )
        assert False, (
            "Initialization should have failed with incorrect R_dict structure."
        )
    except AssertionError as e:
        assert "Action index must be an integer" in str(e), (
            f"Unexpected error message: {str(e)}"
        )
    # wrong type for reward value
    try:
        R = RewardFunction(
            n_states=10,
            n_actions=5,
            R_dict=defaultdict(lambda: defaultdict(float), {0: {0: "not a float"}}),
        )
        assert False, (
            "Initialization should have failed with incorrect R_dict structure."
        )
    except AssertionError as e:
        assert "R_dict[0][0] must be a float" in str(e), (
            f"Unexpected error message: {str(e)}"
        )


def test_empty_initialization():
    """Testing whether the RewardFunction can be initialized with zero states and actions."""
    R = RewardFunction(n_states=0, n_actions=0)
    assert isinstance(R, RewardFunction), (
        "Initialization failed: object is not an instance of RewardFunction."
    )
    assert len(R.R_dict) == 0, "Initialization failed: R_dict should be empty."
    assert R.R_dict[0][0] == 0.0, (
        "Initialization failed: default value for R_dict should be 0.0."
    )
    assert R.n_states == 0, "Initialization failed: n_states should be 0."
    assert R.n_actions == 0, "Initialization failed: n_actions should be 0"


def test_initialization_from_numpy_array():
    """Testing whether the RewardFunction can be initialized from a numpy array and whether the values are correctly set."""
    num_s, num_a = 10, 5
    reward_array = initialize_array(num_s, num_a)

    # Initialize the RewardFunction with the numpy array
    R = RewardFunction.from_array(reward_array)
    assert R.n_states == num_s
    assert R.n_actions == num_a

    for s in range(num_s):
        for a in range(num_a):
            expected_reward = reward_array[s, a]
            actual_reward = R.R_dict[s][a]
            assert np.isclose(expected_reward, actual_reward), (
                f"Initialization from array failed: expected reward {expected_reward} for (s={s}, a={a}), got {actual_reward}."
            )


def test_indexing():
    """Testing whether the RewardFunction can be indexed correctly."""
    num_s, num_a = 10, 5
    reward_array = initialize_array(num_s, num_a)
    R = RewardFunction.from_array(reward_array)

    for s in range(num_s):
        for a in range(num_a):
            expected_reward = reward_array[s, a]
            actual_reward_1 = R[s, a]
            actual_reward_2 = R.R_dict[s][a]
            assert expected_reward == actual_reward_1, (
                f"Indexing test failed: expected reward {expected_reward} for (s={s}, a={a}), got {actual_reward_1} from R[s, a]."
            )
            assert expected_reward == actual_reward_2, (
                f"Indexing test failed: expected reward {expected_reward} for (s={s}, a={a}), got {actual_reward_2} from R.R_dict[s][a]."
            )
