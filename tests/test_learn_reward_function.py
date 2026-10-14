import numpy as np
import pytest

from verigym.abstraction.learn_abstraction import (
    learn_abstraction,
    normalize_aggregated_counts,
)

from utils import initialize_transition_array, generate_dataset


def test_single_sample_dataset():
    n_states, n_actions = 5, 3
    T_array = initialize_transition_array(n_states, n_actions)
    n_trajectories, trajectory_length = 1, 1
    reward = np.array([10], dtype=float)
    dataset = generate_dataset(
        n_states, n_actions, T_array, n_trajectories, trajectory_length, reward
    )
    print(dataset)
    T_dict, R_dict, P_tot, state_distr = learn_abstraction(
        dataset=dataset, n_states=n_states, n_actions=n_actions, multithreading=False
    )
    print("yo")
    print(R_dict)
    _, R, _ = normalize_aggregated_counts(
        T_dict, R_dict, P_tot, state_distr, n_states=n_states, n_actions=n_actions
    )

    sample = dataset[0][0]
    state, action = sample[0], sample[1]
    assert R[state, action] == reward, (
        f"Expected reward of 0 for single entry but found {R[0, 0]}"
    )

    reward_2 = np.array(11)
    dataset_2 = [[(sample[0], sample[1], reward_2, sample[3])]]
    dataset.extend(dataset_2)
    T_dict_2, R_dict_2, P_tot_2, state_distr_2 = learn_abstraction(
        dataset=dataset, n_states=n_states, n_actions=n_actions, multithreading=False
    )
    _, R_combined, _ = normalize_aggregated_counts(
        T_dict_2,
        R_dict_2,
        P_tot_2,
        state_distr_2,
        n_states=n_states,
        n_actions=n_actions,
    )
    assert R_combined[state, action] == (reward + reward_2) / 2, (
        f"Expected reward of {reward} for combined dataset but found {R_combined[0, 0]}"
    )


def test_multiple_samples_deterministic():
    """Dataset with multiple samples, but rewards is always the same."""
    n_states, n_actions = 3, 3
    T_array = initialize_transition_array(n_states, n_actions)
    n_trajectories, trajectory_length = 10, 10
    reward = np.array([10])
    dataset = generate_dataset(
        n_states, n_actions, T_array, n_trajectories, trajectory_length, reward
    )
    T_dict, R_dict, P_tot, state_distr = learn_abstraction(
        dataset=dataset, n_states=n_states, n_actions=n_actions, multithreading=False
    )
    _, R, _ = normalize_aggregated_counts(
        T_dict, R_dict, P_tot, state_distr, n_states=n_states, n_actions=n_actions
    )

    # Check that all state-action pairs have the correct reward
    for s in range(n_states):
        for a in range(n_actions):
            assert R[s, a] == reward, f"Expected reward of {reward} but found {R[s, a]}"


def test_multiple_samples_stochastic():
    """Dataset with multiple samples, but rewards is stochastic."""
    n_states, n_actions = 3, 3
    T_array = initialize_transition_array(n_states, n_actions)
    n_trajectories, trajectory_length = 10, 100
    rewards = np.array([10, 20])
    dataset = generate_dataset(
        n_states, n_actions, T_array, n_trajectories, trajectory_length, rewards
    )
    T_dict, R_dict, P_tot, state_distr = learn_abstraction(
        dataset=dataset, n_states=n_states, n_actions=n_actions, multithreading=False
    )
    _, R, _ = normalize_aggregated_counts(
        T_dict, R_dict, P_tot, state_distr, n_states=n_states, n_actions=n_actions
    )

    # Check that all state-action pairs have a reward between the min and max of the reward array
    for s in range(n_states):
        for a in range(n_actions):
            assert rewards.min() <= R[s, a] <= rewards.max(), (
                f"Expected reward between {rewards.min()} and {rewards.max()} but found {R[s, a]}"
            )


def test_single_sample_dataset_multi_threading():
    n_states, n_actions = 3, 3
    T_array = initialize_transition_array(n_states, n_actions)
    n_trajectories, trajectory_length = 1, 1
    reward = np.array([10])
    dataset = generate_dataset(
        n_states, n_actions, T_array, n_trajectories, trajectory_length, reward
    )

    T_dict, R_dict, P_tot, state_distr = learn_abstraction(
        dataset=dataset, n_states=n_states, n_actions=n_actions, multithreading=True
    )
    _, R, _ = normalize_aggregated_counts(
        T_dict, R_dict, P_tot, state_distr, n_states=n_states, n_actions=n_actions
    )
    del T_dict
    del R_dict
    del P_tot
    del state_distr

    sample = dataset[0][0]
    state, action = sample[0], sample[1]
    assert R[state, action] == reward, (
        f"Expected reward of 0 for single entry but found {R[0, 0]}"
    )

    reward_2 = np.array(11)
    dataset_2 = [[(sample[0], sample[1], reward_2, sample[3])]]
    dataset.extend(dataset_2)

    T_dict, R_dict, P_tot, state_distr = learn_abstraction(
        dataset=dataset, n_states=n_states, n_actions=n_actions, multithreading=True
    )
    _, R_combined, _ = normalize_aggregated_counts(
        T_dict, R_dict, P_tot, state_distr, n_states=n_states, n_actions=n_actions
    )

    assert R_combined[state, action] == pytest.approx((reward + reward_2) / 2), (
        f"Expected reward of {reward} for combined dataset but found {R_combined[0, 0]}"
    )


def test_multiple_samples_deterministic_multi_threading():
    """Dataset with multiple samples, but rewards is always the same."""
    n_states, n_actions = 3, 3
    T_array = initialize_transition_array(n_states, n_actions)
    n_trajectories, trajectory_length = 10, 10
    reward = np.array([10])
    dataset = generate_dataset(
        n_states, n_actions, T_array, n_trajectories, trajectory_length, reward
    )
    T_dict, R_dict, P_tot, state_distr = learn_abstraction(
        dataset=dataset, n_states=n_states, n_actions=n_actions, multithreading=True
    )
    _, R, _ = normalize_aggregated_counts(
        T_dict, R_dict, P_tot, state_distr, n_states=n_states, n_actions=n_actions
    )

    # Check that all state-action pairs have the correct reward
    for s in range(n_states):
        for a in range(n_actions):
            assert R[s, a] == reward, f"Expected reward of {reward} but found {R[s, a]}"


def test_multiple_samples_stochastic_multi_threading():
    """Dataset with multiple samples, but rewards is stochastic."""
    n_states, n_actions = 3, 3
    T_array = initialize_transition_array(n_states, n_actions)
    n_trajectories, trajectory_length = 10, 100
    rewards = np.array([10, 20])
    dataset = generate_dataset(
        n_states, n_actions, T_array, n_trajectories, trajectory_length, rewards
    )
    T_dict, R_dict, P_tot, state_distr = learn_abstraction(
        dataset=dataset, n_states=n_states, n_actions=n_actions, multithreading=True
    )
    _, R, _ = normalize_aggregated_counts(
        T_dict, R_dict, P_tot, state_distr, n_states=n_states, n_actions=n_actions
    )

    # Check that all state-action pairs have a reward between the min and max of the reward array
    for s in range(n_states):
        for a in range(n_actions):
            assert rewards.min() <= R[s, a] <= rewards.max(), (
                f"Expected reward between {rewards.min()} and {rewards.max()} but found {R[s, a]}"
            )


def test_cumsum_matches_mean_of_observed_rewards():
    """`R_cumsum` stores the cumulative reward per (s, a): single- and multithreaded learning agree,
    and normalizing yields the mean of the observed rewards."""
    dataset = [
        [(0, 0, 1.0, 1), (1, 1, 4.0, 0), (0, 0, 2.0, 0)],
        [(0, 0, 6.0, 1), (1, 1, np.float32(0.5), 1)],
        [(1, 0, np.array([3.0]), 0)],
    ]
    observed = {(0, 0): [1.0, 2.0, 6.0], (1, 1): [4.0, 0.5], (1, 0): [3.0]}

    counts_single = learn_abstraction(dataset, n_states=2, n_actions=2, multithreading=False)
    counts_multi = learn_abstraction(dataset, n_states=2, n_actions=2, multithreading=True)
    assert counts_single[1] == counts_multi[1]

    _, R, _ = normalize_aggregated_counts(*counts_single, n_states=2, n_actions=2)
    for (s, a), rewards in observed.items():
        assert type(R[s, a]) is float
        assert R[s, a] == pytest.approx(np.mean(rewards))
