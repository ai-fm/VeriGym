import numpy as np

from verigym.abstraction.learn_abstraction import learn_abstraction

from utils import initialize_transition_array, generate_dataset


def test_single_sample_dataset():
    n_states, n_actions = 5, 3
    T_array = initialize_transition_array(n_states, n_actions)
    n_trajectories, trajectory_length = 1, 1
    reward = np.array([10])
    dataset = generate_dataset(
        n_states, n_actions, T_array, n_trajectories, trajectory_length, reward
    )
    _, R, _ = learn_abstraction(dataset, n_states, n_actions)

    sample = dataset[0][0]
    state, action = sample[0], sample[1]
    assert R[state, action] == reward, (
        f"Expected reward of 0 for single entry but found {R[0, 0]}"
    )

    reward_2 = np.array(11)
    dataset_2 = [[(sample[0], sample[1], reward_2, sample[3])]]
    dataset.extend(dataset_2)
    _, R_combined, _ = learn_abstraction(dataset, n_states, n_actions)
    assert R_combined[state, action] == (reward + reward_2) / 2, (
        f"Expected reward of {reward} for combined dataset but found {R_combined[0, 0]}"
    )


def test_multiple_samples_deterministic():
    """Dataset with multiple samples, but rewards is always the same."""
    n_states, n_actions = 5, 3
    T_array = initialize_transition_array(n_states, n_actions)
    n_trajectories, trajectory_length = 10, 10
    reward = np.array([10])
    dataset = generate_dataset(
        n_states, n_actions, T_array, n_trajectories, trajectory_length, reward
    )
    _, R, _ = learn_abstraction(dataset, n_states, n_actions)

    # Check that all state-action pairs have the correct reward
    for s in range(n_states):
        for a in range(n_actions):
            assert R[s, a] == reward, f"Expected reward of {reward} but found {R[s, a]}"


def test_multiple_samples_stochastic():
    """Dataset with multiple samples, but rewards is stochastic."""
    n_states, n_actions = 5, 3
    T_array = initialize_transition_array(n_states, n_actions)
    n_trajectories, trajectory_length = 10, 1000
    rewards = np.array([10, 20])
    dataset = generate_dataset(
        n_states, n_actions, T_array, n_trajectories, trajectory_length, rewards
    )
    _, R, _ = learn_abstraction(dataset, n_states, n_actions)

    # Check that all state-action pairs have a reward between the min and max of the reward array
    mean_reward = rewards.mean()
    for s in range(n_states):
        for a in range(n_actions):
            assert np.isclose(R[s, a], mean_reward, atol=0.1 * mean_reward), (
                f"Expected reward between {rewards.min()} and {rewards.max()} but found {R[s, a]}"
            )
