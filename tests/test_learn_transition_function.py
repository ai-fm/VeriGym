import numpy as np

from verigym.abstraction.learn_transitions import (
    learn_transition_function,
    learn_initial_state_distribution,
)

from utils import (
    generate_dataset,
    make_discretized_env,
    initialize_transition_array,
)


def test_random_exploration_strategy():
    env, NUM_STEPS = make_discretized_env()
    _dataset = env.simulate("dummy_policy", NUM_STEPS)


def test_learn_transition_function():
    # create a simple trnasition function
    n_states, n_actions = 10, 5
    T_array = initialize_transition_array(n_states, n_actions)

    # generate a fake dataset of trajectories and state, action next_state tuples
    n_trajectories, trajectory_length = 10, 2000
    dataset = generate_dataset(
        n_states, n_actions, T_array, n_trajectories, trajectory_length
    )

    # use learn_transition_function to approximate T
    T = learn_transition_function(
        dataset=dataset, n_states=n_states, n_actions=n_actions
    )

    # check that T is close to T_array
    for s in range(n_states):
        for a in range(n_actions):
            for s_next in range(n_states):
                # print(T[s][a], T_array[s, a], flush=True)
                assert np.allclose(T[s, a, s_next], T_array[s, a, s_next], atol=0.1), (
                    f"Transition probabilities for state {s} and action {a} are not close enough to the true transition function."
                )


def test_learn_initial_state_distribution():
    # dummy action, reward, next state
    a, r, ns = 0, 0, 0
    # initial state occurences for two states s_0 and s_1
    n_s_0, n_s_1 = 3, 7
    init_states = [0] * n_s_0 + [1] * n_s_1
    # fill a dataset with initial states and varying trajectory lengths
    dataset = [
        [(s_init, a, r, ns) for _ in range(i + 1)]
        for i, s_init in enumerate(init_states)
    ]
    assert len(dataset) == (n_s_0 + n_s_1)
    # count occurences
    s_0 = [1 for t in dataset if t[0][0] == 0]
    s_1 = [1 for t in dataset if t[0][0] == 1]
    assert (len(s_0) == n_s_0) and (len(s_1) == n_s_1)

    # learn init state distribution
    S_init = learn_initial_state_distribution(dataset=dataset, n_states=100)

    assert S_init[0] == n_s_0 / (n_s_0 + n_s_1), (
        "Expected a different probability for s_0"
    )
    assert S_init[1] == n_s_1 / (n_s_0 + n_s_1), (
        "Expected a different probability for s_0"
    )
    assert (S_init[2:] == 0).all(), "All other states should have zero probability."
