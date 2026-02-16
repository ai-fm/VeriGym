import numpy as np

from verigym.abstraction.learn_transitions import learn_transition_function

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
