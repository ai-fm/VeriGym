"""
In this module we learn the transition function T through interactions with the environment.

1) Collect data. Through random walks or exploration strategies.
    i) Initialize an empty dataset to store observations.
    ii) For each episode:
        a) Reset the environment to get the initial state.
        b) While the episode is not done:
            - Select an action (randomly or using a policy).
            - Execute the action in the environment.
            - Observe the next state and reward.
            - Store the (state, action, next_state) tuple in the dataset.
2) Use the data to learn T.
    i) Using the frequentist approach, we initialize a count table (and update it with each new observation).
    ii) We discretize all values
    iii) For each (state, action, next_state) tuple in the dataset:
        a) Increment the count for the (state, action, next_state) in the count table.
    iv) Normalize the counts to get probabilities for T.


3) Evaluate the learned T by comparing predicted next states with actual next states from a validation dataset/new interaction with the environment.
    i) As we are compared point samples to our predicted distributions we should look at calibration plots. More precisely; coverage plots
"""

from collections import defaultdict
import logging

from numpy.typing import NDArray
import numpy as np

from verigym.environments.transition_func import TransitionFunction

logger = logging.getLogger(__name__)


def learn_transition_function(
    dataset: list[tuple[NDArray, NDArray, NDArray]], n_states: int, n_actions: int
) -> TransitionFunction:
    """
    Learn the transition function T using a frequentist approach.

    We expect the observations and actions in the dataset (tuples of state, action, next_state) to be integers that can be used as indices.
    This can be achieved by using the `DiscretizeBoxObservation` wrapper with argument `use_box_space=False` for gym environments with continuous spaces and then using the `factored_to_index` function.

    Uses hashmaps (dictionaries) as a proxy for sparse matrices to store the count table.
    """

    T_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    P_tot = defaultdict(lambda: 0)

    # Populate count table
    for trajectory in dataset:
        for s, a, r, s_next in trajectory:
            s, a, s_next = s.item(), a.item(), s_next.item()
            if s not in T_dict:
                T_dict[s] = {}  # do we need this? defaultdict should take care of this
            if a not in T_dict[s]:
                T_dict[s][a] = defaultdict(lambda: 0)
            T_dict[s][a][s_next] += 1
            P_tot[(s, a)] += 1

    for (s, a), tot_count in P_tot.items():
        if P_tot == 0:
            continue
        for s_next in T_dict[s][a].keys():
            T_dict[s][a][s_next] /= tot_count
        sum_tot = sum([prob for s_next, prob in T_dict[s][a].items()])
        assert round(sum_tot, 1) in {0, 1}, (
            f"Counts for {s, a} sum to {sum_tot} != {0, 1}!"
        )

    return TransitionFunction(n_states, n_actions, T_dict)


def learn_initial_state_distribution(
    dataset: list[tuple[NDArray, NDArray, NDArray]], n_states: int
) -> NDArray:
    """
    Learns the initial state distribution by looking at the first state(s) of a dataset of trajectories.

    Parameters
    ----------
    dataset : list[tuple[NDArray, NDArray, NDArray]]
        Dataset of trajectories.
    n_states : int
        Number of states.

    Returns
    -------
    NDArray
        Array of length `n_states` indicating the probability of a state corresponding to the index.
    """
    state_distr = np.zeros(n_states)
    
    for trajectory in dataset:
        # look at first sample only 
        init_state, action, reward, next_state = trajectory[0]
        state_distr[init_state] += 1
        
    # normalize
    state_distr /= state_distr.sum()
    
    return state_distr
        
        
