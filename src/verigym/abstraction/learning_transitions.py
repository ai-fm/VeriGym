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

from typing import Literal
from math import prod
import gymnasium as gym
import numpy as np
from tqdm.auto import trange

from collections import defaultdict

from verigym.abstraction.discretization import centered_pow_bin, generate_box_bins, BinEdges
from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation, DiscretizeBoxObservation
from verigym.environments import ExplicitEnv, VeriGymEnv



def discretize_space(bins_per_dim: int):
    bin_edges = generate_box_bins(env.observation_space, np.linspace, bins_per_dim)
    print("bin_edges: ", bin_edges)
    print("num states: ", prod([len(dimension)+1 for dimension in bin_edges]))
    env = DiscretizeBoxObservation(env, bin_edges=bin_edges, use_box_space=False)
    
    return env, bin_edges

def generate_samples(env: gym.Env, num_episodes: int=1000, max_steps_per_episode: int=200) -> np.ndarray:
    """Let's generate samples for a simple environment such as cart pole via random walk."""
    env = ReplaceInfObservation(env, neg_inf=-10, pos_inf=10)
    

    dataset = []

    for episode in range(num_episodes) if num_episodes < 1000 else trange(num_episodes, desc="Sampling"):
        state, _ = env.reset()
        for step in range(max_steps_per_episode):
            action = env.action_space.sample()  # Random action
            next_state, _reward, terminated, truncated, _ = env.step(action)
            dataset.append((state, action, next_state))
            state = next_state
            # print(state, action, next_state)
            if terminated or truncated:
                break

    env.close()
    # dataset = np.array(dataset)
    return dataset


def learn_transition_function(dataset: list, bin_edges: BinEdges):
    """
        Learn the transition function T using a frequentist approach.
        We will for now discretize all values to keep it simple. Rounding to nearest 0.5

        We expect the observations in the dataset to be integers that can be used as indices. 
        This can be achieved by using the `DiscretizeBoxObservation` wrapper with argument `use_box_space=False`.

        Uses hashmaps (dictionaries) as a proxy for sparse matrices to store the count table. 

        The dictionaries are nested like:

        P = {
                state_index: {
                    action_index:
                        {
                            next_state_index: probability
                            for next_state_index in non_zero_transitions
                        }
                        for action_index in action_space
                } for state_index in state_space
            }
    """

    P = {}
    P_tot = defaultdict(lambda: 0)

    # Populate count table
    # TODO: Could this be sped up? Vectorization? Not sure. But it also isn't too slow at the moment.
    for s, a, s_next in dataset:
        s = tuple(s.ravel().tolist())
        a = int(a.item())
        s_next = tuple(s_next.ravel().tolist())
        if s not in P:
            P[s] = {}
        if a not in P[s]:
            P[s][a] = defaultdict(lambda: 0)
        P[s][a][s_next] += 1
        P_tot[(s, a)] += 1

    for (s, a), tot_count in P_tot.items():
        if P_tot == 0:
            continue
        for s_next in P[s][a].keys():
            P[s][a][s_next] /= tot_count
        sum_tot = sum([prob for s_next, prob in P[s][a].items()])
        assert round(sum_tot, 1) in {
            0, 1}, f"Counts for {s, a} sum to {sum_tot} != {0, 1}!"

    return P


def evaluate_transition_function(env: gym.Env, T, data: np.ndarray):
    pass
    
    
def construct_explicit_env(T) -> ExplicitEnv:
    """ Constructs the `ExplicitEnv` and assigns the learned transition function to the model.

        TODO: Currently we pass `model=None` to the `EplicitEnv` constructor. Do we want to change that?
    """
    explicitenv = ExplicitEnv(model=None, render_mode=None)
    
    explicitenv.transition_function = T
    
    return explicitenv


def create_abstraction(
        original_env: VeriGymEnv, 
        exploration_strategy: Literal["random", "sb3 policy"],
        num_episodes: int,
        verbose: bool=False,
        ) -> ExplicitEnv:
    
    # decide on how to discretize the state and action space
    discretize_space()
    
    # create a dataset of transitions
    NUM_EPISODES = 1000
    # generate_samples(num_episodes=)
    
    # approximate the transition function
    
    # Construct the abstracted ExplicitEnv
    abstracted_env = construct_explicit_env(T)
    
    return abstracted_env


if __name__ == "__main__":
    num_episodes = 5000
    max_steps_per_episode = 100
    BINS_EDGES_PER_DIM = 5
    env = gym.make("CartPole-v1")
    
    data, bin_edges = generate_samples(env, num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode, bins_per_dim=BINS_EDGES_PER_DIM)
    print(f"Generated {len(data)} samples.")
    # print(data)

    T = learn_transition_function(data, bin_edges)
