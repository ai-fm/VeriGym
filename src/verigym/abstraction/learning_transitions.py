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

from math import prod
import gymnasium as gym
import numpy as np
from tqdm.auto import trange

from verigym.abstraction.discretization import centered_pow_bin, generate_box_bins
from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation, DiscretizeBoxObservation



BINS_PER_DIM = 5

def generate_samples(num_episodes: int=1000, max_steps_per_episode: int=200) -> np.ndarray:
    """Let's generate samples for a simple environment such as cart pole via random walk."""
    env = gym.make("CartPole-v1")
    env = ReplaceInfObservation(env, neg_inf=-10, pos_inf=10)
    bin_edges = generate_box_bins(env.observation_space, np.linspace, BINS_PER_DIM)
    print("bin_edges: ", bin_edges)
    print("num states: ", prod([len(dimension) for dimension in bin_edges]))
    env = DiscretizeBoxObservation(env, bin_edges=bin_edges, use_box_space=False)
    

    dataset = []

    for episode in range(num_episodes) if num_episodes<1000 else trange(num_episodes, desc="Sampling"):
        state, _ = env.reset()
        for step in range(max_steps_per_episode):
            action = env.action_space.sample()  # Random action
            next_state, reward, done, _, _ = env.step(action)
            dataset.append((state, action, next_state))
            state = next_state
            # print(state, action, next_state)
            if done:
                break

    env.close()
    # dataset = np.array(dataset)
    return dataset, bin_edges


def learn_transition_function(dataset: list, bin_edges_obs):
    """
        Learn the transition function T using a frequentist approach.
        We will for now discretize all values to keep it simple. Rounding to nearest 0.5
        
        We expect the observations in the dataset to be integers that can be used as indices. 
        This can be achieved by using the `DiscretizeBoxObservation` wrapper with argument `use_box_space=False`.
        
        TODO: Use sparse matrices to store the count table efficiently. 
    """
    # Get the state space and action space
    states = (np.array([data[0] for data in dataset]))
    actions = (np.array([data[1] for data in dataset]))
    next_states = (np.array([data[2] for data in dataset]))
    
    
    # TODO: Something is fishy with the state_space creation
    delta = 5
    state_space = bin_edges
    # print(state_space.shape, state_space.size)
    
    action_space = np.array((0,1))
    
    # n_states = state_space.size
    n_states = prod([len(dimension) for dimension in state_space])
    n_actions = action_space.size
    # n_next_states = len(np.unique(discretized_next_states, axis=0))
    
    # Initialize count table
    # print(f"Len of one dim {len(np.arange(-10, 10.5, delta))} -> {5**4*2*5**4} ")
    print(f"Initializing count_table with shape {n_states, n_actions, n_states} = {n_states*n_states*n_actions}")
    count_table = np.zeros((n_states, n_actions, n_states))
    
    # Populate count table 
    # TODO: Could this be sped up? Vectorization? Not sure. But it also isn't too slow at the moment.
    for s, a, s_next in zip(states, actions, next_states):
        
        # s_idx = np.where(np.all(np.unique(discretized_states, axis=0) == s, axis=1))[0][0]
        # a_idx = np.where(np.unique(discretized_actions) == a)[0][0]
        # s_next_idx = np.where(np.all(np.unique(discretized_next_states, axis=0) == s_next, axis=1))[0][0]
        
        count_table[s, a, s_next] += 1
        
    # Normalize to get probabilities
    normalizer = count_table.sum(axis=2, keepdims=True)
    # set all zero sums to 1, to avoid div by zero error
    normalizer[normalizer==0] = 1
    count_table = count_table / normalizer # TODO: 
    
    assert not np.isnan(count_table).any(), "Transition function contains NAN values."
    return count_table


def evaluate_transition_function(env: gym.Env, T , data: np.ndarray):
    pass
    
    





if __name__ == "__main__":
    num_episodes = 10000
    max_steps_per_episode = 100
    data, bin_edges = generate_samples(num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode)
    print(f"Generated {len(data)} samples.")
    # print(data)
    
    T = learn_transition_function(data, bin_edges)
    
    print(T.shape)
    density = (T!=0).mean()
    print(f"Density: {density:.5%}")
    



