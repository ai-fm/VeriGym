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

from verigym.abstraction.discretization import centered_pow_bin, generate_box_bins, BinEdges
from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation, DiscretizeBoxObservation





def generate_samples(num_episodes: int=1000, max_steps_per_episode: int=200, bins_per_dim=5) -> np.ndarray:
    """Let's generate samples for a simple environment such as cart pole via random walk."""
    env = gym.make("CartPole-v1")
    env = ReplaceInfObservation(env, neg_inf=-10, pos_inf=10)
    bin_edges = generate_box_bins(env.observation_space, np.linspace, bins_per_dim)
    print("bin_edges: ", bin_edges)
    print("num states: ", prod([len(dimension)+1 for dimension in bin_edges]))
    env = DiscretizeBoxObservation(env, bin_edges=bin_edges, use_box_space=False)
    

    dataset = []

    for episode in range(num_episodes) if num_episodes<1000 else trange(num_episodes, desc="Sampling"):
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
    return dataset, bin_edges



def learn_transition_function(dataset: list, bin_edges: BinEdges):
    """
        Learn the transition function T using a frequentist approach.
        We will for now discretize all values to keep it simple. Rounding to nearest 0.5
        
        We expect the observations in the dataset to be integers that can be used as indices. 
        This can be achieved by using the `DiscretizeBoxObservation` wrapper with argument `use_box_space=False`.
        
        TODO: Use sparse matrices to store the count table efficiently. 
    """
    # Get the states and actions
    states = (np.array([data[0] for data in dataset]))
    actions = (np.array([data[1] for data in dataset]))
    next_states = (np.array([data[2] for data in dataset]))
    
    state_shape = tuple(len(edges)+1 for edges in bin_edges)
    
    action_space = np.array((0,1)) # TODO: this is not generic yet
    action_shape = action_space.shape
    
    # Initialize count table
    print(f"Initializing count_table with shape {state_shape, action_shape, state_shape} = {prod(state_shape)*prod(action_shape)*prod(state_shape)}")
    count_table = np.zeros((*state_shape, *action_shape, *state_shape), dtype=int)
    print(f"{count_table.shape =}")
    
    """
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
    
    # Populate count table 
    # TODO: Could this be sped up? Vectorization? Not sure. But it also isn't too slow at the moment.
    for s, a, s_next in zip(states, actions, next_states):
        index = tuple(s) + tuple((a,)) + tuple(s_next) #TODO: putting the action in a tuple might break with complex action spaces
        count_table[index] += 1
        
    # Normalize to get probabilities
    axis = tuple(range(count_table.ndim-len(state_shape),count_table.ndim)) # dimension corresponding to s_next
    normalizer = count_table.sum(axis=axis, keepdims=True)
    # set all zero sums to 1, to avoid div by zero error
    normalizer[normalizer==0] = 1
    
    count_table = count_table / normalizer # TODO:     
    
    temp = count_table.sum(axis=axis)
    assert np.logical_or((temp == 1), (temp == 0)).all()
    
    assert not np.isnan(count_table).any(), "Transition function contains NAN values."
    return count_table


def evaluate_transition_function(env: gym.Env, T , data: np.ndarray):
    pass
    
    





if __name__ == "__main__":
    num_episodes = 50000
    max_steps_per_episode = 100
    BINS_EDGES_PER_DIM = 5
    data, bin_edges = generate_samples(num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode, bins_per_dim=BINS_EDGES_PER_DIM)
    print(f"Generated {len(data)} samples.")
    # print(data)
    
    T = learn_transition_function(data, bin_edges)
    
    print(T.shape)
    density = (T!=0).mean()
    print(f"Density: {density:.5%}")
    



