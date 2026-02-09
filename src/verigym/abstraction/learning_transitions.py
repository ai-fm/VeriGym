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
from collections import defaultdict
from math import prod
from itertools import product
import logging

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm
from numpy.typing import NDArray

from verigym.abstraction.discretization import (
    # centered_pow_bin,
    generate_box_bins,
    BinEdges,
)
from verigym.abstraction.gym_utils.transform_observation import (
    ReplaceInfObservation,
    DiscretizeBoxObservation,
)
from verigym.environments import ExplicitEnv, VeriGymEnv

from verigym.environments.transition_func import TransitionFunction

# Datastructure of transition function. TODO: Should be defined somewhere else.
TransifitionFunctionDict = defaultdict[tuple, dict[int, defaultdict[tuple, float]]]
ExplorationStrategies = Literal["random", "sb3 policy"]
EXPLORATION_STRATEGIES = {"random", "sb3 policy"}

logger = logging.getLogger(__name__)


def generate_samples(
    env: gym.Env,
    num_steps: int = 1000,
    exploration_strategy: ExplorationStrategies = "random",
) -> list[tuple[NDArray, NDArray, NDArray]]:
    """Let's generate samples for a simple environment such as cart pole via random walk."""

    if exploration_strategy not in EXPLORATION_STRATEGIES:
        raise ValueError(
            f"{exploration_strategy = } does not match any of {EXPLORATION_STRATEGIES}"
        )

    if exploration_strategy == "sb3 policy":
        raise NotImplementedError

    dataset: list[tuple[NDArray, NDArray, NDArray]] = []
    progress_bar = tqdm(total=num_steps, desc="Sampling")
    current_step = 0
    state, _ = env.reset()

    while current_step < num_steps:
        action = env.action_space.sample()  # Random action
        next_state, _reward, terminated, truncated, _ = env.step(action)
        dataset.append((state, action, next_state))
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()

        progress_bar.update()
        current_step += 1

    return dataset


def learn_transition_function(
    dataset: list[tuple[NDArray, NDArray, NDArray]], bin_edges: BinEdges
) -> TransitionFunction:
    """
    Learn the transition function T using a frequentist approach.
    We will for now discretize all values to keep it simple. Rounding to nearest 0.5

    We expect the observations in the dataset to be integers that can be used as indices.
    This can be achieved by using the `DiscretizeBoxObservation` wrapper with argument `use_box_space=False`.

    Uses hashmaps (dictionaries) as a proxy for sparse matrices to store the count table.
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
        assert round(sum_tot, 1) in {0, 1}, (
            f"Counts for {s, a} sum to {sum_tot} != {0, 1}!"
        )

    return P


def construct_explicit_env(T) -> ExplicitEnv:
    """Constructs the `ExplicitEnv` and assigns the learned transition function to the model.

    TODO: Currently we pass `model=None` to the `EplicitEnv` constructor. Do we want to change that?
    """
    explicitenv = ExplicitEnv(
        model=None, render_mode=None
    )  # TODO: Instantiate the ExplicitEnv. Currently not possible due to Abstract methods etc.

    explicitenv.transition_function = T

    return explicitenv


def create_abstraction(
    original_env: VeriGymEnv,
    exploration_strategy: Literal["random", "sb3 policy"],
    num_steps: int,
    bin_edges_per_dim: int,
    verbose: bool = False,
) -> ExplicitEnv:
    assert isinstance(original_env, gym.Env), (
        f"original_env is type {type(original_env)} and does not inherit from gym.Env"
    )

    # discretize space
    bin_edges = generate_box_bins(
        original_env.observation_space, np.linspace, bin_edges_per_dim
    )
    logger.info(f"bin_edges: {bin_edges}")
    logger.info(f"num states: {prod([len(dimension) + 1 for dimension in bin_edges])}")
    discretized_env = DiscretizeBoxObservation(
        original_env, bin_edges=bin_edges, use_box_space=False
    )

    # create a dataset of transitions
    dataset = generate_samples(
        env=discretized_env,
        num_steps=num_steps,
        exploration_strategy=exploration_strategy,
    )

    # approximate the transition function
    _T = learn_transition_function(dataset=dataset, bin_edges=bin_edges)

    # Construct the abstracted ExplicitEnv
    # abstracted_env = construct_explicit_env(T) # TODO

    return


def factored_to_index(bin_edges: BinEdges, state: NDArray) -> int:
    """Converts a factored state representation to an index representation.
    Indexes start at 1.
    
    Parameters
    ----------
    bin_edges : BinEdges
        The bin edges used for discretization.
    state : NDArray
        The factored state representation.
    
    Returns
    -------
    int
        The index representation of the state.
    
    """
    lens = [len(dim) for dim in bin_edges]
    index = 1

    for i in range(1,len(state)+1):
        feature, edges = state[-i], bin_edges[-i]
        pos = np.where(feature == edges)[0]
        index += pos * (np.prod(lens[-i+1:]) if i!=1 else 1)

    return index


def index_to_factored(bin_edges: BinEdges, state_index: NDArray) -> NDArray:
    """Converts an index representation of a state to a factored state representation.
    Indexes start at 1.
    
    Parameters
    ----------
    state_index : int
        The index representation of the state.
    bin_edges : BinEdges
        The bin edges used for discretization.
    
    Returns
    -------
    NDArray
        The factored state representation.
    
    """
    state = np.zeros((len(bin_edges),))
    index = state_index - 1

    for i in range(len(bin_edges)-1, -1, -1):
        dim_size = len(bin_edges[i])
        pos = index % dim_size
        state[i] = bin_edges[i][pos.item()] 
        index = index // dim_size

    return state


if __name__ == "__main__":
    NUM_STEPS = 10000
    BIN_EDGES_PER_DIM = 5
    env = gym.make("CartPole-v1")
    env = ReplaceInfObservation(env, neg_inf=-10, pos_inf=10)

    abstracted_env = create_abstraction(
        original_env=env,
        exploration_strategy="random",
        num_steps=NUM_STEPS,
        bin_edges_per_dim=BIN_EDGES_PER_DIM,
    )
