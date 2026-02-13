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
import logging

import gymnasium as gym
import numpy as np
import functools
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
from verigym.environments import ExplicitEnv, VeriGymEnv, GenerativeEnv
from verigym.abstraction.abstractionmapper import (
    AbstractionMap,
    AbstractionMapper,
    IdentityAbstractionMap,
)

from verigym.environments.transition_func import TransitionFunction

# Datastructure of transition function. TODO: Should be defined somewhere else.
# TransifitionFunctionDict = defaultdict[tuple, dict[int, defaultdict[tuple, float]]]
ExplorationStrategies = Literal["random", "sb3 policy"]
EXPLORATION_STRATEGIES = {"random", "sb3 policy"}

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


def create_abstraction(
    original_env: VeriGymEnv,
    exploration_strategy: Literal["random", "sb3 policy"],
    num_steps: int,
    bin_edges_per_dim: int | list[int],
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
        original_env, bin_edges=bin_edges, use_box_space=True
    )

    # discretize actions
    # TODO: Currently we assume that the action space is already discrete and starts at 0. We should add a wrapper to discretize the action space if this is not the case. For now, we just check that the action space is compatible and warn if it isn't.
    assert isinstance(original_env.action_space, gym.spaces.Discrete), (
        f"Currently only Discrete action spaces are supported but found {original_env.action_space}"
    )
    logger.warning(
        "Currently only Discrete action spaces are supported, so no discretization is applied to the action space."
    )
    if original_env.action_space.start != 0:
        logger.warning(
            f"Action space starts at {original_env.action_space.start} instead of 0. This might cause issues with the current implementation as we expect actions to be integers starting from 0."
        )
    # TODO: Create action abstraction map for discretized actions.

    # Convert into VeriGym compatible object
    generative_env = GenerativeEnv.from_gymnasium(discretized_env)

    # create a dataset of transitions
    dataset = generative_env.simulate(policy=exploration_strategy, n_steps=num_steps)

    # convert states to indices
    dataset_indices = []
    for trajectory in dataset:
        trajectory_indices = []
        for s, a, r, s_next in trajectory:
            s_index = factored_to_index(bin_edges, s)
            assert s_index >= 1, f"s_index should be >= 1 but got {s_index}"
            s_next_index = factored_to_index(bin_edges, s_next)
            trajectory_indices.append((s_index, a, r, s_next_index))
        dataset_indices.append(trajectory_indices)

    # approximate the transition function
    n_actions = original_env.action_space.n
    n_states = prod([len(dimension) for dimension in bin_edges])
    T = learn_transition_function(
        dataset=dataset_indices, n_states=n_states, n_actions=n_actions
    )

    # approximate the reward function
    # TODO

    # Create abstraction mapping
    def mapping(x: NDArray) -> int:
        return factored_to_index(bin_edges, discretized_env.func(x))

    state_abstraction_map = AbstractionMap(forward_map=mapping)

    # Construct the abstracted ExplicitEnv
    abstracted_env = ExplicitEnv(
        nr_states=n_states,
        nr_actions=n_actions,
        nr_rewards=None,  # TODO
        initial_state_distr={0: 1.0},  # TODO
        transition_function=T,
        reward_function={},  # TODO
        original_env=original_env,
        render_mode=None,
    )

    # TODO (minor) work around the cross reference of ExplicitEnv and AbstractionMapper in a better way -> Does AbstractionMapper actually need those Envs as class members?
    # TODO: make mapper for discretized actions; action abstraction is identity by default
    mapper = AbstractionMapper(
        original_env, abstracted_env, state_abstraction_map=state_abstraction_map
    )

    abstracted_env.abstraction_map = mapper

    return abstracted_env


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

    for i in range(1, len(state) + 1):
        feature, edges = state[-i], bin_edges[-i]
        pos = np.where(feature == edges)[0]
        index += pos * (np.prod(lens[-i + 1 :]) if i != 1 else 1)

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

    for i in range(len(bin_edges) - 1, -1, -1):
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
