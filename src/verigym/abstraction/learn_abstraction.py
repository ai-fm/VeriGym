import copy
import functools
import logging
import multiprocessing
import time
from typing import Callable
from math import prod
from collections import defaultdict


import gymnasium as gym
import numpy as np

from numpy.typing import NDArray

from ..environments.reward_func import RewardFunction
from ..environments.transition_func import TransitionFunction
from ..environments.explicitenv import ExplicitEnv
from ..environments.verigymenv import VeriGymEnv
from ..environments.learnedexplicitenv import LearnedExplicitEnv, LearnedTransitionFunction, LearnedRewardFunction
from ..policy.policy import PolicyClass
from ..policy.implemented_policies import RandomizedPolicy
from .abstractionmapper import AbstractionMap, AbstractionMapper
from .gym_utils.mapping import box_to_discrete, get_discrete_box_tf
from .gym_utils.transform_observation import (
    DiscretizeBoxObservation,
)
from .discretization import (
    generate_box_bins,
)
from .utils import factored_to_index

logger = logging.getLogger(__name__)


class CachedDiscretizer:
    def __init__(self, discretizer: Callable):
        self.cache = {}
        self.discretizer = discretizer

    def discretize(self, state: NDArray) -> int:
        tupled = tuple(state)
        if tupled not in self.cache:
            self.cache[tupled] = self.discretizer(state)
        return self.cache[tupled]


def mapping(x: NDArray, to_bins: Callable, to_int: Callable):
    return to_int(to_bins(x))


def create_abstraction(
    original_env: VeriGymEnv,
    exploration_policy,
    num_steps: int,
    bin_edges_per_dim: int | list[int],
    use_box_space: bool = True,
    multithreading: bool = True,
    n_iterations: int = 1,
    verbose: bool = False,
) -> ExplicitEnv:
    """
    Creates an abstraction from a VeriGymEnv by discretizing the state space. Returns an `ExplicitEnv`.
    Action space is currently not abstracted.


    Parameters
    ----------
    original_env : VeriGymEnv
        The environment / model to be abstracted.
    exploration_policy : PolicyClass
        The policy of interacting with the `original_env` (e.g. a random policy).
    num_steps : int
        Number of steps to take within the environment (also, see `n_iterations`).
    bin_edges_per_dim : int | list[int]
        Number of discretization bins per feature dimension of the state space.
    multithreading: bool, optional
        Whether to multithread or use single thread.
    n_iterations: int
        Number of (interleaving) iterations. For each iteration the `exploration_policy.update_for_abstraction_refinement(...)` will be called, alowing the policy to adjust based on the gathered interactions. Note, that the total number of steps equal `n_iterations * num_steps`. For policies that are not interleaving, set the n_iterations to 1 (default).
    verbose : bool, optional
        Whether to be verbose, by default False.

    Returns
    -------
    ExplicitEnv
        The abstracted model.
    """

    ###
    #   Validity Checks:
    ##

    assert isinstance(original_env, gym.Env), (
        f"original_env is type {type(original_env)} and does not inherit from gym.Env"
    )
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

    ###
    #   Initialization
    ###

    # Get discretized env &  exploration policy
    bin_edges = generate_box_bins(
        original_env.observation_space, np.linspace, bin_edges_per_dim
    )
    logger.info(f"bin_edges: {bin_edges}")
    logger.info(f"num states: {prod([len(dimension) + 1 for dimension in bin_edges])}")

    # discretize actions
    # TODO: Currently we assume that the action space is already discrete and starts at 0. We should add a wrapper to discretize the action space if this is not the case. For now, we just check that the action space is compatible and warn if it isn't.
    discretized_env = DiscretizeBoxObservation(
        original_env, bin_edges=bin_edges, use_box_space=use_box_space
    )
    
    # Define variables
    n_actions, n_states, T_counts, R_dict_counts, P_tot_counts, state_distr_counts = create_new_objects(original_env, bin_edges)
    dataset = []
    
    # Get abstractionmap
    if use_box_space:
        f = get_discrete_box_tf(discretized_env.observation_space, bin_edges)
    else:
        _, f = box_to_discrete(discretized_env.observation_space, bin_edges)
    discretizer = CachedDiscretizer(
        functools.partial(factored_to_index, bin_edges=bin_edges)
    )
    state_abstraction_map = AbstractionMap(
        forward_map=functools.partial(mapping, to_int=discretizer.discretize, to_bins=f)
    )
    # TODO: make mapper for discretized actions; action abstraction is identity by default
    abstraction_mapper = AbstractionMapper(state_abstraction_map=state_abstraction_map)
    learned_env = LearnedExplicitEnv(
        nr_states=n_states,
        nr_actions=n_actions,
        nr_rewards=1,
        initial_state_distr=defaultdict(int),
        transition_function=LearnedTransitionFunction(n_states, n_actions),
        reward_function=LearnedRewardFunction(n_states, n_actions),
        abstraction_map=abstraction_mapper,      # TODO: I don't fully understand this: aren't abstraction mappers part of policies, not environments?
        original_env=original_env,
        render_mode=None,
    )
    exploration_policy = exploration_policy(learned_env, map=abstraction_mapper)

    ###
    #   Exploration Loop
    ###

    # Loop through iterations. If interleaving abstraction is not required, n_iterations will be just 1.
    for i in range(n_iterations):

        # print(f"Iteration {i}:")
        tik = time.time()
        # generate dataset via simulation
        dataset = original_env.simulate(
            policy=exploration_policy, n_steps=num_steps, verbose=verbose
        )

        tok = time.time()
        # print(f"Simulation done! (in {tok-tik})s")

        # approximate the transition function from new dataset
        # note, we are only getting the counts for state/action/nex_state/reward pairs here
        new_T_counts, new_R_dict_counts, new_P_tot_counts, new_state_distr_counts = (
            learn_abstraction(
                dataset,
                n_states,
                n_actions,
                abstraction_mapper=abstraction_mapper,
                multithreading=multithreading,
            )
        )
        # print("Learning done! Starting aggregation...")
        # if (type(exploration_policy) is not RandomizedPolicy) or (i == n_iterations-1):
        if True:
            # print(new_T_counts)
            learned_env.update_env(
                new_init_counts=new_state_distr_counts, 
                new_transition_counts=new_T_counts,
                new_reward_counts=new_R_dict_counts
            )

        # print(f"Learning & Aggregation done! (in {time.time()-tok}s)")

        if i < n_iterations-1:
            exploration_policy.update_for_abstraction_refinement(learned_env)
            assert isinstance(exploration_policy, PolicyClass)

    return learned_env

def create_new_objects(original_env: VeriGymEnv, bin_edges: list[NDArray]) -> tuple[int, int, dict, dict, dict, NDArray]:
    """ Creates all required objects for the abstraction learning."""
    # number of actions
    n_actions = original_env.action_space.n  # TODO, update for discretized actions
    # number of states
    n_states = prod([len(dimension) for dimension in bin_edges])
    # number of counts (occurences) for each state-action-next_state pair
    T_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    # list of rewards for all occured state-action pairs
    R_dict_counts = defaultdict(lambda: defaultdict(lambda: list()))
    # dict with occurences for each state-action pair
    P_tot_counts = defaultdict(lambda: 0)
    # list with occurences of each state as initial state
    state_distr_counts = np.zeros(n_states)
    return n_actions, n_states, T_counts, R_dict_counts, P_tot_counts, state_distr_counts


# Define named functions for defaultdict factories
def make_int_dict():  # pragma: no cover
    return defaultdict(int)


def make_list_dict():  # pragma: no cover
    return defaultdict(list)


def make_middle_dict():  # pragma: no cover
    return defaultdict(make_int_dict)


def collect_data_from_trajectories(
    trajectories: list[list[tuple[int, int, float, int]]],
    num_states: int,
    mapper: AbstractionMapper,
):
    # Initialize local storage for this thread
    data = {
        "T": defaultdict(make_middle_dict),
        "R": defaultdict(make_list_dict),
        "init": defaultdict(int),
        "tot": defaultdict(int),
    }

    # mapper = copy.deepcopy(mapper)

    # print(len(trajectories))

    T_dict = data["T"]
    R_dict = data["R"]
    state_distr = data["init"]
    P_tot = data["tot"]

    for trajectory in trajectories:
        for i, (s, a, r, s_next) in enumerate(trajectory):
            if isinstance(a, np.ndarray):
                a = a.item()
            if isinstance(r, np.ndarray):
                r = r.item()
            s = mapper.state_abstraction_map.forward_map(s)
            a = mapper.action_abstraction_map.forward_map(a)
            s_next = mapper.state_abstraction_map.forward_map(s_next)
            if i == 0:
                state_distr[s] += 1
            T_dict[s][a][s_next] += 1
            P_tot[(s, a)] += 1
            R_dict[s][a].append(r)

    return data


def learn_abstraction_multithreaded(
    dataset: list[list[tuple[int, int, float, int]]],
    n_states: int,
    n_actions: int,
    abstraction_mapper: AbstractionMapper,
):
    T_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    R_dict = defaultdict(lambda: defaultdict(lambda: list()))
    P_tot = defaultdict(lambda: 0)
    state_distr = defaultdict(int)

    num_threads = max(min(4, multiprocessing.cpu_count() - 1), 1)
    chunk_size = len(dataset) // num_threads

    if chunk_size == 0:  # For handling super small datasets (like in the tests)
        print("Chunk size is zero!")
        num_threads = 1
        chunks = [(dataset, n_states, abstraction_mapper)]
    else:
        chunks = [
            (
                dataset[i * chunk_size : i * chunk_size + chunk_size],
                n_states,
                copy.deepcopy(abstraction_mapper),
            )
            for i in range(num_threads - 1)
        ]
        chunks.append(
            (
                dataset[(num_threads - 1) * chunk_size :],
                n_states,
                copy.deepcopy(abstraction_mapper),
            )
        )
        lens = [len(chunk[0]) for chunk in chunks]
        assert sum(lens) == len(dataset), f"{sum(lens)=} and {len(dataset)=}"

    tik = time.time()

    with multiprocessing.Pool(num_threads) as executor:
        results = executor.starmap(collect_data_from_trajectories, chunks)

    tok = time.time()

    # print("processing in", tok - tik)
    # print("aggregating..")

    for r in results:
        for (s, count) in r["init"].items():
            state_distr[s] += count

        tot = r["tot"]
        for (s, a), tot_count in tot.items():
            P_tot[(s, a)] += tot_count

    for (s, a), tot_count in P_tot.items():
        if tot_count == 0:
            continue
        for r in results:
            if s in r["T"] and a in r["T"][s]:
                for s_next, count in r["T"][s][a].items():
                    T_dict[s][a][s_next] += count

    for r in results:
        for s in r["R"]:
            for a in r["R"][s]:
                R_dict[s][a].extend(r["R"][s][a])

    # print("aggregating in", time.time() - tok)

    return T_dict, R_dict, P_tot, state_distr


def learn_abstraction(
    dataset: list[list[tuple[int, int, float, int]]],
    n_states: int,
    n_actions: int,
    abstraction_mapper: AbstractionMapper = AbstractionMapper(),
    multithreading: bool = True,
) -> tuple[TransitionFunction, RewardFunction, NDArray]:
    # print(f"{len(dataset)=}")
    if multithreading:
        return learn_abstraction_multithreaded(
            dataset, n_states, n_actions, abstraction_mapper
        )
    else:
        return learn_abstraction_single_threaded(
            dataset, n_states, n_actions, abstraction_mapper
        )

"""Defunct: incorporated into learned_env instead!"""
def normalize_aggregated_counts(
    T_dict, R_dict, P_tot, state_distr:dict, n_states, n_actions
):
    # state_distr = state_distr.astype(float)
    distr_sum = np.sum(list(state_distr.values()))
    if distr_sum > 0:
        for (s, p) in state_distr.items():
            state_distr[s] = p / distr_sum

    for (s, a), tot_count in P_tot.items():
        if tot_count == 0:
            continue
        for s_next in T_dict[s][a].keys():
            T_dict[s][a][s_next] /= tot_count
        sum_tot = sum([prob for s_next, prob in T_dict[s][a].items()])
        assert round(sum_tot, 1) in {0, 1}, (
            f"Counts for {s, a} sum to {sum_tot} != {0, 1}!"
        )

    for s in R_dict:
        for a in R_dict[s]:
            if not R_dict[s][a]: # is empty
                R_dict[s][a] = 0.0
            else:
                R_dict[s][a] = np.mean(R_dict[s][a])

    return (
        TransitionFunction(n_states, n_actions, T_dict),
        RewardFunction(n_states, n_actions, R_dict),
        state_distr,
    )


def learn_abstraction_single_threaded(
    dataset: list[list[tuple[int, int, float, int]]],
    n_states: int,
    n_actions: int,
    abstraction_mapper: AbstractionMapper,
) -> tuple[TransitionFunction, RewardFunction, NDArray]:

    results = collect_data_from_trajectories(dataset, n_states, abstraction_mapper)
    P_tot = results["tot"]
    T_dict = results["T"]
    R_dict = results["R"]
    state_distr = results["init"]

    return T_dict, R_dict, P_tot, state_distr
