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
from ..environments.generativeenv import GenerativeEnv
from ..policy.policy import PolicyClass, RandomizedPolicy
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


def learn_model_by_discretization(
        original_env: VeriGymEnv,
        exploration_strategy: PolicyClass,
        action_discretization_stragegy: str, # as for states
        state_discretization_strategy: str, # equal bins, sampling bins, KD trees, ...
        num_steps: int,
        action_kwargs: dict,
        state_kwargs: dict,
        multithreading: bool = True,
        verbose: bool = False,
) -> ExplicitEnv:
    assert isinstance(original_env, VeriGymEnv), \
        f"original_env is type {type(original_env)} and does not inherit from VeriGymEnv"

    # Generate action Box bins
    # Generate state Box bins

    # create discretized env
    
    # explore with exploration_strategy
    original_env.simulate(
        policy=exploration_strategy, n_steps=num_steps, verbose=verbose
    )

    # create abstraction mapper

    # learn transition function, reward function, initial state distribution

    # create explicit env

    # return abstract explicit env


def learn_model_by_neural_network( # TODO later
        original_env: VeriGymEnv,
) -> VeriGymEnv:
    # does not necessarily have to discretize. approximate transition function + reward function with NN
    ...

def create_abstraction(
    original_env: VeriGymEnv,
    exploration_policy: PolicyClass,
    num_steps: int,
    bin_edges_per_dim: int | list[int],
    use_box_space: bool = True,
    multithreading: bool = True,
    verbose: bool = False,
) -> ExplicitEnv:
    """
    Creates an abstraction from a VeriGymEnv by discretizing the state space. Returns an `ExplicitEnv`.
    Action space is currently not abstracted.


    Parameters
    ----------
    original_env : VeriGymEnv
        The environment / model to be abstracted.
    exploration_strategy : Literal[&quot;random&quot;, &quot;sb3 policy&quot;]
        The strategy of interacting with the `original_env` (e.g. a random policy).
    num_steps : int
        Number of steps to take within the environment.
    bin_edges_per_dim : int | list[int]
        Number of discretization bins per feature dimension of the state space.
    verbose : bool, optional
        Whether to be verbose, by default False.

    Returns
    -------
    ExplicitEnv
        The abstracted model.
    """
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
        original_env, bin_edges=bin_edges, use_box_space=use_box_space
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

    # Convert into VeriGym compatible object
    if isinstance(original_env, gym.Env):
        # TODO: Get rid of the conversion to GenerativeEnv. This not be part of create_abstraction(). Also, policies might require the env during instantiation, and we want to allow any type of PolicyClass, which is not possible if we instantiate here.
        generative_env = GenerativeEnv.from_gymnasium(original_env)
        
    if exploration_policy == "random":
        exploration_policy = RandomizedPolicy(generative_env)
        
    assert isinstance(exploration_policy, PolicyClass)

    tik = time.time()
    dataset = generative_env.simulate(
        policy=exploration_policy, n_steps=num_steps, verbose=verbose
    )

    tok = time.time()
    print("simulate", tok - tik)

    # Create state abstraction mapping
    # def mapping(x: NDArray) -> int:
    # return factored_to_index(bin_edges, discretized_env.func(x))

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

    newtok = time.time()
    print("update dataset", newtok - tok)

    # print(dataset_indices)

    # approximate the transition function
    n_actions = original_env.action_space.n
    n_states = prod([len(dimension) for dimension in bin_edges])
    T, R, S_init = learn_abstraction(
        dataset=dataset,
        n_states=n_states,
        n_actions=n_actions,
        abstraction_mapper=abstraction_mapper,
        multithreading=multithreading
    )

    print("learning:", time.time() - newtok)

    # Construct the abstracted ExplicitEnv
    abstracted_env = ExplicitEnv(
        nr_states=n_states,
        nr_actions=n_actions,
        nr_rewards=1,  # TODO rename + compatability for multi objective gym envs
        initial_state_distr=S_init,  # TODO
        transition_function=T,
        reward_function=R,
        abstraction_map=abstraction_mapper,
        original_env=original_env,
        render_mode=None,
    )

    return abstracted_env


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
        "init": np.zeros(num_states, dtype=int),
        "tot": defaultdict(int),
    }

    # mapper = copy.deepcopy(mapper)

    print(len(trajectories))

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


def learn_abstraction(
    dataset: list[list[tuple[int, int, float, int]]],
    n_states: int,
    n_actions: int,
    abstraction_mapper: AbstractionMapper = AbstractionMapper(),
    multithreading: bool = True,
) -> tuple[TransitionFunction, RewardFunction, NDArray]:
    print(f"{len(dataset)=}")
    if multithreading:
        T_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        R_dict = defaultdict(lambda: defaultdict(lambda: list()))
        P_tot = defaultdict(lambda: 0)

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

        print("processing in", tok - tik)
        print("aggregating..")

        state_distr = np.zeros(n_states, dtype=int)

        for r in results:
            state_distr += r["init"]
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
            for s_next in T_dict[s][a].keys():
                T_dict[s][a][s_next] /= tot_count
            sum_tot = sum([prob for s_next, prob in T_dict[s][a].items()])
            assert round(sum_tot, 1) in {0, 1}, (
                f"Counts for {s, a} sum to {sum_tot} != {0, 1}!"
            )

        state_distr = state_distr.astype(float)
        state_distr /= state_distr.sum()

        for r in results:
            for s in r["R"]:
                for a in r["R"][s]:
                    R_dict[s][a].extend(r["R"][s][a])

        for s in R_dict:
            for a in R_dict[s]:
                R_dict[s][a] = np.mean(R_dict[s][a])

        print("aggregating in", time.time() - tok)

    else:
        T_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        P_tot = defaultdict(lambda: 0)

        state_distr = np.zeros(n_states)

        R_dict = defaultdict(lambda: defaultdict(lambda: list()))

        # Populate count table
        for trajectory in dataset:
            for i, (s, a, r, s_next) in enumerate(trajectory):
                if isinstance(a, np.ndarray):
                    a = a.item()
                s = abstraction_mapper.state_abstraction_map.forward_map(s)
                a = abstraction_mapper.action_abstraction_map.forward_map(a)
                s_next = abstraction_mapper.state_abstraction_map.forward_map(s_next)
                if i == 0:
                    state_distr[s] += 1
                T_dict[s][a][s_next] += 1
                P_tot[(s, a)] += 1
                R_dict[s][a].append(r)

        state_distr /= state_distr.sum()

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
                R_dict[s][a] = np.mean(R_dict[s][a])

    return (
        TransitionFunction(n_states, n_actions, T_dict),
        RewardFunction(n_states, n_actions, R_dict),
        state_distr,
    )
