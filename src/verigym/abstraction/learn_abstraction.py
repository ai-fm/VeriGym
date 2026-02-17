from concurrent.futures import ThreadPoolExecutor
import functools
import logging
import time
from typing import Literal
from math import prod

import gymnasium as gym
import numpy as np

from numpy.typing import NDArray

from verigym.abstraction.abstractionmapper import AbstractionMap, AbstractionMapper
from verigym.environments import ExplicitEnv, VeriGymEnv, GenerativeEnv
from verigym.abstraction.gym_utils.transform_observation import (
    DiscretizeBoxObservation,
)
from verigym.abstraction.discretization import (
    # centered_pow_bin,
    generate_box_bins,
)
from verigym.abstraction.utils import factored_to_index
from verigym.policy.policy import RandomizedPolicy

logger = logging.getLogger(__name__)

from collections import defaultdict
import logging

from numpy.typing import NDArray
import numpy as np

from verigym.abstraction.abstractionmapper import AbstractionMapper
from verigym.environments.reward_func import RewardFunction
from verigym.environments.transition_func import TransitionFunction

def create_abstraction(
    original_env: VeriGymEnv,
    exploration_strategy: Literal["random", "sb3 policy"],
    num_steps: int,
    bin_edges_per_dim: int | list[int],
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
    discretized_env =  DiscretizeBoxObservation(
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

    # Convert into VeriGym compatible object
    generative_env = GenerativeEnv.from_gymnasium(original_env)

    tik = time.time()
    dataset = generative_env.simulate(policy=RandomizedPolicy(generative_env), n_steps=num_steps)
    
    tok = time.time()
    print('simulate', tok - tik)
    
    # Create state abstraction mapping
    def mapping(x: NDArray) -> int:
        return factored_to_index(bin_edges, discretized_env.func(x))

    state_abstraction_map = AbstractionMap(forward_map=mapping)

    # TODO: make mapper for discretized actions; action abstraction is identity by default
    abstraction_mapper = AbstractionMapper(
        state_abstraction_map=state_abstraction_map
    )
    
    newtok = time.time()
    print('update dataset', newtok - tok)
    
    # print(dataset_indices)

    # approximate the transition function
    n_actions = original_env.action_space.n
    n_states = prod([len(dimension) for dimension in bin_edges])
    T, R, S_init = learn_abstraction(
        dataset=dataset, n_states=n_states, n_actions=n_actions, abstraction_mapper=abstraction_mapper
    )
    
    print('learning:', time.time() - newtok)

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

import threading

# Thread-local storage for each thread's aggregates
thread_local = threading.local()

def collect_data_from_trajectory(trajectories : list[list[tuple[int, int, float, int]]], num_states : int, abstraction_mapper : AbstractionMapper):
    # Initialize local storage for this thread
    if not hasattr(thread_local, 'data'):
        thread_local.data = {'T': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))), 'R': defaultdict(lambda: defaultdict(lambda: list())), 'init': np.zeros(num_states), 'tot' : defaultdict(lambda: 0)}
    
    T_dict = thread_local.data['T']
    R_dict = thread_local.data['R']
    state_distr = thread_local.data['init']
    P_tot = thread_local.data['tot']

    for trajectory in trajectories:
        for i, (s, a, r, s_next) in enumerate(trajectory):
            if isinstance(a, np.ndarray): a = a.item()
            if isinstance(r, np.ndarray): r = r.item()
            s = abstraction_mapper.state_abstraction_map.forward_map(s)
            a = abstraction_mapper.action_abstraction_map.forward_map(a)
            s_next = abstraction_mapper.state_abstraction_map.forward_map(s_next)
            if i == 0: state_distr[s] += 1
            if s not in T_dict:
                T_dict[s] = {}  # do we need this? defaultdict should take care of this
            if a not in T_dict[s]:
                T_dict[s][a] = defaultdict(lambda: 0)
            T_dict[s][a][s_next] += 1
            P_tot[(s, a)] += 1
            R_dict[s][a].append(r)
    
    return thread_local.data

def nested_sum(target, source):
    """
    Recursively sum values of overlapping keys in target and source dictionaries.
    If a key exists in both, their values are added (if both are numbers),
    or merged recursively (if both are dictionaries).
    """
    for key, value in source.items():
        if key in target:
            if isinstance(target[key], (defaultdict, dict)) and isinstance(value, (defaultdict, dict)):
                nested_sum(target[key], value)
            elif isinstance(target[key], (int, float)) and isinstance(value, (int, float)):
                target[key] += value
            elif isinstance(target[key], list) and isinstance(value, list):
                target[key].extend(value)
            else:
                print(key, value, target[key])
                raise ValueError("?")
        else:
            target[key] = value
    return target

def learn_abstraction(
    dataset: list[list[tuple[int, int, float, int]]], n_states: int, n_actions: int, abstraction_mapper : AbstractionMapper = AbstractionMapper(), multithreading : bool = True
) -> tuple[TransitionFunction, RewardFunction, NDArray]:
    if multithreading:
        num_threads = 14
        chunk_size = len(dataset) // num_threads
        chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

        process = functools.partial(collect_data_from_trajectory, num_states=n_states, abstraction_mapper=abstraction_mapper)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(process, chunks))
        
        T_dict = functools.reduce(nested_sum, (r['T'] for r in results), defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))))
        R_dict = functools.reduce(nested_sum, (r['R'] for r in results), defaultdict(lambda: defaultdict(lambda: list())))
        P_tot = functools.reduce(nested_sum, (r['tot'] for r in results), defaultdict(lambda: 0))
        state_distr = functools.reduce(lambda x, y : x + y, [r['init'] for r in results], np.zeros(n_states, dtype=int))
    else:
        T_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        P_tot = defaultdict(lambda: 0)

        state_distr = np.zeros(n_states)
        
        R_dict = defaultdict(lambda: defaultdict(lambda: list()))

        # Populate count table
        for trajectory in dataset:
            for i, (s, a, r, s_next) in enumerate(trajectory):
                if isinstance(a, np.ndarray): a = a.item()
                s = abstraction_mapper.state_abstraction_map.forward_map(s)
                a = abstraction_mapper.action_abstraction_map.forward_map(a)
                s_next = abstraction_mapper.state_abstraction_map.forward_map(s_next)
                if i == 0: state_distr[s] += 1
                if s not in T_dict:
                    T_dict[s] = {}  # do we need this? defaultdict should take care of this
                if a not in T_dict[s]:
                    T_dict[s][a] = defaultdict(lambda: 0)
                T_dict[s][a][s_next] += 1
                P_tot[(s, a)] += 1
                R_dict[s][a].append(r)
        
    state_distr /= state_distr.sum()

    for (s, a), tot_count in P_tot.items():
        if P_tot == 0:
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

    return TransitionFunction(n_states, n_actions, T_dict), RewardFunction(n_states, n_actions, R_dict), state_distr
