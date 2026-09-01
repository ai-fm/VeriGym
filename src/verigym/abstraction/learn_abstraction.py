import copy
import logging
import multiprocessing
import time
from typing import Any, Callable
from collections import defaultdict


import gymnasium as gym
import numpy as np

from numpy.typing import NDArray

from ..environments.reward_func import RewardFunction
from ..environments.transition_func import TransitionFunction
from ..environments.explicitenv import ExplicitEnv
from ..environments.verigymenv import VeriGymEnv
from ..policy.policy import PolicyClass
from .abstractionmapper import AbstractionMapper

logger = logging.getLogger(__name__)


class CachedDiscretizer:
    """Wraps a discretization function with memoization to avoid redundant computation.

    During abstraction learning, states and actions are discretized repeatedly as
    trajectories are processed. This class caches discretization results to improve
    performance by avoiding recomputation of the same mappings.

    The cache key is a tuple of the array values, enabling O(1) lookup of previously
    computed discrete indices for both scalar and multi-dimensional inputs.

    Note: For *very* large state-action spaces this could become memory intensive.
    But transition and reward function will be the first points of concern when computing
    the abstraction.
    """

    def __init__(self, discretizer: Callable):
        self.cache = {}
        self.discretizer = discretizer

    def discretize(self, input: NDArray | float | int) -> int:
        """
        Discretizes the `input` which should be a state or an action.

        Scalar inputs are promoted to a 1-D array so that the `key`
        discretizer receives the same datastructure (a tuple).
        """
        arr = np.atleast_1d(input)
        key = tuple(arr)
        if key not in self.cache:
            self.cache[key] = self.discretizer(arr)
        return self.cache[key]


def forward_mapping(x: NDArray, to_bins: Callable, to_int: Callable):
    # Promote scalars / 0-D inputs (e.g. actions from a `Discrete` space) to a
    # 1-D factored representation before `to_bins`: `sample_to_discrete` indexes
    # `sample.shape[0]`, which a 0-D array does not have.
    # TODO: This function is the result of not having consistent types of our states/actions, would like to clean this up and not require the function in the future. (Joshua)
    x = np.atleast_1d(x)
    return to_int(to_bins(x))


def backward_mapping(x: int, backward_map: Callable, space: gym.Space) -> Any:
    """Applies `backward_map` and casts the result to a valid sample of `space`.

    `backward_map` reconstructs a bin-edge value, a real number, even for a
    `Discrete`/`MultiDiscrete` `space` where the bin edges do not necessarily
    fall exactly on integers (e.g. `bin_edges_per_*_dim` not evenly dividing
    the number of discrete values). Left uncast, this fails `space.contains(...)`
    (e.g. a `Discrete` space rejects a float array such as `array([0.])`).
    """
    value = np.atleast_1d(backward_map(x))
    if isinstance(space, gym.spaces.Discrete):
        return int(np.rint(value.reshape(-1)[0]))
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.rint(value).astype(space.dtype).reshape(space.shape)
    return value.astype(space.dtype).reshape(space.shape)


def create_abstraction(
    original_env: VeriGymEnv,
    abstraction_mapper: AbstractionMapper,
    exploration_policy: PolicyClass,
    num_steps: int,
    n_iterations: int = 1,
    multithreading: bool = True,
    verbose: bool = False,
) -> ExplicitEnv:
    """
    Creates an abstraction from a VeriGymEnv by discretizing the state and
    action spaces. Returns an `ExplicitEnv`.


    Parameters
    ----------
    original_env : VeriGymEnv
        The environment / model to be abstracted.
    abstraction_mapper: AbstractionMapper
        Holds the mapping between the original space (of `original_env`) and the space which we are abstracting into.
    exploration_policy : PolicyClass
        The policy of interacting with the `original_env` (e.g. a random policy).
    num_steps : int
        Number of steps to take within the environment (also, see `n_iterations`).
    n_iterations: int
        Number of (interleaving) iterations. For each iteration the `exploration_policy.update_for_abstraction_refinement(...)` will be called, alowing the policy to update based on the gathered interactions (e.g. for state-based exploration policies). Note, that the total number of steps equal `n_iterations * num_steps`. For policies that are not interleaving, set `n_iterations` to 1 (default).
    multithreading: bool, optional
        Whether to multithread or use single thread.
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
    
    # Get discrete states and actions
    n_states = abstraction_mapper.abstract_n_states
    n_actions = abstraction_mapper.abstract_n_actions
    
    assert (n_states is not None) and (n_actions is not None), f"Neither should be none {(n_states, n_actions) = }"

    # Initialize relevant objects for learning the abstraction
    T_counts, R_dict_counts, P_tot_counts, state_distr_counts = _create_count_databases(n_states=n_states, n_actions=n_actions)
    dataset = []

    # Loop through iterations. If interleaving abstraction is not required, n_iterations will be just 1.
    for i in range(n_iterations):
        exploration_policy = exploration_policy.update_for_abstraction_refinement(
            dataset, T_counts, P_tot_counts, R_dict_counts, state_distr_counts
        )
        assert isinstance(exploration_policy, PolicyClass)

        tik = time.time()
        # generate dataset via simulation
        dataset = original_env.simulate( #TODO get rid of this simulate call, is it requires original_env to be of type VeriGymEnv. This should also work for gym.Env
            policy=exploration_policy, n_steps=num_steps, verbose=verbose
        )

        tok = time.time()
        print(f"Simulation time: {tok - tik:.4f}s")

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

        # --- START Aggregate across iterations
        state_distr_counts += new_state_distr_counts

        for (s, a), tot_count in new_P_tot_counts.items():
            P_tot_counts[(s, a)] += tot_count
            R_dict_counts[s][a].extend(new_R_dict_counts[s][a])
            for next_state, count in new_T_counts[s][a].items():
                T_counts[s][a][next_state] += count
        # --- END Aggregate

        print(f"Learning Abstraction: {time.time() - tok:.4f}s")

    # Obtain valid distributions/values by aggregating the variables storing the counts (normalizing via P_tot_counts)
    T, R, S_init = normalize_aggregated_counts(
        T_counts, R_dict_counts, P_tot_counts, state_distr_counts, n_states, n_actions
    )

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

def _create_count_databases(n_states: int, n_actions: int) -> tuple[dict, dict, dict, NDArray]:
    """
    Creates all required objects for the abstraction learning.
    The returned objects `T_counts`, `R_dict_counts` and `state_distr_counts` have the same datatype 
    as their usual counterparts, only that they are intended to hold the absolute number of samples 
    and are not normalized to probability distributions. For that one has to normalize using the 
    total counts stored in `P_tot_counts`. This `dict` is populated for convenience as its usage 
    reduces the amount of times the counts would have to be computed from `T_counts` or `R_dict_counts`.
    
    Parameters
    ----------
    n_states : int
        The number of states.
    n_action : int
        The number of actions.
        
    Returns
    -------
    tuple[dict, dict, dict, NDArray]
        T_counts, R_dict_counts, P_tot_counts, state_distr_counts. 
    
    Example
    -------
    ```
    import math
    from verigym.abstraction.learn_abstraction import _create_count_databases
    
    bin_edges_states = [[2, 3], [0.5, 1.0, 1.5]]
    bin_edges_actions = [[-0.5, 0.0, 0.5]]
    # number of states
    n_states = math.prod([len(dimension) - 1 for dimension in bin_edges_states])
    # number of actions (product over the discretized action dimensions)
    n_actions = math.prod([len(dimension) - 1 for dimension in bin_edges_actions])
    # number of counts (occurences) for each state-action-next_state pair
    (
        T_counts,
        R_dict_counts,
        P_tot_counts,
        state_distr_counts,
    ) = _create_count_databases(n_states, n_actions)
    ```
    """
    T_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    # list of rewards for all occured state-action pairs
    R_dict_counts = defaultdict(lambda: defaultdict(lambda: list()))
    # dict with occurences for each state-action pair
    P_tot_counts = defaultdict(lambda: 0)
    # list with occurences of each state as initial state
    state_distr_counts = np.zeros(n_states)
    return (
        T_counts,
        R_dict_counts,
        P_tot_counts,
        state_distr_counts,
    )


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

    T_dict = data["T"]
    R_dict = data["R"]
    state_distr = data["init"]
    P_tot = data["tot"]

    for trajectory in trajectories:
        for i, (s, a, r, s_next) in enumerate(trajectory):
            if isinstance(r, np.ndarray):
                r = r.item()
            # Go through the mapper wrappers (not `.forward_map` directly) so
            # that size-1 ndarray outputs are normalized to hashable scalars,
            # which is required for use as dict keys / array indices below.
            s = mapper.original_to_abstract_state(s)
            a = mapper.original_to_abstract_action(a)
            s_next = mapper.original_to_abstract_state(s_next)
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
    state_distr = np.zeros(n_states)

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

    print("processing in ", tok - tik)
    print("aggregating..")

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

    for r in results:
        for s in r["R"]:
            for a in r["R"][s]:
                R_dict[s][a].extend(r["R"][s][a])

    print("aggregating in", time.time() - tok)

    return T_dict, R_dict, P_tot, state_distr


def learn_abstraction(
    dataset: list[list[tuple[int, int, float, int]]],
    n_states: int,
    n_actions: int,
    abstraction_mapper: AbstractionMapper = AbstractionMapper(),
    multithreading: bool = True,
) -> tuple[TransitionFunction, RewardFunction, NDArray]:
    print(f"Trajectories in dataset: {len(dataset)}")
    if multithreading:
        return learn_abstraction_multithreaded(
            dataset, n_states, n_actions, abstraction_mapper
        )
    else:
        return learn_abstraction_single_threaded(
            dataset, n_states, n_actions, abstraction_mapper
        )


def normalize_aggregated_counts(
    T_dict, R_dict, P_tot, state_distr, n_states, n_actions
):
    state_distr = state_distr.astype(float)
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
