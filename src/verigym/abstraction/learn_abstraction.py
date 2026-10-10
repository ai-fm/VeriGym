import copy
import logging
import multiprocessing
import time
from collections import defaultdict
import math

import scipy.stats

import gymnasium as gym
import numpy as np

from numpy.typing import NDArray

from verigym.environments.interval_explicitenv import IntervalExplicitEnv

from ..environments.reward_func import RewardFunction, IntervalRewardFunction
from ..environments.transition_func import TransitionFunction, IntervalTransitionFunction
from ..environments.explicitenv import ExplicitEnv
from ..environments.verigymenv import VeriGymEnv
from ..environments.labeling import AbstractStateLabeler
from ..policy.policy import PolicyClass
from .abstractionmapper import AbstractionMapper, validate_for_abstraction

logger = logging.getLogger(__name__)


def get_interval_transition_reward(
    T_counts, R_counts, P_tot_counts, n_states, n_actions, confidence=0.9, iid=False
):
    """
    Construct (confidence) intervals from the observed data.

    If iid, computes Clopper-Pearson Binomial intervals.
    Else, computes Azuma-Hoeffding Martingale intervals.

    Expects the raw counts (see `learn_abstraction()`), not the normalized probabilities.
    Intervals are only constructed for observed successors; unobserved successors keep the interval (0.0, 0.0),
    i.e., the same support as the point estimates of `normalize_aggregated_counts()`.

    Parameters
    ----------
    T_counts : dict
        Mapping s -> a -> s' -> number of observed transitions (s, a, s')
    R_counts : dict
        Mapping s -> a -> list of observed rewards
    P_tot_counts : dict
        Mapping of (s,a) to total counts (e.g., the sum of successor counts in T_counts for each s,a).
        State-action pairs with a total count of 0 are skipped.
    n_states : int
        |S|
    n_actions : int
        |A|
    confidence : float, optional
        The statistical (high) confidence of the iMDP construction, by default 0.9
    iid : boolean, optional
        Whether the data is independently and identically distributed (i.i.d.), by default False
    """
    # TODO: implement reward estimation from IID data.
    delta = 1 - confidence  # Confidence over total model
    # M = len(P_tot_counts) * n_states # TODO: Use actual visited counts for the confidence guarantee instead of whole state space?
    M = n_states**2 * n_actions
    alpha = delta / M  # confidence for each transition
    interval_T = make_interval_transition_dict()
    interval_R = make_reward_dict()
    for (s, a), n in P_tot_counts.items():
        if n == 0:  # unvisited, skipped just like in `normalize_aggregated_counts()`
            continue
        for ss, k in T_counts[s][a].items():
            if not isinstance(k, (int, np.integer)):
                raise TypeError(
                    f"Expected raw (integer) counts, but T_counts[{s}][{a}][{ss}] = {k!r}. "
                    "Intervals must be computed from counts, not from normalized probabilities."
                )
            if iid:  # Use Clopper-Pearson Binomial intervals
                lb = 0.0 if k == 0 else scipy.stats.beta.ppf(alpha / 2, k, n - k + 1)
                ub = 1.0 if k == n else scipy.stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
            else:  # Use Azuma-Hoeffding Martingale intervals
                eps = math.sqrt(math.log(2 / alpha) / (2 * n))
                lb = 0.0 if k == 0 else k / n - eps
                ub = 1.0 if k == n else k / n + eps
            lb = max(0.0, lb)
            ub = min(1.0, ub)
            interval_T[s][a][ss] = (lb, ub)
        mean_reward = np.mean(R_counts[s][a])
        interval_R[s][a] = (mean_reward, mean_reward)  # FIXME? Use expected / MLE for now

    interval_T_function = IntervalTransitionFunction(n_states, n_actions, interval_T)
    interval_R_function = IntervalRewardFunction(n_states, n_actions, interval_R)
    return interval_T_function, interval_R_function


def create_abstraction(
    original_env: VeriGymEnv,
    abstraction_mapper: AbstractionMapper,
    exploration_policy: PolicyClass,
    num_steps: int,
    n_iterations: int = 1,
    multithreading: bool = True,
    verbose: bool = False,
    intervals: bool = False,
    assume_iid: bool = False,
    add_terminal_state: bool = True,
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
    intervals: bool, optional
        Whether to learn transition functions with (confidence) intervals or point-based estimates.
        Default is False, i.e., point-based estimate learning.
    assume_iid: bool, optional.
        Whether to assume that the data distribution is iid or not. This is only relevant for interval learning.
        Defaults to False.
    add_terminal_state: bool, optional
        Whether to add an absorbing terminal state with index `abstraction_mapper.abstract_n_states`. Every step on
        which `original_env` terminated (not truncated) transitions into this state instead of into the abstract
        state of its observation. The terminal state has a self-loop with reward 0 for every action, so the abstract
        model has `abstract_n_states + 1` states. No original state maps to it. If False, terminated steps are learned
        like any other transition and the abstract model has `abstract_n_states` states. Defaults to True.

    Returns
    -------
    ExplicitEnv
        The abstracted model.


    Notes
    -----
    - If intervals == True, the return type is `IntervalExplicitEnv`, which is a sub-type of `ExplicitEnv`.
    - If add_terminal_state == True, the returned model has one more state than `abstraction_mapper`, so its
      `observation_space` is `Discrete(abstract_n_states + 1)`. The terminal state is labelled "terminal" when exported
      to stormpy, so properties like `Pmax=? [F "terminal"]` can be checked.
    """
    assert isinstance(original_env, gym.Env), (
        f"original_env is type {type(original_env)} and does not inherit from gym.Env"
    )

    validate_for_abstraction(abstraction_mapper, multithreading=multithreading)
    if intervals:
        if assume_iid:
            if (
                abstraction_mapper.original_n_states == math.inf
                or abstraction_mapper.original_n_actions == math.inf
            ):
                print(
                    f"WARNING: Can only learn intervals from IID data on discrete environments, but received: n_states={abstraction_mapper.original_n_states} and n_actions={abstraction_mapper.original_n_actions}."
                )
        else:
            if (
                abstraction_mapper._state_abstraction_map.is_identity_map
                and abstraction_mapper._action_abstraction_map.is_identity_map
            ):
                print(
                    "WARNING: Data said to be non-IID when constructing invervals but creating abstraction through identity mapping."
                )

    # Get discrete states and actions
    n_states = abstraction_mapper.abstract_n_states
    n_actions = abstraction_mapper.abstract_n_actions

    assert (n_states is not None) and (n_actions is not None), (
        f"Neither should be none {(n_states, n_actions) = }"
    )
    # cast from np.int64, as the reward function requires python ints as state indices
    n_states, n_actions = int(n_states), int(n_actions)

    # Terminated steps transition into an extra absorbing state behind the states of the mapper
    terminal_state = n_states if add_terminal_state else None
    if add_terminal_state:
        n_states += 1

    # Initialize relevant objects for learning the abstraction
    T_counts, R_dict_counts, P_tot_counts, state_distr_counts = _create_count_databases(
        n_states=n_states
    )
    dataset = []

    # Loop through iterations. If interleaving abstraction is not required, n_iterations will be just 1.
    for i in range(n_iterations):
        exploration_policy = exploration_policy.update_for_abstraction_refinement(
            dataset, T_counts, P_tot_counts, R_dict_counts, state_distr_counts
        )
        assert isinstance(exploration_policy, PolicyClass)

        tik = time.time()
        # generate dataset via simulation
        dataset = original_env.simulate(  # TODO get rid of this simulate call, is it requires original_env to be of type VeriGymEnv. This should also work for gym.Env
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
                terminal_state=terminal_state,
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

    if terminal_state is not None:
        add_absorbing_state(T_counts, R_dict_counts, P_tot_counts, terminal_state, n_actions)

    # Obtain valid distributions/values by aggregating the variables storing the counts (normalizing via P_tot_counts)
    T, R, S_init = normalize_aggregated_counts(
        T_counts, R_dict_counts, P_tot_counts, state_distr_counts, n_states, n_actions
    )

    if intervals:
        interval_T, interval_R = get_interval_transition_reward(
            T_counts, R_dict_counts, P_tot_counts, n_states, n_actions, iid=assume_iid
        )
        if terminal_state is not None:
            # the terminal state is absorbing by construction, not estimated from data
            for a in range(n_actions):
                interval_T.T_dict[terminal_state][a][terminal_state] = (1.0, 1.0)
        abstracted_env = IntervalExplicitEnv(
            nr_states=n_states,
            nr_actions=n_actions,
            initial_state_distr=S_init,  # TODO
            transition_function=T,
            reward_function=R,
            interval_transition_function=interval_T,
            interval_reward_function=interval_R,
            nr_rewards=1,  # TODO rename + compatability for multi objective gym envs
            abstraction_map=abstraction_mapper,
            original_env=original_env,
            render_mode=None,
        )
    else:
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

    abstracted_env.state_labeler = AbstractStateLabeler(
        original_labeler=original_env.state_labeler,
        abstraction_mapper=abstraction_mapper,
    )

    return abstracted_env


def _create_count_databases(n_states: int) -> tuple[dict, dict, dict, NDArray]:
    """
    Creates all required objects for the abstraction learning.
    The returned objects `T_counts`, `R_dict_counts` and `state_distr_counts` have the same datatype
    as their usual counterparts, only that they are intended to hold the absolute number of samples
    and are not normalized to probability distributions. For that one has to normalize using the
    total counts stored in `P_tot_counts`. This `dict` is populated for convenience as its usage
    reduces the amount of times the counts would have to be computed from `T_counts` or `R_dict_counts`.

    Note: We only need `n_states` (and not `n_actions`) due to the initialization of the `state_distr_counts` and need to
    know the size of the array.

    Parameters
    ----------
    n_states : int
        The number of states.

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
    # number of counts (occurences) for each state-action-next_state pair
    (
        T_counts,
        R_dict_counts,
        P_tot_counts,
        state_distr_counts,
    ) = _create_count_databases(n_states)
    ```
    """
    T_counts = make_transition_dict()
    # list of rewards for all occured state-action pairs
    R_dict_counts = make_reward_dict()
    # dict with occurences for each state-action pair
    P_tot_counts = make_int_dict()
    # list with occurences of each state as initial state
    state_distr_counts = np.zeros(n_states, dtype=int)
    return (
        T_counts,
        R_dict_counts,
        P_tot_counts,
        state_distr_counts,
    )


# Define named functions for defaultdict factories (they cannot be lambda functions due to multiprocess pickling issues)


def make_int_dict():  # pragma: no cover
    return defaultdict(int)  # calling int() without argument returns 0


def make_list_dict():  # pragma: no cover
    return defaultdict(list)


def make_middle_dict():  # pragma: no cover
    return defaultdict(make_int_dict)


def make_transition_dict():
    """Dict that can be used for transition function"""
    return defaultdict(make_middle_dict)

def make_zero_interval(): # pragma: no cover
    return (0.0, 0.0)

def make_interval_dict(): # pragma: no cover
    return defaultdict(make_zero_interval)

def make_interval_middle_dict(): # pragma: no cover
    return defaultdict(make_interval_dict)

def make_interval_transition_dict():
    """Dict that can be used for interval transition function"""
    return defaultdict(make_interval_middle_dict)

def make_reward_dict():
    """Dict that can be used for reward function"""
    return defaultdict(make_list_dict)


def add_absorbing_state(
    T_counts: dict, R_dict_counts: dict, P_tot_counts: dict, state: int, n_actions: int
) -> None:
    """
    Makes `state` absorbing in the count databases (see `_create_count_databases()`), in place:
    every action gets a self-loop with a single count and reward 0.
    Used for the terminal state of an abstraction (see `create_abstraction()`).

    Parameters
    ----------
    T_counts : dict
        Mapping s -> a -> s' -> number of observed transitions (s, a, s').
    R_dict_counts : dict
        Mapping s -> a -> list of observed rewards.
    P_tot_counts : dict
        Mapping of (s, a) to total counts.
    state : int
        The state to make absorbing.
    n_actions : int
        Number of actions.
    """
    for a in range(n_actions):
        T_counts[state][a][state] += 1
        P_tot_counts[(state, a)] += 1
        R_dict_counts[state][a].append(0.0)


def collect_data_from_trajectories(
    trajectories: list[list[tuple[int, int, float, int, bool]]],
    n_states: int,
    mapper: AbstractionMapper | None = None,
    terminal_state: int | None = None,
) -> tuple[dict, dict, dict, NDArray]:
    """
    Taking a dataset of `trajectories` populates dicts counting the total
    occurences in the `trajectories` and add them to the objects that can then
    be used for computing the transition and reward function as well as the
    initial state distribution.

    Parameters
    ----------
    trajectories : list[list[tuple[int, int, float, int, bool]]]
        Dataset of trajectories (state, action, reward, next_state, terminated), as returned by
        `VeriGymEnv.simulate()`. Steps without the `terminated` flag, i.e. (state, action, reward, next_state),
        are treated as not terminated.
    n_states : int
        Number of states of the corresponding state space (including `terminal_state`, if given).
    mapper : AbstractionMapper | None
        Maps from the state and action spaces of an original environment to an abstract environment.
        If None, an identity map (no mapping) will be perormed. Defaults to `None`.
    terminal_state : int | None
        State that terminated steps transition into, instead of the (mapped) `next_state`.
        If None, terminated steps are counted like any other step. Defaults to `None`.

    Returns
    -------
    tuple[dict, dict, dict, NDArray]
        T_counts, R_dict_counts, P_tot_counts, state_distr_counts. The count
        databases created by `_create_count_databases`, populated from `trajectories`.
    """
    # If no mapper is passed, we keep states and actions as they are
    if mapper is None:

        def original_to_abstract_state(x):
            return x

        def original_to_abstract_action(x):
            return x
    else:
        # using `*_enum` functions.
        original_to_abstract_state = mapper._state_abstraction_map.original_to_enum
        original_to_abstract_action = mapper._action_abstraction_map.original_to_enum
        assert (original_to_abstract_state is not None) and (
            original_to_abstract_action is not None
        ), (
            f"One of the abstraction maps is None: {original_to_abstract_state = }, {original_to_abstract_action = }"
        )

    # Initialize local storage for this thread
    (
        T_counts,
        R_dict_counts,
        P_tot_counts,
        state_distr_counts,
    ) = _create_count_databases(n_states)

    for trajectory in trajectories:
        for i, (s, a, r, s_next, *terminated) in enumerate(trajectory):
            if isinstance(r, np.ndarray):
                r = r.item()
            # Go through the mapper wrappers (not `.forward_map` directly) so
            # that size-1 ndarray outputs are normalized to hashable scalars,
            # which is required for use as dict keys / array indices below.
            s = original_to_abstract_state(s)
            a = original_to_abstract_action(a)
            if terminal_state is not None and terminated and terminated[0]:
                s_next = terminal_state
            else:
                s_next = original_to_abstract_state(s_next)
            if i == 0:
                state_distr_counts[s] += 1
            T_counts[s][a][s_next] += 1
            P_tot_counts[(s, a)] += 1
            R_dict_counts[s][a].append(r)

    return T_counts, R_dict_counts, P_tot_counts, state_distr_counts


def learn_abstraction_multithreaded(
    dataset: list[list[tuple[int, int, float, int, bool]]],
    n_states: int,
    n_actions: int,
    abstraction_mapper: AbstractionMapper,
    terminal_state: int | None = None,
):
    (
        T_dict,
        R_dict,
        P_tot,
        state_distr,
    ) = _create_count_databases(n_states)

    num_threads = max(min(4, multiprocessing.cpu_count() - 1), 1)
    chunk_size = len(dataset) // num_threads

    if chunk_size == 0:  # For handling super small datasets (like in the tests)
        print("Chunk size is zero!")
        num_threads = 1
        chunks = [(dataset, n_states, abstraction_mapper, terminal_state)]
    else:
        chunks = [
            (
                dataset[i * chunk_size : i * chunk_size + chunk_size],
                n_states,
                copy.deepcopy(abstraction_mapper),
                terminal_state,
            )
            for i in range(num_threads - 1)
        ]
        chunks.append(
            (
                dataset[(num_threads - 1) * chunk_size :],
                n_states,
                copy.deepcopy(abstraction_mapper),
                terminal_state,
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

    for _T_results, _R_results, P_tot_results, state_distr_results in results:
        state_distr += state_distr_results
        for (s, a), tot_count in P_tot_results.items():
            P_tot[(s, a)] += tot_count

    # results are unpacked once rather than once per state-action pair.
    all_T_results = [
        T_results
        for T_results, _R_results, _P_tot_results, _state_distr_results in results
    ]
    for (s, a), tot_count in P_tot.items():
        if tot_count == 0:
            continue
        for T_results in all_T_results:
            if s in T_results and a in T_results[s]:
                for s_next, count in T_results[s][a].items():
                    T_dict[s][a][s_next] += count

    for _T_results, R_results, _P_tot_results, _state_distr_results in results:
        for s in R_results:
            for a in R_results[s]:
                R_dict[s][a].extend(R_results[s][a])

    print("aggregating in", time.time() - tok)

    return T_dict, R_dict, P_tot, state_distr


def learn_abstraction(
    dataset: list[list[tuple[int, int, float, int, bool]]],
    n_states: int,
    n_actions: int,
    abstraction_mapper: AbstractionMapper = None,
    multithreading: bool = True,
    terminal_state: int | None = None,
) -> tuple[dict, dict, dict, NDArray]:
    """
    Abstraction learning for a given dataset. Single- or multithreaded.
    Computes the total counts (!) for transition and reward function and initial state distribution.
    Do not forget to normalize (see `normalize_aggregated_counts()`) for obtaining probability distributions.

    Parameters
    ----------
    dataset : list[list[tuple[int, int, float, int, bool]]]
        The dataset of which we are learning the abstraction, with steps (state, action, reward, next_state, terminated).
    n_states : int
        Number of states in the state space. (corresponds to the abstract state space if `abstraction_mapper` is not `None`)
        Includes `terminal_state`, if given.
    n_actions : int
        Number of actions in the action space. (corresponds to the abstract action space if `abstraction_mapper` is not `None`)
    abstraction_mapper : AbstractionMapper, optional
        Mapping from an original space (samples in dataset) to the abstract space. If no mapping is required, set to `None`, by default None
    multithreading : bool, optional
        Flag for using single- or multithreading, by default True
    terminal_state : int | None, optional
        State that terminated steps transition into (see `collect_data_from_trajectories()`).
        Note that it is not made absorbing here, see `add_absorbing_state()`. By default None

    Returns
    -------
    tuple[dict, dict, dict, NDArray]
        T_counts, R_dict_counts, P_tot_counts, state_distr_counts
    """
    print(f"Trajectories in dataset: {len(dataset)}")
    if multithreading:
        return learn_abstraction_multithreaded(
            dataset, n_states, n_actions, abstraction_mapper, terminal_state
        )
    else:
        return learn_abstraction_single_threaded(
            dataset, n_states, n_actions, abstraction_mapper, terminal_state
        )


def normalize_aggregated_counts(
    T_dict, R_dict, P_tot, state_distr, n_states, n_actions
):
    """
    Normalizes the aggregated counts (see `learn_abstraction()`) into point estimates.
    The inputs are not modified in place.

    Parameters
    ----------
    T_dict : dict
        Mapping s -> a -> s' -> number of observed transitions (s, a, s').
    R_dict : dict
        Mapping s -> a -> list of observed rewards.
    P_tot : dict
        Mapping of (s, a) to total counts. State-action pairs with a total count of 0 are skipped.
    state_distr : NDArray
        Number of occurences of each state as initial state.
    n_states : int
        Number of states.
    n_actions : int
        Number of actions.

    Returns
    -------
    tuple[TransitionFunction, RewardFunction, NDArray]
        T, R, S_init
    """
    state_distr = state_distr.astype(float)
    state_distr /= state_distr.sum()

    T_normalized = make_transition_dict()
    for (s, a), tot_count in P_tot.items():
        if tot_count == 0:
            continue
        for s_next, count in T_dict[s][a].items():
            T_normalized[s][a][s_next] = count / tot_count
        sum_tot = sum(T_normalized[s][a].values())
        assert round(sum_tot, 1) in {0, 1}, (
            f"Counts for {s, a} sum to {sum_tot} != {0, 1}!"
        )

    R_normalized = make_reward_dict()
    for s in R_dict:
        for a in R_dict[s]:
            if np.size(R_dict[s][a]) > 0:  # no observed rewards, no estimate
                R_normalized[s][a] = np.mean(R_dict[s][a])

    return (
        TransitionFunction(n_states, n_actions, T_normalized),
        RewardFunction(n_states, n_actions, R_normalized),
        state_distr,
    )


def learn_abstraction_single_threaded(
    dataset: list[list[tuple[int, int, float, int, bool]]],
    n_states: int,
    n_actions: int,
    abstraction_mapper: AbstractionMapper,
    terminal_state: int | None = None,
) -> tuple[dict, dict, dict, NDArray]:

    T_counts, R_dict_counts, P_tot_counts, state_distr_counts = (
        collect_data_from_trajectories(dataset, n_states, abstraction_mapper, terminal_state)
    )

    return T_counts, R_dict_counts, P_tot_counts, state_distr_counts
