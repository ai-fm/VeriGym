"""
Abstraction Refinement module
"""
from typing import Callable

import numpy as np

import verigym
from verigym.abstraction.learn_abstraction import create_abstraction


def abstraction_refinement(
    original_env: verigym.GenerativeEnv,
    mapper_schedule: list,
    num_steps: int,
    earlystopping_callback: Callable,
    make_exploration_policy: Callable[[verigym.AbstractionMapper | None], verigym.PolicyClass],
    n_iterations: int = 1,
    reuse_dataset: bool = True,
    seed: int | None = None,
    multithreading: bool = True,
) -> list[tuple[tuple[tuple[float, float]], verigym.AbstractedExplicitEnv, verigym.StormpyPolicy]]:
    """
    Learn an abstraction for each mapper in `mapper_schedule` until `earlystopping_callback` says stop.

    Parameters
    ----------
    original_env : verigym.GenerativeEnv
        Environment to abstract.
    mapper_schedule : list
        Abstraction mappers to learn the abstraction for, in order.
    num_steps : int
        Exploration step budget per mapper, split evenly over `n_iterations`.
    earlystopping_callback : Callable. 
        `(original_env, abstracted_model) -> (stop: bool, metric: Any)`.
        User may copmute metrics as well as return a bool flag `stop` indicating if abstraction refinement should stop. May also return arbitrary information such as metrics that are returned by this function as a list.
    make_exploration_policy : Callable[[verigym.AbstractionMapper  |  None], verigym.PolicyClass]
        Constructor for exploration policy.
        `mapper -> policy`. Called with `None` if `reuse_dataset` is set.
    n_iterations : int, optional
        Number of interleaved exploration/learning rounds per mapper, by default 1
    reuse_dataset : bool, optional
        Collect one dataset up front and share it across all mappers, by default True.
        Requires `n_iterations == 1` and a mapper-independent exploration policy.
        Not every exploration strategy is compatible with this setting.
    seed : int | None, optional
        Seeds the exploration: the collection of the reused dataset (env and policy, as `create_abstraction` does) and
        `create_abstraction` for every mapper, so all mappers are learned with the same random numbers.
        If None (default), the env and policies continue with their current random generators.
    multithreading : bool, optional
        Whether `create_abstraction` learns the abstraction with multiple processes, by default True

    Returns
    -------
    list[tuple[tuple[tuple[float, float]], verigym.AbstractedExplicitEnv, verigym.StormpyPolicy]]
        A list of `(metric, abstracted_model)`, one per mapper tried.
    """
    assert (len(mapper_schedule) > 0), "mapper_schedule must have at least one entry."
    assert not reuse_dataset or n_iterations == 1, "A reused dataset requires n_iterations == 1"
    
    results = []

    # Trajectories are stored in the original space, so one dataset can be shared by all mappers
    dataset = None
    if reuse_dataset:
        exploration_policy = make_exploration_policy(None)
        env_seed = None
        if seed is not None:
            exploration_policy.seed(seed)
            env_seed = np.random.SeedSequence(seed).generate_state(1).tolist()[0]  # as create_abstraction does
        dataset = original_env.simulate(policy=exploration_policy, n_steps=num_steps, verbose=False, seed=env_seed)
        num_steps = 0

    for i, mapper in enumerate(mapper_schedule):
        print(f"====== Iteration {i}/{len(mapper_schedule)} ======")
        exploration_policy = make_exploration_policy(mapper)
        abstracted_model = create_abstraction(
                original_env=original_env, 
                abstraction_mapper=mapper, 
                exploration_policy=exploration_policy,
                num_steps=num_steps//n_iterations,
                n_iterations=n_iterations,
                multithreading=multithreading,
                dataset=dataset,
                seed=seed,
            )
        
        task_complete, metric_result = earlystopping_callback(original_env, abstracted_model)

        results.append(((metric_result), abstracted_model))

        if task_complete:
            print(f"Finishing, we hit the threshold at iteration {i}.")
            break
        else:
            print("Did not succeed in satisfying the threshold performance with the given parameters.")  

    return results