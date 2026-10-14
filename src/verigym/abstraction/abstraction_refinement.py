"""
Abstraction Refinement module
"""
from typing import Callable

import stormpy

import verigym
from verigym.abstraction.learn_abstraction import create_abstraction
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp

    
    
    
def abstract(original_env, abstraction_mapper, num_steps, exploration_policy, n_iterations=1, dataset=None):
    """Performs one abstraction based on the parameters given by the schedule."""
    abstracted_model = create_abstraction(
        original_env=original_env, 
        abstraction_mapper=abstraction_mapper, 
        exploration_policy=exploration_policy,
        num_steps=num_steps,
        n_iterations=n_iterations,
        dataset=dataset,
    )


    
    return abstracted_model
    
    
# def optimal_policy_via_model_checking(env: verigym.AbstractedExplicitEnv, discount_factor=0.95):
#     """We compute the optimal policy using the model checker `stormpy`."""
#     stormpy_mdp = build_stormpy_mdp(env)
#     prop = stormpy.parse_properties(f"Rmax=?[Cdiscount={discount_factor}]")[
#         0
#     ]  # checking for cumulative discounted reward

#     # Run the model checking:
#     result = stormpy.check_model_sparse(stormpy_mdp, prop, extract_scheduler=True)
#     scheduler = result.scheduler
    
#     # policy that can be execured on original model
#     policy_on_original = verigym.StormpyPolicy(scheduler, env.abstraction_map, stormpy_mdp)
    
#     # policy that can be execute on the abstracted model
#     identity_mapper = verigym.AbstractionMapper.initialize_identity_mapper(env.observation_space, env.action_space)
#     policy_on_abstracted = verigym.StormpyPolicy(
#         policy=scheduler, 
#         abstraction_mapper=identity_mapper,
#         mdp=stormpy_mdp
#         )
#     return policy_on_original, policy_on_abstracted


def abstraction_refinement(
    original_env: verigym.GenerativeEnv,
    mapper_schedule: list,
    num_steps: int,
    eval_steps: int,
    earlystopping_callback: Callable,
    make_exploration_policy: Callable[[verigym.AbstractionMapper | None], verigym.PolicyClass],
    n_iterations: int = 1,
    reuse_dataset: bool = True,
) -> list[tuple[tuple[tuple[float, float]], verigym.AbstractedExplicitEnv, verigym.StormpyPolicy]]:
    """Iteratively gets discretization parameters from `schedule` then performs an `abstract` step, then executes callback, which decides whether we continue.

    `make_exploration_policy(mapper)` returns the exploration policy for the given mapper. The `num_steps` budget is split
    evenly over `n_iterations` (interleaved exploration/learning). `reuse_dataset` only works for mapper-independent,
    non-interleaving exploration policies (then `make_exploration_policy` is called with `None`).
    """
    assert (len(mapper_schedule) > 0), f"mapper_schedule have at least one entry."
    assert not reuse_dataset or n_iterations == 1, "A reused dataset requires n_iterations == 1"
    
    results = []

    # Trajectories are stored in the original space, so one dataset can be shared by all mappers
    dataset = None
    if reuse_dataset:
        dataset = original_env.simulate(policy=make_exploration_policy(None), n_steps=num_steps, verbose=False)
        num_steps = 0

    for i, mapper in enumerate(mapper_schedule):
        print(f"====== Iteration {i}/{len(mapper_schedule)} ======")
        abstracted_model = abstract(
            original_env, mapper, num_steps // n_iterations, make_exploration_policy(mapper), n_iterations, dataset
        )

        # original_mean, original_std = evaluate_policy(original_env, policy_on_original, eval_steps)
        # abstract_mean, abstract_std = evaluate_policy(abstracted_model, policy_on_abstracted, eval_steps)
        # print(f"Avg Reward on original_env: {original_mean:.2f} ± {original_std:.2f}")
        # print(f"Avg Reward on abstract_env: {abstract_mean:.2f} ± {abstract_std:.2f}")
        
        task_complete, metric_result = earlystopping_callback(original_env, abstracted_model)

        results.append(((metric_result), abstracted_model))

        if task_complete:
            print("Finishing, we hit the threshold.")
            break
        else:
            print(f"Task complete: {task_complete}")

    return results