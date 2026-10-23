"""
Profile a single abstraction run: build an abstraction of `--env` with exploration policy `--policy`
using a total budget of `--budget` environment steps, then evaluate the optimal policy of the
abstraction on both the original environment and the abstraction itself.

The result of each run is written as one JSON file to `--out-dir`, so runs can be collected
(and resumed) by an outer script, see `collect_scaling.sh`.

Example:
    python run_profiling.py --env MountainCarContinuous-v0 --policy Entropy --budget 10000 --seed 0
"""
import verigym
from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp
from verigym.policy.qvalue import ActiveLearningPolicy, QValuePolicy, EntropyLearningPolicy
from verigym.policy.randomized import RandomizedPolicy
import argparse
import json
import gymnasium as gym
import numpy as np
import subprocess
import time
import stormpy
import os
import sys

# make docs/workflows/utils.py (and the plot_style it imports) importable regardless of the working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_explored_state_action_pairs

POLICIES = {
    "Random": lambda env, abstraction_mapper, seed: RandomizedPolicy(env, seed),
    "Q-Learning": lambda env, abstraction_mapper, seed: QValuePolicy(env, abstraction_mapper, seed),
    "ActiveLearning": lambda env, abstraction_mapper, seed: ActiveLearningPolicy(env, abstraction_mapper, seed),
    "Entropy": lambda env, abstraction_mapper, seed: EntropyLearningPolicy(env, abstraction_mapper, seed),
}
EVAL_SEED = 12345  # fixed, so all runs are evaluated on the same initial states


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, help="gymnasium environment id")
    parser.add_argument("--policy", required=True, choices=POLICIES.keys(), help="exploration policy")
    parser.add_argument("--budget", type=int, required=True, help="total number of exploration steps")
    parser.add_argument("--steps-per-iter", type=int, default=1000, help="exploration steps per iteration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins-states", type=int, default=25)
    parser.add_argument("--bins-actions", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--eval-episodes", type=int, default=1000, help="complete episodes used for evaluation")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "tech_eval", "runtimes"))
    args = parser.parse_args()
    if args.budget % args.steps_per_iter != 0:
        parser.error("--budget must be a multiple of --steps-per-iter")
    return args


def output_path(args):
    return os.path.join(args.out_dir, f"{args.env}__{args.policy}__{args.budget}"
                                     f"__s{args.bins_states}a{args.bins_actions}__seed{args.seed}.json")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        return None


def evaluate(env, policy, n_episodes, horizon, seed):
    """Mean undiscounted return over `n_episodes` complete episodes, each capped at `horizon` steps."""
    returns = []
    state, _ = env.reset(seed=seed)
    for _ in range(n_episodes):
        total = 0.0
        for _ in range(horizon):
            state, reward, terminated, truncated, _ = env.step(policy.get_action(state))
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
        state, _ = env.reset()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    args = parse_args()
    n_iterations = args.budget // args.steps_per_iter

    env = verigym.GenerativeEnv.from_gymnasium(gym.make(args.env))
    horizon = env.spec.max_episode_steps

    # seed once; afterwards all generators keep running (create_abstraction is called with seed=None,
    # as passing a seed there would reset to the same initial state in every iteration)
    np.random.seed(args.seed)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    t0 = time.perf_counter()
    abstraction_mapper = linspace_mapper(env=env, n_bins_states=args.bins_states, n_bins_actions=args.bins_actions)
    t_create_mapper = time.perf_counter() - t0
    exploration_policy = POLICIES[args.policy](env, abstraction_mapper, args.seed)

    print(f"{args.env} | {args.policy} | budget {args.budget} ({n_iterations} x {args.steps_per_iter}) | "
          f"{abstraction_mapper.abstract_n_states} states, {args.bins_actions} action bins | seed {args.seed}")

    ### Compute extracted model
    model, info = verigym.create_abstraction(
        env,
        abstraction_mapper,
        exploration_policy,
        num_steps=args.steps_per_iter,
        n_iterations=n_iterations,
        verbose=False,
        return_info=True,
        multithreading=False,
        seed = args.seed
    )

    ### Get coverage model
    explored_S, explored_SA = get_explored_state_action_pairs(model)

    ### Get stormpy build times
    t0 = time.perf_counter()
    stormpy_mdp = build_stormpy_mdp(model)
    t_stormpy_build = time.perf_counter() - t0

    ### Compute optimal policy of the abstraction
    t0 = time.perf_counter()
    prop = stormpy.parse_properties(f"Rmax=?[Cdiscount={args.gamma}]")[0]
    result = stormpy.check_model_sparse(stormpy_mdp, prop, extract_scheduler=True)
    t_model_check = time.perf_counter() - t0

    ### Get policy quality on original
    stormpy_policy = verigym.StormpyPolicy(result.scheduler, model.abstraction_map, stormpy_mdp)
    t0 = time.perf_counter()
    reward_original, reward_original_std = evaluate(env, stormpy_policy, args.eval_episodes, horizon, EVAL_SEED)
    t_eval = time.perf_counter() - t0
    ### Get policy quality on abstracted env
    # identity_mapper = verigym.AbstractionMapper.initialize_identity_mapper(model.observation_space, model.action_space)
    # stormpy_policy_on_abstracted = verigym.StormpyPolicy(result.scheduler, identity_mapper, stormpy_mdp)
    # reward_abstracted, reward_abstracted_std = evaluate(model, stormpy_policy_on_abstracted, args.eval_episodes, horizon, EVAL_SEED)
    # t_eval = time.perf_counter() - t0

    # print(f"  Mean reward original: {reward_original:.2f}, abstract: {reward_abstracted:.2f}")

    record = {
        "env": args.env,
        "policy": args.policy,
        "budget": args.budget,
        "seed": args.seed,
        "n_iterations": n_iterations,
        "steps_per_iter": args.steps_per_iter,
        "bins_states": args.bins_states,
        "bins_actions": args.bins_actions,
        "n_abstract_states": abstraction_mapper.abstract_n_states,
        "gamma": args.gamma,
        "eval_episodes": args.eval_episodes,
        "horizon": horizon,
        **info,
        "t_stormpy_build": t_stormpy_build,
        "t_model_check": t_model_check,
        "t_eval": t_eval,
        "explored_states": explored_S,
        "explored_state_action_pairs": explored_SA,
        "mean_reward_original": reward_original,
        "std_reward_original": reward_original_std,
        "git_commit": git_commit(),
    }
    record["time_model_construction"] += t_create_mapper

    os.makedirs(args.out_dir, exist_ok=True)
    path = output_path(args)
    with open(path + ".tmp", "w") as f:
        json.dump(record, f, indent=2, default=float)
    os.replace(path + ".tmp", path)  # atomic, so an interrupted run never leaves a partial file
    print(f"  Written to {path}")


if __name__ == "__main__":
    main()
