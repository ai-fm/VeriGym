import verigym
from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp
from verigym.policy.qvalue import ActiveLearningPolicy, QValuePolicy, EntropyLearningPolicy
from verigym.policy.randomized import RandomizedPolicy
import json
import gymnasium as gym
import time
import stormpy
from examples.workflow_utils import get_explored_state_action_pairs, get_mean_reward_from_trajectories

QUICK = False

BINS_STATES = 10
BINS_ACTIONS = 3
SEED = 0
STEPS = int(1e3) if not QUICK else int(1e3)
ITERATIONS = int(100) if not QUICK else int(10)
TRIALS = 1 if not QUICK else 2
SIM_STEPS = int(1e5) if not QUICK else int(1e4)  # steps used to evaluate the computed policy
GAMMA = 0.95  # discount factor
ENVS = [  # env id, with d and data type of its observation
    "MountainCarContinuous-v0",  # 2, float32
    "Pendulum-v1",  # 3, float32
    "Acrobot-v1",  # 6, float32
    # "LunarLander-v3",  # 8, float32
    # "Hopper-v5",  # 11, float64
    # "HalfCheetah-v5",  # 17, float64
    # "BipedalWalker-v3",  # 24, float32
    # "Ant-v5",  # 105, float64
    # "Humanoid-v5",  # 348, float64
    # "CarRacing-v3",  # 96x96x3 = 27648, uint8
]
if QUICK:
    ENVS = ["Pendulum-v1"]

POLICIES = [
    lambda env, abstraction_mapper: RandomizedPolicy(env),
    lambda env, abstraction_mapper: QValuePolicy(env, abstraction_mapper),
    lambda env, abstraction_mapper: ActiveLearningPolicy(env, abstraction_mapper),
    lambda env, abstraction_mapper: EntropyLearningPolicy(env, abstraction_mapper)]
POLICY_NAMES = ["Random", "Q-Learning", "ActiveLearning", "Entropy"]

# POLICIES = [POLICIES[2]]
# POLICY_NAMES = [POLICY_NAMES[2]]

timings = {}

def add_dict(d1, d2):
    for (key, val) in d2.items():
        if key in d1:
            d1[key] += val
        else:
            d1[key] = val

def normalize_dict(d1, c):
    for (key,val) in d1.items():
        d1[key] = val / c

for env_name in ENVS:
    print(f"\nRunning {env_name}:")
    timings[env_name] = {}

    for (pol, name) in zip(POLICIES, POLICY_NAMES):
        print(f"Exploring with {name}:")
        env = verigym.GenerativeEnv.from_gymnasium(gym.make(env_name))
        cumulative_times = {}

        abstraction_mapper = linspace_mapper(env=env, n_bins_states=BINS_STATES, n_bins_actions=BINS_ACTIONS)

        for i in range(TRIALS):

            ### Compute extracted model
            model, info = verigym.create_abstraction(
                env,
                abstraction_mapper,
                pol(env, abstraction_mapper),
                # RandomizedPolicy(env),
                num_steps = STEPS,
                n_iterations=ITERATIONS,
                verbose = False,
                return_info = True,
                multithreading=False
            )

            ### Get coverage model
            explored_S, explored_SA = get_explored_state_action_pairs(model)

            ### Get stormpy build times
            t0 = time.time()
            stormpy_mdp = build_stormpy_mdp(model)
            t_stormpy_build = time.time() - t0

            ### Get policy quality
            prop = stormpy.parse_properties(f"Rmax=?[Cdiscount={GAMMA}]")[0]
            result = stormpy.check_model_sparse(stormpy_mdp, prop, extract_scheduler=True)
            # policy mapped to the original model
            stormpy_policy = verigym.StormpyPolicy(result.scheduler, model.abstraction_map, stormpy_mdp)
            # policy on the abstract model
            identity_mapper = verigym.AbstractionMapper.initialize_identity_mapper(
                model.observation_space, model.action_space
            )
            stormpy_policy_on_abstracted = verigym.StormpyPolicy(result.scheduler, identity_mapper, stormpy_mdp)

            # simulation in the original (continuous) model
            trajectories_original = env.simulate(policy=stormpy_policy, n_steps=SIM_STEPS)
            reward_original = get_mean_reward_from_trajectories(trajectories_original)

            # simulation in the abstracted model
            trajectories_abstracted = model.simulate(policy=stormpy_policy_on_abstracted, n_steps=SIM_STEPS)
            reward_abstracted = get_mean_reward_from_trajectories(trajectories_abstracted)

            print(f"  Mean reward original: {reward_original:.2f}, abstract: {reward_abstracted:.2f}")

            add_dict(cumulative_times, info)
            add_dict(cumulative_times, {
                "explored_states": explored_S,
                "explored_state_action_pairs": explored_SA,
                "t_stormpy_build": t_stormpy_build,
                "mean_reward_original": reward_original,
                "mean_reward_abstracted": reward_abstracted,
            })

        normalize_dict(cumulative_times, TRIALS)
        timings[env_name][name] = cumulative_times

print("\nWriting file:")
with open("timings.json", "w") as f:
    json.dump(timings, f, indent=2, default=float)
print("Done!")