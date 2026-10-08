import verigym
from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.policy.qvalue import ActiveLearningPolicy
from verigym.policy.randomized import RandomizedPolicy
import json
import gymnasium as gym

QUICK = True

BINS = 10
SEED = 0
STEPS = int(1e6) if not QUICK else int(1e4)
TRIALS = 5 if not QUICK else 2
ENVS = [  # env id, with d and data type of its observation
    "MountainCarContinuous-v0",  # 2, float32
    "Pendulum-v1",  # 3, float32
    "Acrobot-v1",  # 6, float32
    "LunarLander-v3",  # 8, float32
    "Hopper-v5",  # 11, float64
    "HalfCheetah-v5",  # 17, float64
    "BipedalWalker-v3",  # 24, float32
    "Ant-v5",  # 105, float64
    "Humanoid-v5",  # 348, float64
    # "CarRacing-v3",  # 96x96x3 = 27648, uint8
]
if QUICK:
    ENVS = ["Pendulum-v1"]

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
    env = verigym.GenerativeEnv.from_gymnasium(gym.make(env_name))
    cumulative_times = {}

    abstraction_mapper = linspace_mapper(env=env, n_bins_states=BINS, n_bins_actions=BINS)
    nr_states, nr_actions = abstraction_mapper.abstract_n_states, abstraction_mapper.abstract_n_actions

    for i in range(TRIALS):
        _model, info = verigym.create_abstraction(
            env,
            abstraction_mapper,
            ActiveLearningPolicy(env, nr_states, nr_actions),
            # RandomizedPolicy(env),
            num_steps = STEPS,
            verbose = False,
            return_info = True
        )
        add_dict(cumulative_times, info)

    normalize_dict(cumulative_times, TRIALS)
    timings[env_name] = cumulative_times

with open("timings.json", "w") as f:
    json.dump(timings, f, indent=2, default=float)