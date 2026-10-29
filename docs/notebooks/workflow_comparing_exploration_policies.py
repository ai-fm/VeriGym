# %% [markdown]
# # VeriGym - Comparing exploration policies

# %% [markdown]
# In this workflow, we zoom in on **exploration policies**, which are a critical component of model learning. We show off the 4 exploration policies that are natively supported by `VeriGym`, and show how to use them for model learning. Furthermore, we show how you can define your own exploration policy and use it in the same way.

# %% [markdown]
# ## 1. Setup

# %% [markdown]
# We start by setting up our environment and abstraction: for more details, see the specific workflows on this. In this file, we use the Gymnasiums `MountainCar` environment which (as we'll see later) requires smart exploration to learn.

# %%
import os
# checks if env var is set, defaults to True
QUICK = os.environ.get("QUICK", "1") != "0" 
# uncomment line below for full run
# QUICK = False 

# %%
import stormpy
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import logging

import verigym
from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation
from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp
from verigym.frameworks.stormpy.stormpypolicy import StormpyPolicy
from verigym.policy.randomized import RandomizedPolicy
from verigym.policy.policy import PolicyClass
from verigym.policy.qvalue import (
    QValuePolicy,
    ActiveLearningPolicy,
    EntropyLearningPolicy,
)

from workflow_utils.utils import get_explored_state_action_pairs, get_mean_reward_from_trajectories
logging.getLogger().setLevel(logging.ERROR)

# %%
SEED = 192837465

# %%
# 1. Load MountainCar environment
gym_env = gym.make("MountainCar-v0")
print("Environment type: ", type(gym_env))

# Create a VeriGymEnv from gym env
generative_model = verigym.GenerativeEnv.from_gymnasium(gym_env)

print("Environment type: ", type(generative_model))
print("Observation space shape: ", generative_model.observation_space.shape)
print("Type of observation space: ", type(generative_model.observation_space))

# Define abstraction
abstraction_mapper = linspace_mapper(
    env=generative_model,
    n_bins_states=20,
    n_bins_actions=3,
)

print(f"The abstract observation space has {abstraction_mapper.abstract_n_states} states!")
print(f"The abstract action space has {abstraction_mapper.abstract_n_actions} actions!")

# %% [markdown]
# ## 2. Exploration policies
#
# `VeriGym` makes use of **exploration policies** to collect samples from the environment for model learning. We natively support a number of methods to do this:
#
# 1. `RandomizedPolicy`: a policy that uniformly samples actions.
# 2. `QValuePolicy`: uses an optimistic version of abstraction learned *so far* to estimate Q-values. Actions are picked epsilon-greedily based on these values.
# 3. `ActiveLearningPolicy`: based on [Araya-López et al. (2012)](https://ewrl.wordpress.com/wp-content/uploads/2011/08/ewrl2011_submission_17.pdf). Computes Q-values similarly to `QValuePolicy`, but uses an *exploration reward* inversely proportional to the visitation count instead of the true reward function.
# 4. `EntropyLearningPolicy`: based on [Hazan et al. (2019)](https://proceedings.mlr.press/v97/hazan19a.html). Iteratively computes a maximum-entropy policy over the state distribution.
#
# Let's define these policies as follows:

# %%
epsilon = 0.1 # exploration threshold for the epsilon-greedy policies

# Constructors, such that every abstraction starts from a fresh policy (seeded by `create_abstraction(seed=SEED)`)
exploration_policies = {
    "Randomized": RandomizedPolicy(generative_model),
    "QValue": QValuePolicy(generative_model, abstraction_mapper, epsilon=epsilon),
    "ActiveLearning": ActiveLearningPolicy(generative_model, abstraction_mapper, epsilon=epsilon),
    "EntropyLearning": EntropyLearningPolicy(generative_model, abstraction_mapper),
}

# %% [markdown]
# During model learning, policies may need to periodically be updated based on the current best estimate of the abstracted model. Such updates are implemented in the function `update_for_abstraction_refinement(...)` in `verigym.policy.PolicyClass`, which takes as input the data gathered thus far. For example, when calling this function on `QValuePolicy`, it updates it's estimate of the model, and then updates it's Q-values via a number of value iteration steps on this updated model.
#
# When calling `create_abstraction`, the frequency of such policy updates can be determined by the parameter `n_iterations`. Each iteration consists of `num_steps` interactions with the environment, after which the policy is updated. More frequent updates yield better exploration behaviour, but requires more (computationally expensive) policy updates. For our example, we use the following parameters:

# %%
n_iterations = 100   if not QUICK else 2
num_steps = int(1e3) if not QUICK else 10

# %% [markdown]
# In total, we thus have a budget of `n_iterations * num_steps` which equates to 1M interations with the environment. Assuming most episodes will run for roughly 200 steps (the environments maximum), this translates to around 20k episodes.

# %% [markdown]
# ## 3. Model Learning
# Next, let's use the different policies to learn model abstractions via the `create_abstraction` function:

# %%
abstracted_models = {}
for name, make_policy in exploration_policies.items():
    print(f"--- {name} ---")
    abstracted_models[name] = verigym.create_abstraction(
        original_env=generative_model,
        abstraction_mapper=abstraction_mapper,
        exploration_policy=make_policy,
        num_steps=num_steps,
        n_iterations=n_iterations,
        multithreading=False,
        seed=SEED
    )

# %% [markdown]
# To determine if the policies are any good, let us first compute the **coverage** of each policy: how many states and state-action pairs were explored during model learning?

# %%
coverage = {
    name: get_explored_state_action_pairs(model)
    for name, model in abstracted_models.items()
}

n_state_action_pairs = abstraction_mapper.abstract_n_states * abstraction_mapper.abstract_n_actions
print(f"{'Policy':<16} {'States':>8} {'State-action pairs':>20}")
for name, (n_states, n_pairs) in coverage.items():
    print(f"{name:<16} {n_states:>8} {n_pairs:>20}")
print(f"{'Total':<16} {abstraction_mapper.abstract_n_states:>8} {n_state_action_pairs:>20}")

# %% [markdown]
# We see that the naive `RandomizedPolicy` explores only a small part of the environment, even after a large number of interactions. In contrast, our other exploration policies discover a much larger portion of the state-action space, and thus typically provide much more useful abstract models. 
#
# To get a better idea of the quality of these different learned models, we use `stormpy` to compute a policy on all models, and compare their performance on the original environment. First, let's compute policies:

# %%
# Set up cumulative discounted reward objective
gamma = 0.95
prop = stormpy.parse_properties(f"Rmax=?[Cdiscount={gamma}]")[0]

# Compute all policies using stormpy
stormpy_mdps = {}
schedulers = {}
for name, abstracted_model in abstracted_models.items():
    
    stormpy_mdps[name] = build_stormpy_mdp(abstracted_model)
    result = stormpy.check_model_sparse(
        stormpy_mdps[name], prop, extract_scheduler=True
    )
    schedulers[name] = result.scheduler

# %% [markdown]
# We then evaluate these policies on both the original and abstracted environment, which yields the following:

# %%
N_SIMUATION_STEPS = (10 if QUICK else int(1e5))

def evaluate_policies_from_models():
    mean_rewards_original = {}
    mean_rewards_abstracted = {}
    for name, abstracted_model in abstracted_models.items():
        # policy mapped to the original model
        verigym_policy = StormpyPolicy(
            schedulers[name], abstracted_model.abstraction_map, stormpy_mdps[name]
        )
        # policy on the abstract model
        identity_mapper = verigym.AbstractionMapper.initialize_identity_mapper(
            abstracted_model.observation_space, abstracted_model.action_space
        )
        verigym_policy_on_abstracted = StormpyPolicy(
            schedulers[name], identity_mapper, stormpy_mdps[name]
        )

        # simulation in the original (continuous) model
        trajectories_original = generative_model.simulate(
            policy=verigym_policy,
            n_steps=N_SIMUATION_STEPS,
            seed=SEED,
        )
        mean_rewards_original[name] = get_mean_reward_from_trajectories(
            trajectories_original
        )

        # simulation in the abstracted model
        trajectories_abstracted = abstracted_model.simulate(
            policy=verigym_policy_on_abstracted,
            n_steps=N_SIMUATION_STEPS,
            seed=SEED,
        )
        mean_rewards_abstracted[name] = get_mean_reward_from_trajectories(
            trajectories_abstracted
        )

    print("Comparing the mean cumulative reward of the abstract policies:")
    print(f"{'Policy':<16} {'Original':>12} {'Abstract':>12}")
    for name in abstracted_models:
        print(
            f"{name:<16} {mean_rewards_original[name]:>12.2f} {mean_rewards_abstracted[name]:>12.2f}"
        )

evaluate_policies_from_models()


# %% [markdown]
# Recall that the original environment terminates after 200 steps, while the abstracted model terminates only after reaching a terminal state. Thus, -200 is the minimal possible value in the original, while -10000 is minimal in the abstracted model.
#
# The value found for all abstracted models is larger then -10000. Thus, all corresponding exploration policies uncovered enough of the model to allow for a path to the goal state. However, only the model found by `EntropyLearningPolicy` is sufficiently accurate to compute policies that reach the goal within 200 steps in the original environment.

# %% [markdown]
# ## 4. Defining Exploration Policies
#
# While the natively supported exploration policies are a good starting point, some environments might need more specialized or sophisticated exploration policies to be able to sufficiently sample the entire environments. `VeriGym` supports users to define such policies themselves and use them in place of our native exploration policies.
#
# As a simple example, we might notice that in `MountainCar`, action 1 (*don't accelerate*) is never optimal. Furthermore, just for fun, let's say we want to pick the two remaining actions with different probabilities at each sampling iteration. 
#
# We can implement such an policy as a `PolicyClass`. As can be read in more detail in the docs, implementing a specific (exploration) policy class requires implementing the functions `_action_from_policy`, which maps abstracted observations to actions, and `update_for_abstraction_refinement`, which defines how the policy is updated at the end of each iteration. A policy that samples its actions also implements `seed`, so that the `seed` of `create_abstraction` makes it reproducible. For our example, this implementation looks as follows:

# %%
class CustomRandomPolicy(PolicyClass):
    """
    Custom policy class for MountainCar env. Randomly picks action 0 or 2 based 
    on probabilities defined in `self.policy`. `self.policy` is a 2-element 
    `np.array` that gets randomized update_for_abstraction_refinement."""

    def __init__(self, policy, abstraction_mapper, seed=None):
        super().__init__(policy, abstraction_mapper)
        self.seed(seed)

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _action_from_policy(self, obs):
        return self.rng.choice(a=[0,2], p=self.policy)

    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_dicts_counts, state_distr_counts):
        r = self.rng.random()
        self.policy = [r, 1-r]
        return self


# %% [markdown]
# With less than 25 lines of code, we have now defined a custom policy! This policy can immidiately be used in exactly the same way as the native policies we used before, e.g. as an exploration policy in `create_abstraction`:

# %%
custom_policy = CustomRandomPolicy(policy=[0.5,0.5], abstraction_mapper=abstraction_mapper)

abstracted_models["Custom"] = verigym.create_abstraction(
        original_env=generative_model,
        abstraction_mapper=abstraction_mapper,
        exploration_policy=custom_policy,
        num_steps=num_steps,
        n_iterations=n_iterations,
        multithreading=False,
        seed=SEED
    )

# %% [markdown]
# For completeness, let's compare the coverage achieved by our custom policy to that of the native ones from before:

# %%
coverage["Custom"] = get_explored_state_action_pairs(abstracted_models["Custom"])

print(f"{'Policy':<16} {'States':>8} {'State-action pairs':>20}")
for name, (n_states, n_pairs) in coverage.items():
    print(f"{name:<16} {n_states:>8} {n_pairs:>20}")
print(f"{'Total':<16} {abstraction_mapper.abstract_n_states:>8} {n_state_action_pairs:>20}")

# %% [markdown]
# Unsurprisingly, our policy reaches a smaller number of state-action pairs then `RandomizedPolicy` (since it never takes action 1), but explores slightly more states. Let's see how that effects performance:

# %%
stormpy_mdps["Custom"] = build_stormpy_mdp(abstracted_models["Custom"])
schedulers["Custom"] = stormpy.check_model_sparse(
    stormpy_mdps["Custom"], prop, extract_scheduler=True
    ).scheduler

evaluate_policies_from_models()

# %% [markdown]
# As expected, our custom exploration policy yields a better model then that found by `Randomized`, but is outperformed by all other native policies.
