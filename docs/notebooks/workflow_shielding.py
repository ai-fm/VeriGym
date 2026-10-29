# %% [markdown]
# # Workflow: Implementing a simple shield for a gymnasium environment
#
# **What is the motivation?**
#
# Shielding is a common technique in safe reinforcement learning. 
# A shield blocks actions that are considered unsafe. 
# There are various ways to define a shield.
# In this workflow, we construct a discretized abstraction from a continuous simulation environment.
# In the abstraction, we use action values derived using probabilistic model checking to block actions that have a too large probability of leading to unsafe states.
# Specifically, we implement a memoryless post-shield, which is close in spirit to what was used in Jansen et al. (2020), and more recently by Brorholt et al. (2026) based on learned model estimates.
# We then map the abstract shield back to the continuous environment, to use during RL training.
# To derive and deploy such shields on continuous simulation-based environments is a substantial engineering effort that `VeriGym` makes easy.
#
#
# **What do we show here?**
#
# We show how to use `VeriGym` to compute and deploy a conceptually simple shield for a continuous `gymnasium` environment. 
# As the example we use the `CartPole-v1` environment, where the task is to learn how to move left and right in order to balance a pole on a cart.
# For this task, we consider it `"unsafe"` to move too close to the edges of the screen, which we express using state labels on the original environment.
# Then, the workflow proceeds as follows:
# - We learn a finite-state representation of `CartPole`.
# - We model check the learned MDP using `stormpy` for the worst-case probability of reaching `"unsafe"` states.
# - Based on the model checking results, we compute state-action values, and construct a shield that blocks all actions that have a probability `p>0.1` of reaching an unsafe state.
# - We apply the abstract shield to the original environment using a convenience environment wrapper, that takes care of mapping the shield back to the original state space.
# - Then, we train two Reinforcement Learning agents, one shielded and one non-shielded, using the `PPO` algorithm from stable baselines.
#
# We evaluate the training development, but more importantly, we evaluate both RL agents in both the shielded and non-shielded environment, and compare episode rewards and the number of safety violations.

# %% [markdown]
# ## Imports and helpers

# %%
import os
# checks if env var is set, defaults to True
QUICK = os.environ.get("QUICK", "1") != "0" 
# uncomment line below for full run
# QUICK = False 

# %%
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import stable_baselines3 as sb3
import stormpy

import verigym
from verigym.abstraction.abstractionmapper import AbstractionMap, AbstractionMapper, bin_edges_map
from verigym.abstraction.discretization import generate_box_bins
from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation
from verigym.environments.labeling import StateLabel
from verigym.frameworks.stable_baselines3.policy import SB3Policy
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp
from verigym.policy.randomized import RandomizedPolicy

from workflow_utils.utils import train_with_sb3_ppo, plot_logged_data, plot_compare_policy_eval, set_seeds
from workflow_utils.utils import OUTPATH_WORKFLOWS as OUTPATH

from workflow_utils import plot_style

import stable_baselines3 as sb3

GYM_STRING = "CartPole-v1"
SUBPATH = "shield_files"
RL_OUTPATH = f"{OUTPATH}/{SUBPATH}/ppo/{GYM_STRING}"
PLOT_PATH = f"{OUTPATH}/figures/{SUBPATH}"
os.makedirs(PLOT_PATH, exist_ok=True)

# %% [markdown]
# Import the class that wraps a Gymnasium environment with a post-shield.

# %%
from verigym.supplements.shield_wrapper import AbstractPostShieldWrapper

# %% [markdown]
# Use a seed for reproducing results.

# %%
SEED = 12929582 # randomly generated seed
sb3.common.utils.set_random_seed(SEED)


# %%
def get_interval_abstraction_mapper(bins_per_state_dim, env):

    bin_edges = generate_box_bins(env.observation_space, np.linspace, bins_per_state_dim)
    state_abstraction_map = bin_edges_map(env.observation_space, bin_edges, backward_kind="interval")
    state_abstraction_map.forward_map = bin_edges.orig_to_enum
    state_abstraction_map.original_to_enum = bin_edges.orig_to_enum

    state_abstraction_map.backward_map= bin_edges.enum_to_interval

    abstraction_mapper = AbstractionMapper(
        state_abstraction_map=state_abstraction_map,
        action_abstraction_map=AbstractionMap.initialize_identity_map(env.action_space)
    )
    return abstraction_mapper


# %%
def eval_reward_safety_violations(env, policy, unsafe_label, seed=SEED, eval_episodes=500):
    count_violations = []
    eval_rewards = []
    for i in range(eval_episodes):
        ep_count, ep_reward = 0, 0
        obs, _ = env.reset(seed=seed+i)

        while True:
            action = policy.get_action(obs)
            obs, rew, term, trunc, _ = env.step(action)
            ep_reward += rew
            try:
                if unsafe_label in env.get_labels_of_state(obs):
                    ep_count += 1
            except:
                if unsafe_label in env.env.get_labels_of_state(obs):
                    ep_count += 1
            if term or trunc:
                break

        eval_rewards.append(ep_reward)
        count_violations.append(ep_count)
        
    return eval_rewards, count_violations


# %% [markdown]
# ## The actual workflow

# %% [markdown]
# We are using the `CartPole-v1` environment in this workflow.
# We first load it from Gymnasium and then into `VeriGym`, and place bounds on unbounded dimensions of the observation space.
# This is necessary for us to be able to apply discretization techniques afterwards.

# %%
gym_env = gym.make(GYM_STRING)
gym_env = ReplaceInfObservation(gym_env, neg_inf=-5, pos_inf=5)
set_seeds(SEED, gym_env)
env = verigym.GenerativeEnv.from_gymnasium(gym_env)

# %% [markdown]
# In this workflow, we want to construct a shield that blocks "unsafe" actions.
# For this, we need to define what we mean by "unsafe". 
# Here, we consider it unsafe for the cart to move too close to the edges of the environment.
# We define this by creating a `StateLabel` object and adding it to the environment.

# %%
unsafe_label = StateLabel("unsafe",
                          lambda s: (s[0] < -1.5) | (s[0] > 1.5))
env.add_state_label(unsafe_label)

# %% [markdown]
# Next, we define our desired state-space discretization and learn an abstract representation of the environment.
# In this case, we divide each dimension of the state space into 10 bins. 
# To properly translate the "unsafe" state label to the abstract environment, we need our abstraction map to provide us with an interval type backward mapping. That means that for each bin, the backward mapping provides us with the upper and lower bin edges.
# Then, we learn the model representation by exploring randomly for `1e6` steps and aggregating transitions.
#
# This process - simulation, discretization and learning - usually takes less than 30 seconds in total.

# %%
abstraction_mapper = get_interval_abstraction_mapper(bins_per_state_dim=10, env=env)

num_steps = int(1e6) if not QUICK else int(1e3)

abstracted_model = verigym.create_abstraction(
    original_env=env,
    abstraction_mapper=abstraction_mapper,
    exploration_policy=RandomizedPolicy(env),
    num_steps=num_steps,
    multithreading=False,
    seed=SEED
)

# %% [markdown]
# We then export the learned abstract MDP to `stormpy` (the Python bindings for the Storm model checker) and model check it, again using the `stormpy` interface. 
# The property `Pmin=? [F "unsafe"]` gives for each state the probability to reach an unsafe state under the safest policy possible in the estimated abstract MDP. 
# We then compute the state-action probabilities from the state probabilities returned by probabilistic model checking by doing one step of value iteration.
# These state-action probabilities give us the probabilities under each action, in expectation over successor states, when following the safety-optimal policy afterwards.
#
# Note that these probabilities do not directly carry truth on the original simulator environment, as discretization does not yield such guarantees out of the box. Still, it is a very practical way to create a shield.

# %%
propertystr = 'Pmin=? [F "unsafe"]'
stormpy_mdp = build_stormpy_mdp(abstracted_model)

def get_action_values(mdp, propertystr):
    formula = stormpy.parse_properties(propertystr)[0]
    result = stormpy.check_model_sparse(mdp, formula)
    state_values = result.get_values()

    action_values = np.zeros((mdp.nr_states, len(mdp.choice_labeling.get_labels())))
    action_values.shape

    for state in mdp.states:
        for action in state.actions:
            if len(action.labels) == 0:
                continue
            action_idx = int(action.labels.pop())

            action_value = 0.0
            for tr in action.transitions:
                action_value += tr.value() * state_values[tr.column]
            action_values[state.id, action_idx] = action_value
    return action_values

action_values = get_action_values(stormpy_mdp, propertystr)


# %% [markdown]
# Now it is time to compute the shield. We want to block all actions that have a probability `p>0.1` of reaching unsafe states. 
# We define the shield as an action mask.

# %%
def build_shield(action_values, safety_threshold):
    shield = np.ones_like(action_values, dtype=bool)
    shield[action_values > safety_threshold] = 0
    return shield

unsafe_threshold = 0.1
shield = build_shield(action_values, unsafe_threshold)

# %% [markdown]
# We apply the shield to the environment using the `AbstractShieldWrapper`, that is available as part of the convenience functionality in `verigym.supplements`.
# Notably, this wrapper requires discrete action spaces.

# %%
shielded_env = AbstractPostShieldWrapper(env, shield, abstraction_mapper, seed=SEED)

# %% [markdown]
# ## Training on the original and on the shielded environment
#
# We train two RL agents using the `PPO` implementation from `stable_baselines3`.
# Training for 25000 steps usually takes around 30 seconds (e.g., on a MacBook Pro 2023 with an M3 chip and 18GB RAM). In total, these two training runs should be in the order of a minute, depending on your hardware.

# %%
os.makedirs(f"{RL_OUTPATH}/normal/", exist_ok=True)
os.makedirs(f"{RL_OUTPATH}/shielded/", exist_ok=True)

# %%
EVAL_FREQ = 1_000
TRAIN_STEPS = 25_000 if not QUICK else 1000

# %%
if not os.path.exists(f"{RL_OUTPATH}/normal/ppo_model.zip"):
    print("Training on non-shielded environment.")
    train_with_sb3_ppo(env, env, f"{RL_OUTPATH}/normal/", TRAIN_STEPS, eval_freq=EVAL_FREQ, seed=SEED)
else:
    print("Proceeding with pre-trained model.")

# %%
if not os.path.exists(f"{RL_OUTPATH}/shielded/ppo_model.zip"):
    print("Training on shielded environment.")
    train_with_sb3_ppo(shielded_env, shielded_env, f"{RL_OUTPATH}/shielded/", TRAIN_STEPS, eval_freq=EVAL_FREQ, seed=SEED)
else:
    print("Proceeding with pre-trained model.")

# %% [markdown]
# ## Evaluation

# %% [markdown]
# First, we load the trained policies back into `VeriGym` using the `SB3Policy` class and visualize their reward development during training.
# Since we do not train for a long time, the development looks relatively unstable for both agents, but reaches a high reward towards the end.
#
# Then, we run evaluation episodes in four different settings: 
# 1) default baseline: policy trained without a shield deployed on the unshielded environment
# 2) post-shielding: policy trained without a shield deployed on the shielded environment
# 3) pre-shielding: policy trained with shield deployed on the unshielded environment
# 4) full-shielding: policy trained with shield deployed on the shielded environment
#
# We compare the output rewards and safety violations for all settings.

# %%
standard_policy = SB3Policy(sb3.PPO.load(f"{RL_OUTPATH}/normal/ppo_model"), env, deterministic=True)
shielded_policy = SB3Policy(sb3.PPO.load(f"{RL_OUTPATH}/shielded/ppo_model"), shielded_env, deterministic=True)

# %%
plot_logged_data([f"{RL_OUTPATH}/normal/log/evaluations.npz", f"{RL_OUTPATH}/shielded/log/evaluations.npz"],
                    labels=["PPO", "Shielded PPO"], key="results", ylabel="Avg. Reward",
                    title="PPO training curves", running_mean=5,
                    path_output=f"{PLOT_PATH}/workflow_shield_training_curve.pdf")

# %% [markdown]
# Evaluation the policies under different combinations of training and evaluation. This can take a couple of minutes to run (typically less than 2).

# %%
eval_episodes = 500 if not QUICK else 10

standard_reward, standard_violations = eval_reward_safety_violations(env, standard_policy, "unsafe", seed=SEED, eval_episodes=eval_episodes)
standard_shield_reward, standard_shield_violations = eval_reward_safety_violations(shielded_env, standard_policy, "unsafe", seed=SEED, eval_episodes=eval_episodes)
shielded_reward, shielded_violations = eval_reward_safety_violations(shielded_env, shielded_policy, "unsafe", seed=SEED, eval_episodes=eval_episodes)
shielded_non_shield_reward, shielded_non_shield_violations = eval_reward_safety_violations(env, shielded_policy, "unsafe", seed=SEED, eval_episodes=eval_episodes)

# %% [markdown]
# For rewards in evaluation episodes we make the following observations.
#
# - The policies that are deployed on the non-shielded environment achieve the highest reward. That indicates that the shield blocks actions with positive reward in execution. 
# - However, both shielded and unshielded agent perform similarly in evaluation. That indicates that the shield does not inhibit the learning process strongly.
# - The policies that are deployed on the shielded environment achieve lower rewards. Interestingly, the agent that was shielded during training performs better than the agent that was trained without a shield.

# %%
plot_compare_policy_eval([standard_reward, shielded_reward, standard_shield_reward, shielded_non_shield_reward], 
                         ["T:standard\nE:standard", "T:shielded\nE:shielded", "T:standard,\nE:shielded", "T:shielded\nE:standard"],
                         title="Episodic reward across 500 evaluation episodes", ylabel="Reward",
                         path_output=f"{PLOT_PATH}/workflow_shield_reward.pdf")

# %% [markdown]
# For safety violations in evaluation we make the following observations.
#
# - The agents that are deployed on the shielded environment report no safety violations. That shows that the shield is effective in execution.
# - The agent that was shielded in training achieves lower safety violations than the agent that was trained unshielded, even when evaluated on the unshielded environment.

# %%
plot_compare_policy_eval([standard_violations, shielded_violations, standard_shield_violations, shielded_non_shield_violations], 
                         ["T:standard\nE:standard", "T:shielded\nE:shielded", "T:standard,\nE:shielded", "T:shielded\nE:standard"],
                         title="Number of safety violations across 500 evaluation episodes", ylabel="Safety Violations",
                         path_output=f"{PLOT_PATH}/workflow_shield_safety_violations.pdf")
