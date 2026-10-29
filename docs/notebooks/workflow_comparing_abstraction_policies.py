# %% [markdown]
# # VeriGym - Comparing abstraction policies

# %% [markdown]
# ## Imports

# %%
import os
# checks if env var is set, defaults to True
QUICK = os.environ.get("QUICK", "1") != "0" 
# uncomment line below for full run
# QUICK = False 

# %%
import logging
from pprint import pprint

import gymnasium as gym
import numpy as np
import stormpy
import stable_baselines3 as sb3
from gymnasium.wrappers import TimeLimit

import os

import verigym
from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation
from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp, build_stormpy_imdp
from verigym.policy.randomized import RandomizedPolicy
from verigym.policy.qvalue import EntropyLearningPolicy
from verigym.frameworks.stable_baselines3.policy import SB3Policy
from workflow_utils.utils import OUTPATH_WORKFLOWS as OUTPATH, train_with_sb3_ppo, train_with_sb3_sac, plot_logged_data, plot_compare_policy_eval, get_mean_reward_from_trajectories
from workflow_utils import plot_style

logging.getLogger().setLevel(logging.ERROR)

SEED = 0

GYM_STRING = "MountainCarContinuous-v0"
SUBPATH = "compare_files"
dir = f"{OUTPATH}/{SUBPATH}"
PLOT_PATH = f"{OUTPATH}/figures/{SUBPATH}"
RL_OUTPATH = f"{OUTPATH}/{SUBPATH}/ppo/{GYM_STRING}"
os.makedirs(PLOT_PATH, exist_ok=True)

def get_average_episode_length(trajectories):
    return np.mean([len(traj) for traj in trajectories])


def get_mean_reward_from_trajectories(trajectories):
    rewards = []
    for trajectory in trajectories:
        trajectory_rewards = list(map(lambda tup: tup[2], trajectory))
        rewards.append(np.sum(trajectory_rewards))
    return float(np.mean(rewards))


# %% [markdown]
# ## 1. Environment
#
# We use the  OpenAI gymnasium Cart Pole environment for this demo:
#
# <!-- <img src="workflow_utils/cart_pole.gif" width="400"/> -->
#

# %%
# 1. Load in gym
gym_env = gym.make(GYM_STRING)
gamma = 0.995  # discount factor
gym_env.reset(seed=SEED)
print("Environment type: ", type(gym_env))

# 2. For discretization, replace (-inf, inf) observation bounds
gym_env = ReplaceInfObservation(gym_env, neg_inf=-5, pos_inf=5)
print("Observation space shape: ", gym_env.observation_space.shape)
print("Observation upper bounds: ", gym_env.observation_space.high)
print("Observation lower bounds: ", gym_env.observation_space.low)
print("Type of observation space: ", type(gym_env.observation_space))

# %% [markdown]
# Now, we load the `gym_env` into our framework, turning it into a (`gym`-like) `VeriGymEnv`:

# %%
# Create a VeriGymEnv from gym env
generative_model = verigym.GenerativeEnv.from_gymnasium(gym_env)

print("Environment type: ", type(generative_model))
print("Observation space shape: ", generative_model.observation_space.shape)
print("Observation upper bounds: ", generative_model.observation_space.high)
print("Observation lower bounds: ", generative_model.observation_space.low)
print("Type of observation space: ", type(generative_model.observation_space))

# %% [markdown]
# The observation space is the same as before, but now we can apply learn the underlying model and perform any abstraction!

# %% [markdown]
# ## 2. Abstraction
#
# Now we: 
# 1. Apply a user-specified discretization to the state space
# 2. Learn the transition and reward functions through random search

# %%
n_bins_states = 25
n_bins_action = 10
abstraction_mapper = linspace_mapper(
    env=generative_model, 
    n_bins_states=n_bins_states, 
    n_bins_actions=n_bins_action
)

abstracted_model = verigym.create_abstraction(
    original_env=generative_model,
    abstraction_mapper=abstraction_mapper,
    exploration_policy=EntropyLearningPolicy(generative_model, abstraction_mapper, seed=SEED),
    num_steps=(10 if QUICK else int(1e3)),
    n_iterations=50,
    multithreading=False,
    seed=SEED
)

print("Type of observation space: ", type(abstracted_model.observation_space))
print(f"The abstract observation space has {abstracted_model.observation_space.n} states!")
print("Type of environment: ", type(abstracted_model))

# %% [markdown]
# The abstract model allows us to explicitly access the model's transition and reward functions. Note that not all states are always reachable.

# %%
state = 120

print(f"Transitions from state {state}:")
pprint(abstracted_model.transition_function[state])
print(f"\nState-action rewards of state {state}:")
pprint(abstracted_model.reward_function[state])

# %% [markdown]
# ## 3. Model checking
#
# We now want to model check the abstraction we learned in `stormpy` w.r.t maximal cumulative reward. 
#
# Therefore, we
# 1. Export the model to a `stormpy.storage.SparseMdp`
# 2. Model check the MDP in `stormpy` and receive a scheduler (policy)

# %%
stormpy_mdp = build_stormpy_mdp(abstracted_model)

print(stormpy_mdp)

# %%
prop = stormpy.parse_properties(f"Rmax=?[Cdiscount={gamma}]")[0]  # checking for cumulative discounted reward

# Run the model checking:
result = stormpy.check_model_sparse(stormpy_mdp, prop, extract_scheduler=True)
scheduler = result.scheduler

# %% [markdown]
# ## 4. Policy evaluation
#
# Finally, we evaluate our model checked policy on the original and the abstract model.
#
# 1. We convert the `stormpy` policy to a `verigym.StormpyPolicy`
# 2. We apply it to both the original and the abstract model.

# %%
# policy mapped to the original model
verigym_policy = verigym.StormpyPolicy(scheduler, abstracted_model.abstraction_map, stormpy_mdp)
# policy on the abstract model
identity_mapper = verigym.AbstractionMapper.initialize_identity_mapper(abstracted_model.observation_space, abstracted_model.action_space)
verigym_policy_on_abstracted = verigym.StormpyPolicy(
    policy=scheduler, 
    abstraction_mapper=identity_mapper,
    mdp=stormpy_mdp
)

n_simulation_steps = int(10e4)

# %%
# simulation in the original (continuous) model
trajectories_original = generative_model.simulate(
    policy=verigym_policy,
    n_steps=n_simulation_steps,
    seed = SEED
)
mean_rewards_original = get_mean_reward_from_trajectories(trajectories_original)

# %%
# verify the policy: (2) policy performance on abstracted model
trajectories_abstracted = abstracted_model.simulate(
    policy=verigym_policy_on_abstracted,
    n_steps=n_simulation_steps,
    seed = SEED
)
mean_rewards_abstracted = get_mean_reward_from_trajectories(trajectories_abstracted)

# %%
print("Comparing the mean cumulative reward of the abstract policy:")
print("\tOriginal environment: ", mean_rewards_original)
print("\tAbstract environment: ", mean_rewards_abstracted)

# %% [markdown]
# ## Visualize abstract-policy on the original environment

# %%
visualize = False # set to True for visualization

if visualize:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML

    # Roll out one episode of verigym_policy, capturing frames via gym's rgb_array rendering
    render_env = gym.make(GYM_STRING, render_mode="rgb_array")
    obs, info = render_env.reset()
    frames = [render_env.render()]

    terminated = truncated = False
    while not (terminated or truncated):
        action = verigym_policy.get_action(obs)
        obs, reward, terminated, truncated, info = render_env.step(action)
        frames.append(render_env.render())
    render_env.close()

    print(f"Episode length: {len(frames)} steps")

    fig, ax = plt.subplots()
    ax.axis("off")
    img = ax.imshow(frames[0])


    def _update(i):
        img.set_data(frames[i])
        return (img,)


    anim = animation.FuncAnimation(fig, _update, frames=len(frames), interval=40, blit=True)
    plt.close(fig)
    HTML(anim.to_jshtml())

# %% [markdown]
# ## Extension: IMDP learning instead of MDP learning

# %%
explicit_interval_model = verigym.create_abstraction(
    original_env=generative_model,
    abstraction_mapper=abstraction_mapper,
    exploration_policy=EntropyLearningPolicy(generative_model, abstraction_mapper, seed=SEED),
    num_steps=(10 if QUICK else int(1e3)),
    n_iterations=50,
    multithreading=False,
    intervals=True,
    assume_iid=False,
    seed = SEED
)

stormpy_imdp = build_stormpy_imdp(explicit_interval_model, 
                                  use_reward_uncertainty=False)
print(stormpy_imdp)

# %% [markdown]
# ## Extension: Comparing to RL
# To determine if our abstraction is any good, we compare it's performance against that of an RL agent trained on the original environment, as well as on the abstraction. We use a PPO agent from stable-baselines, with minor alterations to the default settings to push the agent towards more exploration.

# %%
outpath_original = f"{RL_OUTPATH}/original/"
os.makedirs(outpath_original, exist_ok=True)
TRAIN_STEPS = int(2e4)
train_with_sb3_ppo(generative_model, generative_model, outpath_original, n_steps=(10 if QUICK else TRAIN_STEPS), eval_freq=2000, use_sde=True, discount=gamma, seed=SEED)


# %%
outpath_abstracted = f"{RL_OUTPATH}/abstracted/"
os.makedirs(outpath_abstracted, exist_ok=True)
abstracted_model_timelimit = TimeLimit(abstracted_model, max_episode_steps=999)
train_with_sb3_ppo(abstracted_model_timelimit, abstracted_model_timelimit, outpath_abstracted, n_steps=(10 if QUICK else TRAIN_STEPS), eval_freq=2000, discount=gamma, seed=SEED)

# %%
plot_logged_data([outpath_original + "log/evaluations.npz", outpath_abstracted + "log/evaluations.npz", 
                  ], labels=["PPO (orig.)", "PPO (abstr.)", 
                             ], key="results",
                 title="PPO training development", path_output=f"{PLOT_PATH}/ppo_training_curve.pdf")

# %%
rl_model_original = sb3.PPO.load(outpath_original + "ppo_model")
rl_model_abstracted = sb3.PPO.load(outpath_abstracted + "ppo_model")

rl_policy_original = SB3Policy(rl_model_original, generative_model, deterministic=True)
trajectories_rl_original = generative_model.simulate(policy=rl_policy_original, n_steps=(10 if QUICK else n_simulation_steps), seed=SEED)
mean_rewards_rl_original = get_mean_reward_from_trajectories(trajectories_rl_original)

rl_policy_abstracted = SB3Policy(rl_model_abstracted, generative_model, deterministic=True)
trajectories_rl_abstracted = generative_model.simulate(policy=rl_policy_abstracted, n_steps=(10 if QUICK else n_simulation_steps), seed=SEED)
mean_rewards_rl_abstracted = get_mean_reward_from_trajectories(trajectories_rl_abstracted)

# %%
episode_returns = lambda trajs: [float(np.sum([t[2] for t in traj])) for traj in trajs]
print("Mean cumulative reward on the original environment:")
print("\tAbstraction policy (model checked): ", mean_rewards_original)
print("\tPPO policy:                         ", mean_rewards_rl_original)
print("\tPPO policy (abstracted):            ", mean_rewards_rl_abstracted)
plot_compare_policy_eval([episode_returns(trajectories_original), episode_returns(trajectories_rl_original), episode_returns(trajectories_rl_abstracted)],
                         labels=["Abstraction + PMC", "PPO", "Abstraction + PPO", ], title="Policy performance of PMC vs. (abstracted) DRL",
                         path_output=f"{PLOT_PATH}/comparing_abstraction_policies_with_abstracted_ppo.pdf")


# %%
plot_compare_policy_eval([episode_returns(trajectories_original), episode_returns(trajectories_rl_original)],
                         labels=["Abstraction + PMC", "PPO"], title="Policy performance of PMC vs. DRL", path_output=f"{PLOT_PATH}/comparing_abstraction_policies.pdf")

