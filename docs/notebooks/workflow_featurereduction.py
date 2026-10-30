# %% [markdown]
# # Workflow: Feature Selection

# %% [markdown]
# **What is the motivation?**
#
# Many Reinforcement Learning environments have high-dimensional observation spaces. With increasing state space dimensionality, learning an effective policy typically requires more and more data and time.
#
# **What do we show here?**
#
# Using `VeriGym`, we can easily specify which state features we want to exclude from our enviroment and automatically construct a reduced environment, including an abstraction mapping between the original and reduced observation spaces.
#
# We use the Mujoco `"Ant-v5"` robotics environment from [`gymnasium`](https://gymnasium.farama.org/environments/mujoco/ant/) as an example. Its observation space has 105 dimension. The first 27 correspond to the torso pose and joint positions and velocities. The latter 78 dimensions correspond to the center of mass based external forces on the pody parts.
#
# In this workflow notebook, we reduce the observation space to the first 27 dimensions by masking the remaining features. We then train a SAC agent using `stable_baselines3` on both the full and the reduced observation spaces.
# We compare:
# - the training performance on the original environment versus the reduced environment
# - the performance of the policy trained on the reduced environment when evaluated on the original environment.

# %%
import os
# checks if env var is set, defaults to True
QUICK = os.environ.get("QUICK", "1") != "0" 
# uncomment line below for full run
#QUICK = False 

# %% [markdown]
# ## Imports and helpers

# %%
# Imports
import gymnasium as gym
import stable_baselines3 as sb3
import os
import matplotlib.pyplot as plt

# VeriGym
import verigym
import verigym.abstraction.feature_selection as fs
from verigym.frameworks.stable_baselines3.policy import SB3Policy

from workflow_utils.utils import plot_logged_data, run_eval_episodes, train_with_sb3_sac, set_seeds
from workflow_utils.utils import OUTPATH_WORKFLOWS as OUTPATH

from workflow_utils import plot_style

# Paths etc.
GYM_STRING = "Ant-v5"
SUBPATH = "feature_reduction"
RL_OUTPATH = f"{OUTPATH}/{SUBPATH}/sac/{GYM_STRING}"
PLOT_PATH = f"{OUTPATH}/figures/{SUBPATH}"
os.makedirs(PLOT_PATH, exist_ok=True)
SEED=0

# %% [markdown]
# ## Actual workflow

# %% [markdown]
# First, we load the standard `"Ant-v5"` environment into `VeriGym` using `verigym.GenerativeEnv.from_gymnasium`. 
# This maintains all functionality of the original `gym` environment, and adds compatibility with additional `VeriGym` features.

# %%
ant_env = verigym.GenerativeEnv.from_gymnasium(
    gym.make(GYM_STRING, render_mode="rgb_array")
)
set_seeds(SEED, ant_env)

# %% [markdown]
# Then, we specify the features to reduce (here: the 78 features that are not related to position or velocity). 
# Following the description of `Ant`'s observation space, the first 13 elements refer to the positions of the robot's body parts, and the next 14 elements to the velocities of these body parts.
# The latter 78 express external forces.
# Intuitively, if the robot should only learn to walk, could it be enough to use first 27 then?
#
# We try this using `VeriGym`'s feature selection with masking to remove the selected features.

# %%
reduce_features = [i for i in range(27, 105)]
fr_masked_env, abstraction_map = fs.state_feature_selection(original_env=ant_env, method="masking",
                                 reduce_indices=reduce_features)
set_seeds(SEED, fr_masked_env)

# %% [markdown]
# For reference, we compare the original and reduced environments' action and observation spaces, as well as types.
#
# Notice that the observation space dimensionality changes. Both environments' types are `VeriGym` native and `gym` compatible. Action spaces remain the same.

# %%
print("Original environment:")
print("type:", type(ant_env))
print("observation space: ", ant_env.observation_space)
print("action space: ", ant_env.action_space)
print()
print("Reduced environment:")
print("type:", type(fr_masked_env))
print("observation space: ", fr_masked_env.observation_space)
print("action space: ", fr_masked_env.action_space)

# %% [markdown]
# ## Training code
#
# We train models using the `stable_baselines3` implementation of `SAC` for `1.000.000` timesteps.
#
# Note: Training took ca 100min per run on a 2023 MacBook Pro with 18GB RAM and M3 Chip.

# %%
STEPS = 1_000_000 if not QUICK else 100

# %%
if not os.path.exists(f"{RL_OUTPATH}/full_env/sac_model.zip"):
    print("Running training on the original environment.")
    os.makedirs(f"{RL_OUTPATH}/full_env/", exist_ok=True)
    train_with_sb3_sac(ant_env, f"{RL_OUTPATH}/full_env/", n_steps=STEPS)
else:
    print("Proceeding with pre-trained model.")

# %%
if not os.path.exists(f"{RL_OUTPATH}/reduced_env/sac_model.zip"):
    print("Running training on the original environment.")
    os.makedirs(f"{RL_OUTPATH}/reduced_env/", exist_ok=True)
    train_with_sb3_sac(fr_masked_env, f"{RL_OUTPATH}/reduced_env/", n_steps=STEPS)
else:
    print("Proceeding with pre-trained model.")

# %% [markdown]
# ## Evaluation
#
# Using the pre-trained models, we compare the following aspects: 
# 1. Training behavior (reward and episode length) between the agents trained on the environment with 105 dimensions (`full_model`) versus the reduced one (`reduced_model`).
# 2. Policy transfer by mapping the policy trained on the reduced environment back to the original observation space using `VeriGym`'s `SB3Policy` wrapper and the automatically generated `AbstractionMapper`.
#

# %%
full_model = sb3.SAC.load(f"{RL_OUTPATH}/full_env/sac_model")
reduced_model = sb3.SAC.load(f"{RL_OUTPATH}/reduced_env/sac_model")

# %%
plot_logged_data([f"{RL_OUTPATH}/full_env/log/evaluations.npz", f"{RL_OUTPATH}/reduced_env/log/evaluations.npz"],
                    labels=["All features (105)", "Reduced features (27)"], key="results",
                    title="Training results using SAC", running_mean=50,
                    path_output=f"{PLOT_PATH}/sac_training_rewards.pdf")

plt.clf()

plot_logged_data([f"{RL_OUTPATH}/full_env/log/evaluations.npz", f"{RL_OUTPATH}/reduced_env/log/evaluations.npz"],
                    labels=["All features (105)", "Reduced features (27)"], key="ep_lengths",
                    title="Training ep_lengths using SAC", running_mean=50,
                    path_output=f"{PLOT_PATH}/sac_training_ep_lengths.pdf")

# %% [markdown]
# Now, we compare policy performance by wrapping the stable baselines policies in to a `verigym.SB3Policy`. `full_policy` wraps the policy learned on the full observation space. `reduced_policy` wraps the policy learned on the reduced observation space.
#
# Additionally, we can use the `abstraction_map` we got when we applied the feature reduction to have `VeriGym` map the policy from the reduced feature space to the original full feature space - we do that in `reduced_on_full_policy`.

# %%
full_policy = SB3Policy(full_model, ant_env)
reduced_policy = SB3Policy(reduced_model, fr_masked_env)
reduced_on_full_policy = SB3Policy(reduced_model, ant_env, abstraction_map)

# %% [markdown]
# We now evaluate the full policy and the reduced policy on the original full environment.

# %%
full_eval = run_eval_episodes(ant_env, full_policy)
reduced_on_full_eval = run_eval_episodes(ant_env, reduced_on_full_policy)
reduced_eval = run_eval_episodes(fr_masked_env, reduced_policy)

# %%
rewards = [full_eval, reduced_on_full_eval, reduced_eval]
labels = ["T:105dim\nE:105dim", "T:27dim\nE:105dim", "T:27dim\nE:27dim"]

plt.title(f"Reward in evaluation for different agent/environment configurations")
plt.boxplot(
    rewards,
    tick_labels=labels,
    showmeans=True,
    showfliers=False
)
#plt.legend()
plt.xlim(0.5, len(rewards)+0.5)
plt.savefig(f"{PLOT_PATH}/eval_rewards.pdf", pad_inches=0, bbox_inches="tight")
plt.show()

# %%
