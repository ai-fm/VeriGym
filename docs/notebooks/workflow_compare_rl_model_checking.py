# %% [markdown]
# # Workflow: Benchmark your Reinforcement Learning policy against a model checked optimal policy
#
# **What is the motivation?**
#
# Say you want to evaluate how a Reinforcement Learning (RL) algorithm performs compared to the actual optimal policy from model checking. 
# Plus, you have an interesting model given as a prism file, or an MDP built in `stormpy`.
#
# With `VeriGym`, you can easily load the MDP into an `ExplicitEnv`, and train a policy on it with your custom RL algorithm.
# Additionally, you can use our `stormpy` interfacing to easily model check and generate an optimal solution and associated policy given the fully specified model.
# Then, you can observe how close your algorithm gets to the theoretically optimal value, and how your algorithm's learned policy behaves to an optimal policy from `stormpy`.
#
# **What do we show here?**
# In this workflow, we benchmark a standard RL algorithm against model checked ground-truth results with the following steps:
# 1. We load a `PRISM` program describing an MDP into `VeriGym` using `stormpy`.
# 2. We train a policy on the resulting `FrameworkExplicitEnv` using `stable_baselines3` implementation of `DQN`.
# 3. We model check the MDP in `stormpy`.
# 4. We compare the reward obtained by the learned policy in simulation with the reward of the optimal policy in simulation using the `PolicyClass` module.
# 5. We check the value of the learned policy by building and checking a Discrete-Time Markov Chain (DTMC) using `stormpy`.
#

# %% [markdown]
# ## Imports and helpers

# %%
import os
# checks if env var is set, defaults to True
QUICK = os.environ.get("QUICK", "1") != "0" 
# uncomment line below for full run
# QUICK = False 

# %%
# Imports
import os
import stable_baselines3 as sb3
from gymnasium.wrappers import TimeLimit
import json

# VeriGym
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv
from verigym.frameworks.stormpy.stormpy_utils import load_stormpy_model
from verigym.supplements.modelchecking import get_policy_from_stormpy, check_policy_value_in_stormpy
from verigym.frameworks.stable_baselines3.policy import SB3Policy
from workflow_utils.utils import OUTPATH_WORKFLOWS as OUTPATH, train_with_sb3_dqn, plot_logged_data, run_eval_episodes, set_seeds
from workflow_utils import plot_style

# Paths, global variables, etc.

GAMMA = 0.95
SEED = 0

SUBPATH = "rl_mc_files"
RL_OUTPATH = f"{OUTPATH}/{SUBPATH}/dqn"
EVAL_OUTPATH = f"{OUTPATH}/{SUBPATH}/eval"
PLOT_PATH = f"{OUTPATH}/figures/{SUBPATH}"
os.makedirs(PLOT_PATH, exist_ok=True)
os.makedirs(EVAL_OUTPATH, exist_ok=True)
prism_path = "workflow_utils/prism_files/"
benchmarks = {
    "Resource Gathering": {
        "file": prism_path + "resource-gathering.v2.pm",
        "constants": "B=200,GOLD_TO_COLLECT=15,GEM_TO_COLLECT=15",
        "property": 'Rmax=?' + f" [Cdiscount={GAMMA}]",
        "maxsteps": 200
    },
    "Consensus":
    {
        "file": prism_path + "consensus.4.v1.prism",
        "constants": "K=2",
        "property": 'Rmax=?' + f" [Cdiscount={GAMMA}]",
        "maxsteps": 1100
    },
    "Refuel": {
        "file": prism_path + "refuel-mdp.prism",
        "constants": "N=25,ENERGY=12",
        "property": 'Rmax=?' + f" [Cdiscount={GAMMA}]",
        "maxsteps": 100
    }
}

# %% [markdown]
# ## Actual workflow

# %% [markdown]
# We are using three benchmarks from qcomp and the Revised Practitioners Guide here.
# For all tasks, we are interested in maximizing the reward before terminating.
#
# - **Resource Gathering** is a grid-world game, in which the goal is to collect as much gold and as many gems as possible. 
# For our experiments here, we focus on collecting gold.
# - **Consensus** Is an internet protocol. We focus on running for a maximal number of steps.
# - **Refuel** Is a grid-world task where an autonomous vehicle must reach a goal without running out of fuel. 
# We focus on maximizing the time before running out of fuel.
#
# We start our worklow by loading each MDP into `stormpy`, and creating a `FrameworkExplicitEnv` from it.
# (The `eval_env` is identical and used for the DRL training evaluation.)
# For the model checking part, we write the reward-maximizing objective by querying the maximal reward (`Rmax=?`), cumulative (`C`), with discount factor $\gamma=0.95$ (`discount=0.95`), using the same factor as for DQN training.
#
# We then train our DRL model using the `stable_baselines3` implementation of DQN. 

# %%
print("Loading Resource Gathering")
mdp_rg = load_stormpy_model(benchmarks["Resource Gathering"]["file"], 
                            constants=benchmarks["Resource Gathering"]["constants"], 
                            property_strs=benchmarks["Resource Gathering"]["property"])
print(mdp_rg)

env_rg = FrameworkExplicitEnv.from_stormpy(mdp_rg)
env_rg = TimeLimit(env_rg, max_episode_steps=benchmarks["Resource Gathering"]["maxsteps"])
eval_env_rg = FrameworkExplicitEnv.from_stormpy(mdp_rg)
eval_env_rg = TimeLimit(eval_env_rg, max_episode_steps=benchmarks["Resource Gathering"]["maxsteps"])
print()

# %%
print("Loading Consensus")
mdp_c = load_stormpy_model(benchmarks["Consensus"]["file"], 
                            constants=benchmarks["Consensus"]["constants"], 
                            property_strs=benchmarks["Consensus"]["property"])
print(mdp_c)

env_c = FrameworkExplicitEnv.from_stormpy(mdp_c)
env_c = TimeLimit(env_c, max_episode_steps=benchmarks["Consensus"]["maxsteps"])
eval_env_c = FrameworkExplicitEnv.from_stormpy(mdp_c)
eval_env_c = TimeLimit(eval_env_c, max_episode_steps=benchmarks["Consensus"]["maxsteps"])
print()

# %%
print("Loading Refuel")
mdp_rf = load_stormpy_model(benchmarks["Refuel"]["file"], 
                            constants=benchmarks["Refuel"]["constants"], 
                            property_strs=benchmarks["Refuel"]["property"])
print(mdp_rf)

env_rf = FrameworkExplicitEnv.from_stormpy(mdp_rf)
env_rf = TimeLimit(env_rf, max_episode_steps=benchmarks["Refuel"]["maxsteps"])
eval_env_rf = FrameworkExplicitEnv.from_stormpy(mdp_rf)
eval_env_rf = TimeLimit(eval_env_rf, max_episode_steps=benchmarks["Refuel"]["maxsteps"])
set_seeds(SEED, env_rf)
set_seeds(SEED, eval_env_rf)
print()

# %% [markdown]
# ### Toggle here to view or hide the training output...

# %%
STEPS = (200 if QUICK else 50000)

# %%
for env_str, env, eval_env in zip(["resource", "consensus", "refuel"], [env_rg, env_c, env_rf], [eval_env_rg, eval_env_c, eval_env_rf]):
    if not os.path.exists(f"{RL_OUTPATH}/{env_str}/dqn_model.zip"):
        print(f"Training on {env_str}")
        os.makedirs(f"{RL_OUTPATH}/{env_str}/", exist_ok=True)
        train_with_sb3_dqn(env, eval_env, f"{RL_OUTPATH}/{env_str}/", n_steps=STEPS, seed=SEED)
    else:
        print("Proceeding with pre-trained model.")

# %% [markdown]
# ### Comparing the policies
#
# We now load our trained DQN policy and look at the reward development during training.
# We train for 50000 steps.

# %%
model_rg = sb3.DQN.load(f"{RL_OUTPATH}/resource/dqn_model")
model_c = sb3.DQN.load(f"{RL_OUTPATH}/consensus/dqn_model")
model_rf = sb3.DQN.load(f"{RL_OUTPATH}/refuel/dqn_model")

# %%
plot_logged_data([f"{RL_OUTPATH}/consensus/log/evaluations.npz",
                  f"{RL_OUTPATH}/refuel/log/evaluations.npz",
                  f"{RL_OUTPATH}/resource/log/evaluations.npz",], 
                 labels=["Consensus", "Refuel", "Resource Gathering",], key="results",
                 title="Training results for DQN on different benchmarks",
                 path_output=f"{PLOT_PATH}/dqn_training_curves.pdf",
                 running_mean=5)

# %% [markdown]
# Now, we load our DRL-based policy as a `SB3Policy`, and additionally model check the MDP that underlies our environment using the reward-maximizing property. 
# The model checked property is returned as a `StormpyPolicy` object.
# Both policy classes can now be deployed on our environment.

# %%
sp_policy_rg = get_policy_from_stormpy(env_rg, benchmarks["Resource Gathering"]["property"])
rl_policy_rg = SB3Policy(model_rg, env_rg, deterministic=True)

sp_policy_c = get_policy_from_stormpy(env_c, benchmarks["Consensus"]["property"])
rl_policy_c = SB3Policy(model_c, env_c, deterministic=True)

sp_policy_rf = get_policy_from_stormpy(env_rf, benchmarks["Refuel"]["property"])
rl_policy_rf = SB3Policy(model_rf, env_rf, deterministic=True)

# %% [markdown]
# For that, we collect some evaluation episodes using a utility function.

# %%
iterations = (1 if QUICK else 100)
rl_eval_rg = run_eval_episodes(env_rg, rl_policy_rg, seed=SEED, iterations=iterations)
sp_eval_rg = run_eval_episodes(env_rg, sp_policy_rg, seed=SEED, iterations=iterations)

rl_eval_c = run_eval_episodes(env_c, rl_policy_c, seed=SEED, iterations=iterations)
sp_eval_c = run_eval_episodes(env_c, sp_policy_c, seed=SEED, iterations=iterations)

rl_eval_rf = run_eval_episodes(env_rf, rl_policy_rf, seed=SEED, iterations=iterations)
sp_eval_rf = run_eval_episodes(env_rf, sp_policy_rf, seed=SEED, iterations=iterations)

eval_summary = {
    "DQN": {
        "resourcegathering": rl_eval_rg,
        "consensus":rl_eval_c,
        "refuel":rl_eval_rf
    },
    "STORM": {
        "resourcegathering": sp_eval_rg,
        "consensus":sp_eval_c,
        "refuel":sp_eval_rf
    }
}
with open(os.path.join(EVAL_OUTPATH, "eval_statistical.json"), "w") as f:
    json.dump(eval_summary, f, indent=2, default=float)

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

plt.title("Comparing DQN and model checked policies")

bp = plt.boxplot(
    [
        [r/max(sp_eval_rg) for r in rl_eval_rg], 
        [r/max(sp_eval_rg) for r in sp_eval_rg], 
        [r/max(sp_eval_c) for r in rl_eval_c], 
        [r/max(sp_eval_c) for r in sp_eval_c], 
        [r/max(sp_eval_rf) for r in rl_eval_rf], 
        [r/max(sp_eval_rf) for r in sp_eval_rf]
     ],
     positions=[1.2, 1.8, 3.2, 3.8, 5.2, 5.8],
     patch_artist=True
)

for i, box in enumerate(bp["boxes"]):
    if i % 2 == 0:
        box.set_facecolor(to_rgba(plot_style.PURPLE, alpha=0.4))
        box.set_edgecolor(plot_style.PURPLE)
    else:
        box.set_facecolor(to_rgba(plot_style.MAGENTA, alpha=0.4))
        box.set_edgecolor(plot_style.MAGENTA)


lh = [
    Patch(edgecolor=plot_style.PURPLE, facecolor=to_rgba(plot_style.PURPLE, alpha=0.4),
          label="DQN"),
    Patch(edgecolor=plot_style.MAGENTA, facecolor=to_rgba(plot_style.MAGENTA, alpha=0.4),
          label="PMC")
]

plt.legend(handles=lh)
plt.xticks([1.5, 3.5, 5.5],
           labels=["Resource Gathering", "Consensus", "Refuel"])

plt.ylabel("Reward")
plt.savefig(f"{PLOT_PATH}/dqn_vs_pmc_rewards.pdf", pad_inches=0, bbox_inches="tight")
plt.show()

# %% [markdown]
# We observe that the model checked policy performs much better than the RL trained policy. It looks like our few training steps were not sufficient yet, we should continue to search for better hyperparameters.

# %% [markdown]
# In addition to comparing the policies in evaluation, we can also model check our trained RL policy.
# We can do that using a utility function in `VeriGym`. 
# Under the hood, the following happens:
# - We export the underlying MDP from our environment and combine it with the RL policy to build a Discrete Time Markov Chain (DMTC).
# - Then, we model check the DTMC in stormpy, and get a value vector with a value per state.
#
# Finally, we can compare the value of that DTMC with the value in the original MDP.
#
# The result again shows us, that we need to train our RL agent more!

# %%
import importlib
import verigym.supplements.modelchecking as mc
import verigym.frameworks.stormpy.stormpy_utils as su
importlib.reload(su)
importlib.reload(mc)

sp_value_rg = mc.check_policy_value_in_stormpy(env_rg, sp_policy_rg, benchmarks["Resource Gathering"]["property"], only_initial_states=True)
rl_value_rg = mc.check_policy_value_in_stormpy(env_rg, rl_policy_rg, benchmarks["Resource Gathering"]["property"], only_initial_states=True)

sp_value_c = mc.check_policy_value_in_stormpy(env_c, sp_policy_c, benchmarks["Consensus"]["property"], only_initial_states=True)
rl_value_c = mc.check_policy_value_in_stormpy(env_c, rl_policy_c, benchmarks["Consensus"]["property"], only_initial_states=True)

sp_value_rf = mc.check_policy_value_in_stormpy(env_rf, sp_policy_rf, benchmarks["Refuel"]["property"], only_initial_states=True)
rl_value_rf = mc.check_policy_value_in_stormpy(env_rf, rl_policy_rf, benchmarks["Refuel"]["property"], only_initial_states=True)


# save data for future use
eval_summary_dtmc = {
    "DQN": {
        "resourcegathering": rl_value_rg,
        "consensus":rl_value_c,
        "refuel":rl_value_rf
    },
    "STORM": {
        "resourcegathering": sp_value_rg,
        "consensus":sp_value_c,
        "refuel":sp_value_rf
    }
}
with open(os.path.join(EVAL_OUTPATH, "eval_dtmc.json"), "w") as f:
    json.dump(eval_summary_dtmc, f, indent=2, default=float)

# %%
print("Resource Gathering")
print("Comparing values in the initial state:")
print("Initial state value based on the DTMC from StormpyPolicy:", sp_value_rg)
print("Initial state value based on the DTMC from SB3Policy:", rl_value_rg)

print("Consensus")
print("Comparing values in the initial state:")
print("Initial state value based on the DTMC from StormpyPolicy:", sp_value_c)
print("Initial state value based on the DTMC from SB3Policy:", rl_value_c)

print("Refuel")
print("Comparing values in the initial state:")
print("Initial state value based on the DTMC from StormpyPolicy:", sp_value_rf)
print("Initial state value based on the DTMC from SB3Policy:", rl_value_rf)
