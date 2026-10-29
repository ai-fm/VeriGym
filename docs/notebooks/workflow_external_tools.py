# %% [markdown]
# # Workflow: External Tools (Julia)
#
# One of the main goals of VeriGym is to allow integration between a variety of existing tools and frameworks.
# In this workflow, we show off the intergration with one such external framework: the `POMDPs.jl` format used to define and solve (PO)MDPs in the `Julia` programming language.
#
# **What do we show here?**
# In this workflow, we benchmark a standard RL algorithm against model checked ground-truth results with the following steps:
# 1. We construct an abstraction for the Gymnasium environment `X` and export it to UMB format.
# 2. In `Julia`, we import the model, solve it via an existing solver, and export a policy using `PRISM`s action list format.
# 3. We import the policy into VeriGym, and evaluate the policy on both the abstracted and original model.
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
import gymnasium as gym
import os
import subprocess
import stormpy

# VeriGym
import verigym
from verigym.policy.randomized import RandomizedPolicy
from verigym.policy.qvalue import EntropyLearningPolicy
from verigym.io.exporter import export_to_umb, export_to_drn
from verigym.abstraction.abstractionmapper import AbstractionMapper, linspace_mapper
from verigym.frameworks.prism.prismpolicy import PrismPolicy
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp

# Helpers
from workflow_utils.utils import OUTPATH_WORKFLOWS as OUTPATH, run_eval_episodes, plot_compare_policy_eval
from workflow_utils import plot_style

# Paths etc.
SUBPATH = "ext_tools_files/"
dir = f"{OUTPATH}/{SUBPATH}"
PLOT_PATH = f"{OUTPATH}/figures/{SUBPATH}"
os.makedirs(PLOT_PATH, exist_ok=True)
os.makedirs(dir, exist_ok=True)
SEED = 1285821

# %% [markdown]
# ## 1 - Construct and export abstraction
# To start of, we pick an environment that we want to make an abstraction for. To keep things simple, we use the standard `gym` environment *CartPole*, and use a simple box discretization. For more details on how to construct and discretize models, see the other workflows.

# %%
gamma = 0.95  # discount factor

original_model = verigym.GenerativeEnv.from_gymnasium(gym.make("CartPole-v1"))

abstraction_mapper = linspace_mapper(
        env=original_model, 
        n_bins_states=11, 
        n_bins_actions=1
    )

abstracted_model = verigym.create_abstraction(
    original_model, 
    abstraction_mapper,
    exploration_policy=EntropyLearningPolicy(original_model, abstraction_mapper, seed=SEED),
    num_steps=(10 if QUICK else int(1e4)),
    n_iterations = 50,
    seed=SEED,
    multithreading=False
)

# %% [markdown]
# Next, to use any external tool, we need to be able to *write* our constructed model to a file. VeriGym makes use of existing `stormpy` functionality to supports writing to the following formats:
# * [Unified Markov Binary (UMB)](https://arxiv.org/abs/2606.17811);
# * [Direct Encoding (DRN)](https://www.stormchecker.org/documentation/background/drn.html).
#
# We chose DRN format, in which case exporting looks as follows:

# %%
export_to_umb(abstracted_model, dir + "model.umb")
export_to_drn(abstracted_model, dir + "model.drn")

# %% [markdown]
# ## 2 - External Policy Computation
#
# Since our model is now exported into a standard format, we can use external solvers to compute a policy. For this example, we use `MCTS.jl`, which is a `Julia` program that requires the model in `POMDPs.jl` format. VeriGym includes a number of `Julia` scripts that simplify this process. More precisely, this includes code to import and export MDPs in UMB format, and export policies in [Prisms explicit action list format](https://prismmodelchecker.org/manual/RunningPRISM/Strategies). Using this functionality, we can run a simple `Julia` script to compute and export a policy as follows:
#
#
# ```julia
# using POMDPs
# using MCTS
#
# # Import our code
# include("../../../src/verigym/frameworks/juliapomdp/POMDPs_UMB.jl")
# import .POMDPs_UMB
#
# # Read the model
# dir = "./working_directory/"
# input, output = dir * "model.umb", dir * "policy.txt"
# model = POMDPs_UMB.read_umb(input; discount=0.95)
#
# # Solve using POMDPs.jl solver
# solver = MCTSSolver()
# planner = solve(solver, model)
#
# # Export policy
# policy = POMDPs_UMB.Explicit_policy(planner, model)
# POMDPs_UMB.write_policy(output, policy)
# println("Policy computation complete!")
# ```
#
# The resulting policy file is already available, but can be re-computed by running the code below. Note that this requires you to have installed `Julia`, as well as the `POMDPs` and `MCTS` modules.

# %%
if True: # set to false to skip (e.g., if julia is not installed):
    n_iterations = (10 if QUICK else 5000)
    subprocess.run(["julia", "workflow_utils/workflow_external_tools_script.jl", "--seed", str(SEED), "--n_iterations", str(n_iterations)], check=True)

# %% [markdown]
# To use the computed policy, we use `VeriGym`'s functionality for importing policies in Prism format:

# %%
policy_path = dir + "policy.txt"

julia_policy = PrismPolicy(policy_path=policy_path, abstraction_mapper=abstraction_mapper)

# %% [markdown]
# # 3 - Policy Evaluation

# %% [markdown]
# We can now use the imported policy the way we would any other policy. As an example, we here compare it's performance agains that of a Stormpy policy:

# %%
stormpy_mdp = build_stormpy_mdp(abstracted_model)

prop = stormpy.parse_properties(f"Rmax=?[Cdiscount={gamma}]")[
    0
]  # checking for cumulative discounted reward

# Run the model checking:
result = stormpy.check_model_sparse(stormpy_mdp, prop, extract_scheduler=True)
stormpy_policy = verigym.StormpyPolicy(result.scheduler, abstracted_model.abstraction_map, stormpy_mdp)

# %%
jl_eval = run_eval_episodes(original_model, julia_policy)
sp_eval = run_eval_episodes(original_model, stormpy_policy)

plot_compare_policy_eval([jl_eval, sp_eval], ["MCTS (Julia)", "PMC (Storm)"],
                         path_output=f"{PLOT_PATH}/mcts_vs_model_checking.pdf")
