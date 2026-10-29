# %% [markdown]
# # Technical Evaluation: How does memory consumption of VeriGym's datastructures for explicit models compare to that of other tools?
#
# We compare the memory consumption of `VeriGym`'s `TransitionFunction` and `FrameworkExplicitEnv` with the memory consumption of `stormpy.SparseTransitionMatrix` and `stormpy.storage.SparseMdp`.
# Further, we compare the sparse transition function representations from `VeriGym` and `stormpy` with non-sparse representation using `numpy`-arrays.
#
# The goal is to 1) see whether/how much memory overhead `VeriGym` introduces over `stormpy`, and 2) compare the extend to which our sparse representation reduces memory consumption as opposed to non-sparse representations.
#
# We use four PRISM MDP benchmarks from the Quantitative Verification Benchmark Set [(qcomp)](https://qcomp.org/benchmarks/) at different sizes. 
# Specifically, we consider two grid-worlds (Pacman and Resource Gathering), and two internet protocols (Consensus and Zeroconf) to cover models with different structures. 
# Models have states spaces $S$ with $400 \leq |S| \leq 1.000.000$.

# %% [markdown]
# ## Imports and Helpers

# %%
import os
# checks if env var is set, defaults to True
QUICK = os.environ.get("QUICK", "1") != "0" 
# uncomment line below for full run
# QUICK = False 

# %%
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy

from verigym.frameworks.stormpy.stormpy_utils import load_stormpy_model
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv

import workflow_utils.plot_style as plot_style
from workflow_utils.utils import OUTPATH_TECH_EVAL

FIGPATH = f"{OUTPATH_TECH_EVAL}/memory_eval"
RESULTSPATH = f"{OUTPATH_TECH_EVAL}/memory_eval/logs"


# %%
def recursive_sizeof(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)

    if obj_id in seen:
        return 0

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(
            recursive_sizeof(key, seen) + 
            recursive_sizeof(value, seen)
            for key, value in obj.items()
        )

    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(recursive_sizeof(item, seen) for item in obj)
    else:
        # Attributes stored in __dict__
        if hasattr(obj, "__dict__"):
            size += recursive_sizeof(obj.__dict__, seen)
        # Attributes stored using __slots__
        slots = getattr(type(obj), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)

        for slot in slots:
            if slot in ("__dict__", "__weakref__"):
                continue
            try:
                value = getattr(obj, slot)
            except AttributeError:
                continue

            size += recursive_sizeof(value, seen)
    return size


# %%
def compute_transition_matrix_size(mdp):
    # Stormpy transition matrix is CSA sparse matrix, can compare like so
    transitions_bit = mdp.nr_transitions * 64
    choice_bits = mdp.nr_choices * 64 * 2
    state_bits = mdp.nr_states * 64

    bits = transitions_bit + choice_bits + state_bits
    return bits

def transition_function_to_np(n_states, n_actions, t_dict):
    non_sparse = np.zeros((n_states, n_actions, n_states), dtype=np.float64)

    for s in t_dict.keys():
        for a in t_dict[s].keys():
            for s_prime in t_dict[s][a].keys():
                non_sparse[s, a, s_prime] = t_dict[s][a][s_prime]

    return non_sparse

def get_csr(tr_array):
    return scipy.sparse.csr_array(tr_array.reshape(-1, tr_array.shape[-1]))

def compute_non_sparse_size(transition_matrix):
    """
    compute the size of a non sparse transition matrix
    assuming data type float64 i.e. 64b per float
    """
    nr_rows = transition_matrix.nr_rows
    nr_columns = transition_matrix.nr_columns
    return 64 * nr_rows * nr_columns

def compute_csr_size(transition_matrix):
    """
    approximate the size of a sparse SciPy CSR matrix
    assuming data type float64 for probabilities and int32 for indices and pointers, these are default values
    """
    n_rows = transition_matrix.nr_rows
    n_entries = transition_matrix.nr_entries
    return 2 * 32 * n_entries + 64 * n_rows



# %%
in_folder = "workflow_utils/prism_files/"
scripts_folder = "tech_eval_utils/"

benchmarks = [
    {
        "name": "Consensus.4",
        "file": os.path.abspath(in_folder + "consensus.4.v1.prism"),
        "constants": ["K=2", "K=4"]
    }, {
        "name": "Pacman",
        "file": in_folder + "pacman.v2.nm",
        "constants": ["MAXSTEPS=5"],
    }, {
        "name": "Resource Gathering",
        "file": in_folder + "resource-gathering.v2.pm",
        "constants": ["B=200,GOLD_TO_COLLECT=15,GEM_TO_COLLECT=15",
                      "B=400,GOLD_TO_COLLECT=30,GEM_TO_COLLECT=30",
                      "B=1300,GOLD_TO_COLLECT=100,GEM_TO_COLLECT=100"],
    }, {
        "name": "Zeroconf",
        "file": in_folder + "zeroconf.v1.prism",
        "constants": ["N=20,K=2,reset=true", "N=20,K=4,reset=true",
                      "N=20,K=2,reset=false", "N=20,K=4,reset=false",
                      "N=20,K=6,reset=false"
                      ]
    }, 
]

if QUICK:
    # If QUICK, reduce number of benchmarks
    for b in benchmarks:
        b['constants'] = b['constants'][:1]

# %% [markdown]
# ## Evaluation

# %%
MAX_NR_STATES = 4e4 if not QUICK else 1e2

transition_function_sizes = {}
env_or_mdp_sizes = {}

for benchmark in benchmarks:
    print(benchmark["name"])
    transition_function_sizes[benchmark["name"]] = {}
    env_or_mdp_sizes[benchmark["name"]] = {}
    for i, constants in enumerate(benchmark["constants"]):
        print(constants)
        mdp = load_stormpy_model(
            benchmark["file"], constants=constants
        )            
        env = FrameworkExplicitEnv.from_stormpy(mdp)

        res = subprocess.run(
            [
                sys.executable, f"{scripts_folder}/memory_eval_runner.py",
                "stormpy",
                benchmark["file"],
                "--constants", constants,
            ],
            capture_output=True,
            text=True,
            check=True
        )
        mdp_out = json.loads(res.stdout)

        res = subprocess.run(
            [
                sys.executable, f"{scripts_folder}/memory_eval_runner.py",
                "env",
                benchmark["file"],
                "--constants", constants,
            ],
            capture_output=True,
            text=True,
            check=True
        )
        env_out = json.loads(res.stdout)
        transition_function_sizes[benchmark["name"]][constants] = {
            "stormpy": compute_transition_matrix_size(mdp),
            "verigym": recursive_sizeof(env.transition_function),
            "n_states": mdp.nr_states,
            "n_transitions": mdp.nr_transitions,
        }
        transition_function_sizes[benchmark["name"]][constants]["flat"] = compute_non_sparse_size(mdp.transition_matrix)
        transition_function_sizes[benchmark["name"]][constants]["sparse"] = compute_csr_size(mdp.transition_matrix)
        env_or_mdp_sizes[benchmark["name"]][constants] = {
            "verigym,sizeoff": recursive_sizeof(env),
            "verigym,process": env_out["change"],
            "verigym,process,w.build": env_out["change_mdp"],
            "stormpy,process": mdp_out["change"],
            "n_states": mdp.nr_states,
            "n_transitions": mdp.nr_transitions,
        }
    print()

# Save the results
RESULTS_FILE = f"{RESULTSPATH}/memory_eval_results.json"
os.makedirs(RESULTSPATH, exist_ok=True)
with open(RESULTS_FILE, "w") as f:
    json.dump({"transition_function_sizes": transition_function_sizes, "env_or_mdp_sizes": env_or_mdp_sizes}, f, indent=2)

# %%
# Load the saved results (written by the experiment cell above)
with open(f"{RESULTSPATH}/memory_eval_results.json") as f:
    saved = json.load(f)
transition_function_sizes = saved["transition_function_sizes"]
env_or_mdp_sizes = saved["env_or_mdp_sizes"]

fig, axs = plt.subplots(figsize=(4,3))

# axs[0].set_title("Size of transition functions (bit)")
for i, (tb, tvariants) in enumerate(transition_function_sizes.items()):
    for j, (tc, data) in enumerate(tvariants.items()):
        axs.scatter(data["n_states"], data["stormpy"], 
                       label="stormpy" if i == 0 and j == 0 else "", 
                       color="blue")
        axs.scatter(data["n_states"], data["verigym"], 
                       label="verigym" if i == 0 and j == 0 else "", 
                       color="green" )
        if "flat" in data.keys():
            axs.scatter(data["n_states"], data["flat"], 
                        label="flat" if i == 0 and j == 0 else "", 
                        color="purple")
            axs.scatter(data["n_states"], data["sparse"],
                                       label="sparse" if i == 0 and j == 0 else "",
                                       color=plot_style.MAGENTA)
        

axs.set_yscale("log")
axs.set_xscale("log")
axs.set_xlabel("number of states")
axs.set_ylabel("transition function size (bit)")
axs.legend(frameon=True)

fig.savefig(f"{FIGPATH}/memory_eval_1a_transition_func.pdf", pad_inches=0, bbox_inches="tight")


# ------

fig, axs = plt.subplots(figsize=(4, 3))

# axs[1].set_title("Size of environment/MDP (bit)")
for i, (eb, evariants) in enumerate(env_or_mdp_sizes.items()):
    for j, (ec, data) in enumerate(evariants.items()):
        axs.scatter(data["n_states"], data["stormpy,process"], 
                       label="stormpy" if i == 0 and j == 0 else "", 
                       color="blue")
        axs.scatter(data["n_states"], data["verigym,process"], 
                       label="verigym (process)" if i == 0 and j == 0 else "", 
                       color="darkgreen")
        axs.scatter(data["n_states"], data["verigym,process,w.build"], 
                       label="verigym (process + MDP)" if i == 0 and j == 0 else "", 
                       color="lime")

axs.set_yscale("log") 
axs.set_xscale("log")
axs.legend(frameon=True)
axs.set_xlabel("number of states")
axs.set_ylabel("MDP size (bit)")

fig.savefig(f"{FIGPATH}/memory_eval_1b_size_environment.pdf", pad_inches=0, bbox_inches="tight")

plt.show()

# %%
fig, axs = plt.subplots(figsize=(4,3))

# axs.set_title("Size of transition function relative to stormpy")
for i, (tb, tvariants) in enumerate(transition_function_sizes.items()):
    for j, (tc, data) in enumerate(tvariants.items()):
        axs.scatter(data["n_states"], data["verigym"]/data["stormpy"], 
                       label="verigym" if i == 0 and j == 0 else "", 
                       color="green" )
        if "flat" in data.keys():
            axs.scatter(data["n_states"], data["flat"]/data["stormpy"], 
                        label="flat" if i == 0 and j == 0 else "", 
                        color="purple")
            axs.scatter(data["n_states"], data["sparse"]/data["stormpy"],
                           label="sparse" if i == 0 and j == 0 else "",
                           color=plot_style.MAGENTA)

axs.set_yscale("log")
axs.set_xscale("log")
axs.set_xlabel("number of states")
axs.set_ylabel("relative size increase transition function\n(verigym/stormpy)")
axs.grid(axis='y')
axs.legend(frameon=True)

fig.savefig(f"{FIGPATH}/memory_eval_2a_transition_func.pdf", pad_inches=0, bbox_inches="tight")


# ----


fig, axs = plt.subplots(figsize=(4,3))
# axs.set_title("Size of environment relative to stormpy process")
for i, (eb, evariants) in enumerate(env_or_mdp_sizes.items()):
    for j, (ec, data) in enumerate(evariants.items()):
        axs.scatter(data["n_states"], data["verigym,process"]/data["stormpy,process"], 
                       label="verigym (process)" if i == 0 and j == 0 else "", 
                       color="darkgreen")
        axs.scatter(data["n_states"], data["verigym,process,w.build"]/data["stormpy,process"], 
                       label="verigym (process + MDP)" if i == 0 and j == 0 else "", 
                       color="lime")

axs.set_xscale("log")
axs.set_xlabel("number of states")
axs.set_ylabel("environment relative size increase\n(verigym/stormpy)")

axs.legend()
fig.savefig(f"{FIGPATH}/memory_eval_2b_environment_size.pdf", pad_inches=0, bbox_inches="tight")

plt.show()

# %%
fig, axs = plt.subplots(figsize=(4,3))

# axs.set_title("Size of transition function relative to stormpy")
for i, (tb, tvariants) in enumerate(transition_function_sizes.items()):
    for j, (tc, data) in enumerate(tvariants.items()):
        axs.scatter(data["n_transitions"], data["verigym"]/data["stormpy"], 
                       label="verigym" if i == 0 and j == 0 else "", 
                       color="green" )
        if "flat" in data.keys():
            axs.scatter(data["n_transitions"], data["flat"]/data["stormpy"], 
                        label="flat" if i == 0 and j == 0 else "", 
                        color="purple")
            axs.scatter(data["n_transitions"], data["sparse"]/data["stormpy"],
                                       label="sparse" if i == 0 and j == 0 else "",
                                       color=plot_style.MAGENTA)

axs.set_yscale("log")
axs.set_xscale("log")
axs.set_xlabel("number of transitions")
axs.set_ylabel("relative size increase transition function\n(verigym/stormpy)")
axs.legend(frameon=True)

fig.savefig(f"{FIGPATH}/memory_eval_3b_transition_func.pdf", pad_inches=0, bbox_inches="tight")

# -----

fig, axs = plt.subplots(figsize=(4,3))

# axs.set_title("Size of environment relative to stormpy process")
for i, (eb, evariants) in enumerate(env_or_mdp_sizes.items()):
    for j, (ec, data) in enumerate(evariants.items()):
        axs.scatter(data["n_transitions"], data["verigym,process"]/data["stormpy,process"], 
                       label="verigym (process)" if i == 0 and j == 0 else "", 
                       color="darkgreen")
        axs.scatter(data["n_transitions"], data["verigym,process,w.build"]/data["stormpy,process"], 
                       label="verigym (process + MDP)" if i == 0 and j == 0 else "", 
                       color="lime")

axs.legend()
axs.set_xscale("log")
axs.set_xlabel("number of transitions")
axs.set_ylabel("relative environment size increase\n(verigym/stormpy)")

fig.savefig(f"{FIGPATH}/memory_eval_3b_environment_size.pdf", pad_inches=0, bbox_inches="tight")

plt.show()
