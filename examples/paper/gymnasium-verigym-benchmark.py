# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Benchmark to compare the runtime between the discretization features of `verigym` and `gymnasium`

# %%
from time import perf_counter_ns
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers.transform_observation import DiscretizeObservation
from gymnasium import make as make_env

from verigym.abstraction.discretization import generate_box_linspace_bins
from verigym.abstraction.gym_utils.mapping import sample_to_discrete


# %%
def get_env(env_id):
    env = make_env(env_id)
    return env


# %%
N = 10_000
N_SAMPLES = 10  # discrete samples per dimension
ENVS = ["Acrobot-v1", "MountainCarContinuous-v0", "Pendulum-v1"]
gymnasium_times = defaultdict(list)
for env_id in ENVS:
    env = get_env(env_id)
    wrapper = DiscretizeObservation(env, bins=N_SAMPLES, multidiscrete=True)
    for i in range(N):
        sample = env.observation_space.sample()
        start = perf_counter_ns()
        discrete_sample = wrapper.observation(sample)
        gymnasium_times[env_id].append(
            (perf_counter_ns() - start) / 1e6
        )  # milliseconds

# %%
verigym_times = defaultdict(list)
for env_id in ENVS:
    env = get_env(env_id)
    bin_edges = generate_box_linspace_bins(env.observation_space, 10)
    # warmup to compile the njit function
    sample = env.observation_space.sample()
    discrete_sample = sample_to_discrete(sample, bin_edges, return_idx=True)
    for i in range(N):
        sample = env.observation_space.sample()
        start = perf_counter_ns()
        discrete_sample = sample_to_discrete(sample, bin_edges, return_idx=True)
        verigym_times[env_id].append((perf_counter_ns() - start) / 1e6)  # milliseconds

# %%
# %matplotlib widget

labels = ENVS
gym_mean_times = [np.mean(gymnasium_times[env_id]) for env_id in ENVS]
verigym_mean_times = [np.mean(verigym_times[env_id]) for env_id in ENVS]
gym_std_times = [np.std(gymnasium_times[env_id]) for env_id in ENVS]
verigym_std_times = [np.std(verigym_times[env_id]) for env_id in ENVS]
width = 0.3

x = np.arange(len(labels))

fig, ax = plt.subplots(constrained_layout=True, figsize=(9, 6))
fig.suptitle("Verigym vs. Gymnasium Average Discretization Speed per sample")
gym_rects = ax.bar(
    x=x - width / 2, height=gym_mean_times, width=width, label="Gymnasium",
    yerr=gym_std_times
)
verigym_rects = ax.bar(
    x=x + width / 2, height=verigym_mean_times, width=width, label="VeriGym",
    yerr=verigym_std_times
)
ax.set_ylabel("Time [ms]")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{round(height, 4)}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel(gym_rects)
autolabel(verigym_rects)
plt.show()
