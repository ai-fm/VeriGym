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
# # VeriGym vs. Gymnasium: how fast is discretization?
#
# Both libraries can turn a continuous observation into bin numbers.
# Gymnasium does it with `DiscretizeObservation`, VeriGym with `BinEdges`.
#
# The discretizer only sees an array of numbers. Where the environment comes
# from (classic control, Box2D, MuJoCo, ...) or if it represents an image does not matter directly. What matters:
#
# - **number of dimensions** `d` (the main factor),
# - **data type** (float32, float64, int32, uint8),
# - **bins per dimension**,
# - **shape** (flat vector or image) and **space type** (`Box`, `Discrete`, ...).
#
# We look at:
#
# 1. Which space types each library can handle at all.
# 2. Time per observation on real environments, ordered by `d`.
# 3. How the time grows with `d` and with the number of bins.
# 4. The effect of the data type: speed, and VeriGym's one-time compile cost
#    (numba JIT) and when it pays off.
# 5. How much discretization slows down a full `env.step()`. Here the
#    environment does matter, because a slow step hides a small extra cost.
# 6. When the flat state number gets too big for a 64-bit integer.
#
# Set `QUICK = True` for a short test run.

# %%
import gc
import itertools
import json
import platform
import subprocess
import sys
import textwrap
from importlib.metadata import version
from time import perf_counter, perf_counter_ns

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.wrappers import FlattenObservation, TransformObservation
from gymnasium.wrappers.transform_observation import DiscretizeObservation
from tqdm.auto import tqdm

from verigym.abstraction.discretization import generate_box_bins
from verigym.abstraction.gym_utils.transform_observation import DiscretizeBoxObservation

QUICK = False

BINS = 10  # bins per dimension
SEED = 0
N_OBS = 500 if QUICK else 2_000  # recorded observations per environment
N_STEPS = 200 if QUICK else 1_000  # steps for the env.step() test
REPEATS = 3 if QUICK else 7  # repeats per timing
TARGET_S = 0.01 if QUICK else 0.05  # minimum length of one timed batch
DIMS = [2**i for i in range(0, 11 if QUICK else 15)]  # 1 ... 16384
BIN_SWEEP = [2, 10, 100, 1000]
BIN_SWEEP_DIM = 64
DTYPES = ["float32", "float64", "int32", "uint8"]
DTYPE_DIM = 64
COLD_REPEATS = 3 if QUICK else 5  # fresh processes per data type

INT64_MAX = np.iinfo(np.int64).max
COLORS = {"gymnasium": "#4C72B0", "verigym": "#DD8452"}
NAMES = {"gymnasium": "Gymnasium", "verigym": "VeriGym"}

print(f"CPU:        {platform.processor() or platform.machine()} ({platform.platform()})")
print(f"Python:     {platform.python_version()}")
for pkg in ["numpy", "numba", "gymnasium", "verigym"]:
    print(f"{pkg + ':':<12}{version(pkg)}")

# %% [markdown]
# ## Environments
#
# Gymnasium's discretization methods need finite bounds on every dimension. Some environments (MuJoCo)
# have infinite bounds. For those we record some observations first and set the
# smallest and largest seen values as bounds.
#
# Gymnasium also only works on flat (1-D) observations, so for the image
# environment (CarRacing) we flatten the image first on the Gymnasium side.
#
# The environments are picked to cover a wide range of state-/observation-dimensions `d`, including datatypes float32, float64, and uint8 (representing a greyscale image). They are listed from small to large `d`.

# %%
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
    "CarRacing-v3",  # 96x96x3 = 27648, uint8
]
if QUICK:
    ENVS = ["Pendulum-v1", "HalfCheetah-v5", "CarRacing-v3"]


def collect_obs(env, n, seed=SEED):
    """Run random actions and record `n` observations."""
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    out = [obs]
    while len(out) < n:
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
        out.append(obs)
    return np.asarray(out)


def make_bounded_env(env_id):
    """Make the env and replace infinite bounds with the seen min/max values.

    Returns the env and the recorded observations.
    """
    env = gym.make(env_id)
    obs = collect_obs(env, N_OBS)
    space = env.observation_space
    if not (np.all(np.isfinite(space.low)) and np.all(np.isfinite(space.high))):
        low, high = obs.min(axis=0), obs.max(axis=0)
        margin = 1e-6 + 1e-3 * (high - low)
        # Not `ReplaceInfObservation`: it builds `Box(low, high)` without the
        # dtype, which would turn the float64 MuJoCo spaces into float32.
        bounded = Box(low - margin, high + margin, dtype=space.dtype)
        env = TransformObservation(env, lambda o: o, bounded)
    return env, obs


def gym_discretizer(env, bins, multidiscrete=True):
    """Gymnasium wrapper; image spaces are flattened first."""
    if len(env.observation_space.shape) > 1:
        env = FlattenObservation(env)
    return DiscretizeObservation(env, bins=bins, multidiscrete=multidiscrete)


class BoxEnv(gym.Env):
    """Tiny env that only exists to hold an observation space."""

    def __init__(self, space):
        self.observation_space = space
        self.action_space = gym.spaces.Discrete(1)


def space_info(space, bins):
    d = int(np.prod(space.shape))
    log10_size = d * np.log10(bins)
    return {
        "shape": tuple(space.shape),
        "dtype": str(space.dtype),
        "d": d,
        "bytes": d * space.dtype.itemsize,
        "log10_size": log10_size,
        "fits_int64": bins**d <= INT64_MAX,
    }


def make_space(d, dtype):
    """A made-up `Box` space with `d` dimensions; ints use 0..255."""
    if np.dtype(dtype).kind in "iu":
        return Box(0, 255, (d,), dtype=dtype)
    return Box(-1.0, 1.0, (d,), dtype=dtype)


# %% [markdown]
# ## How we measure time
#
# One call takes only a few microseconds, which is close to the cost of
# reading the clock itself. So we do not time single calls. Instead we time a batch of
# multiple calls and divide by the batch size. The batch is made bigger until it
# runs for at least `TARGET_S` seconds. We repeat this `REPEATS` many times and keep
# the median (middle value) and the 25%/75% values (IQR ranges).
#
# Before timing, each function is called once, so VeriGym's compile step is
# not part of these numbers. We measure the compile step on its own later.

# %%
def _run_batch(fn, obs_list, k):
    calls = itertools.islice(itertools.cycle(obs_list), k)
    gc.disable()
    start = perf_counter()
    for o in calls:
        fn(o)
    elapsed = perf_counter() - start
    gc.enable()
    return elapsed


def time_per_call(fn, obs):
    """Return the per-call times in microseconds, one per repeat."""
    obs_list = list(obs)
    fn(obs_list[0])  # warm-up (compiles numba code)
    k = 1
    while _run_batch(fn, obs_list, k) < TARGET_S:
        k *= 2
    return np.array([_run_batch(fn, obs_list, k) / k for _ in range(REPEATS)]) * 1e6


def summarize(times):
    q25, med, q75 = np.percentile(times, [25, 50, 75])
    return {"median_us": med, "q25_us": q25, "q75_us": q75}


# %% [markdown]
# ## 1. Which space types are supported?
#
# Before timing anything: can each library discretize the space at all?
# We wrap a tiny env with each space type and discretize one sample.
#
# - `ok`: it works.
# - Otherwise we show the error type.

# %%
SUPPORT_SPACES = {
    "Box, 1-D, float": Box(-1.0, 1.0, (4,), dtype=np.float32),
    "Box, 1-D, int": Box(0, 255, (4,), dtype=np.int32),
    "Box, image (8x8x3, uint8)": Box(0, 255, (8, 8, 3), dtype=np.uint8),
    "Box, infinite bounds": Box(-np.inf, np.inf, (4,)),
    "Discrete(16)": Discrete(16),
    "MultiDiscrete([16, 16])": MultiDiscrete([16, 16]),
}


def try_discretize(lib, space):
    try:
        env = BoxEnv(space)
        if lib == "gymnasium":
            wrapper = DiscretizeObservation(env, bins=4, multidiscrete=True)
        else:
            wrapper = DiscretizeBoxObservation(env, n_samples=4, use_box_space=False)
        wrapper.observation(space.sample())
        return "ok"
    except Exception as e:
        print(e)
        return f"error ({type(e).__name__})"


print(f"{'space':<28}{'Gymnasium':<22}{'VeriGym':<22}")
print("-" * 72)
for name, space in SUPPORT_SPACES.items():
    print(f"{name:<28}{try_discretize('gymnasium', space):<22}"
          f"{try_discretize('verigym', space):<22}")

# %% [markdown]
# Gymnasium only accepts flat `Box` spaces with finite bounds. VeriGym also
# accepts images and discrete spaces (there it can merge states into fewer
# bins). Neither accepts infinite bounds directly: you first have to pick
# finite bounds, as we do for MuJoCo below.
#
# Gymnasium fails on any observation with more than one axis (even a `(4, 3)`
# array), so images must first be flattened with `FlattenObservation`; for real
# images this then only works with `multidiscrete=True`, because one single
# state number for a whole image does not fit into 64 bits.

# %% [markdown]
# ## 2. Time per observation on real environments
#
# We time two kinds of output:
#
# - `idx`: one bin number per dimension (a `MultiDiscrete` sample).
# - `enum`: one single number for the whole observation (a `Discrete` sample).
#   This only works while the total number of states fits in 64 bits (see
#   section 6), so we skip it for the bigger environments.
#
# We also check that both libraries give the same bins.

# %%
env_rows = []  # one row per (env, library, output kind)
env_cache = {}  # env id -> (env, observations), reused below

for env_id in tqdm(ENVS, desc="Environments"):
    env, obs = make_bounded_env(env_id)
    env_cache[env_id] = (env, obs)
    space = env.observation_space
    info = space_info(space, BINS)
    bin_edges = generate_box_bins(space, np.linspace, BINS)
    flat_obs = obs.reshape(len(obs), -1)

    # Do both libraries put each value in the same bin?
    gym_wrap = gym_discretizer(env, BINS)
    gym_idx = np.array([gym_wrap.observation(o) for o in flat_obs])
    vg_idx = np.array([bin_edges.orig_to_idx(o).ravel() for o in obs])
    agreement = np.mean(gym_idx == vg_idx)

    base = {"env": env_id, "agreement": agreement, **info}
    fns = {
        ("gymnasium", "idx"): (gym_wrap.observation, flat_obs),
        ("verigym", "idx"): (bin_edges.orig_to_idx, obs),
    }
    if info["fits_int64"]:
        gym_enum = gym_discretizer(env, BINS, multidiscrete=False)
        fns[("gymnasium", "enum")] = (gym_enum.observation, flat_obs)
        fns[("verigym", "enum")] = (bin_edges.orig_to_enum, obs)

    for (lib, mode), (fn, inputs) in fns.items():
        env_rows.append(
            {**base, "lib": lib, "mode": mode, **summarize(time_per_call(fn, inputs))}
        )

for r in env_rows:
    if r["lib"] == "gymnasium" and r["mode"] == "idx":
        print(f"{r['env']:<26} d={r['d']:<6} same bins: {r['agreement']:.4%}")

# %% [markdown]
# ## 3. How time grows with dimensions and bins
#
# Here we use a made-up space `Box(-1, 1, (d,))` (float32) with random samples,
# so the only thing that changes is the size. We change:
#
# - the number of dimensions `d` (with 10 bins each), and
# - the number of bins (with `d = 64` fixed).

# %%
def bench_synthetic(d, bins, dtype="float32"):
    space = make_space(d, dtype)
    space.seed(SEED)
    obs = np.array([space.sample() for _ in range(min(N_OBS, 200))])
    gym_wrap = gym_discretizer(BoxEnv(space), bins)
    bin_edges = generate_box_bins(space, np.linspace, bins)
    return {
        "gymnasium": summarize(time_per_call(gym_wrap.observation, obs)),
        "verigym": summarize(time_per_call(bin_edges.orig_to_idx, obs)),
    }


dim_rows = [{"d": d, **bench_synthetic(d, BINS)} for d in tqdm(DIMS, desc="Dimensions")]
bin_rows = [
    {"bins": b, **bench_synthetic(BIN_SWEEP_DIM, b)} for b in tqdm(BIN_SWEEP, desc="Bins")
]

# %% [markdown]
# ## 4. Data type: speed and compile cost
#
# Here the size stays fixed (`d = 64`, 10 bins) and only the data type changes.
# So any difference comes from the data type alone.
#
# **Warm speed:** time per observation once everything is ready.
#
# **Compile cost:** VeriGym's core loop is compiled by numba the first time it
# is called. numba compiles a new version for **each data type**, so each one
# pays this cost once. The compiled code is not kept between Python sessions,
# so every new process pays it again. To see the real cost we start a fresh
# Python process for each data type and time the first and second call there.
# We do this `COLD_REPEATS` times per data type and keep the median and IQR.
#
# From that we get the **break-even point**: how many calls it takes before
# VeriGym has made up for the compile time.

# %%
dtype_rows = [
    {"dtype": dt, **bench_synthetic(DTYPE_DIM, BINS, dt)}
    for dt in tqdm(DTYPES, desc="Data types")
]
for r in dtype_rows:
    print(f"{r['dtype']:<8} Gymnasium {r['gymnasium']['median_us']:7.2f} µs, "
          f"VeriGym {r['verigym']['median_us']:6.2f} µs")

# %%
COLD_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from time import perf_counter
    import numpy as np
    from gymnasium.spaces import Box
    from verigym.abstraction.discretization import generate_box_bins

    dtype = np.dtype(sys.argv[1])
    high = 255 if dtype.kind in "iu" else 1
    space = Box(0, high, (16,), dtype=dtype)
    bin_edges = generate_box_bins(space, np.linspace, 10)
    x = space.sample()
    t0 = perf_counter(); bin_edges.orig_to_idx(x); t1 = perf_counter()
    bin_edges.orig_to_idx(x); t2 = perf_counter()
    print(json.dumps({"first_s": t1 - t0, "second_s": t2 - t1}))
    """
)


def cold_start(dtype):
    out = subprocess.run(
        [sys.executable, "-c", COLD_SCRIPT, dtype],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


# Take turns between the data types, so that no type always runs first.
cold = {dt: [] for dt in DTYPES}
for _ in tqdm(range(COLD_REPEATS), desc="Compile (fresh process)"):
    for dt in DTYPES:
        cold[dt].append(cold_start(dt))

jit = {}
for dt, runs in cold.items():
    q25, med, q75 = np.percentile([r["first_s"] for r in runs], [25, 50, 75])
    second = np.median([r["second_s"] for r in runs])
    jit[dt] = {"first_s": med, "first_q25_s": q25, "first_q75_s": q75, "second_s": second}
    print(f"{dt:<8} first call {med * 1e3:6.1f} ms (IQR {q25 * 1e3:.1f}-{q75 * 1e3:.1f}), "
          f"second call {second * 1e6:5.1f} µs")


def break_even(env_id):
    """Number of calls after which VeriGym (incl. compile) is faster in total."""
    rows = {r["lib"]: r for r in env_rows if r["env"] == env_id and r["mode"] == "idx"}
    saved_per_call = (rows["gymnasium"]["median_us"] - rows["verigym"]["median_us"]) * 1e-6
    if saved_per_call <= 0:
        return np.inf
    return jit[rows["verigym"]["dtype"]]["first_s"] / saved_per_call


# %% [markdown]
# ## 5. Cost inside a full `env.step()`
#
# In practice the discretization runs once per environment step. Here we run
# the same actions with the same seed three times:
#
# - plain env,
# - env + Gymnasium discretizer,
# - env + VeriGym discretizer (`DiscretizeBoxObservation`).
#
# The extra time compared to the plain env is the cost of discretization.
# Steps here are timed one by one, which is fine because a step is much slower
# than reading the clock. All three runs use the same seed and actions, so
# step number `i` is the same state in each run. So we can take the extra time
# for each step and report its median and IQR.
#
# The environments are sorted by how long a plain step takes. A cheap step
# makes the extra cost stand out, an expensive step hides it.

# %%
def time_steps(env, actions):
    env.reset(seed=SEED)
    times = []
    for a in actions:
        start = perf_counter_ns()
        _, _, terminated, truncated, _ = env.step(a)
        times.append(perf_counter_ns() - start)
        if terminated or truncated:
            env.reset()
    return np.array(times) / 1e3  # µs, one value per step


step_rows = []
for env_id in tqdm(ENVS, desc="env.step()"):
    env, _ = env_cache[env_id]
    env.action_space.seed(SEED)
    actions = [env.action_space.sample() for _ in range(N_STEPS)]
    variants = {
        "plain": env,
        "gymnasium": gym_discretizer(env, BINS),
        "verigym": DiscretizeBoxObservation(env, n_samples=BINS, use_box_space=False),
    }
    for v in variants.values():  # warm-up, includes numba compile
        v.reset(seed=SEED)
        v.step(actions[0])
    t = {name: time_steps(v, actions) for name, v in variants.items()}
    row = {"env": env_id, "plain": np.median(t["plain"])}
    for lib in ["gymnasium", "verigym"]:
        row[lib] = np.median(t[lib])
        # Same seed and actions, so step i is the same state in every run.
        # This lets us take the extra time step by step.
        row[f"{lib}_extra"] = np.percentile(t[lib] - t["plain"], [25, 50, 75])
    step_rows.append(row)

step_rows.sort(key=lambda r: r["plain"])
for r in step_rows:
    q = {lib: r[f"{lib}_extra"] for lib in ["gymnasium", "verigym"]}
    print(f"{r['env']:<26} plain {r['plain']:8.1f} µs | extra per step (IQR): "
          f"gymnasium {q['gymnasium'][1]:8.1f} ({q['gymnasium'][0]:.1f} to "
          f"{q['gymnasium'][2]:.1f}) µs | verigym {q['verigym'][1]:6.1f} "
          f"({q['verigym'][0]:.1f} to {q['verigym'][2]:.1f}) µs")

# %% [markdown]
# ## 6. When is the state space too big for one number?
#
# With `b` bins in each of `d` dimensions there are `b^d` states. Turning a
# state into one single number (`enum` / `Discrete`) only works while `b^d`
# fits into a 64-bit integer (at most about 9.2·10^18, so roughly 19 digits).
#
# So the largest number of states that one int64 state number can cover is
# `2^63 - 1 = 9,223,372,036,854,775,807`.
#
# The largest `d` that still fits is about `63 / log2(b)`.
#
# Below we test both libraries right around this limit. For each test we check
# whether the library:
#
# - works (`ok`),
# - stops with an error (`error`), or
# - keeps going with a wrong state count (`WRONG`). This is the worst case,
#   because nothing tells you that something went wrong.

# %%
def max_dims(b):
    """Largest d with b**d <= INT64_MAX (exact, with Python ints)."""
    d = 0
    while b ** (d + 1) <= INT64_MAX:
        d += 1
    return d


def probe(lib, d, b):
    space = make_space(d, "float32")
    try:
        if lib == "gymnasium":
            n = gym_discretizer(BoxEnv(space), b, multidiscrete=False).observation_space.n
            return "ok" if int(n) == b**d else "WRONG"
        bin_edges = generate_box_bins(space, np.linspace, b)
        last = bin_edges.orig_to_enum(space.high)  # the largest state number
        return "ok" if last == b**d - 1 else "WRONG"
    except Exception:
        return "error"


overflow_rows = []
print(f"{'bins':>5} {'d':>4}  {'b^d fits?':<10} {'Gymnasium':<10} {'VeriGym':<10}")
for b in BIN_SWEEP:
    d_max = max_dims(b)
    for d in range(d_max - 1, d_max + 3):
        row = {
            "bins": b, "d": d, "fits": b**d <= INT64_MAX,
            "gymnasium": probe("gymnasium", d, b), "verigym": probe("verigym", d, b),
        }
        overflow_rows.append(row)
        print(f"{b:>5} {d:>4}  {str(row['fits']):<10} "
              f"{row['gymnasium']:<10} {row['verigym']:<10}")

# %% [markdown]
# ## Figures

# %%
def idx_rows(lib):
    return [r for r in env_rows if r["lib"] == lib and r["mode"] == "idx"]


def err(rows):
    med = np.array([r["median_us"] for r in rows])
    return med, [med - [r["q25_us"] for r in rows], [r["q75_us"] for r in rows] - med]


# %% [markdown]
# ### Figure 1: time per observation for each environment
#
# Lower is better. The y axis is logarithmic. Environments are sorted by `d`.
# The number above each pair says how many times faster VeriGym is.
#
# The whiskers show the IQR over the repeats. They are mostly too small to
# see, which means the measurement is stable.

# %%
gym_r, vg_r = idx_rows("gymnasium"), idx_rows("verigym")
order = np.argsort([r["d"] for r in gym_r])
gym_r, vg_r = [gym_r[i] for i in order], [vg_r[i] for i in order]
x, width = np.arange(len(gym_r)), 0.38

fig, ax = plt.subplots(num="Figure 1: Time per observation", figsize=(11, 5),
                       constrained_layout=True)
for offset, lib, rows in [(-width / 2, "gymnasium", gym_r), (width / 2, "verigym", vg_r)]:
    med, yerr = err(rows)
    ax.bar(x + offset, med, width, yerr=yerr, color=COLORS[lib], label=NAMES[lib])
for i, (g, v) in enumerate(zip(gym_r, vg_r)):
    top = max(g["q75_us"], v["q75_us"])
    ax.annotate(f"{g['median_us'] / v['median_us']:.1f}×", (i, top), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=9)
ax.set_yscale("log")
ax.set_xticks(x, [f"{r['env']}\n(d={r['d']}, {r['dtype']})" for r in gym_r],
              rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Time per observation [µs]")
ax.set_title(f"Discretization time per observation ({BINS} bins per dimension)")
ax.legend()
plt.show()

# %% [markdown]
# ### Figure 2: how time grows with dimensions and with bins
#
# Both axes are logarithmic. Lines are the made-up `Box` spaces, dots are the
# real environments from Figure 1 (name next to the Gymnasium dot, the VeriGym
# dot of the same env is straight below it). If the dots sit on the lines, the
# number of dimensions alone explains the time. Bands and whiskers show the IQR.

# %%
fig, (ax_d, ax_b) = plt.subplots(1, 2, num="Figure 2: Scaling with dimensions and bins",
                                figsize=(12, 4.5), constrained_layout=True)
for lib in ["gymnasium", "verigym"]:
    med = np.array([r[lib]["median_us"] for r in dim_rows])
    ax_d.plot(DIMS, med, "-", color=COLORS[lib], label=NAMES[lib])
    ax_d.fill_between(DIMS, [r[lib]["q25_us"] for r in dim_rows],
                      [r[lib]["q75_us"] for r in dim_rows], color=COLORS[lib], alpha=0.2)
    real = idx_rows(lib)
    ax_d.scatter([r["d"] for r in real], [r["median_us"] for r in real],
                 color=COLORS[lib], edgecolor="black", zorder=3, s=30)
    if lib == "gymnasium":
        for r in real:
            ax_d.annotate(r["env"], (r["d"], r["median_us"]), xytext=(-4, 6),
                          textcoords="offset points", fontsize=7, ha="right")

    med, yerr = err([r[lib] for r in bin_rows])
    ax_b.errorbar(BIN_SWEEP, med, yerr=yerr, fmt="o-", capsize=3,
                  color=COLORS[lib], label=NAMES[lib])
ax_d.set(xscale="log", yscale="log", xlabel="Number of dimensions d",
         ylabel="Time per observation [µs]", title=f"Scaling with dimensions ({BINS} bins)")
ax_b.set(xscale="log", yscale="log", xlabel="Bins per dimension",
         ylabel="Time per observation [µs]",
         title=f"Scaling with bins (d = {BIN_SWEEP_DIM})")
ax_d.legend()
ax_b.legend()
plt.show()

# %% [markdown]
# ### Figure 3: effect of the data type
#
# Left: time per observation (`d = 64`, 10 bins), log scale. Right: VeriGym's
# one-time compile cost per data type, measured in a fresh Python process.
# Whiskers show the IQR.

# %%
x = np.arange(len(DTYPES))
fig, (ax_t, ax_c) = plt.subplots(1, 2, num="Figure 3: Data type", figsize=(12, 4.5),
                                constrained_layout=True)
for offset, lib in [(-width / 2, "gymnasium"), (width / 2, "verigym")]:
    rows = [r[lib] for r in dtype_rows]
    med, yerr = err(rows)
    ax_t.bar(x + offset, med, width, yerr=yerr, color=COLORS[lib], label=NAMES[lib])
ax_t.set_yscale("log")
ax_t.set_xticks(x, DTYPES)
ax_t.set(ylabel="Time per observation [µs]",
         title=f"Warm speed by data type (d = {DTYPE_DIM}, {BINS} bins)")
ax_t.legend(loc="center right")

med = np.array([jit[dt]["first_s"] for dt in DTYPES]) * 1e3
low = med - [jit[dt]["first_q25_s"] * 1e3 for dt in DTYPES]
high = [jit[dt]["first_q75_s"] * 1e3 for dt in DTYPES] - med
ax_c.bar(x, med, 0.5, yerr=[low, high], capsize=3, color=COLORS["verigym"])
ax_c.set_xticks(x, DTYPES)
ax_c.set(ylabel="First call [ms]", title="VeriGym compile cost by data type")
plt.show()

# %% [markdown]
# ### Figure 4: when does the compile cost pay off?
#
# Total time after `n` calls. VeriGym starts higher (compile time) but grows
# slower. The dashed line marks the break-even point. One plot per data type.
#
# Why is the VeriGym line curved? Its total time is
# `compile time + n · time per call`. That is a straight line on normal axes,
# but here both axes are logarithmic. As long as `n · time per call` is much
# smaller than the compile time, the total is almost just the compile time, so
# the line looks flat. Once the calls add up to more than the compile time, the
# line bends up and runs parallel to the Gymnasium line. The gap between the two
# parallel lines is the speedup. Gymnasium has no compile time, so its line is
# straight from the start.

# %%
examples = {}
for r in vg_r:  # pick the smallest env for each data type
    examples.setdefault(r["dtype"], r["env"])

fig, axes = plt.subplots(1, len(examples), num="Figure 4: Compile cost break-even",
                         figsize=(5 * len(examples), 4),
                         constrained_layout=True, squeeze=False)
for ax, (dtype, env_id) in zip(axes[0], examples.items()):
    g = next(r for r in gym_r if r["env"] == env_id)["median_us"] * 1e-6
    v = next(r for r in vg_r if r["env"] == env_id)["median_us"] * 1e-6
    n_star = break_even(env_id)
    n = np.logspace(0, 7, 200)  # same x range (1 to 10^7 calls) in every plot
    ax.plot(n, n * g, color=COLORS["gymnasium"], label=NAMES["gymnasium"])
    ax.plot(n, jit[dtype]["first_s"] + n * v, color=COLORS["verigym"],
            label=f"{NAMES['verigym']} (incl. compile)")
    if np.isfinite(n_star):
        ax.axvline(n_star, ls="--", color="grey")
        ax.annotate(f"break-even\n≈ {n_star:,.0f} calls", (n_star, n[0] * g),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="Number of calls n",
           ylabel="Total time [s]", title=f"{env_id} ({dtype})")
    ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ### Figure 5: extra time per `env.step()`
#
# How much slower one step gets because of discretization, compared to the
# plain env. The y axis is logarithmic. Environments are sorted by plain step
# time, which is the number in brackets.

# %%
x = np.arange(len(step_rows))
fig, ax = plt.subplots(num="Figure 5: Extra time per env.step()", figsize=(11, 4.5),
                       constrained_layout=True)
for offset, lib in [(-width / 2, "gymnasium"), (width / 2, "verigym")]:
    q25, med, q75 = np.array([r[f"{lib}_extra"] / r["plain"] * 100 for r in step_rows]).T
    ax.bar(x + offset, med, width, yerr=[med - q25, q75 - med], capsize=3,
           color=COLORS[lib], label=NAMES[lib])
ax.set_yscale("log")
ax.set_xticks(x, [f"{r['env']}\n({r['plain']:.0f} µs)" for r in step_rows],
              rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Extra time per step [% of plain step]")
ax.set_title("Cost of discretization inside env.step()")
ax.legend()
plt.show()

# %% [markdown]
# ### Figure 6: when does `b^d` stop fitting into 64 bits?
#
# Left: the largest number of dimensions that still fits, for each number of
# bins. Right: how many digits the number of states has for each real
# environment (with 10 bins per dimension). Everything above the red line
# cannot be turned into one single state number.

# %%
fig, (ax_l, ax_r) = plt.subplots(1, 2, num="Figure 6: int64 limit", figsize=(12, 4.5),
                                constrained_layout=True)
bs = np.unique(np.logspace(np.log10(2), 3, 60).astype(int))
ax_l.step(bs, [max_dims(int(b)) for b in bs], where="post", color="black")
ax_l.set(xscale="log", xlabel="Bins per dimension b",
         ylabel="Largest d that fits in int64",
         title="Max. dimensions for a single state number")
ax_l.grid(alpha=0.3)

ax_r.axhline(np.log10(INT64_MAX), color="red", ls="--", label="int64 limit (19 digits)")
for r in gym_r:
    ax_r.scatter(r["d"], r["log10_size"], color="black", s=20)
    ax_r.annotate(r["env"], (r["d"], r["log10_size"]), xytext=(4, -3),
                  textcoords="offset points", fontsize=7)
ax_r.set(xscale="log", yscale="log", xlabel="Number of dimensions d",
         ylabel=f"Digits in the number of states ({BINS} bins)",
         title="Size of the discrete state space")
ax_r.legend()
plt.show()

# %% [markdown]
# ## Summary table
#
# Times are medians in µs. `speedup` is Gymnasium time / VeriGym time.
# `break-even` is the number of calls before VeriGym's compile time is paid back.

# %%
steps = {r["env"]: r for r in step_rows}
header = (f"{'env':<26}{'shape':<14}{'dtype':<9}{'d':>6}{'log10|S|':>10}{'int64?':>8}"
          f"{'step':>10}{'gym':>10}{'verigym':>10}{'speedup':>9}{'JIT ms':>8}"
          f"{'break-even':>12}{'same bins':>11}")
print(header)
print("-" * len(header))
for g, v in zip(gym_r, vg_r):
    n_star = break_even(g["env"])
    print(f"{g['env']:<26}{str(g['shape']):<14}{g['dtype']:<9}{g['d']:>6}"
          f"{g['log10_size']:>10.1f}{'yes' if g['fits_int64'] else 'no':>8}"
          f"{steps[g['env']]['plain']:>10.1f}{g['median_us']:>10.2f}{v['median_us']:>10.2f}"
          f"{g['median_us'] / v['median_us']:>8.1f}×{jit[g['dtype']]['first_s'] * 1e3:>8.0f}"
          f"{n_star:>12,.0f}{g['agreement']:>11.2%}")
