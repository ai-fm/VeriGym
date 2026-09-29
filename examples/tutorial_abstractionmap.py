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
# # Tutorial - Abstraction Mapping
# This tutorial explains abstraction mapping: What it is and how to use it.
# 
# When deriving an explicit MDP from a simulator environment, it is most often the case that our original environment (simulator) has a multidimensional continuous state space and we want to discretize that space. 
# The means we establish a mapping to a (multidemensional) discrete space, which we call the abstract space.
#
# Verigym makes it easy to define a mapping between the original environment and the abstracted environment by using an `AbstractionMapper`.
# This class can hold two `AbstractionMap`s; one for the state space and one for the action space.
# We demonstrate the creation and usage below.
#
# The tutorial is structured as follows:
#
# 1. Gym spaces
# 2. Bin edges: cutting a continuous space into pieces
# 3. Different discretization types (linspace, polynomial, exponential, data-based)
# 4. From bin edges to an `AbstractionMap`
# 5. Combining two maps into an `AbstractionMapper`
# 6. Identity mapping
# 7. Putting it all together on a real environment

# %%
import functools

import gymnasium as gym
import numpy as np

from verigym.abstraction.abstractionmapper import AbstractionMap, AbstractionMapper

# %% [markdown]
# ## 1. Gym Spaces
# `gymnasium` has a set of classes for standard types of state and action spaces.
# The two we care about most are 
# - `gym.spaces.Box` for real numbers and 
# - `gym.spaces.MultiDiscrete` for integer numbers.

# %%
# 1-dimensional space on the interval [-1,1]
box_space = gym.spaces.Box(low=-1, high=1)
print(box_space)

# 1-dimensional space for natural numbers from {0, ..., 2}
discrete_space = gym.spaces.MultiDiscrete([3])
print(discrete_space)

# %% [markdown]
# The whole point of an abstraction is to go from the first (uncountably many elements)
# to the second (finitely many elements). `AbstractionMap` stores how many elements each side has.
# %%
from verigym.abstraction.gym_utils.spaces import get_n_elements_of_space

print("elements in the Box space:            ", get_n_elements_of_space(box_space))
print("elements in the MultiDiscrete space:  ", get_n_elements_of_space(discrete_space))

# %% [markdown]
# ## 2. Bin edges: discretizing a continuous space
#
# Before we can map anything, we have to decide *where* to cut the continuous space.
# These points are called **bin edges**. In VeriGym they are stored in a `BinEdges` object,
# which holds one array of bin edges per dimension of the space.
#
# The easiest way to create one is `generate_box_bins(space, bin_func, n_samples)`:
#
# - `space`: the space we want to discretize (`Box`, `Discrete` or `MultiDiscrete`),
# - `bin_func`: a function `(low, high, n_samples) -> array of edges`, e.g. `np.linspace`,
# - `n_samples`: how many bin edges to use per dimension.
#
# A value is always mapped to the *closest bin edge below or equal to it*.
# So `n_samples=5` means every dimension is described by 5 possible values.
# %%
from verigym.abstraction.discretization import generate_box_bins

# A 2-dimensional space: dimension 0 lives in [-1, 1], dimension 1 lives in [0, 10].
space = gym.spaces.Box(low=np.array([-1.0, 0.0], dtype=np.float32), high=np.array([1.0, 10.0], dtype=np.float32))

bin_edges = generate_box_bins(space, np.linspace, 5)

print("edges of dimension 0:", bin_edges[0])
print("edges of dimension 1:", bin_edges[1])
print("bins per dimension:  ", bin_edges.lengths)
print("nvec (shape of the discrete space):", bin_edges.nvec)

# %% [markdown]
# We can now discretize a sample. `sample_to_discrete` either returns the bin edge *value*
# (`return_idx=False`) or the *index* of that bin edge (`return_idx=True`).
# %%
from verigym.abstraction.gym_utils.mapping import sample_to_discrete

sample = np.array([0.3, 7.4])
print("original sample: ", sample)
print("bin edge values: ", sample_to_discrete(sample, bin_edges, return_idx=False))
print("bin edge indices:", sample_to_discrete(sample, bin_edges, return_idx=True))

# %% [markdown]
# Note that we do not have to use the identical number of bins for each dimension.
# If `n_samples` is an array (with the same shape as the space), each dimension gets its own resolution.
# This is useful when one dimension matters more than another.
# %%
bin_edges_uneven = generate_box_bins(space, np.linspace, np.array([3, 6]))
print("edges of dimension 0:", bin_edges_uneven[0])
print("edges of dimension 1:", bin_edges_uneven[1])

# %% [markdown]
# ## 3. Different discretization types
#
# `np.linspace` puts the bin edges at equal distances. That may be a good default,
# not always the best choice: maybe we care a lot about small values and barely about large ones.
# Since `generate_box_bins` accepts *any* function `(low, high, n_samples) -> array`,
# we can freely change how the space is cut.
#
# ### 3.1 Uniform bins (`np.linspace`)
# Every bin has the same width. Use it when all regions of the space are equally important.
# %%
one_dim_space = gym.spaces.Box(low=0.0, high=10.0, shape=(1,))

linspace_bins = generate_box_bins(one_dim_space, np.linspace, 7)
print("linspace:      ", np.round(linspace_bins[0], 2))

# %% [markdown]
# ### 3.2 Polynomial bins (`centered_pow_bin`)
# `centered_pow_bin` places the edges according to a polynomial. The bins are narrow around
# the center of the interval and wide towards the borders. Use it when the interesting
# behaviour happens in the middle of the interval (e.g. a pole angle around 0).
#
# The `power` argument is a keyword argument, so we fix it with `functools.partial`
# to get a function with the required `(low, high, n_samples)` signature.
# %%
from verigym.abstraction.discretization import centered_pow_bin

pow_bins = generate_box_bins(one_dim_space, centered_pow_bin, 7)
print("centered_pow (power=2):", np.round(pow_bins[0], 2))

pow4_bin_func = functools.partial(centered_pow_bin, power=4)
pow4_bins = generate_box_bins(one_dim_space, pow4_bin_func, 7)
print("centered_pow (power=4):", np.round(pow4_bins[0], 2))

# %% [markdown]
# ### 3.3 Own/custom function (sinus)
# Writing our own bin function is just as easy: it has to take `(low, high, n_samples)`
# and return an array of edges *sorted in ascending order*, starting at `low`.
# Here we want to discretize using spacing following the sine function.
# %%
# TODO code for the sine function


# %% [markdown]
# ### 3.4 Data-based bins
# The bin functions above only look at the boundaries of the space. 
# But perhaps certain regions in the state-space are visited more frequently. 
# Given a dataset of samples (states or actions) we can create a discretization that evenly distributes those sampling into bins.
# As a result, the space will be discretized into finer segments where there are many samples, and more coarse where there are less samples. 
# We build the `BinEdges` object directly from the dataset, by placing the
# edges at the quantiles of the data. Every bin then contains roughly the same number of samples.
#
# A `BinEdges` object consists of three things:
#
# - `space`: the space it belongs to,
# - `edges`: **all** bin edges of **all** dimensions in one flat array,
# - `ranges`: for every dimension the `(start, end)` slice into `edges`.
# %%
from verigym.abstraction.discretization import BinEdges, data_discretization_function

# %% [markdown]
# ### 3.5 Which one should I use?
#
# | Discretization | How to get it | When to use it |
# | --- | --- | --- |
# | uniform | `np.linspace` | good default, no prior knowledge |
# | polynomial | `centered_pow_bin` | interesting behaviour in the middle of the interval |
# | custom | own bin function | when you know data follows a particular distribution |
# | data-based | build `BinEdges` from samples | we already have trajectories/observations |
#
# Whatever we choose, the result is always a `BinEdges` object, and everything that follows
# stays exactly the same.

# %% [markdown]
# ## 4. From bin edges to an `AbstractionMap`
#
# An `AbstractionMap` maps a *single* space (either state space or action space) between the original
# and the abstract environment. Constructing it requires four things:
#
# - `forward_map`: original sample -> abstract sample (required),
# - `backward_map`: abstract sample -> original sample (optional, not always definable),
# - `original_space`: the `gym.Space` of the original environment,
# - `abstract_space`: the `gym.Space` of the abstract environment.
#
# There are two common ways to represent an abstract state, and VeriGym supports both.
#
# ### 4.1 Factored representation (one integer per dimension)
# `box_to_discrete` gives us the abstract `MultiDiscrete` space plus the two mapping functions.
# A state is then a vector of bin indices, e.g. `[2, 4]`.
# %%
from verigym.abstraction.gym_utils.mapping import box_to_discrete

abstract_space, to_discrete, to_continuous = box_to_discrete(space, bin_edges)

factored_map = AbstractionMap(
    forward_map=to_discrete,
    backward_map=to_continuous,
    original_space=space,
    abstract_space=abstract_space,
)

print("abstract space:", factored_map.abstract_space)
print("forward: ", sample, "->", factored_map.forward_map(sample))
print("backward:", factored_map.forward_map(sample), "->", factored_map.backward_map(factored_map.forward_map(sample)))

# %% [markdown]
# ### 4.2 Index representation (a single integer)
# For building an explicit MDP we want *one* integer per state, not a vector.
# `factored_to_index` flattens the vector of bin indices into a single index, and
# `index_to_factored` reverses that.
#
# This is the representation used by `create_abstraction`, so it is the one we usually need.
# %%
from verigym.abstraction.utils import factored_to_index, index_to_factored

n_abstract_states = int(np.prod(bin_edges.lengths))
print("number of abstract states:", n_abstract_states)

index_map = AbstractionMap(
    forward_map=functools.partial(factored_to_index, bin_edges=bin_edges),
    backward_map=functools.partial(index_to_factored, bin_edges=bin_edges),
    original_space=space,
    abstract_space=gym.spaces.MultiDiscrete([n_abstract_states]),
)

abstract_state = index_map.forward_map(sample)
print("forward: ", sample, "->", abstract_state)
print("backward:", abstract_state, "->", index_map.backward_map(abstract_state))

# %% [markdown]
# Note that the backward map does not return the original sample. That is expected:
# the abstraction throws information away, so we only get the bin edge back that the sample was mapped to.
# Also note that `has_backward_map` tells us whether a backward map was provided at all.
# %%
print("has backward map:", index_map.has_backward_map)
print("original space is continuous:", index_map.from_continuous_space)
print("elements original / abstract:", index_map.original_n_elements, "/", index_map.abstract_n_elements)

# %% [markdown]
# ## 5. Combining two maps into an `AbstractionMapper`
#
# An `AbstractionMapper` is simply the pair of maps for the *whole* environment:
# one `AbstractionMap` for the states and one for the actions.
# It offers four convenience methods:
#
# - `original_to_abstract_state` / `abstract_to_original_state`
# - `original_to_abstract_action` / `abstract_to_original_action`
# %%
# The action space of our toy example: a 1-dimensional continuous action in [-2, 2].
action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1,))
action_bin_edges = generate_box_bins(action_space, np.linspace, 5)
n_abstract_actions = int(np.prod(action_bin_edges.lengths))

action_map = AbstractionMap(
    forward_map=functools.partial(factored_to_index, bin_edges=action_bin_edges),
    backward_map=functools.partial(index_to_factored, bin_edges=action_bin_edges),
    original_space=action_space,
    abstract_space=gym.spaces.MultiDiscrete([n_abstract_actions]),
)

mapper = AbstractionMapper(
    state_abstraction_map=index_map,
    action_abstraction_map=action_map,
)

print("abstract states / actions:", mapper.abstract_n_states, "/", mapper.abstract_n_actions)
print("original states / actions:", mapper.original_n_states, "/", mapper.original_n_actions)
print("continuous states / actions:", mapper.from_continuous_states, "/", mapper.from_continuous_actions)

# %%
state = np.array([-0.9, 2.5])
action = np.array([1.3])

abstract_state = mapper.original_to_abstract_state(state)
abstract_action = mapper.original_to_abstract_action(action)

print("state  ", state, "->", abstract_state, "->", mapper.abstract_to_original_state(abstract_state))
print("action ", action, "->", abstract_action, "->", mapper.abstract_to_original_action(abstract_action))

# %% [markdown]
# If we did not provide a backward map, the backward direction raises a `ValueError`.
# %%
forward_only_map = AbstractionMap(
    forward_map=functools.partial(factored_to_index, bin_edges=action_bin_edges),
    original_space=action_space,
    abstract_space=gym.spaces.MultiDiscrete([n_abstract_actions]),
)
forward_only_mapper = AbstractionMapper(index_map, forward_only_map)

try:
    forward_only_mapper.abstract_to_original_action(0)
except ValueError as error:
    print("ValueError:", error)

# %% [markdown]
# ## 6. Identity mapping
#
# Sometimes no abstraction is needed, because the environment is already discrete
# (e.g. `Taxi-v3`). Then we use the identity map, which returns every input unchanged.
# %%
identity_mapper = AbstractionMapper.initialize_identity_mapper(
    state_space=gym.spaces.Discrete(500),
    action_space=gym.spaces.Discrete(6),
)

print("state 3 ->", identity_mapper.original_to_abstract_state(3))
print("action 1 ->", identity_mapper.original_to_abstract_action(1))
print("abstract states / actions:", identity_mapper.abstract_n_states, "/", identity_mapper.abstract_n_actions)

# %% [markdown]
# A single map can be built the same way with `AbstractionMap.initialize_identity_map(space)`.
# This is handy when only one of the two spaces needs to be abstracted, e.g. continuous states
# but already discrete actions.

# %% [markdown]
# ## 7. Putting it all together on a real environment
#
# Finally we build an `AbstractionMapper` for `CartPole-v1`, which is what we would hand over to
# `create_abstraction`. Two practical details show up here:
#
# 1. The observation space of `CartPole-v1` is unbounded (`inf`). We cannot cut an infinite
#    interval into bins, so we first replace the infinities with `ReplaceInfObservation`.
# 2. The action space is `gym.spaces.Discrete(2)`, so an action is a plain `int` and not an array.
#    VeriGym's `forward_mapping` and `backward_mapping` helpers take care of this: they promote
#    scalars to arrays and cast the result back into a valid sample of the original space.
#    `CachedDiscretizer` additionally caches results, because during abstraction learning the same
#    states and actions are discretized over and over again.
# %%
import verigym.abstraction.learn_abstraction as learn_abstraction
from verigym.abstraction.gym_utils.mapping import get_discrete_box_tf
from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation


def build_abstraction_mapper(env: gym.Env, bin_func, n_state_bins: int, n_action_bins: int) -> AbstractionMapper:
    """Build an `AbstractionMapper` that discretizes states and actions into single integers."""
    state_bin_edges = generate_box_bins(env.observation_space, bin_func, n_state_bins)
    action_bin_edges = generate_box_bins(env.action_space, bin_func, n_action_bins)

    n_states = int(np.prod(state_bin_edges.lengths))
    n_actions = int(np.prod(action_bin_edges.lengths))

    state_discretizer = learn_abstraction.CachedDiscretizer(
        functools.partial(factored_to_index, bin_edges=state_bin_edges)
    )
    action_discretizer = learn_abstraction.CachedDiscretizer(
        functools.partial(factored_to_index, bin_edges=action_bin_edges)
    )

    state_map = AbstractionMap(
        forward_map=functools.partial(
            learn_abstraction.forward_mapping,
            to_bins=get_discrete_box_tf(env.observation_space, state_bin_edges),
            to_int=state_discretizer.discretize,
        ),
        backward_map=functools.partial(
            learn_abstraction.backward_mapping,
            backward_map=functools.partial(index_to_factored, bin_edges=state_bin_edges),
            space=env.observation_space,
        ),
        original_space=env.observation_space,
        abstract_space=gym.spaces.MultiDiscrete([n_states]),
    )
    action_map = AbstractionMap(
        forward_map=functools.partial(
            learn_abstraction.forward_mapping,
            to_bins=get_discrete_box_tf(env.action_space, action_bin_edges),
            to_int=action_discretizer.discretize,
        ),
        backward_map=functools.partial(
            learn_abstraction.backward_mapping,
            backward_map=functools.partial(index_to_factored, bin_edges=action_bin_edges),
            space=env.action_space,
        ),
        original_space=env.action_space,
        abstract_space=gym.spaces.MultiDiscrete([n_actions]),
    )

    return AbstractionMapper(state_abstraction_map=state_map, action_abstraction_map=action_map)


env = gym.make("CartPole-v1")
env = ReplaceInfObservation(env, neg_inf=-10, pos_inf=10)

cartpole_mapper = build_abstraction_mapper(env, np.linspace, n_state_bins=5, n_action_bins=2)
print("abstract states / actions:", cartpole_mapper.abstract_n_states, "/", cartpole_mapper.abstract_n_actions)

observation, _info = env.reset(seed=42)
print("observation", observation, "->", cartpole_mapper.original_to_abstract_state(observation))
print("action     ", 1, "->", cartpole_mapper.original_to_abstract_action(1))

# %% [markdown]
# Changing the discretization is now a one-line change: we pass a different `bin_func`.
# The number of abstract states stays the same, but *where* the states lie changes,
# and therefore observations can end up in a different abstract state.
# %%
pow_mapper = build_abstraction_mapper(env, centered_pow_bin, n_state_bins=5, n_action_bins=2)

env.observation_space.seed(0)
observations = [env.observation_space.sample() for _ in range(5)]

print(f"{'observation':>44} | linspace | polynomial")
for observation in observations:
    print(
        f"{str(np.round(observation, 2)):>44} |"
        f"{cartpole_mapper.original_to_abstract_state(observation):>9} |"
        f"{pow_mapper.original_to_abstract_state(observation):>11}"
    )

# %% [markdown]
# The resulting `AbstractionMapper` is all that `create_abstraction` needs to turn the
# simulator into an explicit MDP:
#
# ```python
# generative_env = GenerativeEnv.from_gymnasium(env)
# abstracted_env = create_abstraction(
#     original_env=generative_env,
#     abstraction_mapper=cartpole_mapper,
#     exploration_policy=RandomizedPolicy(generative_env),
#     num_steps=1000,
# )
# ```
#
# ## Summary
#
# - `BinEdges` describe *where* a continuous space is cut; create them with `generate_box_bins`.
# - The `bin_func` decides the *type* of discretization: uniform (`np.linspace`), polynomial
#   (`centered_pow_bin`), exponential (own function) or data-based (built from samples).
# - An `AbstractionMap` combines a forward map, an optional backward map and the two spaces,
#   for either the state or the action space.
# - An `AbstractionMapper` holds both maps and is what the rest of VeriGym works with.
# - Use `initialize_identity_mapper` when a space is already discrete and needs no abstraction.
