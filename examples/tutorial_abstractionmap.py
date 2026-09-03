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
# ## Gym Spaces
# `gymnasium` has a set of classes for standard types of state and action spaces.spaces.Box` for real numbers and `gym.spaces.MultiDiscrete` for integer numbers.
# %%
import gymnasium as gym
from verigym.abstraction.abstractionmapper import AbstractionMap, AbstractionMapper

# 1-dimensional space on the interval [-1,1]
box_space = gym.spaces.Box(low=-1, high=1)
print(box_space)

# 1-dimensional space for natural numbers from {0, ..., 2}
discrete_space = gym.spaces.MultiDiscrete([3])
print(discrete_space)

# %%

AbstractionMapper.original_to_abstract_state

# %% [markdown]
# # Creating an AbstractionMapper
# # Different discretization types
# # Identity Mapping