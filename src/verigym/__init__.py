from importlib.metadata import version as _version

import gymnasium as gym

# Read from the installed package metadata, so pyproject.toml stays the only place to change it
__version__ = _version("verigym")

# Environments
from .io import exporter as exporter
from .io import importer as importer
from .environments.explicitenv import AbstractedExplicitEnv as AbstractedExplicitEnv
from .environments.frameworkexplicitenv import (
    FrameworkExplicitEnv as FrameworkExplicitEnv,
)
from .environments.generativeenv import GenerativeEnv as GenerativeEnv
from .environments.reward_func import RewardFunction as RewardFunction
from .environments.transition_func import TransitionFunction as TransitionFunction
from .environments.verigymenv import VeriGymEnv as VeriGymEnv

# Abstraction
from . import abstraction as abstraction
from .abstraction.learn_abstraction import create_abstraction as create_abstraction
from .abstraction.abstractionmapper import AbstractionMapper as AbstractionMapper

# Frameworks
from .frameworks import juliapomdp as juliapomdp
from .frameworks import mujoco as mujoco
from .frameworks import stormpy as stormpy

# Policies
from .policy.policy import PolicyClass as PolicyClass
from .frameworks.stormpy.stormpypolicy import StormpyPolicy as StormpyPolicy

# __all__ = [
#     "VeriGymEnv",
#     "ExplicitEnv",
#     "FrameworkExplicitEnv",
#     "GenerativeEnv",
#     "RewardFunction",
#     "TransitionFunction",
#     "exporter",
#     "create_abstraction",
#     "juliapomdp",
#     "mujoco",
#     "stormpy",
#     "PolicyClass",
#     "AbstractionMapper"
# ]


gym.register(
    id="ExplicitEnv-v0",
    entry_point=AbstractedExplicitEnv,
)

gym.register(id="FrameworkExplicitEnv-v0", entry_point=FrameworkExplicitEnv)
