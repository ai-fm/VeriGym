import gymnasium as gym

from verigym.environments.explicitenv import ExplicitEnv
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv

gym.register(
    id="ExplicitEnv-v0",
    entry_point=ExplicitEnv,
)

gym.register(id="FrameworkExplicitEnv-v0", entry_point=FrameworkExplicitEnv)
