import gymnasium as gym

from verigym.frameworks.stormpy.stormpyenv import StormpyEnv

gym.register(
    id="StormpyEnv-v0",
    entry_point=StormpyEnv,
)
