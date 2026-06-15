import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium import ObservationWrapper
import numpy as np

from ..environments import VeriGymEnv
from .abstractionmapper import AbstractionMapper

def state_feature_selection(
        original_env: VeriGymEnv,
        method: str,
        reduce_indices: list
) -> tuple[VeriGymEnv, AbstractionMapper]:
    """
    Applies feature selection to the observation space of a VeriGymEnv. The returned environment is of the same type as the input.
    Feature selection does not include model learning.

    Parameters
    ----------
    original_env: VeriGymEnv
        The environment / model to apply feature selection to. 
        The environment's observation space should be gym.spaces.MultiDiscrete or gym.spaces.Box.
    method: str
        The method of feature selection. 
        "binning" maintains the shape of the observation space. The selected features will be represented by a single valuation.
        "masking" adapts the shape of the observation space by removing selected features.
    reduce_indices : list
        The features selected for reduction. If the list is empty, the original model is retained.
    """
    assert isinstance(original_env.observation_space, gym.spaces.MultiDiscrete) or isinstance(original_env.observation_space, gym.spaces.Box), \
        f"Cannot select features from observation_space of type {type(original_env.observation_space)}. Please ensure it is either MultiDiscrete or Box."
    
    if method == "binning":
        feature_env = BinFeaturesWrapper(original_env, reduce_indices)
        state_abstraction_map = ...
    elif method == "masking":
        feature_env = ReduceFeaturesWrapper(original_env, reduce_indices)
        state_abstraction_map = ...
    else:
        raise NotImplementedError(f"The given method {method} is not implemented for feature selection.")
    
    abstraction_mapper = AbstractionMapper(
        state_abstraction_map=state_abstraction_map
    )
    
    return feature_env, abstraction_mapper


class ReduceFeaturesWrapper(ObservationWrapper):
    def __init__(self, env, reduce_indices):
        super().__init__(env)

        self.keep_indices = [i for i in range(env.observation_space.shape[0])
                             if i not in reduce_indices]

        if isinstance(env.observation_space, gym.spaces.MultiDiscrete):
            self.observation_space = gym.spaces.MultiDiscrete(
                env.observation_space.nvec[self.keep_indices]
            )

        elif isinstance(env.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=env.observation_space.low[self.keep_indices],
                high=env.observation_space.high[self.keep_indices],
                dtype=env.observation_space.dtype,
            )
        
        else:
            raise TypeError(
                f"Unsupported observation space of type {type(env.observation_space)}"
            )

    def observation(self, obs):
        return obs[self.keep_indices]
    
class BinFeaturesWrapper(ObservationWrapper):
    def __init__(self, env, reduce_indices):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.reduce_indices = reduce_indices

        if isinstance(self.observation_space, gym.spaces.MultiDiscrete):
            self.reduce_vals = np.zeros_like(self.observation_space.nvec)

        elif isinstance(self.observation_space, gym.spaces.Box):
            # Take the midpoint of each dimension interval
            self.reduce_vals = (self.observation_space.low + self.observation_space.high) / 2.0

        else:
            raise TypeError(
                f"Unsupported observation space of type {type(env.observation_space)}"
            )
    
    def observation(self, obs):
        binned_obs = obs.copy()
        for i in self.reduce_indices:
            binned_obs[i] = self.reduce_vals[i]
        return binned_obs