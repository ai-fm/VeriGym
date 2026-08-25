import gymnasium as gym
from gymnasium import ObservationWrapper
import numpy as np
from itertools import product

from ..environments import VeriGymEnv
from .abstractionmapper import AbstractionMapper, AbstractionMap

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
    if not (isinstance(original_env.observation_space, gym.spaces.MultiDiscrete) or isinstance(original_env.observation_space, gym.spaces.Box)):
        raise ValueError(f"Cannot select features from observation_space of type {type(original_env.observation_space)}. Please ensure it is either MultiDiscrete or Box.")
    
    if method == "binning":
        feature_env = BinFeaturesWrapper(original_env, reduce_indices)

    elif method == "masking":
        feature_env = ReduceFeaturesWrapper(original_env, reduce_indices)
    else:
        raise NotImplementedError(f"The given method {method} is not implemented for feature selection.")

    if isinstance(original_env.observation_space, gym.spaces.MultiDiscrete):
        def backward_map(abs_obs):
            # returns a concrete set of states
            orig_states = []
            vals = [np.arange(original_env.observation_space.nvec[i]).tolist() for i in reduce_indices]
            base_state = []
            abs_idx = 0
            for i in range(original_env.observation_space.shape[0]):
                if i not in reduce_indices:
                    base_state.append(abs_obs[abs_idx])
                    abs_idx += 1
                else:
                    base_state.append(0)
                    if len(abs_obs) == original_env.observation_space.shape[0]:
                        abs_idx += 1

            for element in product(*vals):
                state = base_state.copy()
                for e, idx in enumerate(reduce_indices):
                    state[idx] = element[e]
                orig_states.append(np.array(state))

            return orig_states

    elif isinstance(original_env.observation_space, gym.spaces.Box):
        def backward_map(abs_obs):
            # returns upper and lower bounds
            lower = []
            upper = []
            reduced_idx = 0
            for i in range(original_env.observation_space.shape[0]):
                if i in reduce_indices:
                    lower.append(original_env.observation_space.low[i])
                    upper.append(original_env.observation_space.high[i])
                else:
                    lower.append(abs_obs[reduced_idx])
                    upper.append(abs_obs[reduced_idx])
                    reduced_idx += 1
            return (lower, upper)

    state_abstraction_map = AbstractionMap(
        forward_map=lambda obs: feature_env.observation(obs),
        backward_map=lambda abs_obs: backward_map(abs_obs),
        original_space=original_env.observation_space,
        abstract_space=feature_env.observation_space
    )
    
    action_abstraction_map = AbstractionMap.create_identity_map(original_env.action_space)

    abstraction_mapper = AbstractionMapper(
        state_abstraction_map=state_abstraction_map,
        action_abstraction_map=action_abstraction_map
    )
    
    return feature_env, abstraction_mapper


class ReduceFeaturesWrapper(ObservationWrapper):
    """
    Wrapper that can be applied to observation spaces of types `gym.spaces.MultiDiscrete` and `gym.spaces.Box`
    for feature selection.

    The wrapper modifies the dimensionality of the original observation space by "removing" features specified in `reduce_indices` upon initialization.
    I.e., the output observation space will have less features than the original.
    The type of observation space and limits/number of possible values remain as in the original.
    """
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
    """
    Wrapper that can be applied to observation spaces of types `gym.spaces.MultiDiscrete` and `gym.spaces.Box`
    for feature selection.

    The wrapped observation space has the same shape, limits, and type as the original observation.
    `reduce_indices` specify the features to be reduced. 
    In case of `gym.spaces.Box`, for an observation `obs`, `obs[reduce_indices]` will take the center of the feature's interval as value.
    In case of `gym.spaces.MultiDiscrete`, for an observation `obs`, `obs[reduce_indices] == 0`.
    """
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