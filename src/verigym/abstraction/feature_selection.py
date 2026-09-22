import functools

import gymnasium as gym
from gymnasium import ObservationWrapper
import numpy as np
from itertools import product
from numpy.typing import NDArray

from ..environments import VeriGymEnv
from .abstractionmapper import AbstractionMapper, AbstractionMap, enumeration_of_space


def _multidiscrete_backward_map(
    reduced_obs: NDArray, *, nvec: NDArray, reduced_indices: list[int], n_dims: int
):
    """List every original observation that feature selection maps onto `reduced_obs`.

    Feature selection throws away some dimensions of an observation, namely those
    in `reduce_indices`. Going backwards, their values are unknown: each removed
    dimension could have held any of its possible values. This function returns
    all of those candidates (a set). The kept dimensions are copied from
    `reduced_obs`, and the removed ones run through every combination of their
    values.

    Works for both feature-selection methods:

    - "masking" deletes the removed dimensions, so `reduced_obs` has
      `n_dims - len(reduce_indices)` entries.
    - "binning" keeps them but sets them to 0, so `reduced_obs` has `n_dims` entries
      and those placeholder zeros are skipped.

    Parameters
    ----------
    reduced_obs : NDArray
        An observation after feature selection, where the observation may be missing 
        one or more dimensions.
    nvec : NDArray
        Number of possible values of each dimension of the original
        `MultiDiscrete` space.
    reduced_indices : list[int]
        The dimensions that feature selection removed.
    n_dims : int
        Number of dimensions of the original observation.

    Returns
    -------
    list[NDArray]
        All `prod(nvec[reduce_indices])` original observations. This is the
        `"set"`-type BackwardKind.

    Examples
    --------
    Original space `MultiDiscrete([2, 3, 4])`, dimension 1 removed. The abstract
    observation `[1, 3]` could have come from any of the 3 values of dimension 1:

    >>> _multidiscrete_backward_map(
    ...     np.array([1, 3]), nvec=np.array([2, 3, 4]), reduce_indices=[1], n_dims=3
    ... )
    [array([1, 0, 3]), array([1, 1, 3]), array([1, 2, 3])]
    """
    orig_states = []
    # all possible values of each removed dimension
    vals = [np.arange(nvec[i]).tolist() for i in reduced_indices]

    # template observation: kept dimensions copied, removed ones filled in below
    base_state = []
    reduced_idx = 0  # read position in `reduced_obs`
    for i in range(n_dims):
        if i not in reduced_indices:
            base_state.append(reduced_obs[reduced_idx])
            reduced_idx += 1
        else:
            base_state.append(0)  # placeholder
            if len(reduced_obs) == n_dims:  # "binning": skip the placeholder zero
                reduced_idx += 1

    # one original observation per combination of removed-dimension values
    for element in product(*vals):
        state = base_state.copy()
        for e, idx in enumerate(reduced_indices):
            state[idx] = element[e]
        orig_states.append(np.array(state))

    return orig_states


def _box_backward_map(
    reduced_obs: NDArray,
    *,
    low: NDArray,
    high: NDArray,
    reduced_indices: list[int],
    n_dims: int,
):
    """Return the region of original observations that map onto `reduced_obs`.

    Feature selection throws away some dimensions of an observation, namely those
    in `reduce_indices`. Going backwards, a removed dimension could have held any
    value within its bounds, while a kept dimension is known exactly. So the
    answer is a box, given by its lower and upper corner:

    - a kept dimension is the single point `[value, value]` taken from `reduced_obs`,
    - a removed dimension spans its whole original range `[low, high]`.

    Only observations from the "masking" method are handled correctly, i.e.
    `reduced_obs` must not contain the removed dimensions. With "binning" they are
    still present (set to their midpoints) and are not skipped, so the values of
    the kept dimensions end up shifted.

    Parameters
    ----------
    reduced_obs : NDArray
        An observation after feature selection, where the observation may be missing 
        one or more dimensions.
    low : NDArray
        Lower bounds of the original `Box` space.
    high : NDArray
        Upper bounds of the original `Box` space.
    reduced_indices : list[int]
        The dimensions that feature selection removed.
    n_dims : int
        Number of dimensions of the original observation.

    Returns
    -------
    NDArray
        Shape `(2, n_dims)`: row 0 holds the lower bounds, row 1 the upper bounds.
        This is the payload of the `"interval"` backward kind.

    Examples
    --------
    Original bounds `low = [0, 10, 20]`, `high = [1, 11, 21]`, dimension 0
    removed. Dimension 0 spans its full range; the other two are pinned to the
    observed values:

    >>> _box_backward_map(
    ...     np.array([10.3, 20.7]),
    ...     low=np.array([0.0, 10.0, 20.0]),
    ...     high=np.array([1.0, 11.0, 21.0]),
    ...     reduce_indices=[0],
    ...     n_dims=3,
    ... )
    array([[ 0. , 10.3, 20.7],
           [ 1. , 10.3, 20.7]])
    """
    lower = []
    upper = []
    reduced_idx = 0  # read position in `reduced_obs`
    for i in range(n_dims):
        if i in reduced_indices:
            # removed dimension: unknown, so its whole original range
            lower.append(low[i])
            upper.append(high[i])
        else:
            # kept dimension: known exactly, so a single point
            lower.append(reduced_obs[reduced_idx])
            upper.append(reduced_obs[reduced_idx])
            reduced_idx += 1
    return np.stack([np.asarray(lower), np.asarray(upper)])  # row 0 lower, row 1 upper


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

    n_dims = original_env.observation_space.shape[0]
    if isinstance(original_env.observation_space, gym.spaces.MultiDiscrete):
        backward_map = functools.partial(
            _multidiscrete_backward_map,
            nvec=original_env.observation_space.nvec,
            reduced_indices=reduce_indices,
            n_dims=n_dims,
        )
        backward_kind = "set"
        # The reduced observation space stays MultiDiscrete, so it can be enumerated.
        abstract_to_enum, enum_to_abstract = enumeration_of_space(feature_env.observation_space)
    else:  # gym.spaces.Box, per the isinstance check above
        backward_map = functools.partial(
            _box_backward_map,
            low=original_env.observation_space.low,
            high=original_env.observation_space.high,
            reduced_indices=reduce_indices,
            n_dims=n_dims,
        )
        backward_kind = "interval"
        # A Box abstract space has infinitely many elements: no enumeration exists.
        abstract_to_enum, enum_to_abstract = None, None

    # `feature_env.observation` is a bound method (not a lambda), so the map --
    # and therefore the mapper -- stays picklable for `multithreading=True`
    # (a lambda-based map raises `PicklingError` there; see `AbstractionMap`'s
    # factory notes).
    state_abstraction_map = AbstractionMap(
        forward_map=feature_env.observation,
        backward_map=backward_map,
        original_space=original_env.observation_space,
        abstract_space=feature_env.observation_space,
        backward_kind=backward_kind,
        abstract_to_enum=abstract_to_enum,
        enum_to_abstract=enum_to_abstract,
    )

    action_abstraction_map = AbstractionMap.initialize_identity_map(original_env.action_space)

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

    def observation(self, observation):
        return observation[self.keep_indices]
    
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