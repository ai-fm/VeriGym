from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box
from gymnasium.wrappers import TransformObservation

from verigym.abstraction.discretization import (
    BinEdges,
    generate_box_bins,
    BinEdgeGenFunc,
)
from verigym.abstraction.gym_utils.mapping import box_to_discrete, get_discrete_box_tf
from verigym.abstraction.gym_utils.finite_space import is_bounded_space

__all__ = [
    "DiscretizeBoxObservation",
    "ReplaceInfObservation",
]


class DiscretizeBoxObservation(TransformObservation):
    """Discretization Wrapper for Box observation spaces


    Parameters
    ----------
    env : gym.Env
        The environment containing a continuous observation space
    n_samples : optional, int | npt.NDarray[np.integer], default=None
        The amount of samples to use for each dimension.
        If `n_samples` is an array it must have the same shape as the
        observation space.
        If `n_samples` is not provided, then a valid Bins array must be
        provided
    bin_edges : BinEdges
        A valid BinEdges array describing how to discretize each dimension
    use_box_space : bool,default=True
        Whether to keep using the Box space or construct a discrete space
        (setting this to False will change the observation space)
    **kwargs
        Additional keyword arguments that ar e forwarded to `generate_box_bins`

    Raises
    ------
    AssertionError
        Raises an AssertionError if the observation space is not
        a continuous Box space or if one of the boundaries is infinite
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        n_samples: int | npt.NDArray[np.integer[Any]] | None = None,
        bin_edges: BinEdges | BinEdgeGenFunc = np.linspace,
        use_box_space: bool = True,
        **kwargs,
    ):
        # assert isinstance(env.observation_space, Box), (
        #     f"The observation space must be of type Box but found {env.observation_space}"
        # )
        assert is_bounded_space(env.observation_space)
        
        if isinstance(bin_edges, Callable):
            assert n_samples is not None, (
                "If bins is defined as a string, n_samples must be either a valid integer\
                        or a numpy array with the same shape as the space" 
            )
            bin_func = bin_edges
            bin_edges = generate_box_bins(
                env.observation_space, bin_func, n_samples, **kwargs
            )
        space = env.observation_space
        f = None
        if use_box_space:
            f = get_discrete_box_tf(env.observation_space, bin_edges)
        else:
            space, f, _ = box_to_discrete(env.observation_space, bin_edges)
        self._bin_edges = bin_edges
        super().__init__(env, f, space)

    @property
    def bin_edges(self):
        return self._bin_edges


class ReplaceInfObservation(TransformObservation):
    """Replace the infinite boundaries of a Box observation_space

    Parameters
    ----------
    env : gym.Env
        A gymnasium environment with a Box observation space
    neg_inf : int | float | npt.NDArray
        The value to replace all negative infinity boundaries with
        If `neg_inf` is an array it must have the same shape as the space
    pos_inf : int | float | npt.NDArray
        The value to replace all positive infinity boundaries with
        If `pos_inf` is an array it must have the same shape as the space

    Raises
    ------
    AssertionError
        Whenever the observation space is not of type Box
        or whenever the shape of `neg_inf` or `pos_inf` is invalid
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        neg_inf: int | float | npt.NDArray[Any],
        pos_inf: int | float | npt.NDArray[Any],
    ):
        space = env.observation_space
        assert isinstance(space, Box), "Only Box observation spaces are supported"
        low, high = space.low.copy(), space.high.copy()
        if isinstance(neg_inf, np.ndarray):
            assert neg_inf.shape == low.shape, (
                f"If neg_inf is an array it must have the same shape as \
                        the space boundary, space boundary has shape {low.shape} \
                        but neg_inf has shape {neg_inf.shape}"
            )
            low[np.isneginf(low)] = neg_inf[np.isneginf(low)]
        else:
            low[np.isneginf(low)] = neg_inf
        if isinstance(pos_inf, np.ndarray):
            assert pos_inf.shape == low.shape, (
                f"If pos_inf is an array it must have the same shape as \
                        the space boundary, space boundary has shape {low.shape} \
                        but neg_inf has shape {neg_inf.shape}"
            )
            high[np.isposinf(high)] = pos_inf[np.isposinf(high)]
        else:
            high[np.isposinf(high)] = pos_inf
        observation_space = space.__class__(low, high)

        def map_obs(obs):
            if np.any(obs < low):
                pass
            if np.any(obs > high):
                pass
            return obs

        super().__init__(env, map_obs, observation_space)


class MDiscreteWrapper(gym.wrappers.TransformObservation):
    def __init__(self, env):
        assert isinstance(env.observation_space, gym.spaces.Discrete), (
            "Can only be used on gym.spaces.Discrete observation spaces"
        )
        self._obs_start = env.observation_space.start
        super().__init__(
            env,
            self.transform,
            gym.spaces.MultiDiscrete(nvec=[env.observation_space.n]),
        )

    def transform(self, obs):
        return [obs - self._obs_start]
