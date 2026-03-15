from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box
from gymnasium.wrappers import TransformAction

from verigym.abstraction.discretization import (
    BinEdges,
    generate_box_bins,
    BinEdgeGenFunc,
)
from verigym.abstraction.gym_utils.mapping import box_to_discrete, get_discrete_box_tf

__all__ = [
    "DiscretizeBoxAction",
]


class DiscretizeBoxAction(TransformAction):
    """Discretization Wrapper for Box action spaces


    Parameters
    ----------
    env : gym.Env
        The environment containing a continuous action space
    n_samples : optional, int | npt.NDarray[np.integer], default=None
        The amount of samples to use for each dimension.
        If `n_samples` is an array it must have the same shape as the
        action space.
        If `n_samples` is not provided, then a valid Bins array must be
        provided
    bin_edges : BinEdges
        A valid BinEdges array describing how to discretize each dimension
    use_box_space : bool,default=True
        Whether to keep using the Box space or construct a discrete space
        (setting this to False will change the action space)
    **kwargs
        Additional keyword arguments that are forwarded to `generate_box_bins`

    Raises
    ------
    AssertionError
        Raises an AssertionError if the action space is not
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
        assert isinstance(env.action_space, Box), (
            f"The action space must be of type Box but found {env.action_space}"
        )
        assert not np.any(
            np.isinf(env.action_space.low) | np.isinf(env.action_space.high)
        ), (
            "Unable to discretize space with infinity bound, you might want to use the"
            "ClipAction wrapper"
        )
        if isinstance(bin_edges, Callable):
            assert n_samples is not None, (
                "If bin_edges is defined as a callable, n_samples must be either a valid integer\
                        or a numpy array with the same shape as the space"
            )
            bin_func = bin_edges
            bin_edges = generate_box_bins(
                env.action_space, bin_func, n_samples, **kwargs
            )
        space = env.action_space
        f = None
        if use_box_space:
            f = get_discrete_box_tf(env.action_space, bin_edges)
        else:
            space, _, to_continuous = box_to_discrete(env.action_space, bin_edges)
        self._bin_edges = bin_edges
        super().__init__(env, to_continuous, space)

    @property
    def bin_edges(self):
        return self._bin_edges
