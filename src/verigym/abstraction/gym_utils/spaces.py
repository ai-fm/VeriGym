import math

import gymnasium.spaces
import numpy as np

__all__ = [
    "is_bounded_space",
    "get_n_elements_of_space"
]

infty = math.inf

def is_bounded_space(space: gymnasium.spaces.Space) -> bool:
    """Return True if the space has finite (non-infinite) bounds, False otherwise.

    Parameters
    ----------
    space:
        The gymnasium space to check.
    """

    if isinstance(space, (gymnasium.spaces.Discrete, gymnasium.spaces.MultiBinary, gymnasium.spaces.MultiDiscrete)):
        return True

    if isinstance(space, gymnasium.spaces.Box):
        bounds_finite = bool(np.any(space.bounded_above) or np.any(space.bounded_below))
        return True if bounds_finite else False

    if isinstance(space, gymnasium.spaces.Dict):
        return all(is_bounded_space(s) for s in space.spaces.values())

    if isinstance(space, gymnasium.spaces.Tuple):
        return all(is_bounded_space(s) for s in space.spaces)

    if isinstance(space, gymnasium.spaces.OneOf):
        return all(is_bounded_space(s) for s in space.spaces)

    if isinstance(space, gymnasium.spaces.Text):
        # max_length is a required integer argument in this gymnasium version,
        # so Text is always bounded over a finite charset.
        return True

    # Graph and Sequence have variable structure/length → infinite
    # if we reach until here, we could not assure that we space is finite
    return False


def get_n_elements_of_space(space: gymnasium.spaces.Space) -> int | float:
    """
    Returns the number of elements of a given `gym.Space`.

    Parameters
    ----------
    space : gymnasium.spaces.Space
        The gym space.

    Returns
    -------
    int | float
        Returns `int` if finite elements and `float('inf')` if infinite.
    """
    
    if isinstance(space, gymnasium.spaces.Discrete):
        return int(space.n)
    
    if isinstance(space, gymnasium.spaces.MultiDiscrete):
        return  np.prod(space.nvec) 
    
    if isinstance(space, gymnasium.spaces.MultiBinary):
        return 2**np.sum(space.n)

    if isinstance(space, gymnasium.spaces.Box):
        return infty
    
    raise ValueError(f"Unsupported gym.Space type. It is not clear / not implemented on how to compute the number of elements in this space. {type(space) = }")
    