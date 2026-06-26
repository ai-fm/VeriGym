import gymnasium.spaces
import numpy as np


def is_finite_space(space: gymnasium.spaces.Space, raise_if_infinite: bool = False) -> bool:
    """Return True if the space has a finite number of elements, False otherwise.

    Parameters
    ----------
    space:
        The gymnasium space to check.
    raise_if_infinite:
        If True, raise a ValueError when the space is not finite instead of
        returning False.
    """

    def _not_finite() -> bool:
        if raise_if_infinite:
            raise ValueError(
                f"Space {space} is not finite. Consider using the ClipAction wrapper "
                "to bound the space before proceeding."
            )
        return False

    if isinstance(space, (gymnasium.spaces.Discrete, gymnasium.spaces.MultiBinary, gymnasium.spaces.MultiDiscrete)):
        return True

    if isinstance(space, gymnasium.spaces.Box):
        # Integer dtypes are always bounded: gymnasium clamps ±inf to the dtype's iinfo range.
        # Float dtypes are continuous (uncountably infinite) regardless of bounds.
        return True if np.issubdtype(space.dtype, np.integer) else _not_finite()

    if isinstance(space, gymnasium.spaces.Dict):
        return all(is_finite_space(s, raise_if_infinite) for s in space.spaces.values())

    if isinstance(space, gymnasium.spaces.Tuple):
        return all(is_finite_space(s, raise_if_infinite) for s in space.spaces)

    if isinstance(space, gymnasium.spaces.OneOf):
        return all(is_finite_space(s, raise_if_infinite) for s in space.spaces)

    if isinstance(space, gymnasium.spaces.Text):
        # max_length is a required integer argument in this gymnasium version,
        # so Text is always bounded over a finite charset.
        return True

    # Graph and Sequence have variable structure/length → infinite
    return _not_finite()
