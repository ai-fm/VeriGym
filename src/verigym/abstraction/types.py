import numpy.typing as npt
from collections.abc import Sequence, Callable
from typing import SupportsIndex



# --- type aliases -------------------------------------------------------------

type BinEdge = npt.NDArray
type BinEdgeGenFunc = Callable[[float, float, SupportsIndex], BinEdge]

# Backward-map payload shapes. Defined here rather than in `abstractionmapper`
# so that `BinEdges.idx_to_interval` can be annotated without importing upward.
type Point = npt.NDArray  # shape (*space.shape,)   -- one sample of original_space
type Interval = npt.NDArray  # shape (2, *space.shape) -- [0] = lower, [1] = upper
type StateSet = Sequence[npt.NDArray]  # iterable of samples of original_space