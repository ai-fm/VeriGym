from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, SupportsIndex
from itertools import product

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete, MultiDiscrete


__all__ = [
    "BinEdge",
    "BinEdges",
    "generate_box_bins",
    "generate_box_linspace_bins",
    "centered_pow_bin",
]


type BinEdge = npt.NDArray
# type BinEdges = Sequence[BinEdge | Sequence] | BinEdge
type BinEdgeGenFunc = Callable[[float, float, SupportsIndex], BinEdge]


@dataclass
class BinEdges:
    space: Box
    edges: npt.NDArray
    # sample_indices: npt.NDArray
    ranges: npt.NDArray

    def __getitem__(self, i: int):
        start, end = self.ranges[i]
        return self.edges[start:end]

    @property
    def nvec(self) -> npt.NDArray:
        lengths = self.ranges[:, 1] - self.ranges[:, 0]
        return lengths.reshape(self.space.shape)

    @cached_property
    def lengths(self) -> npt.NDArray:
        return self.ranges[:, 1] - self.ranges[:, 0]


def generate_box_bins(
    space: Box,
    bin_func: BinEdgeGenFunc,
    n_samples: int | npt.NDArray[np.integer[Any]],
) -> BinEdges:
    """Generate a Bins array from a Box space using the `bin_func` to generate the
    individual bins

    Parameters
    ----------
    space : Box
        The continuous space to create the BinEdges for
    bin_func : Callable[[float, float, int], npt.NDArray]
        A function taking in a start, end and the amount of samples as input
        and returns a numpy array with bin boundaries sorted in ascending order
    n_samples : int | array_like
        The amount of samples used to discretize each dimension
        If `n_samples` is an array it must have the same shape as the `space`

    Returns
    -------
    BinEdges
        A nested ``BinEdges`` structure (one ``BinEdge`` per dimension) describing
        how to discretize each dimension of the sample. A scalar ``Discrete`` space
        is treated as a single-dimension space, so the result is always nested
        (e.g. ``[array([...])]``), consistent with vector ``Box`` spaces.
    """
    if isinstance(space, Box):
        low = np.asarray(space.low)
        high = np.asarray(space.high)
    elif isinstance(space, Discrete):
        # Treat a scalar Discrete space as a single-dimension space so that the
        # returned BinEdges is nested (one BinEdge), consistent with Box spaces.
        low = np.array([space.start])
        high = np.array([space.start + space.n - 1])
    elif isinstance(space, MultiDiscrete):
        low = np.asarray(space.start)
        high = np.asarray(space.start + space.nvec - 1)
    else:
        raise TypeError(f"Unknown or unsupported type for gym.Space: {type(space) = }")

    if isinstance(n_samples, int):
        n_samples = np.full(low.shape, n_samples, dtype=np.int64)
    n_samples = np.asarray(n_samples)
    assert n_samples.shape == low.shape, (
        "If n_samples is an array it must have the same shape as the space"
    )
    assert np.all(n_samples >= 1), "Each bin must have at least one datapoint"

    edges, lengths = [], []
    for low_, high_, n_samples_ in zip(
        low.ravel(), high.ravel(), n_samples.ravel(), strict=True
    ):
        bin_edge = bin_func(low_, high_, n_samples_)
        edges.extend(bin_edge)
        lengths.append(len(bin_edge))
    ranges = np.lib.stride_tricks.sliding_window_view(np.cumsum([0] + lengths), 2)
    return BinEdges(space=space, edges=np.asarray(edges), ranges=ranges)


def generate_box_linspace_bins(space: Box, n_samples: int | npt.NDArray) -> BinEdges:
    """Generate a BinEdges array from a Box space using the `np.linspace` function to
    generate the individual bins.

    Parameters
    ----------
    space : Box
        The continuous space to create the Bins for
    n_samples : int | array_like
        The amount of samples used to discretize each dimension
        If `n_samples` is an array it must have the same shape as the `space`

    Returns
    -------
    BinEdges
        An array_like structure containing the BinEdges describing how to discretize
        each dimension of the sample. The BinEdges array usually has the same shape
        as the provided `space` with an additional dimension where individual bin
        arrays are located
    """
    return generate_box_bins(space, np.linspace, n_samples)


def centered_pow_bin(
    start: float | int, end: float | int, n_samples: int, power: int = 2
) -> BinEdge:
    """Generate an array containing bin boundaries following a polynomial function
    with one coefficient. The samples are taken from the interval [-1, 1] and then
    taken to the `power` of the provided value. After the application of the function
    the result is transformend to the target interval [`start`, `end`].

    Parameters
    ----------
    start : float | int
        The lowest value of the bin array
    end : float | int
        The highest value of the bin array
    n_samples : int
        How many samples to include in the final array

    Returns
    -------
    BinEdge
        The bin boundary array consisting of `n_samples` values
    """
    b = np.linspace(-1, 1, n_samples)
    sign = np.sign(b)
    b = np.pow(b, power)
    if power % 2 == 0:
        b *= sign
    b /= b.max()
    b = (b + 1) / 2
    b *= end - start
    return b + start


def nvec_from_samples(a: npt.NDArray):
    """Extract the `nvec` parameter for a `gymnasium.spaces.MultiDiscrete` space from a samples array.

    The function extracts all the unique observations for each dimension and creates an `nvec` from
    them.

    Parameters
    ----------
    a : npt.NDArray
        A numpy array containing observations from an environment or an MDP

    Returns
    -------
    npt.NDArray
        A vector that can be used to initialize

    Note
    ----
    The `nvec` only describes the amount of unique values in each dimension. It does
    not describe the bounds for these dimensions. Use `lookup_table_from_mdiscrete_samples` with
    the same input to retrieve a lookup table mapping the index in the `gym.spaces.MultiDiscrete` space
    to the actual values.
    """
    # TODO: Might be better in utils
    assert a.ndim > 1, (
        f"In order to generate an nvec, the array needs to have at least two dimension but has {a.ndim}"
    )
    nvec = np.zeros(a.shape[1:], dtype=int)
    for subview, subview_idx in subview_iter(a):
        unique_values = np.unique(subview)
        nvec[*subview_idx] = len(unique_values)
    return nvec


def subview_iter(
    a: npt.NDArray,
) -> Generator[tuple[npt.NDArray, tuple[int, ...]], None, None]:
    """
    Iterates over the subviews of an NDarray.

    This function generates subviews for all dimensions except the first
    (i.e., all but the batch dimension). Each yield returns a tuple containing
    the subview array and its corresponding index.

    Parameters
    ----------
    a : np.ndarray
        The input array to iterate over. Must have at least two dimensions.

    Yields
    ------
    tuple[npt.NDArray, tuple[int]]
        A tuple containing:
            - `subview`: ndarray, Subview of the input array for given indices.
            - `subview_idx`: tuple, Indices corresponding to the subview.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> list(subview_iter(a))
    [(array([1, 3]), (0,)), (array([2, 4]), (1,))]
    """
    shape_iter = [list(range(s_i)) for s_i in a.shape[1:]]
    for subview_idx in product(*shape_iter):
        yield a[:, *subview_idx], subview_idx
