from collections.abc import Callable, Sequence, Generator
from typing import Any
from itertools import product

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box



__all__ = [
    "BinEdge",
    "BinEdges",
    "verify_bin_edges",
    "generate_box_bins",
    "generate_box_linspace_bins",
    "centered_pow_bin",
    "generate_full_depth_bins",
]

type BinEdge = npt.NDArray
type BinEdges = Sequence[BinEdge | Sequence] | BinEdge
type BinEdgeGenFunc = Callable[[float, float, int], BinEdge]


def verify_bin_edges(sample, bin_edges: BinEdges) -> bool:
    """Return a boolean whether or not this `sample` can be discretized with the
    provided `bin_edges`. 

    Parameters
    ----------
    sample : npt.NDArray
        A sample from an n-dimensional space
    bin_edges : BinEdges
        The corresponding BinEdges structure used for discretization

    Returns
    -------
    bool
        True if `bin_edges` can be used for discretization
    """
    valid = True
    if np.isscalar(sample):
        return isinstance(bin_edges, np.ndarray | Sequence) and np.isscalar(
            bin_edges[0]
        )
    if len(sample) != len(bin_edges):
        return False
    for sub_sample, sub_bin_edges in zip(sample, bin_edges):
        valid &= verify_bin_edges(sub_sample, sub_bin_edges)
    return valid


def even_bin_edges(
    a: npt.NDArray,
    n_samples: int,
    start: int | float | None = None,
    end: int | float | None = None,
) -> BinEdge:
    """Generate bin edges following the density of the provided samples.
    The generated bin edges have the property that each bin has roughly the same amount of
    samples inside them.

    Parameters
    ----------
    a : npt.NDArray
        A one dimensional array of scalar values to generate the bin edges from
    n_samples : int
        The amount of edges in the generated bins
    start : int | float | None
        An optional value for the lowest bin edge, will automatically default to a.min()
        if not provided and has to be lower than or equal to a.min()
    end : int | float | None
        An optional value for the highest bin edge, will automatically default to a.max()
        if not provided and has to be higher than or equal to a.max()

    Returns
    -------
    BinEdge
        An array with exactly n_sample bin edges
    """
    assert a.ndim == 1, (
        f"a must be a one dimensional array of scalar values but has {a.ndim} dimensions"
    )
    assert n_samples <= len(a), "n_samples must be smaller than the length of a"
    s_a = np.sort(a, axis=0)
    if n_samples == 1:
        return np.array([s_a[len(s_a) // 2]])

    if start is not None:
        assert start <= s_a[0], (
            f"start must be smaller than or equal to the lowest \
                    value of a which is {s_a[0]}"
        )
        s_a[0] = start
    if end is not None:
        assert end >= s_a[-1], (
            f"end must be larger than or equal to the highest \
                    value of a which is {s_a[-1]}"
        )
        s_a[-1] = end
    bin_edge_idx = np.round(
        np.arange(n_samples) * (len(a) - 1) / (n_samples - 1)
    ).astype(int)
    bin_edges = s_a[bin_edge_idx]
    return bin_edges


def samples_to_bin_edges(
    a: npt.NDArray,
    n_samples: int | npt.NDArray[np.integer[Any]],
    low: int | float | npt.NDArray[np.integer | np.floating] | None = None,
    high: int | float | npt.NDArray[np.integer | np.floating] | None = None,
    f: Callable[[npt.NDArray, int, int | float, int | float], npt.NDArray]
    | None = None,
) -> BinEdges:
    """Generate BinEdges from a sequence of observations `a`
    The generated bin edges have the property that each bin has roughly the same amount of
    samples inside them. The resulting BinEdges have the same shape as one of the
    observations in a with an extra dimension which can be irregular and depends
    on n_samples. If n_samples is an integer the last dimension is equal to `n_samples`.
    If `n_samples` is an array the last dimension is inhomogenous
    and has the lengths of `n_samples`

    Parameters
    ----------
    a : npt.NDArray
        A one dimensional array of scalar values to generate the bin edges from
    n_samples : int | array_like
        The amount of edges in the generated bins. If n_samples is an array it must have
        the same shape as one of the observations in `a`
    f : Callable[[npt.NDArray, int, int | float, int | float], npt.NDArray], \
        default=even_bin_edges
        A function taking in a 1D array with values, the amount of samples and optional
        low and high limits to create one BinEdge array.
        Defaults to the even_bin_edges function
    low : int | float | array_like, default=None
        An optional lower boundary for the generated bin edges.
        Must be lower than a.min(axis=0) in all
        dimensions if provided as a scalar.
        If low is an array it must have the same shape as a single observation in `a`
    high : int | float | array_like, default=None
        An optional lower boundary for the generated bin edges.
        Must be lower than a.max(axis=0) in all
        dimensions if provided as a scalar.
        If high is an array it must have the same shape as a single observation in `a`
    """
    if f is None:
        f = even_bin_edges
    arr = np.sort(a, axis=0)
    if low is None:
        low = np.min(arr, axis=0)
    if high is None:
        high = np.max(arr, axis=0)
    if not isinstance(low, np.ndarray):
        low = (
            np.zeros(arr.shape[1:]) + low
        )  # TODO: No idea how to make the type checker happy...
    if not isinstance(high, np.ndarray):
        high = (
            np.zeros(arr.shape[1:]) + high
        )  # TODO: No idea how to make the type checker happy...
    if isinstance(n_samples, int):
        n_samples = (
            np.zeros(arr.shape[1:]) + n_samples
        )  # TODO: No idea how to make the type checker happy...
    low = np.array(low)
    high = np.array(high)
    n_samples = np.array(n_samples).astype(int)
    assert n_samples.shape == arr.shape[1:], (
        f"If n_samples is an array it must have the same shape \
                    as an element from a but they have shape \
                    {n_samples.shape} and {a.shape[1:]}"
    )
    assert low.shape == high.shape and low.shape == n_samples.shape, (
        f"low, high and n_samples must have the same shape but have \
                {low.shape}, {high.shape}, {n_samples.shape}"
    )

    def inner(arr, n_samples, low, high):
        if arr.ndim == 1:
            return f(arr, n_samples, low, high)
        # Recurse over each sub-array along axis 1
        return [
            inner(arr[:, i], n_samples[i], low[i], high[i]) for i in range(arr.shape[1])
        ]

    return inner(arr, n_samples, low, high)


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
        An array_like structure containing the `BinEdges` describing how to discretize
        each dimension of the sample. The `BinEdges` array usually has the same shape
        as the provided `space` with an additional dimension where individual bin
        arrays are located
    """
    if isinstance(n_samples, int):
        n_samples = np.zeros(space.shape, dtype=np.int64) + n_samples
    assert n_samples.shape == space.shape, (
        "If n_samples is an array it must have the same shape as the space"
    )
    assert np.all(n_samples >= 1), "Each bin must have at least one datapoint"

    def add_bin(low, high, samples):
        if not isinstance(low, np.ndarray):
            return bin_func(low, high, samples)
        level_bin = []
        for low_, high_, sample in zip(low, high, samples, strict=False):
            level_bin.append(add_bin(low_, high_, sample))
        return level_bin

    return add_bin(space.low, space.high, n_samples)


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


def nvec_from_bin_edges(bin_edges: BinEdges) -> npt.NDArray:
    """Create the `nvec` array that can be used for a `gym.spaces.MultiDiscrete` space
    from BinEdges

    Parameters
    ----------
    bin_edges : BinEdges
        The structure describing the discrete space

    Returns
    -------
    npt.NDArray
        A numpy array that can be used to specify the dimensionality
        of a `gym.spaces.MultiDiscrete` space
    """

    def inner(b):
        level = []
        if np.isscalar(b[0]):
            return len(b)
        for dim in b:
            level.append(inner(dim))
        return np.array(level, dtype=int)

    nvec = inner(bin_edges)
    assert isinstance(nvec, np.ndarray), (
        f"The provided bin_edges is too shallow, the nvec would be a scalar: {nvec}"
    )
    return nvec


def generate_full_depth_bins(shape: tuple, bins: BinEdges) -> BinEdges:
    """If the bins are shallow and do not provide a bin for every sample
    this function will extend the bins to do that by branching out
    from the lowest dimension

    If the Bins are already in the correct shape the result will simply equal the input

    Parameters
    ----------
    shape : Tuple
        The shape of the space the bins are supposed to operate on
    bins : Bins
        The shallow Bins to extend

    Returns
    -------
    Bins
        The complete Bins for the provided shape
    """

    # TODO: Add assertions for the correct bin shape
    def inner(d, template_bin):
        if len(shape) <= d:
            return template_bin.copy()
        level = []
        for i in range(shape[d]):
            if isinstance(template_bin[i], np.ndarray | Sequence):
                level_bin = inner(d + 1, template_bin[i])
            else:
                level_bin = inner(d + 1, template_bin)
            level.append(level_bin)
        return level

    return inner(0, bins)


def extract_arr_paths(structure, path=()):
    """Recursively extract the deepest sequences using DFS and return the
    sequence along with its path (indices) through the structure"""
    if not isinstance(structure[0], np.ndarray | Sequence):
        yield path, structure
    elif isinstance(structure, list):
        for i, item in enumerate(structure):
            assert isinstance(structure, np.ndarray | Sequence), (
                f"Expected a sequence but found: {structure}"
            )
            yield from extract_arr_paths(item, path + (i,))
    else:
        raise TypeError("Expected nested lists and numpy arrays only.")


def carthesian_bin_edges_iter(bin_edges: BinEdges):
    """Generator that yields Cartesian product samples preserving structure."""
    array_paths_and_values = list(extract_arr_paths(bin_edges))
    paths = [p for p, _ in array_paths_and_values]
    arrays = [a for _, a in array_paths_and_values]
    # For bin edges, the sample shape is the maximum
    # of the path array + 1 (shape is 1 indexed)
    sample_shape = np.max(paths, axis=0) + 1
    # Generate Cartesian product
    for combo in product(*arrays):
        sample = np.zeros(sample_shape)
        for path, value in zip(paths, combo):
            sample[path] = value
        yield sample



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