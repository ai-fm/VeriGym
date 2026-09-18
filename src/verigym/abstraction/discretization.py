"""
BinEdges, and the functions that translate between the four representations of a sample.

This module must not import from `abstractionmapper.py`! ; the dependency is
one-directional (`abstractionmapper -> discretization`).
"""

from collections.abc import Generator
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from numba import njit

from verigym.abstraction.types import Point, Interval, BinEdge, BinEdgeGenFunc

__all__ = [
    "BinEdge",
    "BinEdgeGenFunc",
    "BinEdges",
    "generate_box_bins",
    "centered_pow_bin",
    "nvec_from_samples",
    "subview_iter",
]





# --- njit kernels ------------------------------------------------------------


@njit
def _orig_to_idx_njit(
    flat_sample: npt.NDArray, edges: npt.NDArray, ranges: npt.NDArray
) -> npt.NDArray:
    """Map a flat sample from R^d to per-dimension bin indices in N^d via binary search.

    Parameters
    ----------
    flat_sample : npt.NDArray
        Sample from R^d as a flattened numpy array or a flat view.
    edges : npt.NDArray
        Flat array of bin edges for every dimension, concatenated; indexed via `ranges`.
    ranges : npt.NDArray
        Array of `(start, end)` pairs. The i-th value of `flat_sample` is
        discretized using `edges[start:end]` with `(start, end) = ranges[i]`.

    Returns
    -------
    npt.NDArray
        Discretized sample in N^d, dtype int64, same shape as `flat_sample`.
        
    Note
    ----
    The function was previously named `_sample_to_discrete_idx`.
    """
    discrete_sample = np.empty(flat_sample.shape, dtype=np.int64)
    for idx, (value, range_) in enumerate(zip(flat_sample, ranges)):
        start, end = range_
        bin_edges = edges[start:end]
        bin_edge_idx = np.searchsorted(bin_edges, value)
        if bin_edges[bin_edge_idx] != value:  # round to the left bin if not exact
            bin_edge_idx -= 1
        discrete_sample[idx] = bin_edge_idx
    return discrete_sample


@njit
def _orig_to_value_njit(
    flat_sample: npt.NDArray, edges: npt.NDArray, ranges: npt.NDArray
) -> npt.NDArray:
    """Map a flat sample from R^d onto the nearest lower bin edge, staying in R^d.

    Same binary search as `_orig_to_idx_njit`, but the bin edge value is
    written out instead of its index.

    Parameters
    ----------
    flat_sample : npt.NDArray
        Sample from R^d as a flattened numpy array or a flat view.
    edges : npt.NDArray
        Flat array of bin edges for every dimension, concatenated; indexed via `ranges`.
    ranges : npt.NDArray
        Array of `(start, end)` pairs, as in `_orig_to_idx_njit`.

    Returns
    -------
    npt.NDArray
        Snapped sample in R^d, same shape as `flat_sample`.
        
    Note
    ----
    The function was previously named `_sample_to_discrete_values`.
    """
    discrete_sample = np.empty(flat_sample.shape)
    for idx, (value, range_) in enumerate(zip(flat_sample, ranges)):
        start, end = range_
        bin_edges = edges[start:end]
        bin_edge_idx = np.searchsorted(bin_edges, value)
        if bin_edges[bin_edge_idx] != value:  # round to the left bin if not exact
            bin_edge_idx -= 1
        discrete_value = bin_edges[bin_edge_idx]
        discrete_sample[idx] = discrete_value
    return discrete_sample


# --- shared ravel/unravel helpers ---------------------------------------------
#
# Used by both `BinEdges` and `AbstractionMap`.


def _ravel(idx: npt.NDArray, nvec: npt.NDArray) -> int:
    """
    Flatten per-dimension indices `idx` into a single index over `nvec`, C order.
    """
    return int(np.ravel_multi_index(np.atleast_1d(idx).ravel(), nvec))


def _unravel(enum: int, nvec: npt.NDArray) -> npt.NDArray:
    """
    Expand a flat index `enum` into per-dimension indices over `nvec`, C order.
    """
    return np.asarray(np.unravel_index(enum, nvec))


# --- BinEdges ----------------------------------------------------------------


@dataclass(frozen=True)
class BinEdges:
    """A per-dimension binning of `space`, plus the conversions between representations.

    The four representations of a sample:

    - `O` (`orig`)  an arbitrary sample of the original space
    - `V` (`value`) a sample snapped onto bin edges; a finite subset of `O`
    - `I` (`idx`)   per-dimension bin indices; a sample of `MultiDiscrete(nvec)`
    - `E` (`enum`)  a single flat index; a sample of `Discrete(prod(nvec))`

    Because `V` lives inside the original space, `value_to_orig` is a dtype/shape
    cast rather than a real conversion, and `value_to_idx` is the same operation as
    `orig_to_idx`. 
    Six fundamental conversions: `orig_to_idx`, `orig_to_value`,
    `idx_to_value`, `idx_to_enum`, `enum_to_idx`, `value_to_orig`.
    The remaining conversions are compositions of these. 

    Attributes
    ----------
    space : Box
        The space this binning was built for. Populated for `Discrete` and
        `MultiDiscrete` inputs too, since `value_to_orig` needs a space.
    edges : npt.NDArray
        Flat concatenation of the bin edges of every dimension.
    ranges : npt.NDArray
        `(start, end)` index pairs into `edges`, one per flattened dimension.

    Notes
    -----
    The conversions are **methods** so that forward/backward maps handed to an
    `AbstractionMap` are bound methods (=picklable, as lambda funcs are not).
    """

    space: Box
    edges: npt.NDArray
    ranges: npt.NDArray

    def __getitem__(self, i: int) -> BinEdge:
        """Return the bin edges of the i-th flattened dimension.

        Parameters
        ----------
        i : int
            Index of the flattened dimension.

        Returns
        -------
        BinEdge
            The slice `edges[start:end]` with `(start, end) = ranges[i]`.
        """
        start, end = self.ranges[i]
        return self.edges[start:end]

    @cached_property
    def nvec(self) -> npt.NDArray:
        """Per-dimension bin counts, reshaped to `space.shape`.

        For a `Discrete` space this is 0-d (`array(3)`, shape `()`), because it
        reshapes to `space.shape == ()`. Use `lengths` instead when a flat array
        is needed, e.g. to build an abstract space.
        """
        lengths = self.ranges[:, 1] - self.ranges[:, 0]
        return lengths.reshape(self.space.shape)

    @cached_property
    def lengths(self) -> npt.NDArray:
        """Per-dimension bin counts as a flat 1-D array.

        Prefer this over `nvec` for ravel/unravel and for constructing
        abstract spaces -- it stays 1-D for a `Discrete` space.
        """
        return self.ranges[:, 1] - self.ranges[:, 0]

    # -- forward conversions ------------------------------------------

    def orig_to_idx(self, x: npt.NDArray) -> npt.NDArray:
        """Map an original sample to per-dimension bin indices (O -> I).

        Parameters
        ----------
        x : npt.NDArray
            A sample of `space`.

        Returns
        -------
        npt.NDArray
            Bin indices, shape `space.shape`, integer dtype. A valid sample of
            `MultiDiscrete(self.lengths)`.
        """
        flat = np.atleast_1d(x).ravel()
        idx = _orig_to_idx_njit(flat, self.edges, self.ranges)
        return idx.reshape(self.space.shape)

    def orig_to_value(self, x: npt.NDArray) -> npt.NDArray:
        """Discretize an original sample onto its bin edges (O -> V).

        Parameters
        ----------
        x : npt.NDArray
            A sample of `space`.

        Returns
        -------
        npt.NDArray
            The snapped sample, shape `space.shape`, float dtype. Still a
            member of the original space.
        """
        flat = np.atleast_1d(x).ravel()
        value = _orig_to_value_njit(flat, self.edges, self.ranges)
        return value.reshape(self.space.shape)

    def orig_to_enum(self, x: npt.NDArray) -> int:
        """Map an original sample to a single flat abstract index (O -> E).

        Composition of `idx_to_enum` and `orig_to_idx`.

        Parameters
        ----------
        x : npt.NDArray
            A sample of `space`.

        Returns
        -------
        int
            A flat index in `[0, prod(self.lengths))`.
        """
        return self.idx_to_enum(self.orig_to_idx(x))

    def value_to_idx(self, x: npt.NDArray) -> npt.NDArray:
        """Map an already discretized sample to per-dimension bin indices (V -> I).

        Essentially the same operation as `orig_to_idx`: We are using a sample from the
        original space's dtype, so we can use the same `orig_to_idx` function.
        
        Note:
        Exists for clarity: so call sites can name the representation they hold.
        If you, as a user find it unnecessary, please provide the feedback.

        Parameters
        ----------
        x : npt.NDArray
            A sample of `space` already sitting on bin edges.

        Returns
        -------
        npt.NDArray
            Bin indices, shape `space.shape`.
        """
        return self.orig_to_idx(x)

    def value_to_enum(self, x: npt.NDArray) -> int:
        """Map a discretized sample to a single flat abstract index (V -> E).

        Parameters
        ----------
        x : npt.NDArray
            A sample of `space` already sitting on bin edges.

        Returns
        -------
        int
            A flat index in `[0, prod(self.lengths))`.
        """
        return self.idx_to_enum(self.value_to_idx(x))

    def idx_to_enum(self, x: npt.NDArray) -> int:
        """Flatten per-dimension bin indices into a single enumerated index (I -> E).

        Parameters
        ----------
        x : npt.NDArray
            Bin indices, shape `space.shape`.

        Returns
        -------
        int
            `np.ravel_multi_index` of the raveled indices over `self.lengths`.
        """
        return _ravel(x, self.lengths)

    # -- backward conversions -------------------------------------------------

    def enum_to_idx(self, x: int) -> npt.NDArray:
        """Expand an enumerated index back into per-dimension bin indices (factored) (E -> I).

        Parameters
        ----------
        x : int
            A flat index in `[0, prod(self.lengths))`.

        Returns
        -------
        npt.NDArray
            Bin indices reshaped to `space.shape`. Exact inverse of `idx_to_enum`.
        """
        idx = _unravel(x, self.lengths)
        return idx.reshape(self.space.shape)

    def enum_to_value(self, x: int) -> npt.NDArray:
        """Map a enumeration index to its representative discretized (factored) sample (E -> V).

        Parameters
        ----------
        x : int
            A flat index in `[0, prod(self.lengths))`.

        Returns
        -------
        npt.NDArray
            The representative bin-edge value, shape `space.shape`, float dtype.
        """
        return self.idx_to_value(self.enum_to_idx(x))

    def enum_to_orig(self, x: int) -> Point:
        """Map an enumerated index to a valid sample of `space` (E -> O).

        Parameters
        ----------
        x : int
            A flat index in `[0, prod(self.lengths))`.

        Returns
        -------
        Point
            A representative sample satisfying `self.space.contains(...)`.
        """
        return self.value_to_orig(self.enum_to_value(x))

    def idx_to_value(self, x: npt.NDArray) -> npt.NDArray:
        """Map per-dimension bin indices to their bin-edge values (I -> V).

        Parameters
        ----------
        x : npt.NDArray
            Bin indices, shape `space.shape`.

        Returns
        -------
        npt.NDArray
            The lower edge of each indexed bin, shape `space.shape`, float dtype.
        """
        flat = np.atleast_1d(x).ravel()
        result = np.empty(flat.shape)
        for i, idx in enumerate(flat):
            result[i] = self[i][idx]
        return result.reshape(self.space.shape)

    def idx_to_orig(self, x: npt.NDArray) -> Point:
        """Map per-dimension bin indices to a valid sample of `space` (I -> O).

        Parameters
        ----------
        x : npt.NDArray
            Bin indices, shape `space.shape`.

        Returns
        -------
        Point
            A representative sample satisfying `self.space.contains(...)`.
        """
        return self.value_to_orig(self.idx_to_value(x))

    def value_to_orig(self, x: npt.NDArray) -> Point:
        """Cast a discretized value (factored) to the dtype and shape of `space` (V -> O).

        Only a cast, no numerical conversion happens. Bin edges do not land on
        integers when the bin count does not evenly divide a
        `Discrete`/`MultiDiscrete` range, so the result is rounded; left uncast,
        it would fail `space.contains(...)`.

        Parameters
        ----------
        x : npt.NDArray
            A snapped bin-edge value.

        Returns
        -------
        Point
            The same value as a valid sample of `space`.
        """
        value = np.atleast_1d(x)
        if isinstance(self.space, Discrete):
            return int(np.rint(value.reshape(-1)[0]))
        if isinstance(self.space, MultiDiscrete):
            return np.rint(value).astype(self.space.dtype).reshape(self.space.shape)
        return value.astype(self.space.dtype).reshape(self.space.shape)

    def idx_to_interval(self, x: npt.NDArray) -> Interval:
        """Map bin indices to the interval they could come from in the original space (I -> Interval).

        Parameters
        ----------
        x : npt.NDArray
            Bin indices, shape `space.shape`.

        Returns
        -------
        Interval
            Array of shape `(2, *space.shape)`; `[0]` holds the lower bound per
            dimension, `[1]` the upper. This is the canonical
            `BackwardKind.INTERVAL` payload.

        Notes
        -----
        The last interval per dimension is the degenerate `[high, high]`: index
        `lengths[i] - 1` is reachable only by exactly `high` along that
        dimension. See the `n_bins` note on `generate_box_bins`.
        """
        flat = np.atleast_1d(x).ravel()
        lower = np.empty(flat.shape)
        upper = np.empty(flat.shape)
        for i, k in enumerate(flat):
            edges_i = self[i]
            lower[i] = edges_i[k]
            upper[i] = edges_i[min(k + 1, len(edges_i) - 1)]
        return np.stack([lower.reshape(self.space.shape), upper.reshape(self.space.shape)])


# --- bin generation ----------------------------------------------------------


def generate_box_bins(
    space: Box,
    bin_func: BinEdgeGenFunc,
    n_bins: int | npt.NDArray[np.integer[Any]],
) -> BinEdges:
    """Build a `BinEdges` for `space`, using `bin_func` per dimension.
    Use this function with custom `bin_func` parameter for custom binning.

    Parameters
    ----------
    space : Box
        The space to create the `BinEdges` for. `Discrete` and `MultiDiscrete`
        are also accepted and are treated as integer-valued ranges.
    bin_func : BinEdgeGenFunc
        A function `(start, end, n) -> NDArray` returning bin boundaries sorted
        ascending, e.g. `np.linspace` or `centered_pow_bin`.
    n_bins : int | array_like
        Bins per dimension. If an array, it must have the same shape as `space`.

    Returns
    -------
    BinEdges
        One `BinEdge` per flattened dimension of `space`, with `space` retained
        on the result. A scalar `Discrete` space is treated as a single-dimension
        space, so the result is always nested, consistent with vector `Box` spaces.

    Raises
    ------
    TypeError
        If `space` is not a `Box`, `Discrete` or `MultiDiscrete`.

    Notes
    -----
    TODO: 
    - `n_bins` is the number of representative values per dimension, so the
    last interval per dimension is the degenerate `[high, high]` (see
    `BinEdges.idx_to_interval`) -- reachable only by exactly `high`. An
    alternative where `n_bins` counts cells instead would avoid the degenerate
    interval but renumber previously exported MDPs; not adopted for now.
    - Also see the PR where BinEdges are supposed to be changed.
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

    if isinstance(n_bins, int):
        n_bins = np.full(low.shape, n_bins, dtype=np.int64)
    n_bins = np.asarray(n_bins)
    assert n_bins.shape == low.shape, (
        "If n_samples is an array it must have the same shape as the space"
    )
    assert np.all(n_bins >= 1), "Each bin must have at least one datapoint"

    edges, lengths = [], []
    for low_, high_, n_samples_ in zip(
        low.ravel(), high.ravel(), n_bins.ravel(), strict=True
    ):
        bin_edge = bin_func(low_, high_, n_samples_)
        edges.extend(bin_edge)
        lengths.append(len(bin_edge))
    ranges = np.lib.stride_tricks.sliding_window_view(np.cumsum([0] + lengths), 2)
    return BinEdges(space=space, edges=np.asarray(edges), ranges=ranges)




def centered_pow_bin(
    start: float | int, end: float | int, n_samples: int, power: int = 2
) -> BinEdge:
    """Generate bin boundaries following a one-coefficient polynomial spacing.

    An input `x` is transformed as `x**power`. Samples are taken from the
    interval `[-1, 1]`, raised to `power` (sign-preserved for even powers),
    normalised, and then affinely mapped onto `[start, end]`. The result is
    denser near the centre of the interval than `np.linspace`.

    Parameters
    ----------
    start : float | int
        The lowest value of the bin array.
    end : float | int
        The highest value of the bin array.
    n_samples : int
        How many boundaries to include in the final array.
    power : int, default 2
        The power to raise each element by.

    Returns
    -------
    BinEdge
        A bin boundary array of `n_samples` values, sorted ascending.
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


# --- validation --------------------------------------------------------------


def _check_compatible(space: Box, bin_edges: BinEdges) -> None:
    """Raise if `bin_edges` was not built for a space shaped like `space`.

    Parameters
    ----------
    space : Box
        The space whose samples are to be discretized.
    bin_edges : BinEdges
        The discretization structure to check against `space`.

    Raises
    ------
    ValueError
        If `space.shape != bin_edges.space.shape`, with both shapes in the message.
    """
    if space.shape != bin_edges.space.shape:
        raise ValueError(
            "The provided BinEdges are incompatible with the space "
            f"The provided space has shape: {space.shape} and the BinEdges "
            f"were create with a space of shape {bin_edges.space.shape}."
        )
