from collections.abc import Callable
from functools import partial
from typing import Any

from numba import njit
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
import gymnasium as gym
from numpy.typing import NDArray

from verigym.abstraction.discretization import (
    BinEdges,
)


@njit
def _sample_to_discrete_values(
    flat_sample: npt.NDArray, edges: npt.NDArray, ranges: npt.NDArray
) -> npt.NDArray:
    """Returns the sample discretized in the `Box` space"""
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


@njit
def _sample_to_discrete_idx(
    flat_sample: npt.NDArray, edges: npt.NDArray, ranges: npt.NDArray
) -> npt.NDArray:
    """Returns the sample discretized in a `Discrete` space"""
    discrete_sample = np.empty(flat_sample.shape, dtype=np.int64)
    for idx, (value, range_) in enumerate(zip(flat_sample, ranges)):
        start, end = range_
        bin_edges = edges[start:end]
        bin_edge_idx = np.searchsorted(bin_edges, value)
        if bin_edges[bin_edge_idx] != value:  # round to the left bin if not exact
            bin_edge_idx -= 1
        discrete_sample[idx] = bin_edge_idx
    return discrete_sample


def sample_to_discrete(
    sample: npt.NDArray, bin_edges: BinEdges, return_idx: bool = False
) -> npt.NDArray:
    """
    Parameters
    ---------
    sample : npt.NDArray
        A numpy array to be discretized
    bin_edges : BinEdges
        Discretization structure
    return_idx : bool
        If `True`, return the index into the bin_edges structure,
        this is the same as the index in the discrete space defined
        by the `bin_edges` structure (the discrete space in the natural numbers).
        If `False`, transform the sample into the discrete space.

    Returns
    -------
    npt.NDArray
        An array with the same shape as `sample` but depending on `return_idx`
        either containing the discretized samples with the original space boundaries
        (real numbers) or the index into the discrete space (natural numbers)

    Examples
    --------
    >>> space = Box(shape=(2, 2), low=-1, high=1)
    >>> sample = space.sample()
    >>> sample
    array([[-0.64394224, -0.41881064],
           [-0.07708957,  0.9520732 ]], dtype=float32)
    >>> bin_edges = generate_box_bins(space, n_samples=3, bin_func=np.linspace)
    >>> bin_edges
    [[array([-1.,  0.,  1.], dtype=float32),
      array([-1.,  0.,  1.], dtype=float32)],
     [array([-1.,  0.,  1.], dtype=float32),
      array([-1.,  0.,  1.], dtype=float32)]]
    >>> sample_to_discrete(sample, bin_edges, return_idx=False)
    array([[-1., -1.],
           [-1.,  0.]], dtype=float32)
    >>> sample_to_discrete(sample, bin_edges, return_idx=True)
    array([[0, 0],
           [0, 1]])
    """
    if return_idx:
        flat_discrete_sample = _sample_to_discrete_idx(
            sample.ravel(), bin_edges.edges, bin_edges.ranges
        )
    else:
        flat_discrete_sample = _sample_to_discrete_values(
            sample.ravel(), bin_edges.edges, bin_edges.ranges
        )
    return flat_discrete_sample.reshape(sample.shape)


def _continuous_to_discrete(
    sample: npt.NDArray[np.floating], bin_edges: BinEdges
) -> npt.NDArray[np.integer]:
    return sample_to_discrete(sample, bin_edges, return_idx=True)


def _discrete_to_continuous(
    sample: npt.NDArray[np.integer], bin_edges: BinEdges
) -> npt.NDArray[np.floating]:
    result = np.empty(sample.ravel().shape)
    for i, (idx, (start, end)) in enumerate(
        zip(sample.ravel(), bin_edges.ranges, strict=True)
    ):
        bin_edge = bin_edges.edges[start:end]
        result[i] = bin_edge[
            idx
        ]  # TODO: an idx error would indicate that the MultiDiscrete space was invalid
    return result.reshape(sample.shape)


def box_to_discrete(
    space: Box, bin_edges: BinEdges
) -> tuple[
    gym.spaces.MultiDiscrete,
    Callable[[npt.NDArray], npt.NDArray],
    Callable[[npt.NDArray], npt.NDArray],
]:
    """Construct a discrete space and a transformation function to and from the continuous space
    using the BinEdges structure provided

    Returns two functions:
    - The first function maps samples from the continuous `Box` space to the discrete space. The
    discrete samples are defined as the index of the interval defined by the `BinEdges`.
    - The second function transforms samples from the discrete space back to the continuous space.
    The result will still be discretized since only the information about the interval index in
    the `BinEdges` is retained and the rest is lost by the transformation to the discrete space.


    Parameters
    ----------
    space : Box
        A continuous space
    bin_edges : BinEdges
        The discretization structure

    Returns
    -------
    tuple[gym.spaces.MultiDiscrete, Callable[[npt.NDArray], npt.NDArray], Callable[[npt.NDArray], npt.NDArray]]
        Either a Discrete space or a MultiDiscrete space with the corresponding
        transformation function mapping from the continuous space into the discrete space
    """
    if space.shape != bin_edges.space.shape:
        raise ValueError(
            "The provided BinEdges are incompatible with the space "
            f"The provided space has shape: {space.shape} and the BinEdges "
            f"were create with a space of shape {bin_edges.space.shape}."
        )
    to_discrete_tf = partial(_continuous_to_discrete, bin_edges=bin_edges)
    to_continuous_tf = partial(_discrete_to_continuous, bin_edges=bin_edges)
    return (gym.spaces.MultiDiscrete(bin_edges.nvec), to_discrete_tf, to_continuous_tf)


def get_discrete_box_tf(
    space: Box, bin_edges: BinEdges
) -> Callable[[NDArray], NDArray]:
    """Returns a parametrized function using the space and bins that is able
    to perform an inplace discretization of a given sample

    Raises
    ------
    AssertionError
        If the space is incompatible to the BinEdges

    Parameters
    ----------
    space : Box
        The space the samples are from
    bin_edges: BinEdges
        The bin edges used to discretize each dimension of the sample

    Returns
    -------
    Callable[[NDArray], NDArray]
        A function that applies inplace discretization on a given sample
        originating from the provided space
    """
    if space.shape != bin_edges.space.shape:
        raise ValueError(
            "The provided BinEdges are incompatible with the space "
            f"The provided space has shape: {space.shape} and the BinEdges "
            f"were create with a space of shape {bin_edges.space.shape}."
        )
    return partial(sample_to_discrete, bin_edges=bin_edges, return_idx=False)
