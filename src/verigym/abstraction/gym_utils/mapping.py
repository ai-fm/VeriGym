from collections.abc import Callable, Sequence, Iterable
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
import gymnasium as gym
from numpy.typing import NDArray

from verigym.abstraction.discretization import (
    BinEdges,
    is_single_dim_nested,
    nvec_from_bin_edges,
    nvec_from_samples,
    verify_bin_edges,
)

from verigym.abstraction.discretization import subview_iter


def sample_to_discrete(
    sample: npt.NDArray, bin_edges: BinEdges, return_idx: bool = False
) -> npt.NDArray:
    """In place conversion of a sample into the discrete space
    defined over the provided `bin_edges`

    Parameters
    ---------
    sample : npt.NDArray
        A numpy array to be discretized
    bin_edges : BinEdges
        Array containing the discretization steps
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
    sample_dim = []
    if (
        not isinstance(sample, np.ndarray)
        or sample.shape[0] != len(bin_edges)
        or (not isinstance(bin_edges[0], Sequence | np.ndarray))
    ):
        # A scalar sample may be paired with a single-dimension nested BinEdges
        # (e.g. [array([...])] generated from a Discrete space); unwrap to the
        # underlying BinEdge before digitizing.
        if np.isscalar(sample) and is_single_dim_nested(bin_edges):
            bin_edges = bin_edges[0]
        idx = np.digitize(sample, bin_edges) - 1
        if return_idx:
            return idx
        else:
            return bin_edges[idx]
    for idx, (sub_sample, sub_bin) in enumerate(zip(sample, bin_edges, strict=False)):
        sample_dim.append(
            sample_to_discrete(sub_sample, sub_bin, return_idx=return_idx)
        )
    return np.array(sample_dim, dtype=int if return_idx else sample.dtype)


def index_bin_edges(idx: npt.NDArray, bin_edges: BinEdges):
    """
    It is pretty much the inverse of `sample_to_discrete`
    Index a BinEdges structure using a sample from a discrete space
    created by a space using the BinEdges for discretization
    like `box_to_discrete`. A sample of the resulting space can be used
    in this function to retrieve the discretized equivalent in the original
    continuous Box space. 

    Parameters
    ----------
    idx: npt.NDArray
        A sample from a discrete space created over `bin_edges`
    bin_edges: BinEdges
        A BinEdges structure that has been used to create the discrete space
        the `idx` is a sample of
    """
    sample_dim = []
    if not isinstance(idx, Iterable):
        # here we only have BinEdge arrays left (one dimensional)
        assert not isinstance(bin_edges[0], Iterable), (
            "found BinEdges instead of BinEdge. "
            "This indicates that the idx is not a sample of a space related to bin_edges"
        )
        return bin_edges[idx]
    for sub_idx, sub_bin_edge in zip(idx, bin_edges):
        sample_dim.append(index_bin_edges(sub_idx, sub_bin_edge))
    return np.array(sample_dim)


def box_to_discrete(
    space: Box, bin_edges: BinEdges
) -> (
    tuple[
        gym.spaces.Discrete,
        Callable[[npt.NDArray], int],
        Callable[[int], npt.NDArray],
    ]
    | tuple[
        gym.spaces.MultiDiscrete,
        Callable[[npt.NDArray], npt.NDArray],
        Callable[[npt.NDArray], npt.NDArray],
    ]
):
    """Construct a discrete space and a transformation function to and from the continuous space
    using the BinEdges structure provided

    The first function maps samples from the continuous `Box` space to the discrete space. The
    discrete samples are defined as the index of the interval defined by the `BinEdges`.
    The second function transforms samples from the discrete space back to the continuous space.
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
    tuple[gym.spaces.Discrete, Callable[[npt.NDArray], int], Callable[[int], npt.NDArray]] |
    tuple[gym.spaces.MultiDiscrete, Callable[[npt.NDArray], npt.NDArray], Callable[[npt.NDArray], npt.NDArray]]
        Either a Discrete space or a MultiDiscrete space with the corresponding
        transformation function mapping from the continuous space into the discrete space
    """
    assert verify_bin_edges(space.sample(), bin_edges), (
        "The provided BinEdges are incompatible with the space"
    )
    nvec = nvec_from_bin_edges(bin_edges)
    if nvec.shape == (1,):
        to_discrete_tf = partial(
            sample_to_discrete, bin_edges=bin_edges[0], return_idx=True
        )
        to_continuous_tf = partial(index_bin_edges, bin_edges=bin_edges[0])
        return (
            gym.spaces.Discrete(nvec[0]),
            lambda x: to_discrete_tf(x)[0],
            lambda y: np.asarray([to_continuous_tf(y)]),
        )

    to_discrete_tf = partial(sample_to_discrete, bin_edges=bin_edges, return_idx=True)
    to_continuous_tf = partial(index_bin_edges, bin_edges=bin_edges)
    return (gym.spaces.MultiDiscrete(nvec), to_discrete_tf, to_continuous_tf)


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
    assert verify_bin_edges(space.sample(), bin_edges), (
        "The provided BinEdges are incompatible with the space"
    )
    return partial(sample_to_discrete, bin_edges=bin_edges, return_idx=False)


def lookup_table_from_mdiscrete_samples(a: npt.NDArray) -> list:
    """Extract a lookup table from an enumerated index to an actual sample
    The returned structure is a list with the same shape as one of the input samples
    but with dictionaries as values mapping the unique values to an
    enumerated index (starting at zero)

    Parameters
    ----------
    a : npt.NDArray
        A numpy describing a collection of samples from some space

    Returns
    -------
    list
        A nested list structure with the same shape as one of the samples
        but with dictionaries as values containing the mapping
    """
    table_structure = np.zeros(a.shape[1:]).tolist()
    for subview, subview_idx in subview_iter(a):
        unique_values = np.unique(subview)
        lookup = {i: idx for idx, i in enumerate(unique_values)}
        temp_structure = table_structure
        idx = list(subview_idx)
        while len(idx) > 1:
            temp_structure = temp_structure[idx.pop(0)]
        temp_structure[idx[0]] = lookup
    return table_structure


def sample_to_mult_discrete(
    sample: npt.NDArray, lookup: list[list | dict]
) -> npt.NDArray:
    """Given a sample and a lookup table from , return the sample in the enumerated
    space where each unique value is mapped to a consecutive integer
    (compatible to `gym.spaces.MultiDiscrete`)

    Parameters
    ----------
    sample : npt.NDArray
        A sample from some discrete space

    lookup: list[list | dict]
        The lookup table mapping the same space to a `gym.spaces.MultiDiscrete` space

    Returns
    -------
    npt.NDArray
        The original sample in the `gym.spaces.MultiDiscrete` space
    """
    s = sample.copy()

    def lookup_nd_sample(a, lookup):
        if not isinstance(a, np.ndarray):
            return lookup[a]
        for idx, i in enumerate(a):
            a[idx] = lookup_nd_sample(a[idx], lookup[idx])

    lookup_nd_sample(s, lookup)
    return s


def dict_space_from_obs(
    obs: dict[str, list[Any] | dict],
) -> tuple[gym.spaces.Dict, Callable[[dict], dict]]:
    """Create a `gym.spaces.Dict` space from a collection of valuations.
    This structure can be obtained from the `obs_from_model` function.

    Parameters
    ----------
    obs : dict[str, list[Any] | dict]
        A dictionary mapping a valuations key to a list of observed values

    Returns
    -------
    tuple[gym.spaces.Dict, dict[str, Any]]
        A tuple containing the `gym.spaces.Dict` space as well as a lookup structure
        to map a sample from the valuations to the space
    """
    # TODO: Utils?
    spaces = dict()
    lookups = dict()
    for name, observations in obs.items():
        if isinstance(observations, dict):
            # More complex so we create a dict space
            spaces[name], lookups[name] = dict_space_from_obs(observations)
            continue
        # Now first check if it is multi dimensionl
        obs_arr = np.array(observations)
        if obs_arr.ndim == 1:
            # Its a one dimensional sequence so we can create a Discrete space
            unique_values = np.unique(obs_arr)
            spaces[name] = gym.spaces.Discrete(len(unique_values))
            lookups[name] = {v: idx for idx, v in enumerate(unique_values)}
            continue
        # If we arrive here we have a multi dimensional structure and need to use either
        # a discrete space again or a multi discrete
        # Discrete would map from index to subsequence completely masking the complexity of the space
        # Multi Discrete would be more sensual since it gives some impression about the underlying structure
        # Which likely contains important information
        # since we are working with MDPs as of now, lets just use a MultiDiscrete space here and map everything
        nvec = nvec_from_samples(obs_arr)
        spaces[name] = gym.spaces.MultiDiscrete(nvec)
        lookups[name] = lookup_table_from_mdiscrete_samples(obs_arr)
        # But how to do the lookup for that?
    space = gym.spaces.Dict(**spaces)

    return space, partial(sample_to_dict_space, lookup=lookups)


def sample_to_dict_space(sample: dict, lookup: dict[str, Any]):
    """Convert a sample from a valuations space to a sample in a corresponding `gym.spaces.Dict` space
    using the lookup structure

    Parameters
    ----------
    sample : dict
        A sample from as a set of valuations from a `stormvogel.model.State` instance

    lookup : dict[str, Any]
        A structure describing a mapping from the valuations space into a `gym.spaces.Dict` space

    Returns
    -------
    dict
        The transformed sample in the `gym.spaces.Dict` space defined by the `lookup` mapping
    """
    space_sample = dict()
    # TODO: Add an assertion to verify that the lookup is compatible?
    for key, value in sample.items():
        if isinstance(value, dict):
            space_sample[key] = sample_to_dict_space(value, lookup[key])
        if isinstance(value, np.ndarray | Sequence):
            space_sample[key] = sample_to_mult_discrete(np.array(value), lookup[key])
        else:
            space_sample[key] = lookup[key][value]

    return space_sample
