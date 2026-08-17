from utils import make_original_env, vector_to_int, int_to_vector, get_vector
from verigym.abstraction.abstractionmapper import (
    AbstractionMap,
    AbstractionMapper,
    IdentityAbstractionMap,
)
from verigym.abstraction.learn_abstraction import create_abstraction
from verigym.abstraction.gym_utils.spaces import DummySpace
from verigym.environments.generativeenv import GenerativeEnv
from verigym.policy.policy import RandomizedPolicy

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
import pytest

import functools


def test_default_identity_abstraction_mapping():
    map = IdentityAbstractionMap()
    abstract_state = 5
    numpy_state = get_vector()
    assert map.forward_map(abstract_state) == abstract_state
    assert map.backward_map(abstract_state) == abstract_state
    assert np.array_equal(map.forward_map(numpy_state), numpy_state)
    assert np.array_equal(map.backward_map(numpy_state), numpy_state)


def test_abstraction_mapping():
    array = get_vector()
    map = AbstractionMap(
        forward_map=vector_to_int, 
        backward_map=functools.partial(int_to_vector, length=array.size),
        original_space=DummySpace(),
        abstract_space=DummySpace(),
    )
    abstract = map.forward_map(array)
    recoveredarray = map.backward_map(abstract)
    assert map.forward_map(recoveredarray) == abstract
    assert np.array_equal(array, recoveredarray)
    
# each entry holds a space and the attributes an `AbstractionMap` should derive from it: (space, n_elements, is_continuous)
SPACES = [
    (Box(np.array((-1,3)), np.array((5,6))), np.inf, True),
    (Discrete(5), 5, False),
    (MultiDiscrete([2,3,4,5]), 2*3*4*5, False),
    (MultiBinary(4), 2**4, False),
    (DummySpace(), 1, False),
]

@pytest.mark.parametrize(["space", "n_elements", "is_continuous"], SPACES)
def test_abstraction_map_attributes(space, n_elements, is_continuous):
    """`AbstractionMap` derives its element counts and its continuity flag from the two spaces it is given."""
    # the fixed abstract space has a size that none of the `SPACES` shares, so a mix-up of the two spaces would be caught
    map = AbstractionMap(
            forward_map=None,
            backward_map=None,
            original_space=space,
            abstract_space=Discrete(7),
    )
    assert map.from_continuous_space == is_continuous
    assert map.original_n_elements == n_elements
    assert map.abstract_n_elements == 7


@pytest.mark.parametrize(["space", "n_elements", "is_continuous"], SPACES)
def test_abstraction_mapper_attributes(space, n_elements, is_continuous):
    """`AbstractionMapper` exposes the attributes of its state and action map, without mixing the two up."""
    # the parametrized space is the original space of the state map and the abstract space of the action map, so that
    # it is covered in both roles, while the fixed spaces have sizes that none of the `SPACES` shares
    state_map = AbstractionMap(
            forward_map=None,
            backward_map=None,
            original_space=space,
            abstract_space=Discrete(3),
    )
    action_map = AbstractionMap(
            forward_map=None,
            backward_map=None,
            original_space=Discrete(7),
            abstract_space=space,
    )
    mapper = AbstractionMapper(state_abstraction_map=state_map, action_abstraction_map=action_map)
    assert mapper.original_n_states == n_elements
    assert mapper.abstract_n_states == 3
    assert mapper.original_n_actions == 7
    assert mapper.abstract_n_actions == n_elements
    assert mapper.from_continuous_states == is_continuous
    assert mapper.from_continuous_actions is False


def test_abstraction_mapping_from_abstraction():
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    _abstracted_env = create_abstraction(
        original_env=generative_env,
        exploration_policy=RandomizedPolicy(env),
        num_steps=NUM_STEPS,
        bin_edges_per_state_dim=BIN_EDGES_PER_DIM,
        bin_edges_per_action_dim=BIN_EDGES_PER_DIM,
    )
    abstraction_map: AbstractionMapper = _abstracted_env.abstraction_map
    assert abstraction_map is not None
    assert not isinstance(abstraction_map._state_abstraction_map, IdentityAbstractionMap)
    assert abstraction_map._state_abstraction_map is not None
    init_state, *_ = env.reset()
    init_abstract = abstraction_map.original_to_abstract_state(init_state)
    assert abstraction_map.original_to_abstract_state(init_state) == init_abstract
