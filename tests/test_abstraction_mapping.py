from utils import make_original_env, vector_to_int, int_to_vector, get_vector
from verigym.abstraction.abstractionmapper import (
    AbstractionMap,
    AbstractionMapper,
    IdentityAbstractionMap,
)
from verigym.abstraction.learn_abstraction import create_abstraction
from verigym.abstraction.gym_utils.spaces import DummySpace
from verigym.environments.generativeenv import GenerativeEnv
from verigym.policy.implemented_policies import RandomizedPolicy

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
    
@pytest.mark.parametrize(["original_space", "from_continuous_space"], [(Box(np.array((-1,3)), np.array((5,6))), True), (Discrete(5), False), (MultiDiscrete([2,3,4,5]), False), (MultiBinary([2,3,4]), False)])
@pytest.mark.parametrize("abstract_space", [Box(np.array((-1,3)), np.array((5,6))), Discrete(5), MultiDiscrete([2,3,4,5]), MultiBinary([2,3,4])])
def test_attribute_from_continuous_space(original_space, abstract_space, from_continuous_space):
    map = AbstractionMap(
            forward_map=None, 
            backward_map=None,
            original_space=original_space,
            abstract_space=abstract_space,
    )
    assert map.from_continuous_space == from_continuous_space


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
