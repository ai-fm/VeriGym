import gymnasium as gym
import numpy as np

from verigym.abstraction.feature_selection import state_feature_selection, ReduceFeaturesWrapper, BinFeaturesWrapper
from verigym.environments.generativeenv import GenerativeEnv

def test_observation_space_types():
    """
    This tests checks that feature selection only works on gym.spaces.Box and gym.spaces.MulitDiscrete, 
    but not gym.spaces.Discrete (since that does not make sense).
    """
    # Test that box observation spaces do not throw an error
    box_env = GenerativeEnv.from_gymnasium(gym.make("LunarLander-v3"))
    assert isinstance(box_env.observation_space, gym.spaces.Box)
    threw_error = False
    try:
        state_feature_selection(box_env, method="binning", reduce_indices=[])
    except ValueError:
        threw_error = True
    assert not threw_error
    
    # Test that MultiDiscrete observation spaces do not throw an error
    # I did not find a MD observation space env, so converting the box space here.
    md_obs = gym.spaces.MultiDiscrete([3 for _ in range(box_env.observation_space.shape[0])])
    md_env = gym.wrappers.TransformObservation(box_env, lambda obs: np.round(obs, dtype=int),
                                               observation_space=md_obs)
    assert isinstance(md_env.observation_space, gym.spaces.MultiDiscrete)
    threw_error = False
    try:
        state_feature_selection(md_env, method="binning", reduce_indices=[])
    except ValueError:
        threw_error = True
    assert not threw_error
    
    # Test that feature selection throws an error when using Discrete observation space.
    d_env = GenerativeEnv.from_gymnasium(gym.make("Taxi-v3"))
    assert isinstance(d_env.observation_space, gym.spaces.Discrete)
    threw_error = False
    try:
        state_feature_selection(d_env, method="binning", reduce_indices=[])
    except ValueError:
        threw_error = True
    assert threw_error


def test_selection_by_binning_on_box():
    """
    Tests feature selection by binning on a box space env.
    """
    # Two test runs: 1) removing one feature (idx 5), 2) removing several features
    reduce_indices = [[5],
                      [2, 3, 6, 7]]
    for idcs in reduce_indices:
        box_env = gym.make("LunarLander-v3")
        reduce_env, abstraction_mapper = state_feature_selection(box_env, 
                                             "binning",
                                             idcs)
        
        # Assert that the correct wrapper is used
        assert isinstance(reduce_env, BinFeaturesWrapper)
        # Assert that the observation space shape and limits stay intact
        assert reduce_env.observation_space.shape[0] == box_env.observation_space.shape[0]
        assert (reduce_env.observation_space.low == box_env.observation_space.low).all()
        assert (reduce_env.observation_space.high == box_env.observation_space.high).all()

        # Assert that the sampled observation is valid in the environments observation space
        obs, _ = reduce_env.reset()
        assert obs in reduce_env.observation_space
        assert (obs[idcs] == reduce_env.reduce_vals[idcs]).all()

        # Test abstraction mapping between the reduced and original samples
        orig_states = abstraction_mapper.abstract_to_original_state(obs)
        lower = orig_states[0]
        upper = orig_states[1]
        for idx in range(box_env.observation_space.shape[0]):
            if idx in idcs:
                assert lower[idx] == box_env.observation_space.low[idx]
                assert upper[idx] == box_env.observation_space.high[idx]
            else:
                assert lower[idx] == upper[idx], f"lower: {lower}, upper: {upper}, obs: {obs}, reduce: {idcs}"
        assert (abstraction_mapper.original_to_abstract_state(obs) == obs).all()

def test_selection_by_binning_on_md():
    """
    Tests feature selection by binning on a multi discrete observation space env.
    """
    # Two test runs: 1) removing one feature (idx 5), 2) removing several features
    reduce_indices = [[5],
                      [2, 3, 6, 7]]
    md_obs = gym.spaces.MultiDiscrete([5, 5, 21, 21, 13, 21, 2, 2])
    for idcs in reduce_indices:
        box_env = gym.make("LunarLander-v3")
        md_env = gym.wrappers.TransformObservation(box_env, lambda obs: (np.round(obs).astype(int) + [2, 2, 10, 10, 6, 10, 1, 1]),
                                               observation_space=md_obs)
        reduce_env, abstraction_mapper = state_feature_selection(md_env, 
                                             "binning",
                                             idcs)
        # Assert that the correct wrapper is used
        assert isinstance(reduce_env, BinFeaturesWrapper)
        # Assert that the observation space shape and limits stay intact
        assert reduce_env.observation_space.shape[0] == md_env.observation_space.shape[0]
        assert (reduce_env.observation_space.nvec == md_env.observation_space.nvec).all()            

        obs, _ = reduce_env.reset()
        assert obs in reduce_env.observation_space
        for idx in idcs:
            assert obs[idx] == 0

        # Test abstraction mapping between the reduced and original samples
        orig_states = abstraction_mapper.abstract_to_original_state(obs)
        for orig_state in orig_states:
            assert orig_state in md_env.observation_space
            for i in range(reduce_env.observation_space.shape[0]):
                if i not in idcs:
                    assert orig_state[i] == obs[i], f"original: {orig_state}, obs: {obs}, i: {i}, reduce: {idcs}"
        assert len(orig_states) == np.prod(md_obs.nvec[idcs])

def test_selection_by_masking_on_box():
    """
    Tests feature selection by masking on a box observation space env.
    """
    # Two test runs: 1) removing one feature (idx 5), 2) removing several features
    reduce_indices = [[5],
                      [2, 3, 6, 7]]
    out_lows = [[-2.5, -2.5, -10., -10., -6.2831855, -0.0, -0.0],
                [-2.5, -2.5, -6.2831855, -10.]]
    out_highs = [[2.5, 2.5, 10., 10., 6.2831855, 1., 1.],
                [2.5, 2.5, 6.2831855, 10.]]
    out_shapes = [7, 4]
    for idcs, lows, highs, shape in zip(reduce_indices, out_lows, out_highs, out_shapes):
        box_env = gym.make("LunarLander-v3")
        reduce_env, abstraction_mapper = state_feature_selection(box_env, 
                                             "masking",
                                             idcs)
        # Assert that the correct wrapper is used
        assert isinstance(reduce_env, ReduceFeaturesWrapper)
        assert reduce_env.observation_space.shape[0] == shape
        assert (reduce_env.observation_space.low == np.array(lows, dtype=np.float32)).all()
        assert (reduce_env.observation_space.high == np.array(highs, dtype=np.float32)).all()

        obs, _ = reduce_env.reset()
        assert obs in reduce_env.observation_space

        # test mapping
        orig_states = abstraction_mapper.abstract_to_original_state(obs)
        lower = orig_states[0]
        upper = orig_states[1]
        for idx in range(box_env.observation_space.shape[0]):
            if idx in idcs:
                assert lower[idx] == box_env.observation_space.low[idx]
                assert upper[idx] == box_env.observation_space.high[idx]
            else:
                assert lower[idx] == upper[idx], f"lower: {lower}, upper: {upper}, obs: {obs}, reduce: {idcs}"


def test_selection_by_masking_on_md():
    """
    Tests feature selection by masking on a multi discrete observation space env.
    """
    # Two test runs: 1) removing one feature (idx 5), 2) removing several features
    reduce_indices = [[5],
                      [2, 3, 6, 7]]
    out_nvecs = [[5, 5, 21, 21, 13, 2, 2],
                 [5, 5, 13, 21]]
    md_obs = gym.spaces.MultiDiscrete([5, 5, 21, 21, 13, 21, 2, 2])
    out_shapes = [7, 4]
    for idcs, nvec, shape in zip(reduce_indices, out_nvecs, out_shapes):
        box_env = gym.make("LunarLander-v3")
        md_env = gym.wrappers.TransformObservation(box_env, lambda obs: (np.round(obs).astype(int) + [2, 2, 10, 10, 6, 10, 1, 1]),
                                               observation_space=md_obs)
        reduce_env, abstraction_mapper = state_feature_selection(md_env, 
                                             "masking",
                                             idcs)
        # Assert that the correct wrapper is used
        assert isinstance(reduce_env, ReduceFeaturesWrapper)
        assert reduce_env.observation_space.shape[0] == shape
        assert (reduce_env.observation_space.nvec == np.array(nvec)).all(), \
            f"shape: {reduce_env.observation_space.nvec.shape}, {np.array(nvec).shape}, \n" \
            f"not equal: {reduce_env.observation_space.nvec} != {np.array(nvec)}"

        obs, _ = reduce_env.reset()
        assert obs in reduce_env.observation_space

        # test mapping
        orig_states = abstraction_mapper.abstract_to_original_state(obs)
        for orig_state in orig_states:
            assert orig_state in md_env.observation_space
            abs_idx = 0
            for i in range(md_env.observation_space.shape[0]):
                if i not in idcs:
                    assert orig_state[i] == obs[abs_idx]
                    abs_idx += 1
        assert len(orig_states) == np.prod(md_obs.nvec[idcs])