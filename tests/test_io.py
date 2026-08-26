""" "
Tests for exporting explicit environments to different formats.
Specifically:
    build_stormpy_mdp_from_explicit_env
    export_to_drn
WIP:
    export_to_julia_pomdp (not implemented)
    export_to_prism (not implemented)

Test stormpy_utils.build_stormpy_mdp
"""

import os
import stormpy
import umbi

from verigym.io.io_utils import _export_umbi_ats_to_umb, _load_umb_to_umbi_ats, export_to_drn, export_to_umb, load_from_umb, load_stormpy_model, base_explicit_env_to_umbi
from verigym.frameworks.stormpy.stormpy_utils import (
    build_stormpy_mdp
)
from verigym.frameworks.stormpy.formatter import StormpyFormatter
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv

PRISM_TEST = os.path.join(os.getcwd(), "tests/test_2d.prism")
UMB_FILENAME = "test.umb"

def compare_mdps(mdp1, mdp2, from_drn=False):
    """
    Helper function to compare two stormpy MDPs
    """
    # compare the MDPs, should have rebuilt the same thing
    # Transitions/shape
    assert mdp1.nr_states == mdp2.nr_states
    assert mdp1.transition_matrix.nr_columns == mdp2.transition_matrix.nr_columns
    assert mdp1.transition_matrix.nr_rows == mdp2.transition_matrix.nr_rows
    assert mdp1.transition_matrix.nr_entries == mdp2.transition_matrix.nr_entries
    assert mdp1.nr_choices == mdp2.nr_choices
    assert mdp1.nr_states == mdp2.nr_states

    # Choice labeling
    if mdp1.has_choice_labeling:
        assert mdp2.has_choice_labeling
        assert mdp1.choice_labeling.get_labels() == mdp2.choice_labeling.get_labels()
    else:
        assert not mdp2.has_choice_labeling

    # State labeling
    assert mdp1.labeling.get_labels() == mdp2.labeling.get_labels()

    if not from_drn:
        # State valuations
        if mdp1.has_state_valuations:
            assert mdp2.has_state_valuations
            for s in range(mdp1.nr_states):
                assert mdp1.states[s].valuations == mdp2.states[s].valuations
        else:
            assert not mdp2.has_state_valuations

    # Rewards
    assert len(mdp1.reward_models) == len(mdp2.reward_models)


def test_build_stormpy_mdp():
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))
    mdp_2 = build_stormpy_mdp(env)

    compare_mdps(mdp, mdp_2)


def test_export_to_stormpy():
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))

    mdp_2 = build_stormpy_mdp(env)

    assert isinstance(mdp_2, stormpy.storage.SparseMdp)

    compare_mdps(mdp, mdp_2)


def test_export_to_drn():
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))

    out_path = os.path.join(os.path.join(os.getcwd(), "tests"), "out_mdp.drn")
    export_to_drn(env, out_path)

    # check whether the exported drn can be re-used for a new sp env
    options_2 = stormpy._core.DirectEncodingParserOptions()
    if mdp.has_choice_labeling:
        options_2.build_choice_labels = True
    mdp_2 = stormpy.build_model_from_drn(out_path, options_2)

    compare_mdps(mdp, mdp_2, from_drn=True)

def test_export_to_umb():
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))
    export_to_umb(env, UMB_FILENAME, use_stormpy=True)
    mdp2 = stormpy.build_from_umb(UMB_FILENAME)
    os.remove(UMB_FILENAME)
    compare_mdps(mdp, mdp2)
    

def test_umbi_io():
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))
    ats = base_explicit_env_to_umbi(env)
    umbi.ats.write(ats, UMB_FILENAME)
    _export_umbi_ats_to_umb(ats, UMB_FILENAME)
    ats2 = _load_umb_to_umbi_ats(UMB_FILENAME)
    os.remove(UMB_FILENAME)
    assert ats == ats2

def test_stormpy_umb_io():
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))
    export_to_umb(env, UMB_FILENAME, True) # use Stormpy
    env2 = load_from_umb(UMB_FILENAME, True) # use Stormpy
    os.remove(UMB_FILENAME)
    mdp2 = build_stormpy_mdp(env2)
    compare_mdps(mdp, mdp2)

def test_export_from_abstraction():
    """
    More of an integration test between export and abstraction learning.
    """
    import gymnasium as gym
    import verigym
    from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation
    from verigym.policy.policy import RandomizedPolicy
    # Load the gym env
    gym_env = gym.make("CartPole-v1")
    gym_env = ReplaceInfObservation(
        gym_env, neg_inf=-10, pos_inf=10
    )

    # Create a VeriGymEnv from gym env
    generative_model = verigym.GenerativeEnv.from_gymnasium(gym_env)
    del gym_env

    # Create abstraction
    abstracted_model = verigym.create_abstraction(  # TODO add different discretisation functions as arguments
        original_env=generative_model,
        bin_edges_per_state_dim=5,  # Discretization: dim 1 has 10 bins, dim 2 has 5 bins, ...
        bin_edges_per_action_dim=1,
        exploration_policy=RandomizedPolicy(
            generative_model
        ),
        num_steps=int(1e4),
        multithreading=False
    )
    assert isinstance(abstracted_model, verigym.ExplicitEnv)

    export_to_umb(abstracted_model, UMB_FILENAME)
    abstracted_model2 = load_from_umb(UMB_FILENAME)
    os.remove(UMB_FILENAME)

    assert abstracted_model == abstracted_model2

def test_export_to_julia_pomdp(): ...  # TODO functionality not implemented


def test_export_to_prism(): ...  # TODO functionality not implemented
