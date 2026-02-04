""""
Tests for exporting explicit environments to different formats.
Specifically:
    export_to_stormpy_mdp
    export_to_drn
WIP:
    export_to_julia_pomdp (not implemented)
    export_to_prism (not implemented)

Test stormpy_utils.build_stormpy_mdp
"""
import os
import stormpy

from verigym.environments.exporter import export_to_drn, export_to_stormpy_mdp
from verigym.frameworks.stormpy.stormpy_utils import load_stormpy_model, build_stormpy_mdp
#from verigym.frameworks.stormpy.stormpyenv import StormpyEnv
from verigym.frameworks.stormpy.formatter import StormpyFormatter
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv

PRISM_TEST = os.path.join(os.getcwd(), "tests/test_2d.prism")

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

    mdp_2 = export_to_stormpy_mdp(env)

    assert isinstance(mdp_2, stormpy.storage.SparseMdp)

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

def test_export_to_julia_pomdp():
    ... # TODO functionality not implemented

def test_export_to_prism():
    ... # TODO functionality not implemented