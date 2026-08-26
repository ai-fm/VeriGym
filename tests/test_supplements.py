from verigym.supplements.modelchecking import get_policy_from_stormpy, check_policy_value_in_stormpy
from verigym.frameworks.stormpy.stormpy_utils import load_stormpy_model, build_stormpy_dtmc
from verigym.frameworks.stormpy.formatter import StormpyFormatter
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv
from verigym.environments.explicitenv import ExplicitEnv
from verigym.environments.labeling import StateLabel
from verigym.environments.exporter import export_to_stormpy_mdp

from copy import deepcopy
import os
import stormpy

PRISM_TEST = os.path.join(os.getcwd(), "tests/test_2d.prism")

def test_stormpy_policy_wrapper():
    """
    Test for supplements.modelchecking.get_stormpy_policy
    """
    property_str = 'Pmin=? [F "unsafe"]'
    mdp = load_stormpy_model(PRISM_TEST, property_str)

    # Try FrameworkExplicitEnv
    stormpy_env = FrameworkExplicitEnv.from_stormpy(mdp)
    stormpy_env_policy = get_policy_from_stormpy(stormpy_env, property_str)
    
    # Try ExplicitEnv with the same R, T
    env_labels = ExplicitEnv(
        nr_states = stormpy_env.nr_states,
        nr_actions = stormpy_env.nr_actions,
        initial_state_distr=deepcopy(stormpy_env.formatter.initial_states),
        transition_function = deepcopy(stormpy_env.transition_function),
        reward_function = deepcopy(stormpy_env.reward_function)
    )
    unsafe_states = [s for s in range(mdp.nr_states) if "unsafe" in mdp.labeling.get_labels_of_state(s)]
    # In PRISM file: label "unsafe" = unsafe; formula unsafe = y=2 & (x=1 | x=2 | x=3);
    unsafe_label = StateLabel("unsafe",
                              lambda s: s in unsafe_states)
    env_labels.add_state_label(unsafe_label)
    labels_policy = get_policy_from_stormpy(env_labels, property_str)
    
    for s in range(stormpy_env.nr_states):
        se_action = stormpy_env_policy.get_action(s)
        l_action = labels_policy.get_action(s)

        assert se_action in stormpy_env.action_space
        assert l_action in env_labels.action_space

        assert se_action == l_action

def test_build_dtmc():
    property_str = 'Pmin=? [F "unsafe"]'
    prop = stormpy.parse_properties(property_str)[0]

    # from framework explicit env
    mdp = load_stormpy_model(PRISM_TEST)
    fw_env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))
    fw_policy = get_policy_from_stormpy(fw_env, property_str)
    fw_dtmc = build_stormpy_dtmc(fw_env, fw_policy)

    # from normal explicit env
    # (using the same transition function, checking if this works without underlying StormpyFormatter)
    ex_env = ExplicitEnv(
        nr_states=fw_env.nr_states, nr_actions=fw_env.nr_actions,
        initial_state_distr=deepcopy(fw_env.initial_states),
        transition_function=deepcopy(fw_env.transition_function),
        reward_function=deepcopy(fw_env.reward_function)
    )
    unsafe_label = StateLabel("unsafe", lambda s: s in fw_env.formatter.labels_to_states["unsafe"])
    ex_env.add_state_label(unsafe_label)

    ex_policy = get_policy_from_stormpy(ex_env, property_str)
    ex_dtmc = build_stormpy_dtmc(ex_env, ex_policy)

    # check that model structure is correct
    assert fw_env.nr_states == fw_dtmc.nr_states
    assert ex_env.nr_states == ex_dtmc.nr_states

    # check that model values are correct
    res_mdp = stormpy.check_model_sparse(mdp, prop)
    val_vec_mdp = res_mdp.get_values()

    val_vec_fw_env = stormpy.check_model_sparse(
        export_to_stormpy_mdp(fw_env), prop
    ).get_values()

    val_vec_ex_env = stormpy.check_model_sparse(
        export_to_stormpy_mdp(ex_env), prop
    ).get_values()

    val_vec_ex_dtmc = stormpy.check_model_sparse(ex_dtmc, prop).get_values()
    val_vec_fw_dtmc = stormpy.check_model_sparse(fw_dtmc, prop).get_values()
    
    assert val_vec_fw_dtmc == val_vec_fw_env
    assert val_vec_fw_dtmc == val_vec_mdp
    assert val_vec_ex_dtmc == val_vec_ex_env
    assert val_vec_ex_dtmc == val_vec_mdp


def test_dtmc_policy_checker():
    property_str = 'Pmin=? [F "unsafe"]'
    # from framework explicit env
    mdp = load_stormpy_model(PRISM_TEST)
    fw_env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))
    fw_policy = get_policy_from_stormpy(fw_env, property_str)

    prop = stormpy.parse_properties(property_str)[0]
    res_mdp = stormpy.check_model_sparse(mdp, prop)
    val_vec_mdp = res_mdp.get_values()

    res_env = stormpy.check_model_sparse(
        export_to_stormpy_mdp(fw_env),
        prop
    )
    val_vec_env = res_env.get_values()

    val_vec_dtmc = check_policy_value_in_stormpy(fw_env, fw_policy, property_str,
                                                 only_initial_states=False)
    
    assert val_vec_dtmc == val_vec_mdp
    assert val_vec_dtmc == val_vec_env