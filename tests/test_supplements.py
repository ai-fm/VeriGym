from verigym.supplements.modelchecking import get_policy_from_stormpy
from verigym.frameworks.stormpy.stormpy_utils import load_stormpy_model
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv
from verigym.environments.explicitenv import ExplicitEnv
from verigym.environments.labeling import StateLabel
from copy import deepcopy
import os

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
