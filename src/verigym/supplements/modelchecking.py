from warnings import warn

import verigym
from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.environments.interval_explicitenv import IntervalEnvMixin
from verigym.io.exporter import unwrap_env
from verigym.abstraction.abstractionmapper import AbstractionMapper
from verigym.policy.policy import PolicyClass
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_dtmc, build_stormpy_imdp, build_stormpy_mdp

import stormpy

"""
This file contains supplementary functions for more user-friendly model checking.
That can mean wrapping functionality from external frameworks into a one-line function call.
"""

def get_policy_from_stormpy(env: BaseExplicitEnv, property_str: str) -> verigym.StormpyPolicy:
    """
    Given an explicit env and a property, returns a VeriGym compatible policy from stormpy.

    Parameters
    ----------
    env : BaseExplicitEnv
        The explicit env to model check.
    property_str : str
        The property to check.
    
    Returns
    -------
    policy : verigym.StormpyPolicy
        The policy obtained from stormpy.
    """
    unwrapped_env = unwrap_env(env)
    prop = stormpy.parse_properties(property_str)[0]

    if isinstance(unwrapped_env, IntervalEnvMixin):
        mdp = build_stormpy_imdp(unwrapped_env)
        warn("StormPy (as of 1.14) does not yet support scheduler extraction for interval MDPs.")
    else:
        mdp = build_stormpy_mdp(unwrapped_env)
    abs_map = unwrapped_env.get_abstraction_map()

    # the choices are labelled with the action names if the env has action labels (e.g., loaded from a prism file)
    label_to_action = None
    if hasattr(unwrapped_env, "formatter") and unwrapped_env.formatter.has_action_labels:
        label_to_action = unwrapped_env.formatter.label_to_action

    if abs_map is None:
        # if there is no abstraction map, return identity map
        abs_map = AbstractionMapper.initialize_identity_mapper(env.observation_space, env.action_space)
        

    result = stormpy.check_model_sparse(mdp, prop, 
                                        extract_scheduler = True)
    scheduler = result.scheduler
    policy = verigym.StormpyPolicy(
        scheduler, abs_map, mdp, label_to_action=label_to_action
    )
    return policy

def check_policy_value_in_stormpy(env: BaseExplicitEnv,
                                  policy: PolicyClass,
                                  property_str: str,
                                  only_initial_states=True):
    """
    Given an environment, a policy on that environment, and a property,
    builds a DTMC from the environment's underlying MDP and the policy,
    solves it in stormpy, and returns the value vector.

    Parameters
    ----------
    env : BaseExplicitEnv
        The environment.
    policy : PolicyClass
        The policy on that environment.
    property_str : str
        The property to check.

    Returns
    -------
    value_vector : list
        Value per state in the DTMC.
    """
    
    assert issubclass(type(env), BaseExplicitEnv) \
        or issubclass(type(env.unwrapped), BaseExplicitEnv)

    prop = stormpy.parse_properties(property_str)[0]

    if not issubclass(type(env), BaseExplicitEnv): # has a wrapper
        export_env = env.unwrapped
    else:
        export_env = env

    dtmc = build_stormpy_dtmc(export_env, policy)

    result = stormpy.check_model_sparse(dtmc, prop, only_initial_states=only_initial_states)
    if only_initial_states:
        return [result.at(init) for init in dtmc.initial_states]
    else:
        return result.get_values()
