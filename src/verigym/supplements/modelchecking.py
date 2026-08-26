import verigym
from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp
from verigym.policy.policy import PolicyClass
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_dtmc

import stormpy

"""
This file contains supplementary functions for more user-friendly model checking.
That can mean wrapping functionality from external frameworks into a one-line function call.
"""

def get_policy_from_stormpy(env: BaseExplicitEnv,
                         property_str: str) -> verigym.StormpyPolicy:
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
    assert issubclass(type(env), BaseExplicitEnv) \
        or issubclass(type(env.unwrapped), BaseExplicitEnv)

    prop = stormpy.parse_properties(property_str)[0]

    if not issubclass(type(env), BaseExplicitEnv): # has a wrapper
        mdp = build_stormpy_mdp(env.unwrapped)
        abs_map = env.unwrapped.get_abstraction_map()
    else:
        mdp = build_stormpy_mdp(env)
        abs_map = env.get_abstraction_map()

    result = stormpy.check_model_sparse(mdp, prop, 
                                        extract_scheduler = True)
    scheduler = result.scheduler
    policy = verigym.StormpyPolicy(
        scheduler, abs_map
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
