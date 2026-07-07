import verigym
from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.environments.exporter import export_to_stormpy_mdp

import stormpy

"""
This file contains supplementary functions for more user-friendly model checking.
That can mean wrapping functionality from external frameworks into a one-line function call.
"""

def get_policy_from_stormpy(env: BaseExplicitEnv,
                         property_str: str):
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
    assert issubclass(type(env), BaseExplicitEnv)

    mdp = export_to_stormpy_mdp(env)
    prop = stormpy.parse_properties(property_str)[0]

    result = stormpy.check_model_sparse(mdp, prop, 
                                        extract_scheduler = True)
    scheduler = result.scheduler
    abs_map = env.abstraction_map if hasattr(env, "abstraction_map") else None
    policy = verigym.StormpyPolicy(
        scheduler, abs_map
    )
    return policy