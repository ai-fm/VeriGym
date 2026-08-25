import verigym
from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.environments.exporter import export_to_stormpy_mdp
from verigym.abstraction.abstractionmapper import AbstractionMapper

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
        mdp = export_to_stormpy_mdp(env.unwrapped)
        abs_map = env.unwrapped.get_abstraction_map()
    else:
        mdp = export_to_stormpy_mdp(env)
        abs_map = env.get_abstraction_map()
        
    if abs_map is None:
        # if there is no abstraction map, return identity map
        abs_map = AbstractionMapper.initialize_identity_mapper(env.observation_space, env.action_space)
        

    result = stormpy.check_model_sparse(mdp, prop, 
                                        extract_scheduler = True)
    scheduler = result.scheduler
    policy = verigym.StormpyPolicy(
        scheduler, abs_map
    )
    return policy