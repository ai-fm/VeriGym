import verigym
from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.environments.exporter import export_to_stormpy_mdp
from verigym.abstraction.abstractionmapper import AbstractionMapper
from verigym.policy.policy import PolicyClass
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_dtmc

import stormpy
import numpy as np
import gymnasium as gym

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
        scheduler, abs_map, mdp
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


class AbstractShieldWrapper(gym.Wrapper):
    """
    Implements a simple shield over a gym-environment by overwriting the step function.
    """
    def __init__(self, env, shield, abstraction_mapper):
        """
        Initialises the AbstractShieldWrapper.

        Parameters
        ----------
        env: VeriGymEnv
            The environment to shield.
        shield: np.array
            An array of dimensionality (number abstract states, number actions).
            For each state-action pair: if shield[state, action] == 1, the action is permitted. 
            If 0, the action is blocked.
        abstraction_mapper:
            An abstraction mapper that maps to a discrete abstract state space on which the shield is defined.
        
        Notes
        -----
        The input environment must have a discrete action space.
        The abstraction mapper must map to a discrete abstract state space.
        """
        super().__init__(env)
        self.env = env
        self.shield = shield
        self.abstraction_mapper = abstraction_mapper
        # Store these things in case we have no safe actions available
        self.obs, self.reward, self.info = None, None, None

    def reset(self, **kwargs):
        """
        Overwrites Env.reset to store information of the current observation and info.
        """
        self.obs, self.info = self.env.reset(**kwargs)
        self.reward = None
        return self.obs, self.info


    def step(self, action):
        """
        Take a step in the SHIELDED environment.
        The shield is applied as follows:

        If the abstract state corresponding to the current observation allows the chosen action,
            returns the result of env.step with the chosen action.
        Else, the action is blocked.
        If the action is blocked, but other actions are available in the state,
            randomly chooses an available action, and returns the result of env.step with the random action.
        Else, if no action is available, terminates the episode.
        """
        state = self.abstraction_mapper.original_to_abstract_state_enum(self.obs)
        if self.shield[state, action] > 0:
            obs, reward, terminated, truncated, info = self.env.step(action)            
        elif any(self.shield[state]) > 0:
            available_actions = np.where(self.shield[state] > 0)[0]
            random_safe_action = np.random.choice(available_actions)
            obs, reward, terminated, truncated, info = self.env.step(random_safe_action)
        else:
            return self.obs, self.reward, True, False, self.info
        
        self.obs, self.reward, self.info = obs, reward, info

        return obs, reward, terminated, truncated, info
        
