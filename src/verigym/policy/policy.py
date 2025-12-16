from verigym.environments.verigymenv import VeriGymEnv
from verigym.abstraction.abstractionmap import AbstractionMap

class PolicyClass:

    def __init__(self, 
                 policy,
                 abstraction_map: AbstractionMap
                 ):
        self.policy = policy
        self.abstraction_map = abstraction_map

    def _action_from_policy(self, obs):
        """
        Get an action from the model's policy.
        """
        # This should be implemented in specific child classes
        raise NotImplementedError

    def get_action(self, obs):
        """
        Gets an observation from `env.observation_space`. 
        Queries the model's policy.
        Map the action from the model's policy to `env.action_space`
        """
        o = self.abstraction_map.abstract_to_original_state(obs) #self._obs_to_model(obs)
        a = self._action_from_policy(o)
        action = self.abstract_to_original_action(a)
        return action
        