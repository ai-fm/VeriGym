from verigym.abstraction.abstractionmap import AbstractionMap

class PolicyClass:
    """
    An abstract class providing an interface to any kind of model policy.
    Given an observation in the VeriGym environment,
    return an action for the VeriGym environment based on the abstract model's policy.
    """

    def __init__(self, 
                 policy,
                 abstraction_map: AbstractionMap
                 ):
        self.policy = policy
        self.abstraction_map = abstraction_map

    def _action_from_policy(self, obs):
        """
        Get an action from the model's policy.

        Parameters
        ----------
        obs : object
            an observation in the abstract model's space
        
        Returns
        -------
        action : object
            an action in the abstract model's space according to the policy
        """
        # This should be implemented in specific child classes
        raise NotImplementedError

    def get_action(self, obs):
        """
        Gets an observation from `env.observation_space`. 
        Queries the model's policy.
        Map the action from the model's policy to `env.action_space`
        
        Parameters
        ----------
        obs : object
            An observation in the environment's observation space
        
        Returns
        -------
        action : object
            An action in the environment's action space

        """
        o = self.abstraction_map.original_to_abstract_state(obs) #self._obs_to_model(obs)
        a = self._action_from_policy(o)
        action = self.abstraction_map.abstract_to_original_action(a)
        return action

    def reset(self):
        """
        Resets the internal memory of the policy, if applicable.
        """
        pass