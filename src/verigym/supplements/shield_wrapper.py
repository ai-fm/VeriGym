

import random

import numpy as np

import gymnasium as gym


class AbstractPostShieldWrapper(gym.Wrapper):
    """
    Implements a simple post-shield over a Gymnasium-environment by overwriting the step function.
    """
    def __init__(self, env, shield, abstraction_mapper, seed=None):
        """
        Initialises the AbstractPostShieldWrapper.

        Parameters
        ----------
        env: VeriGymEnv
            The VeriGym environment to apply the shield to.
        shield: np.array
            An array of dimensionality (number abstract states, number actions).
            For each state-action pair: if shield[state, action] == True, the action is permitted. 
            If False, the action is blocked.
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
        self.rng = random.Random(seed)
        # Store these things in case we have no safe actions available
        self.obs, self.reward, self.info = None, None, None

    def reset(self, **kwargs):
        """
        Overwrites Env.reset to store information of the current observation and info.
        If a seed is passed, the shield's RNG for choosing a random safe action is reseeded as well.
        """
        if kwargs.get("seed") is not None:
            self.rng.seed(kwargs["seed"])
        self.obs, self.info = self.env.reset(**kwargs)
        self.reward = None
        return self.obs, self.info


    def step(self, action):
        """
        Take a step in the shielded environment, with post-shield semantics.
        That is, after an action is proposed, the shield checks whether it is permitted.
        The shield is applied as follows:

        If the abstract state corresponding to the current observation allows the chosen action,
            returns the result of env.step with the chosen action.
        Otherwise, the action is blocked. In that case:
            If other actions are available in the state,
                randomly chooses an available action, and returns the result of env.step with the random action.
            Otherwise, if no permitted action is available, terminates the episode.
        """
        state = self.abstraction_mapper.original_to_abstract_state_enum(self.obs)
        if self.shield[state, action]:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            available_actions = np.where(self.shield[state] > 0)[0].ravel().tolist()
            if available_actions != []:
                random_safe_action = self.rng.choice(available_actions)
                obs, reward, terminated, truncated, info = self.env.step(random_safe_action)
            else:
                return self.obs, self.reward, True, False, self.info

        self.obs, self.reward, self.info = obs, reward, info

        return obs, reward, terminated, truncated, info

