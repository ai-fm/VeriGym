from verigym.environments.explicitenv import ExplicitEnv
from verigym.frameworks.stormpy.formatter import StormpyExplicitFormatter

import gymnasium as gym
import random
import stormpy
from typing import Optional

class StormpyEnv(ExplicitEnv):
    """
    VeriGym wrapper for `stormpy` MDP models.
    
    Parameters
    ----------
    stormpy_mdp : stormpy.storage.SparseMdp
        The input stormpy MDP
    """

    def __init__(self, 
                 model: stormpy.storage.SparseMdp,
                 render_mode: str | None = None):
        super().__init__(
              model, 
              render_mode=render_mode
        )

        self.formatter = StormpyExplicitFormatter(self.model)

        self.transition_function = self.formatter.transition_function
        self.reward_function = self.formatter.reward_function

        self.state = random.choice(self.formatter.initial_states)
        self.nr_states = self.model.nr_states
        self.nr_actions = self.formatter.nr_actions

        self.observation_space = gym.spaces.Discrete(self.nr_states)
        self.action_space = gym.spaces.Discrete(self.nr_actions)

        # Which actions are available in a state?
        self.action_mask = self.formatter.action_mask
        
    def _sample_transition(self, state, action):
        """
        Sample a transition from the transition function according to the transition probabilities.

        Parameters
        ----------
        state : int
            The current state index
        action : int
            The chosen action index
        
        Returns
        -------
        next_state : int
            The sampled next state according to the transition probabilities.
        """
        transitions = self.transition_function[state][action]
        next_states = sorted(transitions.keys())
        probs = [transitions[next_state] for next_state in next_states]

        next_state = random.choices(next_states, weights=probs, k=1)[0]
        return next_state
    
    def step(self, action):
        """
        Take a step in the environment from the current state.

        Parameters
        ----------
        action : int
            Chosen action from self.action_space

        Returns : 
            observation : dict if self.formatter.has_state_valuations else int
            reward : list(int)
            terminated : bool
            truncated : bool
            info : dict
        """
        if self.action_mask[self.state][action] > 0:
            print(self.reward_function[self.state])
            reward = self.reward_function[self.state][action]
            self.state = self._sample_transition(self.state, action)
        else: 
            reward = [0.0 for _ in range(self.formatter.n_rewards)]

        # terminal states are those that have no actions available
        terminated = True if sum(self.action_mask[self.state]) == 0.0 else False 
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None):
        """
        Reset to an initial state.
        """
        super().reset(seed=seed,
                      options=options)

        self.state = random.choice(self.formatter.initial_states)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_info(self):
        """
        Accumulate additional information about the state/environment.

        Returns
        -------
        info : dict
        """
        info = {
            "action_mask": self.action_mask[self.state]
        }
        if self.formatter.has_state_valuations:
            info["state_valuations"] = self.formatter.state_to_values[self.state]
        if self.formatter.has_state_labels:
            info["state_labels"] = self.formatter.state_to_labels[self.state]
        if self.formatter.has_reward_labels:
            info["reward_labels"] = list(self.formatter.reward_labels.keys())
        if self.formatter.has_action_labels:
            info["action_labels"] = self.formatter.action_to_label
        return info