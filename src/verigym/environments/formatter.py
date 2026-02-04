import numpy as np
from typing import Protocol

class ExplicitFormatter(Protocol):
    """ 
    Convert an explicit model to a unified format for any `ExplicitEnv`.
    """
    def __init__(self, model):
        self.mdp = model

        self.nr_states = 0
        self.initial_states = np.zeros(self.nr_states) # initial state distribution

        self.n_rewards = 0  # How many reward functions/models in the explicit model?
        # Are the reward models named?
        self.has_reward_labels = False 
        # If self.has_reward_labels == True, 
        # self.reward_labels = {label: idx for label in reward_labels}
        self.reward_labels = None

        # Do actions have labels?
        self.has_action_labels = False
        # If self.has_action_labels == True,
        # self.action_to_label = {idx: label for idx, label in enumerate(action_labels)}
        self.action_to_label = None
        # Inverse mapping to action_to_label
        self.label_to_action = None
        # number of different actions
        self.nr_actions = 0
        # For each state, set action mask to 1 for all available actions
        self.action_mask = np.zeros((self.nr_states, self.nr_actions), dtype=np.int8)

        # Do states have additional labels (!= feature values)?
        self.has_state_labels = False
        # If self.has_state_labels, 
        # self.labels_to_state = {label: set(states_with_label) for all state_labels}
        self.labels_to_state = None
        # Inverse mapping from state to label
        self.state_to_labels = None

        # Does the model have state valuations, i.e., feature values?
        self.has_state_valuations = False
        # If self.has_state_valuations,
        # self.state_to_values = {state_idx: {"var": val for var in features} for state in state_space}
        self.state_to_values = None

        # Transition function must have the format given by self._convert_transition_matrix.
        self.transition_function = None
        # Reward function must have the format given by self._convert_transition_matrix.
        self.reward_function = None


    def _convert_transition_matrix(self,
                                   transition_matrix) -> dict:
        """ Converts the original models transition matrix to a unified format for all `ExplicitEnv`s.
        The output transition function should have the following format:
        P = {
            state_index: {
                action_index:
                    {
                        next_state_index: probability
                        for next_state_index in non_zero_transitions
                    }
                    for action_index in action_space
            } for state_index in state_space
        }
        """
        ...

    def _convert_reward_matrix(self, 
                               reward_models) -> dict:
        """ Converts the original models reward model(s) to a unified format for all `ExplicitEnv`s.
        Represents all state/state-action rewards as state-action rewards
        The output reward function should have the following format:
        R = {
            state_index: {
                action_index: [list of rewards]
                for action_index in action_space
            }
            for state_index in state_space
        }
        """
        ...

    def sample_initial_state(self):
        assert self.initial_states is not None
        
        idx = np.random.choice(len(self.initial_states), p=self.initial_states)
        return idx

