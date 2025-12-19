import numpy as np
from typing import Protocol
from abc import abstractmethod


class ExplicitFormatter(Protocol):
    """ 
    Convert an explicit model to a unified format for any `ExplicitModelEnv`.
    """
    initial_states = None
    nr_states = 0

    n_rewards = 0  # How many reward functions/models in the explicit model?
    # Are the reward models named?
    has_reward_labels = False
    # If self.has_reward_labels == True,
    # self.reward_labels = {label: idx for label in reward_labels}
    reward_labels = None

    # Do actions have labels?
    has_action_labels = False
    action_labels = []
    action_to_label = {idx: label for idx, label in enumerate(action_labels)}
    # Inverse mapping to action_to_label
    label_to_action = None
    # number of different actions
    nr_actions = 0
    # For each state, set action mask to 1 for all available actions
    action_mask = np.zeros((nr_states, nr_actions), dtype=np.int8)

    # Do states have additional labels (!= feature values)?
    has_state_labels = False
    # If self.has_state_labels,
    # self.labels_to_state = {label: set(states_with_label) for all state_labels}
    labels_to_state = None
    # Inverse mapping from state to label
    state_to_labels = None

    # Does the model have state valuations, i.e., feature values?
    has_state_valuations = False
    # If self.has_state_valuations,
    # self.state_to_values = {state_idx: {"var": val for var in features} for state in state_space}
    state_to_values = None

    # Transition function must have the format given by self._convert_transition_matrix.
    transition_function = None
    # Reward function must have the format given by self._convert_transition_matrix.
    reward_function = None

    def format(self, model) -> None:
        self.model = model
        self._format(model)

    @abstractmethod
    def _format(self, model) -> None:
        raise NotImplementedError("Implement in child class.")

    @abstractmethod
    def _convert_transition_matrix(self,
                                   transition_matrix) -> dict:
        """ Converts the original models transition matrix to a unified format for all `ExplicitModelEnv`s.
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

    @abstractmethod
    def _convert_reward_matrix(self,
                               reward_models) -> dict:
        """ Converts the original models reward model(s) to a unified format for all `ExplicitModelEnv`s.
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
