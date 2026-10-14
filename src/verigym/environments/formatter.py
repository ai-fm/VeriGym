from typing import Protocol

import numpy as np

from verigym.environments.transition_func import TransitionFunction
from verigym.environments.reward_func import RewardFunction
from verigym.environments.state_distributions import StateDistribution


class ExplicitFormatter(Protocol):
    """
    Convert an explicit model to a unified format for any `ExplicitEnv`.
    """

    # Set by formatters of models with state valuations, for factored states:
    # the variables, the maximal value of each, and the inverse mapping of `get_full_state_from_idx`
    var_order: list[str]
    max_valuations: list[int]
    values_to_state: dict[tuple, int]
    # Mapping from each state label to the states with the label
    labels_to_states: dict[str, set[int]]

    def __init__(self, model):
        #self.mdp = model

        self.nr_states = 0
        self.initial_states = StateDistribution(self.nr_states)  # initial state distribution
        self.terminal_states = [] # terminal states

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
        # self.labels_to_states = {label: set(states_with_label) for all state_labels}
        self.labels_to_states = None
        # Inverse mapping from state to label
        self.state_to_labels = None

        # Does the model have state valuations, i.e., feature values?
        self.has_state_valuations = False
        # If self.has_state_valuations,
        # self.state_to_values = {state_idx: {"var": val for var in features} for state in state_space}
        self.state_to_values = None


    def _convert_transition_matrix(self, transition_matrix) -> TransitionFunction:
        """Converts the original models transition matrix/function to a `TransitionFunction` compatible with verigym."""
        ...

    def _convert_reward_matrix(self, reward_models) -> RewardFunction:
        """Converts the original models reward model(s) to a `RewardFunction` compatible with verigym."""
        ...

    def sample_initial_state(self):
        assert self.initial_states is not None
        return self.initial_states.sample()

    def get_full_state_from_idx(self, state_idx: int) -> tuple:
        """The factored state (the values of the variables in `var_order`) of the state with index `state_idx`."""
        ...
