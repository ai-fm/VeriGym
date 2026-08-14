from collections import defaultdict

import numpy as np
import stormpy

from verigym.environments.formatter import ExplicitFormatter
from verigym.frameworks.stormpy.stormpy_utils import format_valuations
from verigym.environments.transition_func import TransitionFunction
from verigym.environments.reward_func import RewardFunction


class StormpyFormatter(ExplicitFormatter):
    """
    Convert a stormpy.storage.SparseMdp into a uniform format to support ExplicitEnvs/StormpyEnvs.
    """

    def __init__(self, mdp: stormpy.storage.SparseMdp):
        super().__init__(mdp)

        self._initialize_action_info()
        self._extract_state_labeling()
        self._extract_state_valuations()
        self.initial_states = np.zeros(mdp.nr_states)
        n_init = len(self.mdp.initial_states)
        for s in self.mdp.initial_states:
            self.initial_states[s] = 1.0 / n_init
        self.mdp.initial_states

        self.nr_states = self.mdp.nr_states

        # Initialize the transition function
        self.transition_function = self._convert_transition_matrix(
            self.mdp.transition_matrix
        )

        # Initialize the reward function
        self.n_rewards = len(self.mdp.reward_models)
        if self.n_rewards > 0:
            self.has_reward_labels = True
            self.reward_labels = {
                name: idx for idx, name in enumerate(self.mdp.reward_models.keys())
            }
            self.reward_function = self._convert_reward_matrix(self.mdp.reward_models)
        else:
            self.has_reward_labels = False
            self.reward_labels = None
            self.reward_function = None

    def _convert_transition_matrix(self, transition_matrix) -> TransitionFunction:
        """Store the transition matrix of a stormpy MDP as a uniform dict structure.

        Parameters
        ----------
        transition_matrix : stormpy.storage.SparseMdp.transition_matrix
            The transition matrix of the input model.

        Returns
        -------
        transition_function : dict
            Transition function for the gym-like environment.
        """
        T_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

        for state in range(self.mdp.nr_states):
            T_dict[state] = {}

            rows_from_state = transition_matrix.get_rows_for_group(state)
            # We consider states that only self-loop to be terminal.
            # Otherwise, the gym-like environment will forever be stuck in these states.
            if len(rows_from_state) == 1:
                for row_idx in rows_from_state:
                    row = transition_matrix.get_row(row_idx)
                    for r in row:
                        if r.column == state and r.value() == 1.0:
                            self.terminal_states.append(state)
            for row_idx in rows_from_state:
                if self.has_action_labels:
                    choice_label = self.mdp.choice_labeling.get_labels_of_choice(
                        row_idx
                    )
                    if len(choice_label) > 0:
                        choice_label = choice_label.pop()
                    else:
                        # These are usually set() i.e. empty
                        continue
                    a = self.label_to_action[choice_label]
                else:
                    a = 0

                if a not in T_dict[state].keys():
                    T_dict[state][a] = defaultdict(float)

                row = transition_matrix.get_row(row_idx)
                for r in row:
                    T_dict[state][a][r.column] = r.value()

        T = TransitionFunction(
            n_states=self.mdp.nr_states, n_actions=self.nr_actions, T_dict=T_dict
        )

        return T

    def _convert_reward_matrix(self, reward_models) -> dict:
        """Store the reward models of a stormpy MDP as a uniform dict structure.

        Parameters
        ----------
        reward_models : stormpy.storage.SparseMdp.reward_models
            The reward models of the input model.

        Returns
        -------
        reward_function : dict
            Reward function for the gym-like environment.
        """

        # Build the dict structure
        reward_function = defaultdict(dict)
        for state in self.mdp.states:
            reward_function[state.id] = {}
            for action in state.actions:
                if self.has_action_labels:
                    if len(action.labels) > 0:
                        a = self.label_to_action[action.labels.pop()]
                    else:
                        continue
                else:
                    a = 0
                reward_function[state.id][a] = [0 for _ in range(self.n_rewards)]

        # Iterate over the stormpy mdp's reward models
        for name, reward_model in reward_models.items():
            reward_idx = self.reward_labels[name]
            if reward_model.has_state_action_rewards:
                start_idx = 0
                for state in self.mdp.states:
                    for action in state.actions:
                        if len(action.labels) > 0:
                            a = self.label_to_action[action.labels.pop()]
                            reward_function[state.id][a][reward_idx] += (
                                reward_model.state_action_rewards[start_idx + action.id]
                            )
                    start_idx += len(state.actions)
            if reward_model.has_state_rewards:
                for state, actions in reward_function.items():
                    for a in actions:
                        reward_function[state][a][reward_idx] += (
                            reward_model.state_rewards[state]
                        )

        R = RewardFunction.from_dict(reward_function, self.nr_states, self.nr_actions)
        return R

    def _initialize_action_info(self) -> None:
        """
        Stores information about actions/choices in the formatter.
        Specifically, extracts
            has_action_labels : bool
            action_to_label : dict
            label_to_action : dict
            nr_actions : int
            action_mask : np.array(dtype=np.int8)
        """
        if self.mdp.has_choice_labeling:
            choice_labels = sorted(self.mdp.choice_labeling.get_labels())
            self.has_action_labels = True
            self.action_to_label = {i: label for i, label in enumerate(choice_labels)}
            self.label_to_action = {label: i for i, label in enumerate(choice_labels)}
            self.nr_actions = len(choice_labels)
            self.action_mask = np.zeros(
                (self.mdp.nr_states, self.nr_actions), dtype=np.int8
            )
            for state in self.mdp.states:
                for action in state.actions:
                    if len(action.labels) > 0:
                        label = action.labels.pop()
                        a = self.label_to_action[label]
                        self.action_mask[state.id][a] = 1
        else:
            self.nr_actions = 1
            self.has_action_labels = False
            self.action_to_label = {}
            self.label_to_action = {}
            self.action_mask = np.ones((self.mdp.nr_states, 1), dtype=np.int8)

    def _extract_state_labeling(self) -> None:
        """
        Stores information about state labels in the formatter.
        Specifically, extracts
            has_state_labels : None
            labels_to_state : dict
            state_to_labels : dict
        """
        state_labeling = self.mdp.labeling.get_labels()
        if len(state_labeling) > 0:
            self.has_state_labels = True
            self.labels_to_states = {label: set() for label in state_labeling}
            self.state_to_labels = {s: set() for s in range(self.mdp.nr_states)}
            for s in range(self.mdp.nr_states):
                for label in self.mdp.labels_state(s):
                    self.labels_to_states[label].add(s)
                    self.state_to_labels[s].add(label)
        else:
            self.state_labels = False
            self.labels_to_states = {}
            self.state_to_labels = {}

    def _extract_state_valuations(self):
        """
        Stores information about state valuations in the formatter.
        Specifically, extracts
            has_state_valuations : bool
            state_to_values : dict
        """
        # Ensure a consistent order of the variables
        self.var_order = [name for name in format_valuations(self.mdp.states[0].valuations)]
        self.max_valuations = [0 for _ in self.var_order]
        self.state_to_values = {}
        self.values_to_state = {}

        if self.mdp.has_state_valuations:
            self.has_state_valuations = True
            for s in self.mdp.states:
                valuations = format_valuations(s.valuations)
                self.state_to_values[s.id] = valuations
                self.values_to_state[self._valuation_to_state_tuple(valuations)] = s.id
                for i, name in enumerate(self.var_order):
                    if self.max_valuations[i] < valuations[name]:
                        self.max_valuations[i] = valuations[name]

        else:
            self.has_state_valuations = False

    def _valuation_to_state_tuple(self, valuation):
        """
        Turn the dict state valuation into a consistent tuple
        """
        full_state = tuple([valuation[name] for name in self.var_order])
        return full_state
    
    def get_full_state_from_idx(self, state_idx):
        return self._valuation_to_state_tuple(self.state_to_values[state_idx])