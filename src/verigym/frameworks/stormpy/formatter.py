import operator
from collections import defaultdict

import stormpy

from verigym.environments.formatter import ExplicitFormatter
from verigym.frameworks.stormpy.stormpy_utils import _valuation_variables, _choice_actions
from verigym.environments.transition_func import IntervalTransitionFunction, TransitionFunction
from verigym.environments.reward_func import IntervalRewardFunction, RewardFunction
from verigym.environments.state_distributions import StateDistribution


def _add_intervals(interval, other):
    """The sum of two reward intervals: the sums of their bounds."""
    return interval[0] + other[0], interval[1] + other[1]


class StormpyFormatter(ExplicitFormatter):
    """
    Convert a stormpy.storage.SparseMdp into a uniform format to support ExplicitEnvs/StormpyEnvs.

    Also converts interval MDPs (stormpy.storage.SparseIntervalMdp): their transition probabilities and rewards are
    (lower, upper) tuples, in an `IntervalTransitionFunction` and an `IntervalRewardFunction`. Everything else (actions,
    labels, valuations, initial states) is the same for both.
    """

    def __init__(self, mdp: stormpy.storage.SparseMdp | stormpy.storage.SparseIntervalMdp):
        super().__init__(mdp)

        self.is_interval = isinstance(mdp, stormpy.storage.SparseIntervalMdp)
        self._initialize_action_info(mdp)
        self._extract_state_labeling(mdp)
        self._extract_state_valuations(mdp)
        self.initial_states = StateDistribution(mdp.nr_states)
        n_init = len(mdp.initial_states)
        for s in mdp.initial_states:
            self.initial_states[int(s)] = 1.0 / n_init

        self.nr_states = mdp.nr_states

        # Initialize the reward function
        self.n_rewards = len(mdp.reward_models)
        if self.n_rewards > 0:
            self.has_reward_labels = True
            self.reward_labels = {
                name: idx for idx, name in enumerate(mdp.reward_models.keys())
            }
        else:
            self.has_reward_labels = False
            self.reward_labels = None

    def _values(self, values) -> list:
        """Probabilities or rewards of the model: floats, or (lower, upper) tuples for interval models."""
        if self.is_interval:
            return [(value.lower(), value.upper()) for value in values]
        return list(values)

    def _convert_transition_matrix(self, mdp) -> TransitionFunction | IntervalTransitionFunction:
        """Store the transition matrix of a stormpy MDP as a uniform dict structure.

        Parameters
        ----------
        transition_matrix : stormpy.storage.SparseMdp.transition_matrix
            The transition matrix of the input model.

        Returns
        -------
        transition_function : TransitionFunction | IntervalTransitionFunction
            Transition function for the gym-like environment; an `IntervalTransitionFunction` for interval models,
            which only keeps the successors with an upper bound above 0.
        """
        transition_matrix = mdp.transition_matrix
        row_groups = list(mdp.nondeterministic_choice_indices)
        T_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

        for state in range(mdp.nr_states):
            T_dict[state] = {}

            for row_idx in range(row_groups[state], row_groups[state + 1]):
                a = self._choice_to_action[row_idx]
                if a is None:
                    # Unlabelled choices, e.g., the self-loops that stormpy adds to deadlocks
                    continue

                # Read the row once, since every access of a stormpy row is costly
                entries = [(entry.column, entry.value()) for entry in transition_matrix.get_row(row_idx)]
                if self.is_interval:
                    entries = [(column, (value.lower(), value.upper())) for column, value in entries]

                if a not in T_dict[state].keys():
                    T_dict[state][a] = defaultdict(lambda: (0.0, 0.0)) if self.is_interval else defaultdict(float)

                for column, value in entries:
                    if self.is_interval and value[1] == 0:
                        continue
                    T_dict[state][a][column] = value

        transition_function_class = IntervalTransitionFunction if self.is_interval else TransitionFunction
        T = transition_function_class(
            n_states=mdp.nr_states, n_actions=self.nr_actions, T_dict=T_dict
        )

        return T

    def _convert_reward_matrix(self, mdp) -> RewardFunction | IntervalRewardFunction:
        """Store the reward models of a stormpy MDP as a uniform dict structure.

        Parameters
        ----------
        reward_models : stormpy.storage.SparseMdp.reward_models
            The reward models of the input model.

        Returns
        -------
        reward_function : RewardFunction | IntervalRewardFunction
            Reward function for the gym-like environment; an `IntervalRewardFunction` for interval models.
        """
        reward_models = mdp.reward_models

        # Without reward models, all rewards are 0: none are stored, since missing rewards read as 0
        if len(reward_models) == 0:
            if self.is_interval:
                return IntervalRewardFunction.from_dict({}, self.nr_states, self.nr_actions)
            return RewardFunction(self.nr_states, self.nr_actions)

        row_groups = list(mdp.nondeterministic_choice_indices)
        state_rows = [range(row_groups[state], row_groups[state + 1]) for state in range(mdp.nr_states)]
        zero = (0.0, 0.0) if self.is_interval else 0.0
        add = _add_intervals if self.is_interval else operator.add

        # Build the dict structure, with the actions of the choices (rows) of every state
        reward_function = defaultdict(dict)
        for state, rows in enumerate(state_rows):
            reward_function[state] = {}
            for row in rows:
                a = self._choice_to_action[row]
                if a is not None:
                    reward_function[state][a] = [zero for _ in range(self.n_rewards)]

        # Iterate over the stormpy mdp's reward models
        for name, reward_model in reward_models.items():
            reward_idx = self.reward_labels[name]
            if reward_model.has_state_action_rewards:
                # stormpy returns a copy of the reward vector on every access, so get it once
                state_action_rewards = self._values(reward_model.state_action_rewards)
                for state, rows in enumerate(state_rows):
                    for row in rows:
                        a = self._choice_to_action[row]
                        if a is not None:
                            rewards = reward_function[state][a]
                            rewards[reward_idx] = add(rewards[reward_idx], state_action_rewards[row])
            if reward_model.has_state_rewards:
                state_rewards = self._values(reward_model.state_rewards)
                for state, actions in reward_function.items():
                    for rewards in actions.values():
                        rewards[reward_idx] = add(rewards[reward_idx], state_rewards[state])

        return self._reward_function_from_dict(reward_function)

    def _reward_function_from_dict(self, reward_function) -> RewardFunction | IntervalRewardFunction:
        """The reward function of a dict s -> a -> reward, or a list with a reward per reward model."""
        if not self.is_interval:
            return RewardFunction.from_dict(reward_function, self.nr_states, self.nr_actions)
        if self.n_rewards > 1:
            raise NotImplementedError("Interval models with more than one reward model are not supported.")
        R_dict = {
            s: {a: rewards[0] if isinstance(rewards, list) else rewards for a, rewards in actions.items()}
            for s, actions in reward_function.items()
        }
        return IntervalRewardFunction.from_dict(R_dict, self.nr_states, self.nr_actions)

    def _initialize_action_info(self, mdp) -> None:
        """
        Stores information about actions/choices in the formatter.
        Specifically, extracts
            has_action_labels : bool
            action_to_label : dict
            label_to_action : dict
            nr_actions : int

        The actions are the choice labels; if a state has several choices with the same label, the k-th one is the
        action "label_k" (see `_choice_actions`). Without choice labels, the i-th choice of a state is action i.
        The action of every choice (row) of `mdp` is stored in `_choice_to_action`, from which the transition and
        reward functions are built.
        """
        self.has_action_labels = mdp.has_choice_labeling()
        self._choice_to_action, label_to_action, self.nr_actions = _choice_actions(mdp)
        self.label_to_action = label_to_action
        self.action_to_label = {i: label for label, i in label_to_action.items()}

    def _extract_state_labeling(self, mdp) -> None:
        """
        Stores information about state labels in the formatter.
        Specifically, extracts
            has_state_labels : bool
            labels_to_states : dict
            state_to_labels : dict
        """
        state_labeling = mdp.labeling.get_labels()
        if len(state_labeling) > 0:
            self.has_state_labels = True
            # One bit vector of states per label (cheaper to get from stormpy than the labels of every state)
            self.labels_to_states = {label: set(mdp.labeling.get_states(label)) for label in state_labeling}
            self.state_to_labels = {s: set() for s in range(mdp.nr_states)}
            for label, states in self.labels_to_states.items():
                for s in states:
                    self.state_to_labels[s].add(label)
        else:
            self.has_state_labels = False
            self.labels_to_states = {}
            self.state_to_labels = {}

    def _extract_state_valuations(self, mdp):
        """
        Stores information about state valuations in the formatter.
        Specifically, extracts
            has_state_valuations : bool
            state_to_values : dict
        """
        self.var_order = []
        self.max_valuations = []
        self.state_to_values = {}
        self.values_to_state = {}

        if mdp.has_state_valuations():
            self.has_state_valuations = True
            # Ensure a consistent order of the variables: the order in which stormpy prints them
            variables = _valuation_variables(mdp.state_valuations)
            self.var_order = [variable.name for variable in variables]
            # The values of each variable in all states; booleans are 0 (false) or 1 (true)
            columns = []
            for variable in variables:
                values = mdp.state_valuations.get_values_states(variable)
                if not (variable.has_boolean_type() or variable.has_integer_type()):
                    # Other (rational) variables are only supported if all their values are whole numbers
                    if not variable.has_rational_type() or any(value.denominator != 1 for value in values):
                        raise NotImplementedError(
                            f"State variable {variable.name} has values that are not booleans or integers, "
                            "which is not supported."
                        )
                    values = [float(value) for value in values]
                columns.append([int(value) for value in values])
            self.max_valuations = [max(0, max(column)) for column in columns]
            # The valuation of every state as a tuple of the values in `var_order`
            tuples = list(zip(*columns)) if len(columns) > 0 else [()] * mdp.nr_states
            for s, values in enumerate(tuples):
                self.state_to_values[s] = dict(zip(self.var_order, values))
                self.values_to_state[values] = s

        else:
            self.has_state_valuations = False

    def _valuation_to_state_tuple(self, valuation):
        """
        Turn the dict state valuation into a consistent tuple
        """
        return tuple(valuation[name] for name in self.var_order)
    
    def get_full_state_from_idx(self, state_idx):
        return self._valuation_to_state_tuple(self.state_to_values[state_idx])