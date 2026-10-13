import stormpy
from collections import namedtuple

from verigym.environments.explicitenv import BaseExplicitEnv
from verigym.environments.labeling import AbstractStateLabeler
from verigym.environments.interval_explicitenv import IntervalExplicitEnv
from verigym.policy.policy import PolicyClass

# The stormpy classes to build models with point values (MDPs) or with intervals (iMDPs)
_ModelClasses = namedtuple("_ModelClasses", ["matrix_builder", "reward_model", "components"])
_POINT_CLASSES = _ModelClasses(
    stormpy.SparseMatrixBuilder, stormpy.SparseRewardModel, stormpy.SparseModelComponents
)
_INTERVAL_CLASSES = _ModelClasses(
    stormpy.IntervalSparseMatrixBuilder, stormpy.SparseIntervalRewardModel, stormpy.SparseIntervalModelComponents
)

def build_stormpy_mdp(env: BaseExplicitEnv, overapproximate=True) -> stormpy.storage.SparseMdp:
    """
    Builds a `stormpy.storage.SparseMdp` from a `BaseExplicitEnv`.

    Parameters
    ----------
    env : ExplicitEnv

    Returns
    -------
    mdp : stormpy.storage.SparseMdp

    Notes
    -----
    - Every state gets one choice per available action, i.e., per action with at least one successor in the
      transition function.
    - Choices are labelled with the env's action names if it has action labels (e.g., when loaded from a prism file),
      and with the action indices otherwise. `StormpyPolicy` maps the choices of a scheduler back to the env's
      actions in both cases (see its `label_to_action` parameter).
    - States without available actions are deadlocks. This includes states whose actions have no successors, which
      used to break the export. A deadlock gets a single unlabelled self-loop with probability 1 and zero reward,
      and the state label "deadlock".
    """
    components = _build_model_components(
        env,
        T_dict=env.get_transition_function().T_dict,
        R_dict=env.get_reward_function().R_dict,
        to_value=lambda value: value,
        model_classes=_POINT_CLASSES,
        overapproximate=overapproximate,
    )
    return stormpy.storage.SparseMdp(components)

def build_stormpy_imdp(env: BaseExplicitEnv,
                       use_reward_uncertainty=False,
                       overapproximate=True):
    """
    Builds a stormpy IMDP from any `BaseExplicitEnv`.
    If used with a standard `ExplicitEnv` instead of an `IntervalExplicitEnv`, it will build an IMDP where lower bounds == upper bounds everywhere.
    Apart from the intervals, the IMDP is the same as the MDP of `build_stormpy_mdp` (see its notes on choices,
    choice labels and deadlocks).

    Parameters
    ----------
    env : BaseExplicitEnv
        The explicit env.
    use_reward_uncertainty : bool = False
        Whether to build uncertainty over rewards.
        If True and the env is an `IntervalExplicitEnv`, this will use the `interval_reward_function`.
        Else, it will use the `reward_function` and build interval rewards in stormpy where lower bounds == upper bounds.
        This is because standard (non-interval) reward functions are technically incompatible with the stormpy IMDP.
    overapproximate : bool = True
        How to handle state labels for abstract state.
        If True, an abstract state gets assigned a label if any original state in it has the label.
        Else, an abstract state gets assigned a label only if all original states in it have the label.

    Returns
    -------
    imdp : stormpy.SparseIntervalMdp
    """
    if isinstance(env, IntervalExplicitEnv):
        T_dict = env.get_interval_transition_function().T_dict
    else:
        T_dict = env.get_transition_function().T_dict

    if use_reward_uncertainty:
        assert isinstance(env, IntervalExplicitEnv), "Cannot derive uncertain rewards from non-uncertain environment."
        R_dict = env.get_interval_reward_function().R_dict
    else:
        R_dict = env.get_reward_function().R_dict

    components = _build_model_components(
        env,
        T_dict=T_dict,
        R_dict=R_dict,
        to_value=_to_interval,
        model_classes=_INTERVAL_CLASSES,
        overapproximate=overapproximate,
    )
    return stormpy.storage.SparseIntervalMdp(components)

def _build_model_components(env: BaseExplicitEnv, T_dict, R_dict, to_value, model_classes: _ModelClasses,
                            overapproximate: bool):
    """
    Builds the stormpy model components for `build_stormpy_mdp` and `build_stormpy_imdp`.

    The transition matrix, the reward models and the choice labeling are built in a single pass over the choices,
    so they are aligned by construction: one choice per state and available action (see `_available_actions`),
    labelled with the action's name if the env has action labels (else its index), and a single unlabelled self-loop
    with zero reward for each deadlock.

    Parameters
    ----------
    env : BaseExplicitEnv
        The explicit env, for the number of states, actions and rewards, and the state labels.
    T_dict : dict
        Mapping s -> a -> s' -> transition probability (or (lower, upper) tuple).
    R_dict : dict
        Mapping s -> a -> reward (or (lower, upper) tuple), or a list thereof for multiple reward models.
        Missing rewards are 0.
    to_value : callable
        Converts a probability or reward of `T_dict` and `R_dict` to the value stored in the stormpy model.
    model_classes : _ModelClasses
        The stormpy classes for models with point values (`_POINT_CLASSES`) or intervals (`_INTERVAL_CLASSES`).
    overapproximate : bool
        How to handle state labels for abstract states (see `_build_state_label_map`).

    Returns
    -------
    components : stormpy.SparseModelComponents | stormpy.SparseIntervalModelComponents
    """
    info = _get_info_from_formatter(env)

    if "reward_labels" in info.keys():
        reward_labels = info["reward_labels"]
    else:
        reward_labels = {f"reward{i}": i for i in range(env.nr_rewards)}
    reward_models = {label: [] for label in reward_labels.keys()}

    if "action_labels" in info.keys():
        action_labels = info["action_labels"]
    else:
        action_labels = {a: str(a) for a in range(env.nr_actions)}

    # Build the transition matrix, rewards and choice labels, choice by choice
    builder = model_classes.matrix_builder(
        rows=0,
        columns=env.nr_states,
        entries=0,
        force_dimensions=True,
        has_custom_row_grouping=True,
    )
    choice_counter = 0
    custom_choice_labeling = {}
    # Stormpy needs a row group for every state, so this loop is over all states.
    one, zero = to_value(1.0), to_value(0.0)
    for s in range(env.nr_states):
        builder.new_row_group(choice_counter)
        actions = _available_actions(T_dict, s, env.nr_actions) if s in T_dict else []
        for a in actions:
            for next_s, prob in T_dict[s][a].items():
                if not (isinstance(prob, float) and prob == 0):
                    builder.add_next_value(choice_counter, next_s, to_value(prob))
            rewards = R_dict.get(s, {}).get(a, 0.0)
            for label, idx in reward_labels.items():
                reward = rewards[idx] if isinstance(rewards, list) else rewards
                reward_models[label].append(to_value(reward))
            custom_choice_labeling[choice_counter] = action_labels[a]
            choice_counter += 1
        # self-loop deadlocks with 0 reward # TODO this might be incorrect for min max?
        if len(actions) == 0:
            builder.add_next_value(choice_counter, s, one)
            for label in reward_labels.keys():
                reward_models[label].append(zero)
            choice_counter += 1
    transition_matrix = builder.build()

    # Assemble the components
    components = model_classes.components(
        transition_matrix=transition_matrix,
        rate_transitions=False,
    )

    if len(reward_models) > 0:
        components.reward_models = {
            label: model_classes.reward_model(optional_state_action_reward_vector=reward_vector)
            for label, reward_vector in reward_models.items()
        }

    if "state_labels" in info.keys():
        components.state_labeling = info["state_labels"]
    else:
        state_labels = _build_state_label_map(env, T_dict, overapproximate)
        components.state_labeling = _build_state_labeling(
            env.nr_states, state_labels
        )

    components.choice_labeling = _build_choice_labeling(nr_choices=choice_counter,
                                                        choice_to_label=custom_choice_labeling,
                                                        choice_labels=list(action_labels.values()))

    if "valuations" in info.keys():
        components.state_valuations = info["valuations"]

    return components

def _available_actions(T_dict, s, nr_actions) -> list[int]:
    """
    The actions of state `s` that have at least one successor in `T_dict`, in ascending order.
    A state without available actions is a deadlock.
    Only reads `T_dict` via `.get`, so it does not add keys to (default)dicts.
    """
    actions = T_dict.get(s, {})
    return [a for a in range(nr_actions) if len(actions.get(a, {})) > 0]

def _to_interval(value) -> stormpy.pycarl.Interval:
    """Converts a point value or a (lower, upper) tuple to a stormpy interval."""
    if isinstance(value, tuple):
        return stormpy.pycarl.Interval(value[0], value[1])
    return stormpy.pycarl.Interval(value, value)

def build_stormpy_dtmc(env: BaseExplicitEnv,
                       policy: PolicyClass,
                       overapproximate=True):
    """
    Builds a `stormpy.storage.SparseDtmc` from a `BaseExplicitEnv` and some policy.

    Parameters
    ----------
    env : BaseExplicitEnv
        The environment.
    policy : PolicyClass
        A policy for that environment.
    overapproximate : bool
        Whether to overapproximate state labels on abstract environments.
        Default = True.

    Returns
    -------
    dtmc : stormpy.storage.SparseDtmc
    """

    # Build DTMC reward and transition function
    info = _get_info_from_formatter(env)
    num_s = env.nr_states
    T_dict = env.get_transition_function().T_dict
    R_dict = env.get_reward_function().R_dict
    R = []

    builder = stormpy.SparseMatrixBuilder(rows=0, columns=num_s, entries=0,
                                          force_dimensions=True, has_custom_row_grouping=False,
                                          row_groups=0)

    for s in range(num_s):
        T_s = T_dict.get(s, {})
        transitions = {}
        if len(T_s) == 0:
            R.append(0.0)
        else:
            action = int(policy.get_action(s))
            transitions = T_s.get(action, {})
            R.append(R_dict.get(s, {}).get(action, 0.0))

        for (n_s, prob) in transitions.items():
            builder.add_next_value(s, n_s, prob)
        # self-loop terminal states
        if len(transitions) == 0:
            builder.add_next_value(s, s, 1)

    transition_matrix = builder.build()

    reward_model = {
        "reward": stormpy.SparseRewardModel(optional_state_action_reward_vector=R)
    }

    components = stormpy.SparseModelComponents(transition_matrix=transition_matrix,
                                               reward_models=reward_model)

    # state labels
    if "state_labels" in info.keys():
        components.state_labeling = info["state_labels"]
    else:
        state_labels = _build_state_label_map(env, T_dict, overapproximate)
        components.state_labeling = _build_state_labeling(
            env.nr_states, state_labels
        )

    # state valuations
    if "valuations" in info.keys():
        components.state_valuations = info["valuations"]

    dtmc = stormpy.storage.SparseDtmc(components)

    return dtmc
    

def _get_info_from_formatter(env):
    info = {}
    if not hasattr(env, "formatter"):
        return info

    if env.formatter.has_action_labels:
        info["action_labels"] = env.formatter.action_to_label

    if env.formatter.has_reward_labels:
        reward_labels = env.formatter.reward_labels
        info["reward_labels"] = reward_labels

    if env.formatter.has_state_labels:
        state_labeling = _build_state_labeling(
            env.nr_states, env.formatter.labels_to_states
        )
        info["state_labels"] = state_labeling

    if env.formatter.has_state_valuations:
        state_valuations = _build_state_valuations(env.formatter.state_to_values)
        info["valuations"] = state_valuations

    return info

def _build_state_label_map(env, T_dict, overapproximate) -> dict:
    """
    Builds the state labels of an env without a formatter (e.g., a learned abstraction).

    - "init": the states with positive initial probability.
    - "deadlock": the states without available actions (see `_available_actions`).
    - The env's state labels: only evaluated for the states in `_support_states(env, T_dict)`
    Parameters
    ----------
    env : BaseExplicitEnv
        The env, for the number of states and actions, the initial states and the state labeler.
    T_dict : dict
        The transition function that is exported, mapping s -> a -> s' -> probability (or (lower, upper) tuple).
    overapproximate : bool
        How to handle state labels for abstract states, see `AbstractStateLabeler`.

    Returns
    -------
    labels_to_states : dict
        Mapping label -> states, as a list of states or a `stormpy.BitVector`.
    """
    init_states = sorted(s for s, prob in env.initial_states.distr_dict.items() if prob > 0)

    deadlocks = stormpy.BitVector(env.nr_states, True)
    for s in T_dict.keys():
        if len(_available_actions(T_dict, s, env.nr_actions)) > 0:
            deadlocks.set(s, False)

    labels_to_states = {"init": init_states, "deadlock": deadlocks}

    if env.has_state_labels():
        if isinstance(env.state_labeler, AbstractStateLabeler):
            if overapproximate:
                get_labels_of_state = env.state_labeler.get_labels_of_abstract_state_exist
            else:
                get_labels_of_state = env.state_labeler.get_labels_of_abstract_state_forall
        else:
            get_labels_of_state = env.state_labeler.get_labels_of_state
        # All labels exist in the model (also if no support state has them), so properties can refer to them
        for label in env.state_labeler.get_labels():
            if label not in labels_to_states.keys():
                labels_to_states[label] = []
        for s in sorted(_support_states(env, T_dict)):
            labels = get_labels_of_state(s)
            for label in labels:
                if label not in labels_to_states.keys():
                    labels_to_states[label] = []
                labels_to_states[label].append(s)
            
    return labels_to_states

def _support_states(env, T_dict) -> set[int]:
    """
    The reachable states that can occur in a run of the exported model from an initial state.
    """
    states = set(env.initial_states.distr_dict.keys()) | set(T_dict.keys())
    for T_s in T_dict.values():
        for T_sa in T_s.values():
            states.update(T_sa.keys())
    return states

def _build_state_labeling(nr_states, label_to_states) -> stormpy.storage.StateLabeling:
    """
    Constructs a stormpy StateLabeling object from label-to-state mapping.

    Parameters
    ----------
    nr_states : int
        Number of states in the explicit model
    label_to_states : dict
        Dictionary mapping labels to sets of states (any iterable of state indices, or a `stormpy.BitVector`).

    Returns
    -------
    state_labeling : stormpy.storage.StateLabeling
        Stormpy state labeling object.
    """
    state_labeling = stormpy.storage.StateLabeling(nr_states)
    for label in label_to_states.keys():
        state_labeling.add_label(label)
    for label, states in label_to_states.items():
        if not isinstance(states, stormpy.BitVector):
            states = stormpy.BitVector(nr_states, list(states))
        state_labeling.set_states(label, states)

    return state_labeling


def _build_choice_labeling(
    nr_choices, choice_to_label, choice_labels
) -> stormpy.storage.ChoiceLabeling:
    """
    Constructs a stormpy ChoiceLabeling object from choice-to-label mapping.

    Parameters
    ----------
    choice_to_label : dict
        Action label at each choice/transition.
    choice_labels : list(str)
        All action labels.

    Returns
    -------
    choice_labeling : stormpy.storage.ChoiceLabeling
        Stormpy choice labeling object.
    """
    choice_labeling = stormpy.storage.ChoiceLabeling(nr_choices)
    for label in choice_labels:
        choice_labeling.add_label(label)
    for choice, label in sorted(choice_to_label.items()):
        choice_labeling.add_label_to_choice(label, choice)
    return choice_labeling


def _build_state_valuations(state_values: dict):
    """
    Build stormpy.StateValuations from a dictionary mapping state indices to variable values.

    Parameters
    ----------
    state_values : dict
        Dictionary of the form
        { state_index:
            { "variable_name" : value
                for each "variable_name" in state_variables
            }
            for state_index in state_space
        }

    Returns
    -------
    stormpy.StateValuations
    """
    manager = stormpy.ExpressionManager()
    varnames = list(state_values[0].keys())

    stormpy_variables = [manager.create_integer_variable(name=var) for var in varnames]
    v_builder = stormpy.StateValuationsBuilder()
    for var in stormpy_variables:
        v_builder.add_variable(var)

    for state, state_val in state_values.items():
        s_vals = [state_val[var] if var in state_val.keys() else True
                  for var in varnames]
        v_builder.add_state(state=state, integer_values=s_vals)

    state_valuations = v_builder.build()
    return state_valuations


def load_stormpy_model(
    prismpath: str, property_strs: str | None = None, constants: str | None = None
) -> stormpy.storage.SparseMdp:
    """
    Load a stormpy model from a prism file.

    Parameters
    ---------
    prismpath : str
        The path to the prism model file
    property_strs : str
        Temporal logic formulas. Optional.
    constants : str
        If the prism model contains unresolved constants, need to provide them here in storm parseable format.

    Returns
    -------
    mdp : stormpy.storage.SparseMdp
        The loaded stormpy mdp.

    Note
    ----
    We currently only accept `.prism`, `.pm` or `.nm` files.
    Will extend to other formats, such as `.jani` in the future.
    """
    assert prismpath.endswith(".prism") or prismpath.endswith(".nm") or prismpath.endswith(".pm"), (
        "Wrong file format. Use .prism or .nm files."
    )

    program = stormpy.parse_prism_program(prismpath)
    if constants:
        program = program.define_constants(
            stormpy.parse_constants_string(program.expression_manager, constants)
        )
    # avoid unnamed actions
    program = program.label_unlabelled_commands({})

    if property_strs:
        formulas = stormpy.parse_properties(property_strs, program)
        options = stormpy.BuilderOptions([formula.raw_formula for formula in formulas])
    else:
        formulas = None
        options = stormpy.BuilderOptions()

    options.set_build_state_valuations()
    options.set_build_choice_labels()
    options.set_build_with_choice_origins()
    options.set_build_all_reward_models()
    options.set_build_all_labels()

    mdp = stormpy.build_sparse_model_with_options(program, options)
    return mdp


def _valuation_variables(state_valuations) -> list:
    """
    The variables of stormpy state valuations, in the order in which stormpy prints them (booleans first).

    The values of the variables are read with stormpy's `get_values_states`, but their order is only available from
    the printed valuation, e.g., "[!b\t& x=1]". The names are read from the valuation of state 0, which stormpy prints
    as "name=value" for integers and "!name" for false booleans, but as an empty token for true booleans. The empty
    tokens are filled with the booleans that are true in state 0, in the order of their index (the order in which
    they were declared).

    Parameters
    ----------
    state_valuations : stormpy.storage.Valuations
        The state valuations of a stormpy model, e.g., `mdp.state_valuations`.

    Returns
    -------
    variables : list[stormpy.Variable]
        The variables, in the order in which stormpy prints them.
    """
    variables = {variable.name: variable for variable in state_valuations.get_all_variables()}
    by_index = sorted(variables.values(), key=lambda variable: variable.index)
    if len(variables) == 0:
        return []

    tokens = state_valuations.get_string(0, True).strip("[]").split("&")
    names = [token.strip().lstrip("!").split("=")[0] for token in tokens]
    unnamed = [variable for variable in by_index if variable.name not in names]
    if (len(names) != len(variables) or names.count("") != len(unnamed)
            or any(name != "" and name not in variables for name in names)
            or not all(variable.has_boolean_type() for variable in unnamed)):
        # unexpected format: use the order of the indices, which is the order of stormpy's output for prism files
        return by_index
    unnamed = iter(unnamed)
    return [variables[name] if name != "" else next(unnamed) for name in names]

def _unwrap_scheduler(mdp: stormpy.storage.SparseMdp, scheduler: stormpy.storage.Scheduler,
                      label_to_action: dict | None = None) -> dict:
    """Converts a stormpy policy to a native Python dict mapping states to actions.

    Parameters
    ----------
    mdp : stormpy.storage.SparseMdp
        Stormpy MDP
    scheduler : stormpy.storage.Scheduler
        Stormpy scheduler
    label_to_action : dict | None
        Maps the action names of the choices of `mdp` to the actions of the original env, e.g.,
        `env.formatter.label_to_action`. The action names are the choice labels, plus "label_2", ... for choices with
        the same label as another choice of their state (see `_choice_action_names`).
        If None (default), it is derived from the choice labels as in `StormpyFormatter` (see
        `_action_names_to_indices`): labels that are all non-negative integers are the action indices, and other
        labels are action names whose index is their position among the sorted names.

    Returns
    -------
    dict[int, int]
    """
    # Every MDP built in VeriGym gets a choice labeling with the action of the original env: its name if the env has
    # action labels, else its index. Without choice labeling, we cannot reliably map actions back to the env.
    assert mdp.has_choice_labeling()

    # The action of every choice, as in `StormpyFormatter` (see `_choice_actions`)
    actions, _, _ = _choice_actions(mdp, label_to_action)
    row_groups = list(mdp.nondeterministic_choice_indices)

    unwrapped_policy = {}
    for s in range(mdp.nr_states):
        row = row_groups[s] + scheduler.get_choice(s).get_deterministic_choice()
        # Unlabelled choices (e.g., the self-loops of deadlocks) are mapped to action 0
        unwrapped_policy[s] = 0 if actions[row] is None else actions[row]

    return unwrapped_policy

def _choice_actions(model, label_to_action: dict | None = None) -> tuple[list, dict, int]:
    """
    Maps every choice (row) of a stormpy model to its action in VeriGym. Used by `StormpyFormatter`, by
    `_unwrap_scheduler` and by the importer of interval MDPs, so that they agree on the actions.

    - With choice labels, the actions are the names of `_choice_action_names`: the labels, plus "label_2", ... for
      choices that have the same label as another choice of their state. They are mapped to action indices with
      `label_to_action`, which defaults to `_action_names_to_indices` of all labels and names. Unlabelled choices
      (e.g., the self-loops that stormpy adds to deadlocks) have no action (None).
    - Without choice labels, the i-th choice of a state is action i, and `label_to_action` is empty.

    Parameters
    ----------
    model : stormpy.storage.SparseMdp | stormpy.storage.SparseIntervalMdp
        The model.
    label_to_action : dict | None
        Maps the action names to action indices, e.g., `env.formatter.label_to_action`. Derived from the model if None.

    Returns
    -------
    actions : list[int | None]
        The action index of every choice.
    label_to_action : dict[str, int]
        Maps the action names to action indices.
    nr_actions : int
        The number of actions (the largest action index plus one).
    """
    row_groups = list(model.nondeterministic_choice_indices)
    if not model.has_choice_labeling():
        actions = [row - row_groups[s] for s in range(model.nr_states) for row in range(row_groups[s], row_groups[s + 1])]
        return actions, {}, max(actions, default=-1) + 1

    names = _choice_action_names(model, row_groups)
    if label_to_action is None:
        all_names = set(model.choice_labeling.get_labels()) | {name for name in names if name is not None}
        label_to_action = _action_names_to_indices(all_names)
    actions = [None if name is None else label_to_action[name] for name in names]
    return actions, label_to_action, max(label_to_action.values(), default=-1) + 1

def _choice_action_names(model, row_groups: list) -> list:
    """
    The action name of every choice (row) of a stormpy model with choice labels, or None for unlabelled choices.

    The name of a choice is its label. If a state has several choices with the same label (e.g., two commands of a
    prism file with the same action that are enabled in the same state), they are different actions, so the first
    keeps the label and the k-th one (k >= 2, in the order of the choices) is named "label_k". Names that already are
    labels of the model are skipped (if "label_2" is a label, the second choice is named "label_3"), and the k-th
    choice with a label has the same name in every state.

    Parameters
    ----------
    model : stormpy.storage.SparseMdp | stormpy.storage.SparseIntervalMdp
        The model, with choice labels.
    row_groups : list[int]
        The first choice (row) of every state, and the number of choices at the end, i.e.,
        `model.nondeterministic_choice_indices` (passed in, since stormpy copies it on every access).

    Returns
    -------
    names : list[str | None]
    """
    labeling = model.choice_labeling
    labels = set(labeling.get_labels())
    # For each label that occurs several times in a state: the names of its 2nd, 3rd, ... choice, and the last suffix
    repeated_names = {}

    def repeated_name(label, k):
        names, suffix = repeated_names.get(label, ([], 1))
        while len(names) < k - 1:
            suffix += 1
            if f"{label}_{suffix}" not in labels:
                names.append(f"{label}_{suffix}")
        repeated_names[label] = (names, suffix)
        return names[k - 2]

    names = []
    for s in range(model.nr_states):
        occurrences = {}
        for row in range(row_groups[s], row_groups[s + 1]):
            choice_labels = labeling.get_labels_of_choice(row)
            if len(choice_labels) == 0:
                names.append(None)
                continue
            # choices of prism files have one label; take the smallest one if there are several, to be deterministic
            label = min(choice_labels)
            occurrences[label] = occurrences.get(label, 0) + 1
            names.append(label if occurrences[label] == 1 else repeated_name(label, occurrences[label]))
    return names

def _action_names_to_indices(labels) -> dict:
    """
    Maps choice labels to action indices. Used by `StormpyFormatter` and by `_unwrap_scheduler`, so that both agree
    on the indices.
    - Labels that are all non-negative integers (e.g., "0", ..., "n-1" of models exported from envs without action
      names) are action indices. Files only keep the labels of actions that are available somewhere, so the indices
      can have gaps.
    - Otherwise, the labels are action names (e.g., of a prism file), whose index is their position among the sorted
      names.
    """
    if all(label.isdigit() and str(int(label)) == label for label in labels):
        return {label: int(label) for label in labels}
    return {label: a for a, label in enumerate(sorted(labels))}