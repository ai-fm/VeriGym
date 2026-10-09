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
    for s in range(env.nr_states):
        builder.new_row_group(choice_counter)
        actions = _available_actions(T_dict, s, env.nr_actions)
        for a in actions:
            for next_s, prob in T_dict[s][a].items():
                if not (type(prob) == float and prob == 0):
                    builder.add_next_value(choice_counter, next_s, to_value(prob))
            rewards = R_dict.get(s, {}).get(a, 0.0)
            for label, idx in reward_labels.items():
                reward = rewards[idx] if isinstance(rewards, list) else rewards
                reward_models[label].append(to_value(reward))
            custom_choice_labeling[choice_counter] = action_labels[a]
            choice_counter += 1
        # self-loop deadlocks with 0 reward # TODO this might be incorrect for min max?
        if len(actions) == 0:
            builder.add_next_value(choice_counter, s, to_value(1.0))
            for label in reward_labels.keys():
                reward_models[label].append(to_value(0.0))
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
        state_labels = _build_state_label_map(env, overapproximate)
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
    T = {s: {} for s in range(num_s)}
    R = []

    for s in range(num_s):
        if len(env.transition_function[s]) == 0: 
            R.append(0.0)
            continue

        action = int(policy.get_action(s))
        transitions = env.transition_function[s, action]

        R.append(env.reward_function[s, action])
        for next_state in transitions.keys():
            T[s][next_state] = transitions[next_state]     

    # Build the stormpy dtmc object
    builder = stormpy.SparseMatrixBuilder(rows=0, columns=num_s, entries=0,
                                          force_dimensions=True, has_custom_row_grouping=False,
                                          row_groups=0)

    for s in range(num_s):
        for (n_s, prob) in T[s].items():
            builder.add_next_value(s, n_s, prob)
        # self-loop terminal states
        if len(T[s].keys()) == 0:
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
        state_labels = _build_state_label_map(env, overapproximate)
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

def _build_state_label_map(env, overapproximate):
    T_dict = env.get_transition_function().T_dict

    # create state labeling of initial and deadlock states
    labels_to_states = {"init": [], "deadlock": []}
    for s in range(env.nr_states):
        if env.initial_states[s] > 0:
            labels_to_states["init"].append(s)
        if len(_available_actions(T_dict, s, env.nr_actions)) == 0:
            labels_to_states["deadlock"].append(s)
    
    if env.has_state_labels():
        if isinstance(env.state_labeler, AbstractStateLabeler):
            if overapproximate:
                get_labels_of_state = env.state_labeler.get_labels_of_abstract_state_exist
            else:
                get_labels_of_state = env.state_labeler.get_labels_of_abstract_state_forall
        else:
            get_labels_of_state = env.state_labeler.get_labels_of_state
        for s in range(env.nr_states):
            labels = get_labels_of_state(s)
            for label in labels:
                if label not in labels_to_states.keys():
                    labels_to_states[label] = []
                labels_to_states[label].append(s)
            
    return labels_to_states

def _build_state_labeling(nr_states, label_to_states) -> stormpy.storage.StateLabeling:
    """
    Constructs a stormpy StateLabeling object from label-to-state mapping.

    Parameters
    ----------
    nr_states : int
        Number of states in the explicit model
    label_to_states : dict
        Dictionary mapping labels to sets of states.

    Returns
    -------
    state_labeling : stormpy.storage.StateLabeling
        Stormpy state labeling object.
    """
    state_labeling = stormpy.storage.StateLabeling(nr_states)
    for label in label_to_states.keys():
        state_labeling.add_label(label)
    for label, states in label_to_states.items():
        state_labeling.set_states(label, stormpy.BitVector(nr_states, list(states)))

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
    We currently only accept `.prism` or `.nm` files.
    Will extend to other formats, such as `.jani` in the future.
    """
    assert prismpath.endswith(".prism") or prismpath.endswith(".nm"), (
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


def format_valuations(state_valuation: str) -> dict:
    """
    Utility function that converts the valuation string for a state of a stormpy MDP to a dict {var: val}

    Parameters
    ----------
    state_valuation : str
        The valuations of one state of a stormpy MDP as str, obtained from `state.valuations`.
        The format is "[var1=value & var2=value]" for integer variables.

    Returns
    -------
    vals : dict
        Valuations parsed in a dictionary of the format {"var1": value, "var2": value}
    """
    state_valuation = state_valuation.split()
    vals = {}
    for sval in state_valuation:
        if sval == "&":
            continue

        elif sval.find("=") == -1:
            sval = sval.replace("[", "").replace("]", "")
            if sval.startswith("!"):
                val = 0
                sval = sval.replace("!", "")
            else:
                val = 1
            vals[sval] = val
        else:
            sval = sval.replace("[", "").replace("]", "")
            var, val = sval.split("=")
            val = int(val)
            vals[var] = val
    return vals

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
        Maps the choice labels of `mdp` to the actions of the original env, e.g., `env.formatter.label_to_action`.
        If None (default), it is derived from the choice labels: labels "0", ..., "n-1" are the action indices, and
        other labels are action names whose index is their position among the sorted names (as in
        `StormpyFormatter`). Pass it explicitly if the action names of a prism file are "0", ..., "n-1" with n > 10,
        since their sorted (string) order differs from their indices.

    Returns
    -------
    dict[int, int]
    """
    # Every MDP built in VeriGym gets a choice labeling with the action of the original env: its name if the env has
    # action labels, else its index. Without choice labeling, we cannot reliably map actions back to the env.
    assert mdp.has_choice_labeling()

    if label_to_action is None:
        labels = mdp.choice_labeling.get_labels()
        if labels == {str(a) for a in range(len(labels))}:
            label_to_action = {label: int(label) for label in labels}
        else:
            label_to_action = _action_names_to_indices(labels)

    unwrapped_policy = {}
    
    for s in mdp.states:
        # Get scheduler action
        choice = scheduler.get_choice(s.id)
        idx = choice.get_deterministic_choice()
        # This gets the label for the unique action in the current state corresponding to idx.
        # We cannot index the stormpy action object, therefore the iteration is needed.
        # The list always contains exactly one element.
        state_action_label = [a.labels for a in s.actions if a.id == idx][0]

        # Convert to action index of original env
        if len(state_action_label) == 0:
            action_idx = 0
        else:
            action_idx = label_to_action[state_action_label.pop()]

        unwrapped_policy[s.id] = action_idx

    return unwrapped_policy

def _action_names_to_indices(labels) -> dict:
    """
    Maps action names (e.g., the choice labels of a prism file) to action indices: their position among the sorted
    names. Used by `StormpyFormatter` and by `_unwrap_scheduler`, so that both agree on the indices.
    """
    return {label: a for a, label in enumerate(sorted(labels))}