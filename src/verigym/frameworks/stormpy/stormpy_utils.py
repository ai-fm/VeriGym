import stormpy
from collections import defaultdict

from verigym.environments.explicitenv import BaseExplicitEnv
from verigym.environments.labeling import AbstractStateLabeler
from verigym.environments.transition_func import IntervalTransitionFunction
from verigym.environments.reward_func import IntervalRewardFunction
from verigym.environments.interval_explicitenv import IntervalEpxlicitEnv


def build_stormpy_mdp(env: BaseExplicitEnv, overapproximate=True) -> stormpy.storage.SparseMdp:
    """
    Builds a `stormpy.storage.SparseMdp` from a `BaseExplicitEnv`.

    Parameters
    ----------
    env : ExplicitEnv

    Returns
    -------
    mdp : stormpy.storage.SparseMdp
    """
    info = _get_info_from_formatter(env)

    # Build the stormpy transition matrix
    env_transitions = env.get_transition_function()
    builder = stormpy.SparseMatrixBuilder(
        rows=0,
        columns=env.nr_states,
        entries=0,
        force_dimensions=True,
        has_custom_row_grouping=True,
    )
    choice_counter = 0
    for s in range(env.nr_states):
        builder.new_row_group(choice_counter)
        for a in range(env.nr_actions):
            if a in env_transitions[s].keys():
                for next_s, prob in env_transitions[s][a].items():
                    builder.add_next_value(choice_counter, next_s, prob)
                if len(env_transitions[s][a].items()) > 0:
                    choice_counter += 1
        # self-loop terminal states
        if len(env_transitions[s].keys()) == 0:
            builder.add_next_value(choice_counter, s, 1.0)
            choice_counter += 1
    transition_matrix = builder.build()

    # Build the reward model(s):
    env_rewards = env.get_reward_function().R_dict
    if "reward_labels" in info.keys():
        reward_labels = info["reward_labels"]
    else:
        reward_labels = {f"reward{i}": i for i in range(env.nr_rewards)}
    reward_models = {label: [] for label in reward_labels.keys()}

    for s in range(env.nr_states):
        if s in env_rewards.keys():
            for a in range(env.nr_actions):
                if a in env_rewards[s].keys():
                    rewards = env_rewards[s][a]
                    for label, idx in reward_labels.items():
                        if isinstance(rewards, list):
                            reward_models[label].append(rewards[idx])
                        else:
                            reward_models[label].append(rewards)
        else:
            for label, idx in reward_labels.items(): # TODO this might be incorrect for min max?
                reward_models[label].append(0.0) # TODO

    # 0 reward for terminal self-loops # TODO this might be incorrect for min
    if "choice_labels" in info.keys():
        choice_labeling = info["choice_labels"]
        for choice in range(choice_counter):
            if len(choice_labeling.get_labels_of_choice(choice)) == 0:
                for label, idx in reward_labels.items():
                    reward_models[label].insert(choice, 0.0) # TODO

    stormpy_reward_models = {}
    for label, reward_vector in reward_models.items():
        stormpy_reward_models[label] = stormpy.SparseRewardModel(
            optional_state_action_reward_vector=reward_vector
        )

    # Assemble the components
    components = stormpy.SparseModelComponents(
        transition_matrix=transition_matrix,
        rate_transitions=False,
    )

    if len(stormpy_reward_models) > 0:
        components.reward_models = stormpy_reward_models

    if "state_labels" in info.keys():
        components.state_labeling = info["state_labels"]
    else:
        state_labels = _build_state_label_map(env, overapproximate)
        components.state_labeling = _build_state_labeling(
            env.nr_states, state_labels
        )

    if "choice_labels" in info.keys():
        components.choice_labeling = info["choice_labels"]

    if "valuations" in info.keys():
        components.state_valuations = info["valuations"]

    # Build the MDP from the components
    mdp = stormpy.storage.SparseMdp(components)

    return mdp

def build_stormpy_imdp(env: BaseExplicitEnv,
                       use_reward_uncertainty=False,
                       overapproximate=True):
    info = _get_info_from_formatter(env)

    # Build the stormpy transition matrix
    if isinstance(env, IntervalEpxlicitEnv):
        env_transitions = env.get_interval_transition_function()
    else:
        env_transitions = env.get_transition_function()
    builder = stormpy.IntervalSparseMatrixBuilder(
       rows=0, columns=env.nr_states, entries=0, force_dimensions=True, has_custom_row_grouping=True,
    )
    choice_counter = 0

    if isinstance(env_transitions, IntervalTransitionFunction):
        for s in range(env.nr_states):
            builder.new_row_group(choice_counter)
            for a in range(env.nr_actions):
                if a in env_transitions[s].keys():
                    for next_s, probs in env_transitions[s][a].items():
                        builder.add_next_value(choice_counter, next_s, stormpy.pycarl.Interval(
                            probs[0], probs[1]
                        ))
                    if len(env_transitions[s][a].items()) > 0:
                        choice_counter += 1
            # self-loop terminal states
            if len(env_transitions[s].keys()) == 0:
                builder.add_next_value(choice_counter, s,
                                       stormpy.pycarl.Interval(1.0, 1.0))
                choice_counter += 1
    else: # standard transition functions, set lb=ub=prob
        for s in range(env.nr_states):
            builder.new_row_group(choice_counter)
            for a in range(env.nr_actions):
                if a in env_transitions[s].keys():
                    for next_s, prob in env_transitions[s][a].items():
                        builder.add_next_value(choice_counter, next_s,
                                               stormpy.pycarl.Interval(prob, prob))
                    if len(env_transitions[s][a].items()) > 0:
                        choice_counter += 1
            # self-loop terminal states
            if len(env_transitions[s].keys()) == 0:
                builder.add_next_value(choice_counter, s, 
                                    stormpy.pycarl.Interval(1.0, 1.0))
                choice_counter += 1
    transition_matrix = builder.build()

    # Build the reward model(s):
    env_rewards = env.get_reward_function().R_dict
    if "reward_labels" in info.keys():
        reward_labels = info["reward_labels"]
    else:
        reward_labels = {f"reward{i}": i for i in range(env.nr_rewards)}
    reward_models = {label: [] for label in reward_labels.keys()}

    if use_reward_uncertainty:
        # Use the env's interval reward function
        assert isinstance(env, IntervalEpxlicitEnv), "Cannot derive uncertain rewards from non-uncertain environment."

        env_rewards = env.interval_rewards
    else:
        # Use the env's point estimate reward function
        env_rewards_point = env.reward_function
        # convert to interval transition function with lb==ub
        interval_R_dict = defaultdict(lambda: defaultdict(float))
        for s in range(env.nr_states):
            for a in range(env.nr_actions):
                r = env_rewards_point[s][a]
                interval_R_dict[s][a] = (r, r)
        env_rewards = IntervalRewardFunction(env.nr_states, env.nr_actions,
                                             interval_R_dict)
    # actually build the reward(s)
    
    for s in range(env.nr_states):
        if s in env_rewards.R_dict.keys():
            for a in range(env.nr_actions):
                if a in env_rewards[s].keys():
                    rewards = env_rewards[s][a]
                    for label, idx in reward_labels.items():
                        if isinstance(rewards, list):
                            reward_models[label].append(stormpy.pycarl.Interval(rewards[idx][0], rewards[idx][1]))
                        else:
                            reward_models[label].append(stormpy.pycarl.Interval(rewards[0], rewards[1]))
        else:
            for label, idx in reward_labels.items():
                reward_models[label].append(stormpy.pycarl.Interval((0.0, 0.0))) # TODO
    
    # Rewards for terminal self-loops # TODO differentiate min and max objective
    if "choice_labels" in info.keys():
        choice_labeling = info["choice_labels"]
        for choice in range(choice_counter):
            if len(choice_labeling.get_labels_of_choice(choice)) == 0:
                for label, idx in reward_labels.items():
                    reward_models[label].insert(choice, stormpy.pycarl.Interval(0.0, 0.0)) # TODO

    stormpy_reward_models = {}
    for label, reward_vector in reward_models.items():
        stormpy_reward_models[label] = stormpy.SparseIntervalRewardModel(
            optional_state_action_reward_vector=reward_vector
        )

    # Assemble the components
    components = stormpy.SparseIntervalModelComponents(
        transition_matrix=transition_matrix,
        rate_transitions=False,
    )

    if len(stormpy_reward_models) > 0:
        components.reward_models = stormpy_reward_models

    if "state_labels" in info.keys():
        components.state_labeling = info["state_labels"]
    else:
        state_labels = _build_state_label_map(env, overapproximate)
        components.state_labeling = _build_state_labeling(
            env.nr_states, state_labels
        )
    
    if "choice_labels" in info.keys():
        components.choice_labeling = info["choice_labels"]
    
    if "valuations" in info.keys():
        components.state_valuations = info["valuations"]

    # Build the IMDP from the components
    imdp = stormpy.storage.SparseIntervalMdp(components)

    return imdp


def _get_info_from_formatter(env):
    info = {}
    if not hasattr(env, "formatter"):
        return info

    choice_counter = 0
    env_transitions = env.get_transition_function()

    if env.formatter.has_action_labels:
        choice_to_label = {}
        for s in range(env.nr_states):
            for a in range(env.nr_actions):
                if a in env_transitions[s].keys():
                    choice_to_label[choice_counter] = env.formatter.action_to_label[a]
                    if len(env_transitions[s][a].items()) > 0:
                        choice_counter += 1
            if len(env_transitions[s].keys()) == 0:
                choice_counter += 1

        choice_labeling = _build_choice_labeling(
            choice_counter, choice_to_label, list(env.formatter.label_to_action.keys())
        )
        info["choice_labels"] = choice_labeling

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
    env_transitions = env.get_transition_function()

    # create state labeling of initial and deadlock states
    labels_to_states = {"init": [], "deadlock": []}
    for s in range(env.nr_states):
        if env.initial_states[s] > 0:
            labels_to_states["init"].append(s)
        if len(env_transitions[s].keys()) == 0:
            labels_to_states["deadlock"].append(s)
    
    if env.has_state_labels():
        if isinstance(env.state_labeler, AbstractStateLabeler):
            if overapproximate:
                get_labels_of_state = env.state_labeler.get_labels_of_abstract_state_overapproximate
            else:
                get_labels_of_state = env.state_labeler.get_labels_of_abstract_state_underapproximate
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
        s_vals = [state_val[var] for var in varnames]
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
