from collections import defaultdict

from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv
from verigym.environments.interval_explicitenv import IntervalExplicitEnv
from verigym.environments.labeling import StateLabel
from verigym.environments.reward_func import IntervalRewardFunction
from verigym.environments.state_distributions import StateDistribution
from verigym.environments.transition_func import IntervalTransitionFunction
from verigym.frameworks.stormpy.stormpy_utils import _choice_actions

import stormpy

"""
Methods to import explicit environments from different formal model formats.
"""

def _env_from_stormpy_model(model, flat: bool) -> FrameworkExplicitEnv | IntervalExplicitEnv:
    """Converts a stormpy model loaded from a file to an env."""
    if isinstance(model, stormpy.storage.SparseIntervalMdp):
        if not flat:
            raise ValueError("Requested featured state representation, but interval MDPs only have flat states.")
        return _interval_env_from_stormpy(model)
    if isinstance(model, stormpy.storage.SparseMdp):
        return FrameworkExplicitEnv.from_stormpy(model, flat=flat)
    raise NotImplementedError(
        f"Loading a {type(model).__name__} is not supported, only MDPs and interval MDPs. "
        "Use stormpy directly to load other types of models."
    )

def _interval_env_from_stormpy(imdp: stormpy.storage.SparseIntervalMdp) -> IntervalExplicitEnv:
    """
    Builds an `IntervalExplicitEnv` from a stormpy interval MDP, with the transition and reward intervals of `imdp`.

    The actions are the choice labels, mapped to action indices as in `StormpyFormatter` (see `_choice_actions`;
    several choices of a state with the same label are the actions "label", "label_2", ...). Unlabelled choices of a
    model with choice labels (such as the self-loops VeriGym exports for deadlocks) are skipped. If the model has no
    choice labels, the i-th choice of a state is action i.
    The initial distribution is uniform over the initial states of `imdp`, and its state labels (apart from "init"
    and "deadlock", which are derived from the env) become `StateLabel`s of the env.
    The names of the actions and the reward model, and the state valuations, are not kept.
    """
    if len(imdp.reward_models) > 1:
        raise NotImplementedError("Loading interval MDPs with more than one reward model is not supported.")
    # stormpy returns a copy of the reward vectors on every access, so get them once
    state_rewards, state_action_rewards = None, None
    for reward_model in imdp.reward_models.values():
        if reward_model.has_state_rewards:
            state_rewards = list(reward_model.state_rewards)
        if reward_model.has_state_action_rewards:
            state_action_rewards = list(reward_model.state_action_rewards)

    matrix = imdp.transition_matrix
    actions, _, nr_actions = _choice_actions(imdp)

    T_dict, R_dict = defaultdict(dict), defaultdict(dict)
    for s in range(imdp.nr_states):
        for row in matrix.get_rows_for_group(s):
            a = actions[row]
            if a is None:
                continue

            T_dict[s][a] = defaultdict(lambda: (0.0, 0.0))
            for entry in matrix.get_row(row):
                if entry.value().upper() > 0:
                    T_dict[s][a][entry.column] = (entry.value().lower(), entry.value().upper())

            reward_lb, reward_ub = 0.0, 0.0
            if state_action_rewards is not None:
                reward_lb += state_action_rewards[row].lower()
                reward_ub += state_action_rewards[row].upper()
            if state_rewards is not None:
                reward_lb += state_rewards[s].lower()
                reward_ub += state_rewards[s].upper()
            R_dict[s][a] = (reward_lb, reward_ub)

    initial_states = StateDistribution(
        imdp.nr_states, {int(s): 1.0 / len(imdp.initial_states) for s in imdp.initial_states}
    )

    env = IntervalExplicitEnv(
        nr_states=imdp.nr_states,
        nr_actions=nr_actions,
        initial_state_distr=initial_states,
        interval_transition_function=IntervalTransitionFunction(imdp.nr_states, nr_actions, T_dict),
        interval_reward_function=IntervalRewardFunction.from_dict(R_dict, imdp.nr_states, nr_actions),
        nr_rewards=len(imdp.reward_models),
    )
    for label in sorted(imdp.labeling.get_labels() - {"init", "deadlock"}):
        states = frozenset(imdp.labeling.get_states(label))
        env.add_state_label(StateLabel(label, lambda s, states=states: s in states))
    return env

def load_from_umb(in_file: str, flat: bool = True) -> FrameworkExplicitEnv | IntervalExplicitEnv:
    """
    Loads an explicit env from a Unified Markov Binary (UMB) file, using stormpy.

    Parameters
    ----------
    in_file : str
        Path to the .umb file.
    flat : bool, default=True
        Whether the env uses flat (integer) states, or factored states from the state valuations in the file.
        Interval MDPs only have flat states.

    Returns
    -------
    env : FrameworkExplicitEnv | IntervalExplicitEnv
        For MDPs, a `FrameworkExplicitEnv` with the choice labels of the file as action labels.
        For interval MDPs, an `IntervalExplicitEnv` with the intervals of the file (see `_interval_env_from_stormpy`).

    Notes
    -----
    UMB only stores the set of initial states, so the env starts uniformly in one of them.
    """
    options = stormpy.storage.UmbImportOptions()
    options.build_choice_labeling = True
    options.build_state_valuations = True
    model = stormpy.build_from_umb(in_file, options)
    return _env_from_stormpy_model(model, flat)

def load_from_drn(in_file: str, flat: bool = True) -> FrameworkExplicitEnv:
    """
    Loads an explicit env from a DRN file, using stormpy.

    Parameters
    ----------
    in_file : str
        Path to the .drn file.
    flat : bool, default=True
        Whether the env uses flat (integer) states, or factored states from the state valuations.
        DRN files do not store state valuations, so only flat states are available.

    Returns
    -------
    env : FrameworkExplicitEnv
        The env, with the choice labels of the file as action labels.

    Notes
    -----
    DRN only stores the set of initial states, so the env starts uniformly in one of them.
    Interval MDPs are not supported, since stormpy cannot read the interval values of DRN files; use UMB instead.
    """
    options = stormpy.DirectEncodingParserOptions()
    options.build_choice_labels = True
    model = stormpy.build_model_from_drn(in_file, options)
    return _env_from_stormpy_model(model, flat)
