
import numpy as np
from verigym.environments.base_explicitenv import BaseExplicitEnv

import umbi
import umbi.ats

# TODO: delete example

"""
def random_walk_ats(num_states: int) -> umbi.ats.ExplicitAts:
    # create ATS
    ats = umbi.ats.ExplicitAts()

    # basic parameters
    ats.time = umbi.ats.TimeType.DISCRETE
    ats.num_players = 0
    ats.num_states = num_states
    ats.num_initial_states = 1
    ats.set_initial_states([num_states // 2])  # start in the middle state

    # two actions: left (0) and right (1), each with 0.9 success prob
    ats.num_choice_actions = 2
    ats.choice_action_to_name = ["left", "right"]
    ats.num_choices = 2 * ats.num_states  # each state has 2 choices (left and right)
    ats.num_branches = 2 * ats.num_choices  # each choice has 2 branches (succeed or fail)

    # build structure
    ats.state_to_choice = []
    ats.choice_to_choice_action = []
    ats.choice_to_branch = []
    ats.branch_to_target = []
    ats.branch_probabilities = []

    for state in range(ats.num_states):
        ats.state_to_choice.append(len(ats.choice_to_choice_action))

        # left action
        ats.choice_to_choice_action.append(0)
        ats.choice_to_branch.append(len(ats.branch_to_target))
        left = max(0, state - 1)
        ats.branch_to_target.extend([left, state])
        ats.branch_probabilities.extend([Fraction(9, 10), Fraction(1, 10)])

        # right action
        ats.choice_to_choice_action.append(1)
        ats.choice_to_branch.append(len(ats.branch_to_target))
        right = min(ats.num_states - 1, state + 1)
        ats.branch_to_target.extend([right, state])
        ats.branch_probabilities.extend([0.9, 0.1])

    ats.state_to_choice.append(len(ats.choice_to_choice_action))
    ats.choice_to_branch.append(len(ats.branch_to_target))

    ats.state_is_markovian = [True] * ats.num_states
    ats.state_exit_rate = [1] * ats.num_states

    # example: APs
    ats.add_ap_annotation(
        umbi.ats.AtomicPropositionAnnotation(
            name="is_terminal",
            alias="Terminal State",
            description="Indicates whether the state is terminal.",
        )
    )
    ats.get_ap_annotation("is_terminal").set_state_values(
        [s == 0 or s == ats.num_states - 1 for s in range(ats.num_states)]
    )

    # example: rewards
    ats.add_reward_annotation(
        umbi.ats.RewardAnnotation(
            name="steps",
            alias="step cost",
            description="Cost incurred at each step.",
        )
    )
    ats.get_reward_annotation("steps").set_state_values([-1] * ats.num_states)

    # wall hit penalty
    choice_penalty = [0] * ats.num_choices
    for choice in ats.state_choice_range(0):
        action = ats.choice_to_choice_action[choice]
        if ats.choice_action_to_name[action] == "left":  # left action in first state
            choice_penalty[choice] = -10
    for choice in ats.state_choice_range(ats.num_states - 1):
        action = ats.choice_to_choice_action[choice]
        if ats.choice_action_to_name[action] == "right":  # right action in last state
            choice_penalty[choice] = -10
    ats.add_reward_annotation(
        umbi.ats.RewardAnnotation(
            name="wall_hit_penalty",
            alias="wall hit penalty",
            description="Penalty incurred when hitting a wall.",
        )
    )
    ats.get_reward_annotation("wall_hit_penalty").set_choice_values(choice_penalty)

    # observations: 3 observations, based on state mod 3
    ats.observation_annotation = umbi.ats.ObservationAnnotation(num_observations=3)
    ats.observation_annotation.set_state_values([s % 3 for s in range(ats.num_states)])

    return ats
"""

def base_explicit_env_to_umb(env : BaseExplicitEnv) -> umbi.ats.ExplicitAts: # TODO: finish
    ats = umbi.ats.ExplicitAts()

    # basic parameters
    ats.time = umbi.ats.TimeType.DISCRETE
    ats.num_players = 1 # assumping a (PO)MDP
    ats.num_states = env.nr_states
    ats.num_initial_states = np.count_nonzero(env.initial_states)
    ats.set_initial_states(np.nonzero(env.initial_states)[0].tolist()) # TODO: how to use distribution?
    # ats.state_is_initial = [s.is_initial() for s in model.states.values()]

    # actions_to_ids = {a: state_id for state_id, a in enumerate(model.actions)}
    # ats.num_actions = len(env.nr_actions)  # TODO change once stormvogel is updated
    # ats.action_strings = [
        # none_to_empty_string(a.label) for a in model.actions
    # ]

    ats.num_choice_actions = env.nr_actions
    # ats.choice_action_to_name = ["left", "right"] # TODO: don't know
    ats.num_choices = ats.num_choice_actions * ats.num_states  
    ats.num_branches = 2 * ats.num_choices

    # build structure
    ats.state_to_choice = []
    ats.choice_to_choice_action = []
    ats.choice_to_branch = []
    ats.branch_to_target = []
    ats.branch_probabilities = []

    # env_rewards = env.get_reward_function().R_dict
    # if "reward_labels" in info.keys():
    #     reward_labels = info["reward_labels"]
    # else:
    #     reward_labels = {f"reward{i}": i for i in range(env.nr_rewards)}
    # reward_models = {label: [] for label in reward_labels.keys()}

    # for s in range(env.nr_states):
    #     if s in env_rewards.keys():
    #         for a in range(env.nr_actions):
    #             if a in env_rewards[s].keys():
    #                 rewards = env_rewards[s][a]
    #                 for label, idx in reward_labels.items():
    #                     if isinstance(rewards, list):
    #                         reward_models[label].append(rewards[idx])
    #                     else:
    #                         reward_models[label].append(rewards)
    #     else:
    #         for label, idx in reward_labels.items():
    #             reward_models[label].append(0.0)
    choice_counter = 0
    env_transitions = env.get_transition_function()
    for s in range(env.nr_states):
        if ats.num_players > 0:
            assert ats.state_to_choice is not None, "If players exist, states must have choices."
            ats.state_to_choice.append(len(ats.choice_to_action))
        for a in range(env.nr_actions):
            if a in env_transitions[s].keys():
                if ats.num_players > 0:
                    ats.choice_to_action.append(actions_to_ids[action])
                ats.choice_to_branch.append(len(ats.branch_to_target))
                for next_s, prob in env_transitions[s][a].items():
                    ats.branch_to_target.append(next_s)
                    ats.branch_probabilities.append(prob)
                if len(env_transitions[s][a].items()) > 0:
                    choice_counter += 1

    ats.validate()


from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp

import os
import stormpy

"""
Methods to export to different formal model formats.
"""


def export_to_stormpy_mdp(env: BaseExplicitEnv) -> stormpy.storage.SparseMdp:
    """Exports an `ExplicitEnv` to a `stormpy.storage.SparseMdp`.

    Parameters
    ----------
    env : ExplicitEnv
        The explicit environment.

    Returns
    -------
    stormpy_mdp : stormpy.storage.SparseMdp
        The mdp.
    """
    assert issubclass(type(env), BaseExplicitEnv)
    stormpy_mdp = build_stormpy_mdp(env)
    return stormpy_mdp


def export_to_julia(env: BaseExplicitEnv): ...  # TODO @Merlijn


def export_to_drn(env: BaseExplicitEnv, out_file: str | None = None) -> None:
    """
    Exports an explicit env to an explicit model and writes directly to a .drn file.

    Parameters
    ----------
    env : BaseExplicitEnv
        The explicit environment.
    out_file : str
        Path to the output file. By default, writes to cwd/mdp.drn.
    """
    if not out_file:
        out_file = os.path.join(os.getcwd(), "mdp.drn")

    stormpy_mdp = build_stormpy_mdp(env)
    stormpy.export_to_drn(model=stormpy_mdp, file=out_file)

def export_to_prism(env: BaseExplicitEnv) -> str:
    """
    Exports an explicit env to PRISM-readable files
    .tra for transitions
    .srew/.trew for rewards
    .lab for labels
    """
    ...  # TODO @Jule/Maris?

def export_to_umb(env: BaseExplicitEnv) -> None:
    base_explicit_env_to_umb(env)
