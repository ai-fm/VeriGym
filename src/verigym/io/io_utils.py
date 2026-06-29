
import numpy as np
from verigym.environments.base_explicitenv import BaseExplicitEnv

import umbi
import umbi.ats

def read_umb(filename : str) -> umbi.ats.SimpleAts:
    return umbi.ats.read(filename)

def base_explicit_env_to_umbi(env : BaseExplicitEnv) -> umbi.ats.SimpleAts: # TODO: finish
    ats = umbi.ats.SimpleAts()

    # basic parameters
    ats.time = umbi.ats.TimeType.DISCRETE
    ats.num_players = 1 # assumping a (PO)MDP
    ats.num_states = env.nr_states
    ats.initial_states = np.nonzero(env.initial_states)[0].tolist() # TODO: how to use distribution? -> probably must create additional dummy initial state

    # build the ATS structure from delta and actions_at_state
    # ats.num_choice_actions = env.nr_actions
    # ats.choice_action_to_name = list(range(env.nr_actions))

    tra = env.get_transition_function()

    for s in ats.states:
        for action in range(env.nr_actions):
            if action in tra[s]:
                choice = ats.new_state_choice(state=int(s)) #, targets=targets, target_prob=probs)
                # ats.choice_to_choice_action[choice] = action

                # get all branches for this (s, action) pair
                branches = sorted(tra[s][action].items(), key=lambda x : x[0]) # sort by target state

                for target, prob in branches:
                    ats.new_choice_branch(choice=choice, target=int(target), prob=float(prob))

    ats.validate()
    return ats


from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp

import os
import stormpy

"""
Methods to input/export to/from different formal model formats.
"""

def _load_from_umb_with_stormpy(filename : str) -> BaseExplicitEnv:
    stormpy_mdp = stormpy.build_from_umb(filename)
    env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))
    return env

def load_from_umb(filename : str, umbi=False): # TODO IMPLEMENT FOR UMBI AND SET DEFAULT TO TRUE
    if umbi:
        ats = _load_umb_to_umbi_ats(filename)
        raise NotImplementedError("TODO!")
        env = FrameworkExplicitEnv(ats, UmbiFormatter(ats))
    else:
        return _load_from_umb_with_stormpy(filename)

def _load_umb_to_umbi_ats(filename) -> umbi.ats.SimpleAts:
    return umbi.ats.read(filename)

def _export_umbi_ats_to_umb(ats : umbi.ats.SimpleAts, filename : str) -> None:
    return umbi.ats.write(ats, filename)

def build_stormpy_mdp_from_explicit_env(env: BaseExplicitEnv) -> stormpy.storage.SparseMdp:
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

    stormpy_mdp = build_stormpy_mdp_from_explicit_env(env)
    stormpy.export_to_drn(model=stormpy_mdp, file=out_file)

def export_to_prism(env: BaseExplicitEnv) -> str:
    """
    Exports an explicit env to PRISM-readable files
    .tra for transitions
    .srew/.trew for rewards
    .lab for labels
    """
    ...  # TODO @Jule/Maris?

def export_to_umb(env: BaseExplicitEnv, filename : str) -> None:
    ats = base_explicit_env_to_umbi(env)
    umbi.ats.write(ats, filename)


import os
import stormpy

from verigym.frameworks.stormpy.stormpy_utils import (
    load_stormpy_model,
    build_stormpy_mdp,
)

# from verigym.frameworks.stormpy.stormpyenv import StormpyEnv
from verigym.frameworks.stormpy.formatter import StormpyFormatter
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv

PRISM_TEST = os.path.join(os.getcwd(), "tests/test_2d.prism")

if __name__ in "__main__":
    mdp = load_stormpy_model(PRISM_TEST)
    env = FrameworkExplicitEnv(mdp, StormpyFormatter(mdp))
    ats = base_explicit_env_to_umbi(env)
    umbi.ats.write(ats, "test.umb")
    print(ats)
    ats2 = umbi.ats.read("test.umb")
    mdp2 = stormpy.build_from_umb("test.umb")
    assert mdp == mdp2
    print(ats2)
    assert ats == ats2
