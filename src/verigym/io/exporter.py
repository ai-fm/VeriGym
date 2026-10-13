from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.environments.interval_explicitenv import IntervalExplicitEnv
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp, build_stormpy_imdp

import os
import warnings

import numpy as np
import stormpy

"""
Methods to export to different formal model formats.
"""

def unwrap_env(env: BaseExplicitEnv) -> BaseExplicitEnv:
    assert issubclass(type(env), BaseExplicitEnv) or issubclass(type(env.unwrapped), BaseExplicitEnv), "Input must be an instance of a BaseExplicitEnv."
    if not issubclass(type(env), BaseExplicitEnv):
        return env.unwrapped
    return env

def _warn_if_initial_distribution_lost(env: BaseExplicitEnv, file_format: str) -> None:
    """
    UMB and DRN files only store the set of initial states, not a distribution over them, and loading such a file
    assumes a uniform distribution over the initial states. Warns if this changes the initial distribution of `env`.
    """
    support = np.array([prob for prob in env.initial_states.distr_dict.values() if prob > 0])
    if len(support) > 0 and not np.allclose(support, support[0]):
        warnings.warn(
            f"The initial state distribution is not uniform over the initial states, but {file_format} files only "
            f"store the set of initial states: loading the file assumes a uniform distribution over them.",
            stacklevel=3,
        )

def export_to_stormpy_mdp(env: BaseExplicitEnv, overapproximate=True) -> stormpy.storage.SparseMdp:
    """Exports an `ExplicitEnv` to a `stormpy.storage.SparseMdp`.
    See the notes of `build_stormpy_mdp` on choices, choice labels and deadlocks.

    Parameters
    ----------
    env : ExplicitEnv
        The explicit environment.
    overapproximate : bool, default=True
        Whether to over- or underapproximate state labels if the ExplicitEnv is abstracted.

    Returns
    -------
    stormpy_mdp : stormpy.storage.SparseMdp
        The mdp.
    """
    return build_stormpy_mdp(unwrap_env(env), overapproximate=overapproximate)

def export_to_stormpy_imdp(env: BaseExplicitEnv, overapproximate=True, use_reward_uncertainty=False) -> stormpy.storage.SparseIntervalMdp:
    """Exports an `ExplicitEnv` to a `stormpy.storage.SparseIntervalMdp`.
    Apart from the intervals, it is the same as `export_to_stormpy_mdp`; see the notes of `build_stormpy_mdp` on
    choices, choice labels and deadlocks.

    Parameters
    ----------
    env : ExplicitEnv
        The explicit environment.
    overapproximate : bool, default=True
        Whether to over- or underapproximate state labels if the ExplicitEnv is abstracted
    use_reward_uncertainty : bool, default = False
        If True and `env` is a `IntervalExplicitEnv`, it builds the IMDP using the `interval_reward_function`
        If False, builds an IMDP with a standard reward function using `reward_function`.

    Returns
    -------
    stormpy_mdp : stormpy.storage.SparseIntervalMdp
        The mdp.
    """
    return build_stormpy_imdp(unwrap_env(env), overapproximate=overapproximate, use_reward_uncertainty=use_reward_uncertainty)

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

    _warn_if_initial_distribution_lost(unwrap_env(env), "DRN")
    stormpy_mdp = export_to_stormpy_mdp(env)
    stormpy.export_to_drn(model=stormpy_mdp, file=out_file)

def export_to_umb(env: BaseExplicitEnv, out_file: str | None = None, intervals: bool | None = None,
                  overapproximate=True) -> None:
    """
    Exports an explicit env to the Unified Markov Binary (UMB) format and writes directly to a .umb file.

    Parameters
    ----------
    env : BaseExplicitEnv
        The explicit environment.
    out_file : str
        Path to the output file. By default, writes to cwd/mdp.umb.
    intervals : bool | None, default=None
        Whether to export an interval MDP (IMDP) instead of an MDP. By default, exports an IMDP if and only if `env`
        is an `IntervalExplicitEnv`. The IMDP of an `IntervalExplicitEnv` has its transition and reward intervals,
        the IMDP of any other env has point intervals [p, p].
    overapproximate : bool, default=True
        Whether to over- or underapproximate state labels if the ExplicitEnv is abstracted.

    Notes
    -----
    - The model is built with `export_to_stormpy_mdp` or `export_to_stormpy_imdp`, see the notes of
      `build_stormpy_mdp` on choices, choice labels and deadlocks.
    - UMB only stores the actions that are available in some state, and the set of initial states. Loading the file
      (e.g., with `load_from_umb`) assumes a uniform distribution over the initial states, so a warning is raised if
      the initial distribution of `env` is not uniform.
    """
    if not out_file:
        out_file = os.path.join(os.getcwd(), "mdp.umb")

    unwrapped_env = unwrap_env(env)
    _warn_if_initial_distribution_lost(unwrapped_env, "UMB")
    is_interval_env = isinstance(unwrapped_env, IntervalExplicitEnv)
    if intervals is None:
        intervals = is_interval_env

    if intervals:
        stormpy_model = export_to_stormpy_imdp(unwrapped_env, overapproximate=overapproximate,
                                               use_reward_uncertainty=is_interval_env)
    else:
        stormpy_model = export_to_stormpy_mdp(unwrapped_env, overapproximate=overapproximate)
    stormpy.export_to_umb(stormpy_model, out_file)


def export_to_prism(env: BaseExplicitEnv) -> str:
    """
    Exports an explicit env to PRISM-readable files
    .tra for transitions
    .srew/.trew for rewards
    .lab for labels
    """
    raise NotImplementedError() # TODO

