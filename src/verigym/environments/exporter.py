from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp, build_stormpy_imdp

import os
import stormpy

"""
Methods to export to different formal model formats.
"""


def export_to_stormpy_mdp(env: BaseExplicitEnv, overapproximate=True) -> stormpy.storage.SparseMdp:
    """Exports an `ExplicitEnv` to a `stormpy.storage.SparseMdp`.

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
    assert issubclass(type(env), BaseExplicitEnv) \
        or issubclass(type(env.unwrapped), BaseExplicitEnv)
    if not issubclass(type(env), BaseExplicitEnv):
        stormpy_mdp = build_stormpy_mdp(env.unwrapped, overapproximate)
    else:
        stormpy_mdp = build_stormpy_mdp(env, overapproximate)
    return stormpy_mdp

def export_to_stormpy_imdp(env: BaseExplicitEnv, overapproximate=True, use_reward_uncertainty=False) -> stormpy.storage.SparseIntervalMdp:
    """Exports an `ExplicitEnv` to a `stormpy.storage.SparseIntervalMdp`.

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
    assert issubclass(type(env), BaseExplicitEnv)
    stormpy_imdp = build_stormpy_imdp(env, use_reward_uncertainty, overapproximate)
    return stormpy_imdp


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
