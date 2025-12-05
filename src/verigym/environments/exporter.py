from verigym.environments.explicitenv import ExplicitEnv
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp

import os
import stormpy

"""
Methods to export to different formal model formats.
"""

def export_to_stormpy_mdp(env: ExplicitEnv) -> stormpy.storage.SparseMdp:
    """ Exports an `ExplicitEnv` to a `stormpy.storage.SparseMdp`.

    Parameters
    ----------
    env : ExplicitEnv
        The explicit environment.
    
    Returns
    -------
    stormpy_mdp : stormpy.storage.SparseMdp
        The mdp.
    """
    assert issubclass(type(env), ExplicitEnv)
    stormpy_mdp = build_stormpy_mdp(env)
    return stormpy_mdp

def export_to_julia(env: ExplicitEnv):
    ... # TODO @Merlijn

def export_to_drn(env: ExplicitEnv,
                    out_file: str | None = None
                    ) -> None:
    """
    Exports an explicit env to an explicit model and writes directly to a .drn file.
    
    Parameters
    ----------
    env : ExplicitEnv
        The explicit environment.
    out_file : str
        Path to the output file. By default, writes to cwd/mdp.drn.
    """
    if not out_file:
        out_file = os.path.join(os.getcwd(), "mdp.drn")
    
    stormpy_mdp = build_stormpy_mdp(env)
    stormpy.export_to_drn(model=stormpy_mdp,
                            file=out_file)
    

def export_to_prism(env: ExplicitEnv) -> str:
    ... # TODO @Jule/Maris? 