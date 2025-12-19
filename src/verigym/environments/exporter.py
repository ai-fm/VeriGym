from verigym.environments.explicitmodelenv import ExplicitModelEnv
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp

import os
import stormpy

"""
Methods to export to different formal model formats.
"""

def export_to_stormpy_mdp(env: ExplicitModelEnv) -> stormpy.storage.SparseMdp:
    """ Exports an `ExplicitModelEnv` to a `stormpy.storage.SparseMdp`.

    Parameters
    ----------
    env : ExplicitModelEnv
        The explicit environment.
    
    Returns
    -------
    stormpy_mdp : stormpy.storage.SparseMdp
        The mdp.
    """
    assert issubclass(type(env), ExplicitModelEnv)
    stormpy_mdp = build_stormpy_mdp(env)
    return stormpy_mdp

def export_to_julia(env: ExplicitModelEnv):
    ... # TODO @Merlijn

def export_to_drn(env: ExplicitModelEnv,
                    out_file: str | None = None
                    ) -> None:
    """
    Exports an explicit env to an explicit model and writes directly to a .drn file.
    
    Parameters
    ----------
    env : ExplicitModelEnv
        The explicit environment.
    out_file : str
        Path to the output file. By default, writes to cwd/mdp.drn.
    """
    if not out_file:
        out_file = os.path.join(os.getcwd(), "mdp.drn")
    
    stormpy_mdp = build_stormpy_mdp(env)
    stormpy.export_to_drn(model=stormpy_mdp,
                            file=out_file)
    

def export_to_prism(env: ExplicitModelEnv) -> str:
    ... # TODO @Jule/Maris? 