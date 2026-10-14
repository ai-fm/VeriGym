from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv

import stormpy

"""
Methods to import explicit environments from different formal model formats.
"""

def _env_from_stormpy_model(model, flat: bool) -> FrameworkExplicitEnv:
    """Converts a stormpy MDP or interval MDP loaded from a file to an env (see `FrameworkExplicitEnv.from_stormpy`)."""
    if isinstance(model, (stormpy.storage.SparseMdp, stormpy.storage.SparseIntervalMdp)):
        return FrameworkExplicitEnv.from_stormpy(model, flat=flat)
    raise NotImplementedError(
        f"Loading a {type(model).__name__} is not supported, only MDPs and interval MDPs. "
        "Use stormpy directly to load other types of models."
    )

def load_from_umb(in_file: str, flat: bool = True) -> FrameworkExplicitEnv:
    """
    Loads an explicit env from a Unified Markov Binary (UMB) file, using stormpy.

    Parameters
    ----------
    in_file : str
        Path to the .umb file.
    flat : bool, default=True
        Whether the env uses flat (integer) states, or factored states from the state valuations in the file.

    Returns
    -------
    env : FrameworkExplicitEnv | IntervalFrameworkExplicitEnv
        A `FrameworkExplicitEnv` with the choice labels of the file as action labels; for interval MDPs, an
        `IntervalFrameworkExplicitEnv` with the transition and reward intervals of the file.

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
