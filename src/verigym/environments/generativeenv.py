from .verigymenv import VeriGymEnv
import gymnasium as gym
from typing import Any
from gymnasium import Env, Wrapper
from collections.abc import Callable, Sequence

class GenerativeEnv(VeriGymEnv):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_gymnasium(cls, env: gym.Env):
        """
        Creates a `GenerativeEnv` from a `gymnasium.Env` instance. All attributes (in the `gymnasium.Env`'s `__dict__` and `__slots__`) are shallow-copied over to the new instance.

        Notes:
        - The original env is not used after copying, this might cause issues with C-level definitions in some environments.
        - This method uses dynamic class creation to merge the original gymnasium environment's class with `GenerativeEnv` and returns a class called `GenerativeEnv_from_gym` that should behave like any other `GenerativeEnv` instantiation.
        - Be cautious when using gymnasium envs with applied wrappers as their compatibility has not been tested, including vectorized envs.
        - `GenerativeEnv.__init__` is currently not called for the merged instance.

        Parameters
        ----------
        env : gym.Env
            The gymnasium environment to wrap.

        Returns
        -------
        GenerativeEnv
            A merged instance whose behavior follows the gym env class first, while remaining an instance of GenerativeEnv.
        """
        assert isinstance(env, (gym.Env)), "Input must be a gymnasium environment."
        merged_cls = type(
            f"{cls.__name__}_from_gym",
            (env.__class__, cls),
            {},
        )
        instance = merged_cls.__new__(merged_cls)

        # Copy instance dict if present.
        if hasattr(env, "__dict__"):
            instance.__dict__.update(env.__dict__)

        # Copy slot-backed attributes from the env.
        for slot in _iter_slots(env.__class__):
            if slot in ("__dict__", "__weakref__"):
                continue
            try:
                value = getattr(env, slot)
            except AttributeError:
                continue
            try:
                setattr(instance, slot, value)
            except AttributeError:
                pass
        
        GenerativeEnv.__init__(instance)
        return instance
    
    @classmethod
    def vec_from_gymnasium(cls, env, num_envs: int = 1,
                           vectorization_mode: str | None = "sync",
                           vector_kwargs: dict[str, Any] | None = None,
                           wrappers: Sequence[Callable[[Env], Wrapper]] | None = None):   
        """
        Builds vectorized GenerativeEnv from a gym environment.

        Parameters
        ----------
        env : gym.Env
            The base environment.
        num_envs : int
            How many envs to contain in the vectorized env.
        render_mode : str
            Environment render mode.
        vectorization_mode : str
            The vectorization mode. Can be "sync" or "async"
        vector_kwargs : dict
            Further arguments to apply to vectorization.
        wrappers : Sequence
            List of wrappers to be applied to the environments.

        Returns
        -------
        env : gym.SyncVectorEnv(GenerativeEnv) if vectorization_mode=="sync" or gym.AsyncVectorEnv(GenerativeEnv) if vectorization_mode=="async"
            The vectorized GenerativeEnv.
        """

        return GenerativeEnv.make_vec(num_envs=num_envs, vectorization_mode=vectorization_mode, vector_kwargs=vector_kwargs, wrappers=wrappers, 
                                      make_callback=GenerativeEnv.from_gymnasium,
                                      env=env
                                      )


def _iter_slots(env_cls: type) -> list[str]:
    slots: list[str] = []
    for base in env_cls.__mro__:
        base_slots = getattr(base, "__slots__", ())
        if isinstance(base_slots, str):
            base_slots = (base_slots,)
        for slot in base_slots:
            if slot not in slots:
                slots.append(slot)
    return slots
