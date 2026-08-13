from .verigymenv import VeriGymEnv

import gymnasium as gym
from typing import Any
from gymnasium import Env, Wrapper
from collections.abc import Callable, Sequence
from typing import Optional
import stormpy
import stormpy.simulator
from pathlib import Path
import json
import numpy as np
import warnings

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
            f"{GenerativeEnv.__name__}_from_gym",
            (env.__class__, GenerativeEnv),
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
                           wrappers: Sequence[Callable[[Env], Wrapper]] | None = None,
                           wrapper_kwargs: list | None = None):   
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
        wrapper_kwargs : list
            Further arguments, per wrapper, to apply to wrappers.

        Returns
        -------
        env : gym.SyncVectorEnv(GenerativeEnv) if vectorization_mode=="sync" or gym.AsyncVectorEnv(GenerativeEnv) if vectorization_mode=="async"
            The vectorized GenerativeEnv.
        """

        return GenerativeEnv.make_vec(num_envs=num_envs, vectorization_mode=vectorization_mode, vector_kwargs=vector_kwargs, wrappers=wrappers, wrapper_kwargs=wrapper_kwargs,
                                      make_callback=GenerativeEnv.from_gymnasium,
                                      env=env
                                      )
    
    @classmethod
    def from_prism(cls, prism_filepath: str, seed=None):
        instance = SymbolicGenerativeEnv.__new__(SymbolicGenerativeEnv)

        file_path = Path(prism_filepath)
        if not file_path.is_file():
            raise FileNotFoundError(f"File {file_path} not found.")

        elif file_path.suffix not in [".prism", ".pm", ".nm"]:
            raise ValueError(f"Expected a valid prism format suffix: .prism, .nm, .pm, but received .{file_path.suffix} instead.")

        prism_program = stormpy.parse_prism_program(prism_filepath)
        n_actions = len(prism_program.get_synchronizing_action_indices())

        simulator = stormpy.simulator.create_simulator(prism_program, seed)

        SymbolicGenerativeEnv.__init__(instance, simulator, n_actions)

        return instance
    
    @classmethod
    def vec_from_prism(cls, prism_filepath: str, 
                       num_envs: int = 1,
                       vectorization_mode: str | None = "sync",
                       vector_kwargs: dict[str, Any] | None = None,
                       wrappers: Sequence[Callable[[Env], Wrapper]] | None = None,
                       wrapper_kwargs: list | None = None):
        """
        Builds a vectorized GenerativeEnv from a prism program file.

        Parameters
        ----------
        prism_filepath : str
            The path to the prism program.
        num_envs : int
            How many envs to contain in the vectorized env.
        render_mode : str
            Environment render mode. 
            Note: We currently do not support visualization of envs from PRISM.
        vectorization_mode : str
            The vectorization mode. Can be "sync" or "async"
        vector_kwargs : dict
            Further arguments to apply to vectorization.
        wrappers : Sequence
            List of wrappers to be applied to the environments.
        wrapper_kwargs : list
            Further arguments, per wrapper, to apply to wrappers.

        Returns
        -------
        env : gym.SyncVectorEnv(GenerativeEnv) if vectorization_mode=="sync" or gym.AsyncVectorEnv(GenerativeEnv) if vectorization_mode=="async"
            The vectorized GenerativeEnv.
        """
        if vectorization_mode == "async":
            raise ValueError("CCannot use async vectorization with envs that include stormpy C++ objects (C++ object that cannot be serialized). ")
        return GenerativeEnv.make_vec(num_envs=num_envs, vectorization_mode=vectorization_mode, vector_kwargs=vector_kwargs, wrappers=wrappers, wrapper_kwargs=wrapper_kwargs,
                                      make_callback=GenerativeEnv.from_prism,
                                      prism_filepath=prism_filepath,
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

class SymbolicGenerativeEnv(GenerativeEnv):
    """
    This is an extension to GenerativeEnv meant for simulation environments from symbolic model descriptions (e.g., in PRISM language).
    It overwrites the `step` and `reset` methods to use the `stormpy.simulator`.

    Note that we do **not** build the underlying MDP here, and hence do not have access to an explicit model representation.
    """

    def __init__(self, simulator: stormpy.simulator.Simulator,
                 n_actions: int):
        super().__init__()

        self.simulator = simulator
        self.action_space = gym.spaces.Discrete(n_actions, dtype=int)

        # build the nvec for the MultiDiscrete observation space using program information
        program = simulator._program.substitute_constants()
        obs_space_builder = {}
        for module in program.modules:
            for var in module.integer_variables:
                name = var.name
                lo = var.lower_bound_expression.evaluate_as_int()
                hi = var.upper_bound_expression.evaluate_as_int()
                n = int(hi-lo) + 1 # add 1 to include both "edges"
                obs_space_builder[name] = n
            for var in module.boolean_variables:
                name = var.name
                n = 2
                obs_space_builder[name] = n
        assert len(obs_space_builder.keys()) == len(self._get_obs(simulator._report_state()))
        nvec = np.array([obs_space_builder[k] for k in sorted(obs_space_builder.keys())])
        
        self.observation_space = gym.spaces.MultiDiscrete(nvec=nvec,
                                                dtype=int,
                                                )

        
    def step(self, action):
        """
        Take a step in the environment, using the simulator.
        """
        available_actions = self.simulator.available_actions()
        if action in available_actions:
            act = available_actions[action]
            state, reward, labels = self.simulator.step(act)
            reward = reward[0]
                    
        else:
            warnings.warn(f"Action {action} is not available in state {self.simulator.report_state()}. Staying.", UserWarning)
            state = self.simulator._report_state()
            reward = self.simulator._report_reward()[0]
            labels = self.simulator._report_labels()

        terminated = self.simulator.is_done()

        truncated = False

        info = {
            "state_valuations": state, # valuations
            "state_labels": labels,
        }

        observation = self._get_obs(state)

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the simulator.
        """
        state, _, labels = self.simulator.restart()

        info = {
            "state_valuations": state, # valuations
            "state_labels": labels,
        }

        observation = self._get_obs(state)

        return observation, info

    def _get_obs(self, state):
        # state is a `stormpy.utility._utility.JsonContainerDouble` object
        # that is a python-wrapped C++ object that does not expose normal dictionary functions.
        state = json.loads(str(state))

        # we extract the observation
        observation = tuple([state[k] for k in sorted(state.keys())])

        return observation