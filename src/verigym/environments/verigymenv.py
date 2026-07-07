import logging
from typing import TYPE_CHECKING

import gymnasium as gym
from gymnasium import Env, Wrapper
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from collections.abc import Callable, Sequence
from typing import Any

from .labeling import StateLabeler, StateLabel

if TYPE_CHECKING:
    from ..policy.policy import PolicyClass

logger = logging.getLogger(__name__)

def _make_env(cls, wrappers, **env_kwargs):
    env = cls(**env_kwargs)

    for wrapper in wrappers:
        env = wrapper(env)

    return env

class VeriGymEnv(gym.Env):
    """
    Abstract wrapper around `gymnasium.Env` for a unified VeriGym interface.
    Parent to all types of VeriGym environments.
    """

    def __init__(self):
        self.state_labeler = StateLabeler(set())

    def simulate(
        self, policy: "PolicyClass", n_steps: int = 1, verbose=True
    ) -> list[list[NDArray, NDArray, NDArray, NDArray]]:
        """
        Simulate the environment for `n_steps` using the provided `policy`.

        Args:
            policy: A function that takes an observation and returns an action.
            n_steps (int): Number of steps to simulate.

        Returns:
            A list of trajectories containing a list of tuples (state, action, reward, next_state) for each step.
        """
        dataset, trajectory = [], []
        state, info = self.reset()

        for _ in tqdm(range(n_steps), desc="Simulating", disable=not verbose):
            action = policy.get_action(state)
            next_state, reward, done, truncated, info = self.step(action)
            next_state, action, reward = (
                np.array(next_state),
                np.array(action),
                np.array(reward),
            )
            trajectory.append((state, action, reward, next_state))
            state = next_state
            if done or truncated:
                dataset.append(trajectory)
                state, info = self.reset()
                trajectory = []

        if len(trajectory) > 0:
            # Append remaining trajectories
            dataset.append(trajectory)

        return dataset
    
        
    def has_state_labels(self):
        return len(self.state_labeler.labels) > 0
    
    def add_state_label(self, label: StateLabel):
        self.state_labeler.add_state_label(label)

    def add_state_labels(self, labels: list[StateLabel]):
        for label in labels:
            self.state_labeler.add_state_label(label)
    
    def get_labels_of_state(self, state):
        return self.state_labeler.get_labels_of_state(state)
    
    
    @classmethod
    def make_vec(cls, 
                 num_envs: int = 1,
                 vectorization_mode:  str | None = "sync",
                 vector_kwargs: dict[str, Any] | None = None,
                 wrappers: Sequence[Callable[[Env], Wrapper]] | None = None,
                 make_callback = None, **kwargs):
        """
        Builds vectorized VeriGymEnvs.

        Parameters
        ----------
        num_envs : int
            How many envs to contain in the vectorized env.
        render_mode : str
            Environment render mode. Currently can only be None.
        vectorization_mode : str
            The vectorization mode. Can be "sync" or "async"
        vector_kwargs : dict
            Further arguments to apply to vectorization.
        wrappers : Sequence
            List of wrappers to be applied to the environments.
        **kwargs : 
            Allows to include environment-specific parameters.

        Returns
        -------
        env : gym.SyncVectorEnv(VeriGymEnv) if vectorization_mode=="sync" or gym.AsyncVectorEnv(VeriGymEnv) if vectorization_mode=="async"
            The vectorized FrameworkExplicitEnvs.
        """
        if kwargs is None: 
            kwargs = {}
        if vector_kwargs is None: 
            vector_kwargs = {}
        if wrappers is None: 
            wrappers = []
        if make_callback is None: 
            make_callback = cls

        env_fns = [lambda: _make_env(make_callback, wrappers, **kwargs) for _ in range(num_envs)]

        if vectorization_mode == "sync":
            return SyncVectorEnv(env_fns, **vector_kwargs)
        elif vectorization_mode == "async":
            return AsyncVectorEnv(env_fns, **vector_kwargs)
        else: 
            raise ValueError()
