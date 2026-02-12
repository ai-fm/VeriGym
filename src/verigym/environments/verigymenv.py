import logging

import gymnasium as gym
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class VeriGymEnv(gym.Env):
    """
    Abstract wrapper around `gymnasium.Env` for a unified VeriGym interface.
    Parent to all types of VeriGym environments.
    """

    def simulate(
        self, policy, n_steps: int
    ) -> list[list[NDArray, NDArray, NDArray, NDArray]]:
        """
        Simulate the environment for `n_steps` using the provided `policy`.

        Args:
            policy: A function that takes an observation and returns an action.
            n_steps (int): Number of steps to simulate.

        Returns:
            A list of trajectories containing a list of tuples (state, action, reward, next_state) for each step.
        """
        logger.warning(
            "The VeriGymEnv.simulate() method currently only simlates a random policy."
        )
        dataset, trajectory = [], []
        state, info = self.reset()
        for _ in range(n_steps):
            action = self.action_space.sample()  # TODO Replace with policy(obs)
            next_state, reward, done, truncated, info = self.step(action)
            trajectory.append((state, action, reward, next_state))
            state = next_state
            assert state is not None, "State should not be None after step."
            if done or truncated:
                dataset.append(trajectory)
                state, info = self.reset()
                trajectory = []

        if len(trajectory) > 0:
            # Append remaining trajectories
            dataset.append(trajectory)

        return dataset
