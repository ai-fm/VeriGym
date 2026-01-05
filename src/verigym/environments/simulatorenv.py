from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from gymnasium.envs.mujoco import MujocoEnv as GymnasiumMujocoEnv
import numpy as np
import numpy.typing as npt

from .verigymenv import VeriGymEnv


ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
StateType = TypeVar("StateType")


class SimulatorEnv(VeriGymEnv, ABC, Generic[ObsType, ActionType, StateType]):
    """
    Abstract base class for simulation environments.

    This class defines the interface for simulation environments that can be
    used with the VeriGym framework. It extends VeriGymEnv and provides a
    generic interface for environments with customizable observation, action,
    and state types.

    Parameters
    ----------
    ObsType : TypeVar
        The type of observations returned by the environment.
    ActionType : TypeVar
        The type of actions accepted by the environment.
    StateType : TypeVar
        The type representing the internal state of the environment.

    See Also
    --------
    [MujocoEnv][verigym.environments.simulatorenv.MujocoEnv] : Concrete implementation using the MuJoCo physics engine.

    Notes
    -----
    All methods in this class are abstract and must be implemented by
    subclasses.
    """

    @abstractmethod
    def set_state(self, state: StateType) -> None:
        """
        Set the internal state of the environment.

        Parameters
        ----------
        state : StateType
            The state to set the environment to.

        Returns
        -------
        None
        """
        ...

    @abstractmethod
    def render(self) -> None:
        """
        Render the current state of the environment.

        Returns
        -------
        None
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Clean up environment resources.

        This method should release any resources held by the environment,
        such as rendering windows or file handles.

        Returns
        -------
        None
        """
        ...

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """
        Reset the environment to an initial state.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for reproducibility. If None, no seed is set.
            Default is None.
        options : dict or None, optional
            Additional options for environment reset. Default is None.

        Returns
        -------
        None
        """
        ...

    @abstractmethod
    def do_simulation(self) -> None:
        """
        Advance the simulation by one timestep.

        This method performs the simulation step without processing
        actions or computing rewards.

        Returns
        -------
        None
        """
        ...

    @abstractmethod
    def get_state(self) -> StateType:
        """
        Get the current internal state of the environment.

        Returns
        -------
        StateType
            The current state of the environment.
        """
        ...

    @abstractmethod
    def step(self, action: ActionType) -> tuple[ObsType, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Parameters
        ----------
        action : ActionType
            The action to take in the environment.

        Returns
        -------
        observation : ObsType
            The observation resulting from the action.
        reward : float
            The reward obtained from taking the action.
        terminated : bool
            Whether the episode has terminated (e.g., goal reached, failure).
        truncated : bool
            Whether the episode was truncated (e.g., time limit reached).
        info : dict
            Additional information about the step.
        """
        ...

    @property
    @abstractmethod
    def dt(self) -> float:
        """
        Get the simulation timestep duration.

        Returns
        -------
        float
            The duration of one simulation timestep in seconds.
        """
        ...


class MujocoEnv(GymnasiumMujocoEnv, SimulatorEnv):
    """
    MuJoCo-based implementation of the SimulatorEnv interface.

    This class provides a concrete implementation of SimulatorEnv using the
    MuJoCo physics engine via the Gymnasium MuJoCo environment. It manages
    the simulation state as position and velocity arrays.

    Attributes
    ----------
    data : MjData
        MuJoCo simulation data containing state information.
    model : MjModel
        MuJoCo model specification.
    init_qpos : ndarray
        Initial generalized positions.
    init_qvel : ndarray
        Initial generalized velocities.
    np_random : numpy.random.Generator
        Random number generator for stochastic operations.

    See Also
    --------
    [SimulatorEnv][verigym.environments.simulatorenv.SimulatorEnv] : Abstract base class defining the interface.

    Examples
    --------
    >>> env = MujocoEnv()
    >>> env.reset(seed=42)
    >>> state = env.get_state()
    >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    def set_state(
        self, state: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ) -> None:
        """
        Set the MuJoCo simulation state.

        Parameters
        ----------
        state : tuple of ndarray
            A tuple containing (qpos, qvel) where:
            - qpos : ndarray of floating
                Generalized positions of the model.
            - qvel : ndarray of floating
                Generalized velocities of the model.

        Returns
        -------
        None
        """
        return super().set_state(*state)

    def get_state(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """
        Get the current MuJoCo simulation state.

        Returns
        -------
        qpos : ndarray of floating
            Flat array of generalized positions.
        qvel : ndarray of floating
            Flat array of generalized velocities.
        """
        return self.data.qpos.flat, self.data.qvel.flat

    def render(self) -> None:
        """
        Render the current state of the MuJoCo environment.

        Returns
        -------
        None
        """
        return super().render()

    def close(self) -> None:
        """
        Clean up MuJoCo environment resources.

        Releases rendering contexts and other resources held by the
        environment.

        Returns
        -------
        None
        """
        return super().close()

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """
        Reset the MuJoCo environment to an initial state.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for the environment's random number generator.
            Default is None.
        options : dict or None, optional
            Additional reset options passed to the parent environment.
            Default is None.

        Returns
        -------
        None
        """
        return super().reset(seed=seed, options=options)

    def do_simulation(self) -> None:
        return super().do_simulation()

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict]:
        """
        Execute one step in the MuJoCo environment.

        Parameters
        ----------
        action : ndarray of float32
            The control action to apply to the environment.

        Returns
        -------
        observation : ndarray of float32
            The observation after taking the action.
        reward : float
            The reward obtained from the action.
        terminated : bool
            Whether the episode has terminated.
        truncated : bool
            Whether the episode was truncated.
        info : dict
            Additional step information.
        """
        return super().step(action)

    @property
    def dt(self) -> float:
        """
        Get the MuJoCo simulation timestep duration.

        Returns
        -------
        float
            The duration of one simulation timestep in seconds.
        """
        return super().dt

    def reset_model(self) -> None:
        """
        Reset the model to a randomized initial state.

        Initializes the model with positions and velocities perturbed from
        their default values. Positions are perturbed with uniform noise
        in [-0.1, 0.1], and velocities are perturbed with Gaussian noise
        scaled by 0.1.

        Returns
        -------
        None

        Notes
        -----
        This method is typically called internally by `reset()` to
        initialize the physics state after environment reset.
        """
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        super().set_state(qpos, qvel)
