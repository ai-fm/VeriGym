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
    @abstractmethod
    def set_state(self, state: StateType) -> None: ...

    @abstractmethod
    def render(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict | None = None) -> None: ...

    @abstractmethod
    def do_simulation(self) -> None: ...

    @abstractmethod
    def get_state(self) -> StateType: ...

    @abstractmethod
    def step(self, action: ActionType) -> tuple[ObsType, float, bool, bool, dict]: ...

    @abstractmethod
    def dt(self) -> float: ...


class MujocoEnv(GymnasiumMujocoEnv, SimulatorEnv):
    def set_state(
        self, state: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ) -> None:
        return super().set_state(*state)

    def get_state(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        return self.data.qpos.flat, self.data.qvel.flat

    def render(self) -> None:
        return super().render()

    def close(self) -> None:
        return super().close()

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        return super().reset(seed=seed, options=options)

    def do_simulation(self) -> None:
        return super().do_simulation()

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict]:
        return super().step(action)

    def dt(self) -> float:
        return super().dt()

    def reset_model(self) -> None:
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        super().set_state(qpos, qvel)
