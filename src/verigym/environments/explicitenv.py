from typing import Optional, Any
import gymnasium as gym
from numpy.typing import NDArray

from verigym.environments.base_explicitenv import BaseExplicitEnv
from verigym.environments.verigymenv import VeriGymEnv
from verigym.environments.transition_func import TransitionFunction
from verigym.environments.reward_func import RewardFunction
from verigym.environments.state_distributions import StateDistribution


class AbstractedExplicitEnv(BaseExplicitEnv):
    abstraction_map: Any
    original_env: VeriGymEnv
    observation_space: gym.spaces.Discrete
    action_space: gym.spaces.Discrete

    def __init__(
        self,
        nr_states: int,
        nr_actions: int,
        initial_state_distr: StateDistribution | NDArray, # gets converted to StateDistr if it isn't one
        transition_function: TransitionFunction,
        reward_function: RewardFunction,
        nr_rewards: int = 1,
        abstraction_map=None,
        original_env: VeriGymEnv = None,
        render_mode: Optional[str] = None,
        dummy_terminal_state: int = None
    ):
        super().__init__(
            nr_states=nr_states,
            nr_actions=nr_actions,
            initial_state_distr=initial_state_distr,
            transition_function=transition_function,
            reward_function=reward_function,
            nr_rewards=nr_rewards,
            render_mode=render_mode,
            dummy_terminal_state=dummy_terminal_state,
        )

        self.abstraction_map = abstraction_map
        self.original_env = original_env

    def get_abstraction_map(self):
        return self.abstraction_map
