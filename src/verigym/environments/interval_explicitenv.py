from typing import Optional

from numpy.typing import NDArray

from verigym.environments.explicitenv import AbstractedExplicitEnv
from verigym.environments.state_distributions import StateDistribution
from verigym.environments.transition_func import IntervalTransitionFunction, TransitionFunction
from verigym.environments.reward_func import IntervalRewardFunction, RewardFunction
from verigym.environments.verigymenv import VeriGymEnv


class IntervalEnvMixin:
    """
    Intervals over the transition probabilities and rewards, i.e., an interval MDP (IMDP). Shared by
    `IntervalExplicitEnv` (e.g., learned abstractions) and `IntervalFrameworkExplicitEnv` (e.g., interval MDPs loaded
    with stormpy).

    The env only stores the intervals. Wherever a point estimate is required (e.g., to sample transitions in `step`,
    for the action mask and terminal states, or to export the env as a point MDP), it uses the point estimates of the
    intervals: `IntervalTransitionFunction.point_estimate` and `IntervalRewardFunction.point_estimate`.
    These are the `transition_function` and `reward_function` of the env.
    """

    interval_transitions: IntervalTransitionFunction
    interval_rewards: IntervalRewardFunction

    def _set_intervals(self, interval_transition_function: IntervalTransitionFunction,
                       interval_reward_function: IntervalRewardFunction):
        """Checks the intervals (see `_check_intervals`) and stores them."""
        self._check_intervals(interval_transition_function, interval_reward_function)
        self.interval_transitions = interval_transition_function
        self.interval_rewards = interval_reward_function

    def _point_estimates(self, interval_transition_function: IntervalTransitionFunction,
                         interval_reward_function: IntervalRewardFunction) -> tuple[TransitionFunction, RewardFunction]:
        """Stores the intervals (see `_set_intervals`), and returns their point estimates, which are the transition and
        reward function of the env."""
        self._set_intervals(interval_transition_function, interval_reward_function)
        return self.interval_transitions.point_estimate(), self.interval_rewards.point_estimate()

    def _gather_transition_info(self, state, action, next_state):
        # The available actions are the ones with intervals (the same as the actions of the point estimates)
        has_action = action in self.interval_transitions.T_dict.get(state, {})

        reward_interval = self.interval_rewards[state][action]  if has_action else [0.0, 0.0]
        transition_interval = self.interval_transitions[state][action][next_state] if has_action else [0.0, 0.0]

        return {
            "reward_interval": reward_interval,
            "transition_interval": transition_interval
        }

    @staticmethod
    def _check_intervals(interval_transitions, interval_rewards):
        """
        Check that the intervals of every state-action pair contain a probability distribution, and that no reward
        interval has a lower bound above its upper bound.

        Only stored entries are visited, so the check does not add keys to the (default)dicts.
        """
        if not interval_transitions.sanity_check():
            raise ValueError("Sanity check for interval transition function failed.")
        for s, actions in interval_rewards.R_dict.items():
            for a, (reward_lb, reward_ub) in actions.items():
                if reward_lb > reward_ub:
                    raise ValueError(f"The reward interval for ({s},{a}) is invalid: [{reward_lb}, {reward_ub}].")

    def get_interval_transition_function(self) -> IntervalTransitionFunction:
        """
        Provides access to the interval transition function.
        """
        return self.interval_transitions

    def get_interval_reward_function(self) -> IntervalRewardFunction:
        """
        Provides access to the interval reward function.
        """
        return self.interval_rewards

    def get_interval_reward(self, state, action) -> list:

        return self.interval_rewards[state][action]


class IntervalAbstractedExplicitEnv(IntervalEnvMixin, AbstractedExplicitEnv):
    """
    An `AbstractedExplicitEnv` with intervals over the transition probabilities and rewards, i.e., an interval MDP (IMDP), e.g.,
    a learned abstraction. See `IntervalEnvMixin` for how the intervals and their point estimates are used.
    """

    def __init__(
        self,
        nr_states: int,
        nr_actions: int,
        initial_state_distr: StateDistribution | NDArray,  # gets converted to StateDistr if it isn't one
        interval_transition_function: IntervalTransitionFunction,
        interval_reward_function: IntervalRewardFunction,
        nr_rewards: int = 1,
        abstraction_map=None,
        original_env: VeriGymEnv = None,
        render_mode: Optional[str] = None,
        dummy_terminal_state: int = None
    ):
        transition_function, reward_function = self._point_estimates(interval_transition_function,
                                                                      interval_reward_function)

        super().__init__(
            nr_states=nr_states,
            nr_actions=nr_actions,
            initial_state_distr=initial_state_distr,
            transition_function=transition_function,
            reward_function=reward_function,
            nr_rewards=nr_rewards,
            abstraction_map=abstraction_map,
            original_env=original_env,
            render_mode=render_mode,
            dummy_terminal_state=dummy_terminal_state,
        )
