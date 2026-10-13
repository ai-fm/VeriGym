from verigym.environments.explicitenv import ExplicitEnv
from verigym.environments.transition_func import IntervalTransitionFunction
from verigym.environments.reward_func import IntervalRewardFunction


class IntervalExplicitEnv(ExplicitEnv):
    """
    An `ExplicitEnv` with intervals over the transition probabilities and rewards, i.e., an interval MDP (IMDP).

    The env only stores the intervals. Wherever a point estimate is required (e.g., to sample transitions in `step`,
    for the action mask and terminal states, or to export the env as a point MDP), it uses the point estimates of the
    intervals: `IntervalTransitionFunction.point_estimate` and `IntervalRewardFunction.point_estimate`.
    These are the `transition_function` and `reward_function` of the env.
    """

    def __init__(self,
                 nr_states,
                 nr_actions,
                 initial_state_distr,
                 interval_transition_function: IntervalTransitionFunction,
                 interval_reward_function: IntervalRewardFunction,
                 nr_rewards = 1,
                 abstraction_map=None,
                 original_env = None,
                 render_mode = None):
        self._check_intervals(interval_transition_function, interval_reward_function)

        super().__init__(nr_states,
                         nr_actions,
                         initial_state_distr,
                         interval_transition_function.point_estimate(),
                         interval_reward_function.point_estimate(),
                         nr_rewards,
                         abstraction_map,
                         original_env,
                         render_mode)

        self.interval_transitions = interval_transition_function
        self.interval_rewards = interval_reward_function

    def _gather_transition_info(self, state, action, next_state):
        has_action = self.action_mask[state][action] > 0

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
