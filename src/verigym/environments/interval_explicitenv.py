from collections import defaultdict

from verigym.environments.explicitenv import ExplicitEnv
from verigym.environments.transition_func import TransitionFunction, IntervalTransitionFunction
from verigym.environments.reward_func import RewardFunction, IntervalRewardFunction


class IntervalExplicitEnv(ExplicitEnv):
    def __init__(self, 
                 nr_states, 
                 nr_actions, 
                 initial_state_distr, 
                 transition_function: TransitionFunction, 
                 reward_function: RewardFunction,
                 interval_transition_function: IntervalTransitionFunction = None,
                 interval_reward_function: IntervalRewardFunction = None,
                 nr_rewards = 1, 
                 abstraction_map=None, 
                 original_env = None, 
                 render_mode = None):
        super().__init__(nr_states, 
                         nr_actions, 
                         initial_state_distr, 
                         transition_function, 
                         reward_function, 
                         nr_rewards, 
                         abstraction_map, 
                         original_env, 
                         render_mode)
        
        if interval_transition_function is None:
            self.interval_transitions = self._convert_to_interval_transitions(transition_function)
        else:
            self._check_interval_transitions(transition_function, interval_transition_function)
            self.interval_transitions = interval_transition_function

        if interval_reward_function is None:
            self.interval_rewards = self._convert_to_interval_rewards(reward_function)
        else:
            self._check_interval_rewards(reward_function, interval_reward_function)
            self.interval_rewards = interval_reward_function
    
    def _gather_transition_info(self, state, action, next_state):
        has_action = self.action_mask[state][action] > 0

        reward_interval = self.interval_rewards[state][action]  if has_action else [0.0, 0.0]
        transition_interval = self.interval_transitions[state][action][next_state] if has_action else [0.0, 0.0]
        
        return {
            "reward_interval": reward_interval,
            "transition_interval": transition_interval
        }
        
    def _convert_to_interval_transitions(self, T):
        """
        If the IntervalExplicitEnv is initialized using only a standard transition function, this creates an 
        interval transition function data structure where lower bound == upper bound for all state-action-next state transitions.
        """
        T_interval_dict: defaultdict[int, dict[int, defaultdict[int, float]]] = defaultdict(dict)

        for s in range(self.nr_states):
            for a in range(self.nr_actions):
                T_interval_dict[s][a] = defaultdict(lambda: (0.0, 0.0))
                for s_prime in range(self.nr_states):
                    prob = T[s][a][s_prime]
                    if prob > 0:
                        T_interval_dict[s][a][s_prime] = (prob, prob)

        interval_transitions = IntervalTransitionFunction(self.nr_states, self.nr_actions, T_interval_dict)

        return interval_transitions
    

    def _convert_to_interval_rewards(self, rewards):
        """
        If the IntervalExplicitEnv is initialized using only a standard reward function, this creates an 
        interval reward function data structure where lower bound == upper bound for all state-action-next state transitions.
        """
        R_interval_dict = defaultdict(lambda: defaultdict(float))
        for s in range(self.nr_states):
            for a in range(self.nr_actions):
                r = rewards[s][a]
                R_interval_dict[s][a] = (r, r)

        
        interval_rewards = IntervalRewardFunction.from_dict(R_interval_dict, self.nr_states, self.nr_actions)

        return interval_rewards

    def _check_interval_transitions(self, transitions, interval_transitions):
        """
        Check that for all state-action-next state transitions, the point estimates are within the interval bounds 
        and that state-action transition probabilities sum to 1.
        """
        if not transitions.sanity_check():
            raise ValueError("Sanity check for transition function failed.")
        for s in range(self.nr_states):
            for a in range(self.nr_actions):
                for s_prime in range(self.nr_states):
                    tra = transitions[s][a][s_prime]
                    tra_lb = interval_transitions[s][a][s_prime][0]
                    tra_ub = interval_transitions[s][a][s_prime][1]

                    if tra < tra_lb or tra > tra_ub:
                        raise ValueError(f"The point estimate for ({s},{a},{s_prime}) is invalid: p={tra} not in [{tra_lb}, {tra_ub}].")

    
    def _check_interval_rewards(self, rewards, interval_rewards):
        """
        Check that for all state-action pairs, the point rewards are within the interval reward bounds.
        """
        for s in range(self.nr_states):
            for a in range(self.nr_actions):
                point_reward = rewards[s][a]
                reward_lb = interval_rewards[s][a][0]
                reward_ub = interval_rewards[s][a][1]
                if point_reward < reward_lb or point_reward > reward_ub:
                    raise ValueError(f"The point reward for ({s},{a}) is invalid: r={point_reward} not in [{reward_lb}, {reward_ub}].")
                
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