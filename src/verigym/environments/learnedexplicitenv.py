"""
Contains definitions related to LearnedExplicitEnv, which extends ExplicitEnv such that the transition-
and reward functions can be updated in-place.
"""
import numpy as np
from .explicitenv import ExplicitEnv
from collections import defaultdict
from numpy.typing import NDArray
from .transition_func import TransitionFunction
from .reward_func import RewardFunction

class LearnedTransitionFunction(TransitionFunction):
    """
    Implementation of transition function that can be using new data.
    """

    T_counts: defaultdict[int, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))   # s -> a -> count

    def update_transition(self, s:int, a:int, sps:dict[int,int]):
        
        # Update counter & compute renormalization factor
        prev_count = self.T_counts[s][a]
        self.T_counts[s][a] += sum(sps.values())
        new_count = self.T_counts[s][a]

        # Update all old probabilities
        for (sp, prob) in self.T_dict[s][a].items():
            self.T_dict[s][a][sp] = (prob * prev_count + sps.get(sp,0) ) / new_count
            
        # Add new transitions
        new_states = sps.keys() - self.T_dict[s][a].keys()
        if not len(new_states) == 0:
            for sp in new_states:
                self.T_dict[s][a][sp] = sps[sp] / new_count

class LearnedRewardFunction(RewardFunction):
    """
    Implementation of reward function that can be using new data.
    """

    R_counts: defaultdict[int, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))    # s -> a -> count

    def update_reward(self, s:int, a:int, rewards:dict[float,int]):
        
        # Update counter & compute renormalization factor
        prev_count = self.R_counts[s][a]
        new_count = len(rewards)
        self.R_counts[s][a] += new_count
        self.R_dict[s][a] = (prev_count * self.R_dict[s][a] + new_count * np.mean(rewards)) / self.R_counts[s][a]


def update_init_state_distr(distr:NDArray, prev_count:int, new_counts:dict[int,int]):
    new_total_count = prev_count + sum(new_counts.values())
    for (s,count) in new_counts.items():
        distr[s] = (distr[s] * prev_count + count) / new_total_count
    return new_total_count

class LearnedExplicitEnv(ExplicitEnv):

    init_state_count = 0

    # TODO: this class should have a seperate initialization function
    # def __init__(self,*args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def update_env(self,
            new_init_counts:dict[int,int],
            new_transition_counts:dict[int,dict[int,dict[int,int]]],
            new_reward_counts:dict[int,dict[int,list[float]]]
            ):
        

        self.init_state_count = update_init_state_distr(self.initial_states, self.init_state_count, new_init_counts)

        for s in new_transition_counts.keys():
            for a in new_transition_counts[s].keys():
                self.transition_function.update_transition(s,a,new_transition_counts[s][a])
        
        for s in new_reward_counts.keys():
            for a in new_reward_counts[s].keys():
                self.reward_function.update_reward(s,a,new_reward_counts[s][a])
        
