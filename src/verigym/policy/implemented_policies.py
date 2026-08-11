from .policy import PolicyClass
import numpy as np
import scipy as scp
from collections import defaultdict
from ..abstraction.abstractionmapper import AbstractionMapper
from ..environments.explicitenv import ExplicitEnv
from ..environments.reward_func import RewardFunction
from ..environments.verigymenv import VeriGymEnv
from ..abstraction.learn_abstraction import normalize_aggregated_counts
from copy import deepcopy

class RandomizedPolicy(PolicyClass):
    """
    A policy that returns random actions, as sampled from the provided environment.
    Works for every class inheriting from `VeriGymEnv` (and therefore `gym.Env`).
    """

    def __init__(self, env:VeriGymEnv, map=AbstractionMapper()):
        def policy(obs):
            return env.action_space.sample()

        return super().__init__(policy, map)

    def _action_from_policy(self, obs):
        return self.policy(obs)

class QValuePolicy(PolicyClass):
    """
    A native MDP policy class that selects actions based on (approximate) Q-values.
    """

    def __init__(self, env:ExplicitEnv, nr_states:int, nr_actions:int, map=AbstractionMapper(), Q_init=0, discount=0.95):
        self.env = env
        self.discount = discount
        self.map = map
        self.epsilon_random = 0.0

        # Is there any way to infer this from the other arguments?
        self.nr_states = nr_states
        self.nr_actions = nr_actions

        self.Q_table = np.zeros((nr_states, nr_actions)) + Q_init

        def policy(obs):
            if self.epsilon_random > 0 and np.random.rand() > self.epsilon_random:
                p=scp.special.softmax(self.Q_table[obs,:])
            else:
                p = np.ones(self.nr_actions) / self.nr_actions
            return np.random.choice(a=self.nr_actions, p=p)
        
        return super().__init__(policy, map)
    
    def _action_from_policy(self, obs):
        return self.policy(obs)
    
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_dict_counts, state_distr_counts):
        
        ### Update Q-table

        return NotImplementedError

class ActiveLearningPolicy(QValuePolicy):
    """
    A policy used for active learning of MDPs, based on the state-action count reward method of Araya-Lopéz et. al. (2012).
    """

    def __init__(self, env:ExplicitEnv, nr_states:int, nr_actions:int, map=AbstractionMapper(), Q_init=0, discount=0.95):
        super().__init__(env, nr_states, nr_actions, map, Q_init, discount)
        self.epsilon_random = 0.0
    
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_dict_counts, state_distr_counts):
        
        ### Construct environment

        # normalize_aggregated_counts changes dicts in-place and thus destroys originals: we prevent this with a copy
        # TODO: change code to prevent this.
        T_counts_copy, R_dict_counts_copy = deepcopy(T_counts), deepcopy(R_dict_counts)

        T, R, S_init = normalize_aggregated_counts(
            T_counts_copy, R_dict_counts_copy, P_tot_counts, state_distr_counts, self.nr_states, self.nr_actions
        )
        Rmax = 1

        ### Construct reward function for learning
        R_learning = defaultdict(lambda: defaultdict(lambda: Rmax))
        for sidx in T.T_dict.keys(): # loop only over explored states
            for aidx in range(self.nr_actions):
                this_count = P_tot_counts[(sidx,aidx)]
                if this_count > 0:
                    R_learning[sidx][aidx] = 1 / this_count
        
        ### Update Q-table
        update_Q_table(self.Q_table, R=R_learning, T=T)

        return self

class EntropyLearningPolicy(QValuePolicy):
    """
    A policy class for (iteratively) computing max-entropy policies, based on algorithm from Hazan et. al. (2019).
    
    """

    def __init__(self, env:ExplicitEnv, nr_states:int, nr_actions:int, map=AbstractionMapper(), Q_init=0, discount=0.95):
        self.tabular_policy = np.zeros((nr_states, nr_actions))
        self.learning_rate = 0.2

        def policy(obs):
            return np.choice(self.nr_actions, self.tabular_policy[obs,:])
        
        super().__init__(env, nr_states, nr_actions, map, Q_init, discount)
        
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_dict_counts, state_distr_counts):


        ### Construct environment

        # normalize_aggregated_counts changes dicts in-place and thus destroys originals: we prevent this with a copy
        # TODO: change code to prevent this.
        T_counts_copy, R_dict_counts_copy = deepcopy(T_counts), deepcopy(R_dict_counts)

        T, R, S_init = normalize_aggregated_counts(
            T_counts_copy, R_dict_counts_copy, P_tot_counts, state_distr_counts, self.nr_states, self.nr_actions
        )
        nr_states, nr_actions = self.nr_states, self.nr_actions

        ### Construct reward function for learning

        # TODO: make this sparse!
        T_pi = np.zeros((nr_states, nr_states))
        R_learning = RewardFunction(n_states=self.nr_states, n_actions=self.nr_actions)
        for sidx in T.T_dict.keys(): # loop only over explored states
            Tcount_s = T_counts[sidx]
            for aidx in range(nr_actions):
                for spidx in Tcount_s[aidx].keys():
                    T_pi[sidx,spidx] += self.tabular_policy[sidx,aidx] * Tcount_s[aidx][spidx]


        d_pi = (1-self.discount) * np.linalg.inv(np.eye(nr_states) - self.discount * T_pi) @ S_init

        for sidx in range(nr_states):
            R_learning[sidx][:] = - (np.log(d_pi[sidx]) + 1)
        
        ### Compute new Q-table
        self.Q_table = update_Q_table(self.Q_table, R=R_learning.R_dict, T=T)

        ### Update policy
        for sidx in range(nr_states):
            self.tabular_policy[sidx, :] = (1-self.learning_rate) * self.tabular_policy[sidx, :] + self.learning_rate * scp.special.softmax(self.Q_table[sidx])
        return self
    
def update_Q_table(Q_table, R, T, nr_iterations = 25, discount=0.99, Q_init=0):
    # Unpacking
    nr_states, nr_actions = np.shape(Q_table)
    if Q_table is None:
        Q_table = np.zeros((nr_states, nr_actions)) + Q_init

    Qmax = np.zeros(nr_states)
    for sidx in T.T_dict.keys():
        Qmax[sidx] = max(Q_table[sidx,:])

    # Updates:
    for _i in range(nr_iterations):
        # Qmax = discount * np.max(Q_table,axis=1)
        for (sidx, Ts) in T.T_dict.items():
        # for sidx in range(nr_states):
            this_Qmax = -np.inf
            for aidx in range(nr_actions):
                this_Q = R[sidx][aidx]
                for (spidx, prob) in Ts[aidx].items():
                    this_Q += prob * Qmax[spidx]
                Q_table[sidx,aidx] = this_Q
                this_Qmax = max(this_Qmax, this_Q)
            Qmax[sidx] = discount * this_Qmax

    return Q_table