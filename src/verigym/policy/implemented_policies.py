from .policy import PolicyClass
import numpy as np
import scipy as scp
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve
import numbers
from ..abstraction.abstractionmapper import AbstractionMapper
from ..environments.explicitenv import ExplicitEnv
from ..environments.learnedexplicitenv import LearnedExplicitEnv
from ..environments.reward_func import RewardFunction
from ..environments.verigymenv import VeriGymEnv


class RandomizedPolicy(PolicyClass):
    """
    A policy that returns random actions, as sampled from the provided environment.
    Works for every class inheriting from `VeriGymEnv` (and therefore `gym.Env`).
    """

    def __init__(self, env:VeriGymEnv, map=AbstractionMapper()):
        def policy(obs):
            return env.action_space.sample()

        abstraction_mapper = AbstractionMapper()  # Identity mapping
        return super().__init__(policy, abstraction_mapper)

    def _action_from_policy(self, obs):
        return self.policy(obs)

class QValuePolicy(PolicyClass):
    """
    A native MDP policy class that selects actions based on (approximate) Q-values.
    """

    def __init__(self, env:ExplicitEnv, map=AbstractionMapper()):
        self.Q_table = None
        self.env = env
        self.discount = 0.95 # TODO: where do we get this?
        self.map = map
        self.epsilon_random = 0.0

        def policy(obs):
            # print(f"o={obs}, actions={self.env.nr_actions}, Q={self.Q_table}")
            if self.Q_table is None:
                return np.random.choice(self.env.nr_actions)
            if self.epsilon_random > 0 and np.random.rand() > self.epsilon_random:
                p=scp.special.softmax(self.Q_table[obs,:])
            else:
                p = np.ones(self.env.nr_actions) / self.env.nr_actions
            return np.random.choice(a=self.env.nr_actions, p=p)
        
        return super().__init__(policy, map)
    
    def _action_from_policy(self, obs):
        return self.policy(obs)
    
    def update_for_abstraction_refinement(self, env):
        
        ### Update Q-table
        self.env = env
        self.Q_table = update_Q_table(self.env, self.Q_table)

        return self

class ActiveLearningPolicy(QValuePolicy):
    """
    A policy used for active learning of MDPs, based on the state-action count reward method of Araya-Lopéz et. al. (2012).
    """

    def __init__(self, env:LearnedExplicitEnv, map=AbstractionMapper()):
        super().__init__(env, map)
        self.epsilon_random = 0.25
    
    def update_for_abstraction_refinement(self, env):

        self.env = env
        nr_states, nr_actions = self.env.nr_states, self.env.nr_actions

        ### Construct reward function for learning
        R_learning = RewardFunction(n_states=nr_states, n_actions=nr_actions)
        for sidx in range(nr_states):
            for aidx in range(nr_actions):
                this_count = self.env.transition_function.T_counts[sidx][aidx]
                if this_count < 1:
                    R_learning[sidx][aidx] = 1_000
                else:
                    R_learning[sidx][aidx] = 1 / this_count
        
        ### Update Q-table
        update_Q_table(self.env, self.Q_table, R=R_learning.R_dict)

class EntropyLearningPolicy(QValuePolicy):
    """
    A policy class for (iteratively) computing max-entropy policies, based on algorithm from Hazan et. al. (2019).
    
    """

    def __init__(self, env:LearnedExplicitEnv, map=AbstractionMapper()):
        self.tabular_policy = np.zeros((env.nr_states, env.nr_actions))
        self.learning_rate = 0.2
        
        super().__init__(env, map)

        def policy(obs):
            return np.choice(self.env.nr_actions, self.tabular_policy[obs,:])
        
    def update_for_abstraction_refinement(self,env):
        self.env = env
        nr_states, nr_actions = self.env.nr_states, self.env.nr_actions

        ### Construct reward function for learning
        # TODO: make this sparse!
        T_pi = np.zeros((nr_states, nr_states))
        R_learning = RewardFunction(n_states=self.env.nr_states, n_actions=self.env.nr_actions)
        for sidx in range(nr_states):
            for aidx in range(nr_actions):
                thisT = self.env.transition_function[sidx][aidx]
                for spidx in thisT.keys():
                    T_pi[sidx,spidx] += self.tabular_policy[sidx,aidx] * thisT[spidx]

        init_states_array = np.zeros(nr_states)
        for s, p in self.env.initial_states.items():
            init_states_array[s] = p

        d_pi = (1-self.discount) * np.linalg.inv(np.eye(nr_states) - self.discount * T_pi) @ init_states_array

        for sidx in range(nr_states):
            R_learning[sidx][:] = - (np.log(d_pi[sidx]) + 1)
        
        ### Compute new Q-table
        self.Q_table = update_Q_table(self.env, self.Q_table, R=R_learning.R_dict)

        ### Update policy
        for sidx in range(nr_states):
            self.tabular_policy[sidx, :] = (1-self.learning_rate) * self.tabular_policy[sidx, :] + self.learning_rate * scp.special.softmax(self.Q_table[sidx])

        return self
    
    
def update_Q_table(env:ExplicitEnv, Q_table, R=None, T=None, nr_iterations = 100, discount=0.99):
    # Unpacking
    nr_states, nr_actions = env.nr_states, env.nr_actions
    if Q_table is None:
        Q_table = np.zeros((nr_states, nr_actions))
        nr_iterations = nr_iterations * 10
    if R is None:
        R = env.reward_function.R_dict
    if T is None:
        T = env.transition_function.T_dict

    # Updates:
    for _i in range(nr_iterations):
        Qmax = discount * np.max(Q_table,axis=1)
        for (sidx, Ts) in T.items():
        # for sidx in range(nr_states):
            for aidx in range(nr_actions):
                Q_table[sidx,aidx] = R[sidx][aidx]
                for (spidx, prob) in Ts[aidx].items():
                    Q_table[sidx, aidx] += prob * Qmax[spidx]

    return Q_table