from .policy import PolicyClass
import numpy as np
import scipy as scp
from collections import defaultdict
from ..abstraction.abstractionmapper import AbstractionMapper
from ..environments.reward_func import RewardFunction
from ..environments.verigymenv import VeriGymEnv
from ..abstraction.learn_abstraction import normalize_aggregated_counts
from copy import deepcopy

class QValuePolicy(PolicyClass):
    """
    A native MDP policy class that selects actions based on (approximate) Q-values.

    This policy class allows for static epsilon-greedy (no annealing).
    """

    def __init__(self, env:VeriGymEnv, nr_states:int, nr_actions:int, abstraction_map=AbstractionMapper(), 
                 Q_init_strategy="zero", 
                 discount=0.95, epsilon=0.0,
                 update_iterations = 25,
                 ):
        """
        Parameters
        ----------
        env : VeriGymEnv
            The environment to apply the policy to.
        nr_states : int
            The number of states in the environment.
        nr_actions : int
            The number of actions in the environment
        abstraction_map : AbstractionMapper
            Used in case that env is abstract. Default initializes to an identity map.
        Q_init_strategy: str
            How to initialize the Q_table. Default: "zero". 
            Further options: "random" -> random values.  "uniform": uniform across actions.
        discount : float
            Discount factor, used during refinement. Default 0.95.
        epsilon : float
            Exploration threshold for epsilon-greedy. Default 0.0 (no exploration).
        update_iterations : int
            The number of iterations to update the Q_table for during abstraction refinement.
        """
        self.env = env
        self.discount = discount
        self.map = abstraction_map
        self.epsilon_random = epsilon

        self.nr_iterations = update_iterations

        self.nr_states = nr_states
        self.nr_actions = nr_actions

        if Q_init_strategy == "zero":
            self.Q_table = np.zeros((nr_states, nr_actions))
        elif Q_init_strategy == "random":
            self.Q_table = np.random.rand(nr_states, nr_actions)
            self.Q_table /= self.Q_table.sum(axis=1, keepdims=True)
        elif Q_init_strategy == "uniform":
            self.Q_table = np.full((nr_states, nr_actions), fill_value = 1 / nr_actions)

        def policy(obs):
            if self.epsilon_random > 0 and np.random.rand() > self.epsilon_random:
                p=scp.special.softmax(self.Q_table[obs,:])
            else:
                p = np.ones(self.nr_actions) / self.nr_actions
            return np.random.choice(a=self.nr_actions, p=p)
        
        return super().__init__(policy, abstraction_map)
    
    def _action_from_policy(self, obs):
        return self.policy(obs)
    
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_dict_counts, state_distr_counts):
        
        ### Update Q-table

        return NotImplementedError

    def _update_Q_table(self, R, T):
        """
        This function is called as part of QValuePolicy.update_for_abstraction_refinement.

        Parameters
        ----------
        R : defaultdict
            Updated rewards
        T : defaultdict 
            Updated transitions.
        """
        # Unpacking
        nr_states, nr_actions = np.shape(self.Q_table)

        Qmax = np.zeros(nr_states)
        for sidx in T.T_dict.keys():
            Qmax[sidx] = max(self.Q_table[sidx,:])

        # Updates:
        for _ in range(self.nr_iterations):
            for (sidx, Ts) in T.T_dict.items():
                this_Qmax = -np.inf
                for aidx in range(nr_actions):
                    this_Q = R[sidx][aidx]
                    for (spidx, prob) in Ts[aidx].items():
                        this_Q += prob * Qmax[spidx]
                    self.Q_table[sidx,aidx] = this_Q
                    this_Qmax = max(this_Qmax, this_Q)
                Qmax[sidx] = self.discount * this_Qmax

        return self.Q_table

class ActiveLearningPolicy(QValuePolicy):
    """
    A policy used for active learning of MDPs, based on the state-action count reward method of Araya-Lopéz et. al. (2012).
    """

    def __init__(self, env:VeriGymEnv, nr_states:int, nr_actions:int, abstraction_map=AbstractionMapper(), Q_init_strategy="zero", discount=0.95, epsilon=0.0):
        """
        Parameters
        ----------
        env : VeriGymEnv
            The environment to apply the policy to.
        nr_states : int
            The number of states in the environment.
        nr_actions : int
            The number of actions in the environment
        abstraction_map : AbstractionMapper
            Used in case that env is abstract. Default initializes to an identity map.
        Q_init_strategy: str
            How to initialize the Q_table. Default: "zero". 
            Further options: "random" -> random values.  "uniform": uniform across actions.
        discount : float
            Discount factor, used during refinement. Default 0.95.
        epsilon : float
            Exploration threshold for epsilon-greedy. Default 0.0 (no exploration).
        """
        super().__init__(env, nr_states, nr_actions, abstraction_map, Q_init_strategy, discount, epsilon=epsilon)
    
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_dict_counts, state_distr_counts):
        
        ### Construct environment
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
        self.Q_table = self._update_Q_table(R=R_learning, T=T)

        return self

class EntropyLearningPolicy(QValuePolicy):
    """
    A policy class for (iteratively) computing max-entropy policies, based on algorithm from Hazan et. al. (2019).
    """

    def __init__(self, env:VeriGymEnv, nr_states:int, nr_actions:int, abstraction_map=AbstractionMapper(), Q_init_strategy="zero", discount=0.95, learning_rate=0.2):
        """
        Parameters
        ----------
        env : VeriGymEnv
            The environment to apply the policy to.
        nr_states : int
            The number of states in the environment.
        nr_actions : int
            The number of actions in the environment
        abstraction_map : AbstractionMapper
            Used in case that env is abstract. Default initializes to an identity map.
        Q_init_strategy: str
            How to initialize the Q_table. Default: "zero". 
            Further options: "random" -> random values.  "uniform": uniform across actions.
        discount : float
            Discount factor, used during refinement. Default 0.95.
        learning_rate : float
            The learning rate. Default 0.2
        """

        self.tabular_policy = np.zeros((nr_states, nr_actions))
        self.learning_rate = learning_rate

        def policy(obs):
            return np.choice(self.nr_actions, self.tabular_policy[obs,:])
        
        super().__init__(env, nr_states, nr_actions, abstraction_map, Q_init_strategy, discount, epsilon=0.0)
        
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_dict_counts, state_distr_counts):
        ### Construct environment
        T_counts_copy, R_dict_counts_copy = deepcopy(T_counts), deepcopy(R_dict_counts)

        T, R, S_init = normalize_aggregated_counts(
            T_counts_copy, R_dict_counts_copy, P_tot_counts, state_distr_counts, self.nr_states, self.nr_actions
        )

        ### Construct reward function for learning
        T_pi = np.zeros((self.nr_states, self.nr_states))
        R_learning = RewardFunction(n_states=self.nr_states, n_actions=self.nr_actions)
        for sidx in T.T_dict.keys(): # loop only over explored states
            Tcount_s = T_counts[sidx]
            for aidx in range(self.nr_actions):
                for spidx in Tcount_s[aidx].keys():
                    T_pi[sidx,spidx] += self.tabular_policy[sidx,aidx] * Tcount_s[aidx][spidx]


        d_pi = (1-self.discount) * np.linalg.inv(np.eye(self.nr_states) - self.discount * T_pi) @ S_init

        for sidx in range(self.nr_states):
            R_learning[sidx][:] = - (np.log(d_pi[sidx]) + 1)
        
        ### Compute new Q-table
        self.Q_table = self._update_Q_table(R=R_learning.R_dict, T=T)

        ### Update policy
        for sidx in range(self.nr_states):
            self.tabular_policy[sidx, :] = (1-self.learning_rate) * self.tabular_policy[sidx, :] + self.learning_rate * scp.special.softmax(self.Q_table[sidx])
        return self