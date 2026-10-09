from .policy import PolicyClass
import numpy as np
import scipy as scp
import scipy.sparse
import scipy.sparse.linalg
from collections import defaultdict
from ..abstraction.abstractionmapper import AbstractionMapper
from ..environments.reward_func import RewardFunction
from ..environments.verigymenv import VeriGymEnv
from ..abstraction.learn_abstraction import normalize_aggregated_counts

class ValueTable:
    """
    Sparse table mapping states to rows of per-action values (e.g. Q-values or policy probabilities).
    Only explicitly written states are stored; all other states return `default_value`.
    """
    nr_states:int
    nr_actions:int
    default_value:float
    Q_dict:dict[int, np.ndarray]

    def __init__(self, nr_states, nr_actions, default_value=0):
        self.nr_states = nr_states
        self.nr_actions = nr_actions
        self.default_value = default_value
        self.Q_dict = {}

    def __getitem__(self, key):
        """
        Read Q-values of a state or state-action pair. 
        Returns default values and writes no new data if key is not present.
        """
        # Writing for an (s,a) pair
        if isinstance(key, tuple):
            s, a = key
            row = self.Q_dict.get(s)
            return self.default_value if row is None else row[a]

        # Writing for a state
        else:
            row = self.Q_dict.get(key)
            if row is None:
                row = np.full(self.nr_actions, self.default_value, dtype=float)
                row.flags.writeable = False
            return row

    def __setitem__(self, key, value):
        """Set Q-value of a state or state-action pair."""
        if isinstance(key, tuple):
            s, a = key
            if s not in self.Q_dict:
                self.Q_dict[s] = np.full(self.nr_actions, self.default_value, dtype=float)
            self.Q_dict[s][a] = value
        else:
            row = np.asarray(value, dtype=float)
            assert row.shape == (self.nr_actions,), f"Expected shape ({self.nr_actions},), got {row.shape}"
            self.Q_dict[key] = row.copy()

class QValuePolicy(PolicyClass):
    """
    A native MDP policy class that selects actions based on (approximate) Q-values.

    This policy class allows for static epsilon-greedy (no annealing).
    """

    def __init__(self, env:VeriGymEnv, abstraction_mapper:AbstractionMapper, 
                 Q_init_strategy="zero", 
                 discount=0.95, epsilon=0.0,
                 update_iterations = 25,
                 ):
        """
        Parameters
        ----------
        env : VeriGymEnv
            The environment to apply the policy to.
        abstraction_map : AbstractionMapper
            Used in case that env is abstract.
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
        self.abstraction_mapper = abstraction_mapper

        self.epsilon_random = epsilon

        self.nr_iterations = update_iterations

        self.nr_states = abstraction_mapper.abstract_n_states
        self.nr_actions = abstraction_mapper.abstract_n_actions

        # elif Q_init_strategy == "random":
        #     self.Q_table = np.random.rand(self.nr_states, self.nr_actions)
        #     self.Q_table /= self.Q_table.sum(axis=1, keepdims=True)
        if Q_init_strategy == "uniform":
            self.Q_table = ValueTable(self.nr_states, self.nr_actions, 1 / self.nr_actions)
        else:
            if Q_init_strategy != "zero":
                print("Warning: initialization of Q-table not recognized")
            self.Q_table = ValueTable(self.nr_states, self.nr_actions, 0)

        def policy(obs):
            if self.epsilon_random < 1.0 and np.random.rand() > self.epsilon_random:
                p=scp.special.softmax(self.Q_table[obs][:])
            else:
                p = np.ones(self.nr_actions) / self.nr_actions
            return np.random.choice(a=self.nr_actions, p=p)
        
        super().__init__(policy, abstraction_mapper)
    
    def _action_from_policy(self, obs):
        obs_enum = self.abstraction_mapper._state_abstraction_map.abstract_to_enum(obs) 
        action_enum =  self.policy(obs_enum)
        return self.abstraction_mapper._action_abstraction_map.enum_to_abstract(action_enum)
    
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_dict_counts, state_distr_counts):
        T, R, S_init = normalize_aggregated_counts(
            T_counts_copy, R_dict_counts_copy, P_tot_counts, state_distr_counts, self.nr_states, self.nr_actions
        )

        self.Q_table = self._update_Q_table(R=R, T=T)

        return self

    def _update_Q_table(self, R, T, R_unvisited = 0.0):
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

        Qmax = defaultdict(lambda: self.Q_table.default_value)
        for sidx in T.T_dict.keys():
            Qmax[sidx] = max(self.Q_table[sidx][:])

        # Updates:
        for _ in range(self.nr_iterations):
            for (sidx, Ts) in T.T_dict.items():
                this_Qmax = -np.inf
                for aidx in range(self.nr_actions):
                    Ts_a = Ts.get(aidx, {})
                    if aidx in Ts:
                        this_Q = R[sidx][aidx]
                    else:
                        this_Q = R_unvisited / (1-self.discount)
                    for (spidx, prob) in Ts_a.items():
                        this_Q += self.discount * prob * Qmax[spidx]
                    self.Q_table[sidx, aidx] = this_Q
                    this_Qmax = max(this_Qmax, this_Q)
                Qmax[sidx] = this_Qmax

        return self.Q_table

class ActiveLearningPolicy(QValuePolicy):
    """
    A policy used for active learning of MDPs, based on the state-action count reward method of Araya-Lopéz et. al. (2012).
    """

    def __init__(self, env:VeriGymEnv, abstraction_mapper:AbstractionMapper, Q_init_strategy="zero", discount=0.95, epsilon=0.0, update_iterations=25):
        """
        Parameters
        ----------
        env : VeriGymEnv
            The environment to apply the policy to.
        abstraction_map : AbstractionMapper
            Used in case that env is abstract.
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
        super().__init__(env, abstraction_mapper, Q_init_strategy, discount, epsilon, update_iterations)
    
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_dict_counts, state_distr_counts):
        
        ### Construct environment
        T, R, S_init = normalize_aggregated_counts(
            T_counts_copy, R_dict_counts_copy, P_tot_counts, state_distr_counts, self.nr_states, self.nr_actions
        )
        Rmax = 1

        ### Construct reward function for learning
        R_learning = defaultdict(lambda: defaultdict(lambda: Rmax))
        for sidx in T.T_dict.keys(): # loop only over explored states
            for aidx in range(self.nr_actions):
                this_count = P_tot_counts.get((sidx,aidx), 0)
                if this_count > 0:
                    R_learning[sidx][aidx] = 1 / this_count
        
        ### Update Q-table
        self.Q_table = self._update_Q_table(R=R_learning, T=T, R_unvisited=Rmax)

        return self

class EntropyLearningPolicy(QValuePolicy):
    """
    A policy class for (iteratively) computing max-entropy policies, based on algorithm from Hazan et. al. (2019).
    """

    def __init__(self, env:VeriGymEnv, abstraction_mapper:AbstractionMapper, Q_init_strategy="zero", discount=0.95, learning_rate=0.2, update_iterations=25):
        """
        Parameters
        ----------
        env : VeriGymEnv
            The environment to apply the policy to.
        abstraction_map : AbstractionMapper
            Used in case that env is abstract.
        Q_init_strategy: str
            How to initialize the Q_table. Default: "zero". 
            Further options: "random" -> random values.  "uniform": uniform across actions.
        discount : float
            Discount factor, used during refinement. Default 0.95.
        learning_rate : float
            The learning rate. Default 0.2
        update_iterations : int
            The number of iterations to update the Q_table for during abstraction refinement.
        """

        # Sparse: states that were never updated keep the uniform default policy
        super().__init__(env, abstraction_mapper, Q_init_strategy, discount, 0.0, update_iterations)
        self.tabular_policy = ValueTable(self.nr_states, self.nr_actions, default_value=1/self.nr_actions)
        self.learning_rate = learning_rate

        def policy(obs):
            return np.random.choice(a=self.nr_actions, p=self.tabular_policy[obs])
        self.policy = policy


    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_dict_counts, state_distr_counts):
        ### Construct environment
        T, _, S_init = normalize_aggregated_counts(
            T_counts_copy, R_dict_counts_copy, P_tot_counts, state_distr_counts, self.nr_states, self.nr_actions
        )

        ### Tabulate all reached states for sparse representation
        S_sparse = set(T.T_dict.keys()) | set(int(s) for s in np.flatnonzero(S_init))
        for Ts in T.T_dict.values():
            for Ts_a in Ts.values():
                S_sparse.update(Ts_a.keys())
        S_sparse = list(S_sparse)
        nr_states_sparse = len(S_sparse)
        index_to_sparse_index = {s: i for i, s in enumerate(S_sparse)}

        ### Build transition function as sparse scipy array
        rows, cols, transition_values = [], [], []
        for (sidx, Ts) in T.T_dict.items():
            for (aidx, Ts_a) in Ts.items():
                for spidx, prob in Ts_a.items():
                    rows.append(index_to_sparse_index[sidx])
                    cols.append(index_to_sparse_index[spidx])
                    transition_values.append(self.tabular_policy[sidx, aidx] * prob)
        T_pi = scipy.sparse.coo_matrix((transition_values, (rows, cols)), shape=(nr_states_sparse, nr_states_sparse)).tocsr() # duplicates are summed

        ### Compute density function
        A = scipy.sparse.identity(nr_states_sparse, format="csc") - self.discount * T_pi.T.tocsc()
        d_sparse = np.atleast_1d((1-self.discount) * scipy.sparse.linalg.spsolve(A, S_init[S_sparse]))

        ### Construct reward function for learning (following Hazam et. al. (2019))
        Rmax = 100
        R_learning = RewardFunction(n_states=self.nr_states, n_actions=self.nr_actions)
        with np.errstate(divide="ignore"): # devision-by-zero errors can occur for non-visited states, but are caught by using min(Rmax, ...)
            for (sidx, Ts) in T.T_dict.items():
                R_s = min(Rmax, -(np.log(d_sparse[index_to_sparse_index[sidx]]) + 1))
                for aidx in Ts.keys():
                    R_learning[sidx][aidx] = R_s

        ### Compute new Q-table
        self.Q_table = self._update_Q_table(R=R_learning.R_dict, T=T, R_unvisited=Rmax)

        ### Update policy
        # States without a stored Q-row have softmax(Q) uniform, so their (uniform) policy is unchanged
        for sidx in self.Q_table.Q_dict.keys():
            self.tabular_policy[sidx] = (1-self.learning_rate) * self.tabular_policy[sidx] + self.learning_rate * scp.special.softmax(self.Q_table[sidx])
        return self