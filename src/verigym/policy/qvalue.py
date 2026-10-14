from .policy import PolicyClass
import numpy as np
import scipy as scp
import scipy.sparse
import scipy.sparse.linalg
from collections import defaultdict
from typing import NamedTuple
from ..abstraction.abstractionmapper import AbstractionMapper
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

class SparseModel(NamedTuple):
    """Sparse representation of a TransitionFunction over the states it reaches. See `make_sparse_representations`."""
    S_sparse: list[int]
    index_to_sparse_index: dict[int, int]
    n_explored: int
    P: scipy.sparse.csr_matrix
    visited: np.ndarray

def make_sparse_representations(T, nr_actions, extra_states=()):
    """
    Build sparse representations of a transition function, indexing only the states that occur in it.

    Parameters
    ----------
    T : TransitionFunction
        Transition function (e.g. as estimated by `normalize_aggregated_counts`).
    nr_actions : int
        Number of actions.
    extra_states : iterable of int
        States to index in addition to those in T (e.g. initial states).

    Returns
    -------
    SparseModel
        S_sparse : sparse index -> state. Explored states (keys of T.T_dict) come first, at indices 0..n_explored-1,
            followed by successors that were never explored and the extra states.
        index_to_sparse_index : state -> sparse index.
        n_explored : number of explored states.
        P : csr matrix of shape (n_explored * nr_actions, len(S_sparse)), with P[i*nr_actions + a, j] the probability
            of moving from S_sparse[i] to S_sparse[j] under action a. Rows of untried actions are empty.
        visited : bool array of shape (n_explored, nr_actions), True for tried state-action pairs.
    """
    S_sparse = list(T.T_dict.keys())
    n_explored = len(S_sparse)
    index_to_sparse_index = {s: i for i, s in enumerate(S_sparse)}

    visited = np.zeros((n_explored, nr_actions), dtype=bool)
    rows, cols, transition_values = [], [], []
    for (i, Ts) in enumerate(T.T_dict.values()):
        for (aidx, Ts_a) in Ts.items():
            visited[i, aidx] = True
            for (spidx, prob) in Ts_a.items():
                rows.append(i * nr_actions + aidx)
                cols.append(index_to_sparse_index.setdefault(spidx, len(index_to_sparse_index)))
                transition_values.append(prob)
    for s in extra_states:
        index_to_sparse_index.setdefault(s, len(index_to_sparse_index))
    S_sparse = list(index_to_sparse_index.keys())

    P = scipy.sparse.csr_matrix(
        (transition_values, (rows, cols)), shape=(n_explored * nr_actions, len(S_sparse))
    ) # duplicates are summed
    return SparseModel(S_sparse, index_to_sparse_index, n_explored, P, visited)

class QValuePolicy(PolicyClass):
    """
    A native MDP policy class that selects actions based on (approximate) Q-values.

    This policy class allows for static epsilon-greedy (no annealing).
    """

    def __init__(self, env:VeriGymEnv, abstraction_mapper:AbstractionMapper, 
                 Q_init_strategy="zero", 
                 discount=0.95, epsilon=0.0,
                 update_iterations = 25,
                 seed: int | None = None
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
        seed : int
            Initial seed used for rng calls
        """
        self.env = env
        self.discount = discount
        self.abstraction_mapper = abstraction_mapper

        self.epsilon_random = epsilon

        self.nr_iterations = update_iterations

        self.nr_states = abstraction_mapper.abstract_n_states
        self.nr_actions = abstraction_mapper.abstract_n_actions
        self.rng = np.random.default_rng(seed)

        if Q_init_strategy == "uniform":
            self.Q_table = ValueTable(self.nr_states, self.nr_actions, 1 / self.nr_actions)
        else:
            if Q_init_strategy != "zero":
                print("Warning: initialization of Q-table not recognized")
            self.Q_table = ValueTable(self.nr_states, self.nr_actions, 0)

        def policy(obs):
            if self.epsilon_random < 1.0 and self.rng.random() > self.epsilon_random:
                p=scp.special.softmax(self.Q_table[obs][:])
            else:
                p = np.ones(self.nr_actions) / self.nr_actions
            return self.rng.choice(a=self.nr_actions, p=p)
        
        super().__init__(policy, abstraction_mapper)
    
    def _action_from_policy(self, obs):
        obs_enum = self.abstraction_mapper._state_abstraction_map.abstract_to_enum(obs) 
        action_enum =  self.policy(obs_enum)
        return self.abstraction_mapper._action_abstraction_map.enum_to_abstract(action_enum)
    
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_cumsum, state_distr_counts):
        T, R, S_init = normalize_aggregated_counts(
            T_counts, R_cumsum, P_tot_counts, state_distr_counts, self.nr_states, self.nr_actions
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
        sparse_model = make_sparse_representations(T, self.nr_actions)

        ### Rewards R[s, a] for explored states; actions that were never tried get R_unvisited forever
        R_sparse = np.full((sparse_model.n_explored, self.nr_actions), R_unvisited / (1-self.discount), dtype=float)
        for (sidx, Ts) in T.T_dict.items():
            for aidx in Ts.keys():
                R_sparse[sparse_model.index_to_sparse_index[sidx], aidx] = R[sidx][aidx]

        return self._update_Q_table_sparse(R_sparse, sparse_model)

    def _update_Q_table_sparse(self, R_sparse, sparse_model):
        """
        Value iteration (Jacobi) on a sparse model, warm-started from the current Q-table.
        Only the rows of explored states are written; sinks keep the default value of the Q-table.

        Parameters
        ----------
        R_sparse : np.ndarray
            Rewards of shape (sparse_model.n_explored, nr_actions), indexed by sparse state index.
        sparse_model : SparseModel
            Sparse transition model, see `make_sparse_representations`.
        """
        n_explored = sparse_model.n_explored
        if n_explored == 0 or self.nr_iterations == 0:
            return self.Q_table

        Qmax = np.full(len(sparse_model.S_sparse), self.Q_table.default_value, dtype=float)
        Qmax[:n_explored] = [self.Q_table[sidx].max() for sidx in sparse_model.S_sparse[:n_explored]]
        for _ in range(self.nr_iterations):
            Q = R_sparse + self.discount * (sparse_model.P @ Qmax).reshape(n_explored, self.nr_actions)
            Qmax[:n_explored] = Q.max(axis=1)

        for (i, sidx) in enumerate(sparse_model.S_sparse[:n_explored]):
            self.Q_table[sidx] = Q[i]

        return self.Q_table

class ActiveLearningPolicy(QValuePolicy):
    """
    A policy used for active learning of MDPs, based on the state-action count reward method of Araya-Lopéz et. al. (2012).
    """

    def __init__(self, env:VeriGymEnv, abstraction_mapper:AbstractionMapper, Q_init_strategy="zero", discount=0.95, epsilon=0.0, update_iterations=25, seed=None):
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
        super().__init__(env, abstraction_mapper, Q_init_strategy, discount, epsilon, update_iterations, seed)
    
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_cumsum, state_distr_counts):
        
        ### Construct environment
        T, R, S_init = normalize_aggregated_counts(
            T_counts, R_cumsum, P_tot_counts, state_distr_counts, self.nr_states, self.nr_actions
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

    def __init__(self, env:VeriGymEnv, abstraction_mapper:AbstractionMapper, Q_init_strategy="zero", discount=0.95, learning_rate=0.2, update_iterations=25, seed = None):
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
        super().__init__(env, abstraction_mapper, Q_init_strategy, discount, 0.0, update_iterations, seed)
        self.tabular_policy = ValueTable(self.nr_states, self.nr_actions, default_value=1/self.nr_actions)
        self.learning_rate = learning_rate

        def policy(obs):
            return self.rng.choice(a=self.nr_actions, p=self.tabular_policy[obs])
        self.policy = policy


    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot_counts, R_cumsum, state_distr_counts):
        ### Construct environment
        T, _, S_init = normalize_aggregated_counts(
            T_counts, R_cumsum, P_tot_counts, state_distr_counts, self.nr_states, self.nr_actions
        )

        ### Sparse representation over all reached states, including initial states
        sparse_model = make_sparse_representations(T, self.nr_actions, extra_states=S_init.distr_dict.keys())
        nr_states_sparse = len(sparse_model.S_sparse)
        n_explored = sparse_model.n_explored

        ### Policy-induced transitions T_pi[s, s'] = sum_a pi(a|s) P[(s,a), s']
        pi_explored = np.array([self.tabular_policy[sidx] for sidx in sparse_model.S_sparse[:n_explored]]).reshape(-1)
        Pi = scipy.sparse.csr_matrix(
            (pi_explored, (np.repeat(np.arange(n_explored), self.nr_actions), np.arange(n_explored * self.nr_actions))),
            shape=(nr_states_sparse, n_explored * self.nr_actions)
        )
        T_pi = Pi @ sparse_model.P

        ### Compute density function
        A = scipy.sparse.identity(nr_states_sparse, format="csc") - self.discount * T_pi.T.tocsc()
        S_init_array = np.array([S_init[s] for s in sparse_model.S_sparse])
        d_sparse = np.atleast_1d((1-self.discount) * scipy.sparse.linalg.spsolve(A, S_init_array))

        ### Construct reward function for learning (following Hazam et. al. (2019)); untried actions get Rmax forever
        Rmax = 100
        with np.errstate(divide="ignore"): # devision-by-zero errors can occur for non-visited states, but are caught by using min(Rmax, ...)
            R_s = np.minimum(Rmax, -(np.log(d_sparse[:n_explored]) + 1))
        R_sparse = np.where(sparse_model.visited, R_s[:, None], Rmax / (1-self.discount))

        ### Compute new Q-table
        self.Q_table = self._update_Q_table_sparse(R_sparse, sparse_model)

        ### Update policy
        # States without a stored Q-row have softmax(Q) uniform, so their (uniform) policy is unchanged
        for sidx in self.Q_table.Q_dict.keys():
            self.tabular_policy[sidx] = (1-self.learning_rate) * self.tabular_policy[sidx] + self.learning_rate * scp.special.softmax(self.Q_table[sidx])
        return self