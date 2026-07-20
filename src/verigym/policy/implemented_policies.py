from .policy import PolicyClass
import numpy as np
import scipy as scp
import numbers
from ..abstraction.learn_abstraction import normalize_aggregated_counts
from ..abstraction.abstractionmapper import AbstractionMapper


class RandomizedPolicy(PolicyClass):
    """
    A policy that returns random actions, as sampled from the provided environment.
    Works for every class inheriting from `VeriGymEnv` (and therefore `gym.Env`).
    """

    def __init__(self, env: "VeriGymEnv"):
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

    def __init__(self, env:"ExplicitEnv", map:AbstractionMapper()):
        self.Q_table = None
        self.env = env
        self.discount = 0.95 # TODO: where do we get this?
        self.map = map

        def policy(obs):
            # print(f"o={obs}, actions={self.env.nr_actions}, Q={self.Q_table}")
            if self.Q_table is None:
                return np.random.choice(self.env.nr_actions)
            return np.random.choice(a=self.env.nr_actions, p=scp.special.softmax(self.Q_table[obs,:]))
        
        return super().__init__(policy, map)
    
    def _action_from_policy(self, obs):
        return self.policy(obs)
    
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot, R_counts, S_init_counts):
        
        ### Extract model parameters: 
        ### TODO: make this part of abstraction learning?
        T, R, _S_init = normalize_aggregated_counts(
            T_counts, R_counts, P_tot, S_init_counts, self.env.n_states, self.env.n_actions
        )

        ### Update Q-table
        self.Q_table = update_Q_table(self.env, self.Q_table, T,R)

        return self

class ActiveLearningPolicy(QValuePolicy):
    """
    A policy used for active learning of MDPs, based on the state-action count reward method of Araya-Lopéz et. al. (2012).
    """

    def __init__(self, env:"ExplicitEnv", map=AbstractionMapper()):
        return super().__init__(env, map)
    
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot, R_counts, S_init_counts):
        nr_states, nr_actions = len(T_counts), len(T_counts[0])
        
        ### Extract model parameters:
        T, R, _S_init = normalize_aggregated_counts(
            T_counts, R_counts, P_tot, S_init_counts, nr_states, nr_actions
        )

        ### Construct reward function for learning
        for sidx in range(len(T_counts)):
            for aidx in range(len(T_counts[sidx])):
                if T_counts[sidx][aidx] < 1:
                    R[sidx][aidx] = 10_000
                else:
                    R[sidx][aidx] = 1 / T_counts[sidx][aidx]
        
        ### Update Q-table
        self.Q_table = update_Q_table(self.env, self.Q_table, T,R)

        return self

class EntropyLearningPolicy(PolicyClass):
    """
    A policy class for (iteratively) computing max-entropy policies, based on algorithm from Hazan et. al. (2019).
    
    """

    def __init__(self, env:"ExplicitEnv"):
        self.Q_table = np.zeros((env.nr_states, env.nr_actions))
        self.tabular_policy = np.zeros((env.nr_states, self.nr_actions))
        self.env = env
        self.discount = 0.95 # TODO: where do we get this?
        self.learning_rate = 0.2
        abstraction_mapper = AbstractionMapper()  # Identity mapping

        def policy(obs):
            return np.choice(self.env.nr_actions, self.tabular_policy[obs,:])
        
        return super().__init__(policy, abstraction_mapper)
        
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot, R_counts, S_init_counts):
        nr_states, nr_actions = len(T_counts), len(T_counts[0])

        ### Extract model parameters:
        T, R, S_init = normalize_aggregated_counts(
            T_counts, R_counts, P_tot, S_init_counts, nr_states, nr_actions
        )
        

        ### Construct reward function for learning
        # TODO: make this sparse!
        T_pi = np.zeros(nr_states, nr_states)
        for sidx in range(nr_states):
            for aidx in range(nr_actions):
                for spidx in T[sidx][aidx].keys():
                    T_pi[sidx,spidx] += self.tabular_policy[sidx,aidx] * T[sidx][aidx][spidx]
    
        d_pi = (1-self.discount) * (np.eye(nr_states) - self.discount *T_pi)**(-1) * S_init

        for sidx in range(nr_states):
            R[sidx][:] = - (np.log(d_pi[sidx]) + 1)
        
        ### Compute new Q-table
        self.Q_table = update_Q_table(self.env, self.Q_table, T,R)

        ### Update policy
        for sidx in range(nr_states):
            self.tabular_policy[sidx, :] = (1-self.self.learning_rate) * self.tabular_policy[sidx, :] + self.self.learning_rate * scp.special.softmax(self.Q_table)

        return self
    
    
def update_Q_table(env, Q_table, T, R, nr_iterations = 100, discount=0.95):
    nr_iterations = 1_000
    nr_states, nr_actions = env.nr_states, env.nr_actions
    if Q_table is None:
        Q_table = np.zeros((nr_states, nr_actions))

    for i in range(nr_iterations):
        for sidx in range(nr_states):
            for aidx in range(nr_actions):
                if not isinstance(R[sidx][aidx], numbers.Number):
                    continue
                Q_table[sidx,aidx] = 0
                for spidx in T[sidx][aidx].keys():
                    p_spidx = T[sidx][aidx][spidx]
                    Q_table[sidx, aidx] += p_spidx * np.max(Q_table[spidx,:])
                Q_table[sidx,aidx] = R[sidx][aidx] + discount * Q_table[sidx,aidx]
    return Q_table