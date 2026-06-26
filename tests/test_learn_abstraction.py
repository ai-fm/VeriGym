from verigym.abstraction.learn_abstraction import create_abstraction

from verigym.environments.generativeenv import GenerativeEnv

from verigym.policy.policy import RandomizedPolicy

from utils import (
    make_original_env,
)


def test_create_abstraction():
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    _abstracted_env = create_abstraction(
        original_env=generative_env,
        exploration_policy=RandomizedPolicy(generative_env),
        num_steps=NUM_STEPS,
        bin_edges_per_state_dim=BIN_EDGES_PER_DIM,
        bin_edges_per_action_dim=BIN_EDGES_PER_DIM,
    )


# Test the interleaving abstraction learning
class RandomizedPolicyTest(RandomizedPolicy):
    iterations: int = 0
    
    def __init__(self, env):
        super().__init__(env)
    
    def update_for_abstraction_refinement(
        self, dataset, T_counts, P_tot, R_counts, S_init_counts
    ) -> "RandomizedPolicyTest":
        self.iterations += 1
        return self


def test_policy_call():
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    N_ITERATIONS = 5
    
    policy = RandomizedPolicyTest(generative_env)
    _abstracted_env = create_abstraction(
        original_env=generative_env,
        exploration_policy=policy,
        num_steps=NUM_STEPS,
        bin_edges_per_state_dim=BIN_EDGES_PER_DIM,
        bin_edges_per_action_dim=BIN_EDGES_PER_DIM,
        n_iterations=N_ITERATIONS,
    )
    
    assert policy.iterations == N_ITERATIONS
    