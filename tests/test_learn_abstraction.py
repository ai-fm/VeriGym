from verigym.abstraction.learn_abstraction import create_abstraction
from verigym.environments.generativeenv import GenerativeEnv
from verigym.policy.implemented_policies import RandomizedPolicy, ActiveLearningPolicy, EntropyLearningPolicy
from verigym.environments.learnedexplicitenv import *

from utils import (
    make_original_env,
)

def test_create_abstraction():
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    names = ["Random", "ActiveLearning", "EntropyLearning"]
    policies = [RandomizedPolicy, ActiveLearningPolicy, EntropyLearningPolicy]
    for idx in range(len(policies)):
        print(f" \ntesting {names[idx]} \n")
        abstracted_env = create_abstraction(
            original_env=generative_env,
            exploration_policy=policies[idx],
            num_steps=NUM_STEPS,
            bin_edges_per_dim=BIN_EDGES_PER_DIM,
            n_iterations=5,
            multithreading=True
        )

        S_init, T, R = abstracted_env.initial_states, abstracted_env.transition_function, abstracted_env.reward_function
        random_visited_state = list(S_init.keys())[0]
        assert np.isclose( sum(S_init.values()), 1.0)
        assert np.isclose( sum(T[random_visited_state][0].values()), 1.0)

def test_learned_explicit_env():
    learned_env = LearnedExplicitEnv(
        nr_states=2,
        nr_actions=2,
        nr_rewards=2,
        # These are unused...
        initial_state_distr={0:0.5, 1:0.5},
        transition_function=LearnedTransitionFunction(
            n_states=2, n_actions=2),  # only self-transitions
        reward_function=LearnedRewardFunction(
            n_states=2, n_actions=2),    # No rewards 
    )

    learned_env.update_env(
        new_init_counts={0:10,1:10},
        new_transition_counts={0:{0:{0:10,1:10},1:{0:10}},1:{0:{1:10},1:{1:10}}},
        new_reward_counts={0:{0:[0,0,1,1],1:[0]},1:{0:[0],1:[0]}}
    )
    assert learned_env.initial_states[0] == 0.5
    assert learned_env.transition_function.T_dict[0][0][0] == 0.5
    assert learned_env.reward_function.R_dict[0][0] == 0.5

    learned_env.update_env(
        new_init_counts={0:20},
        new_transition_counts={0:{0:{0:20}}},
        new_reward_counts={0:{0:[1,1,1,1]}}
    )
    assert learned_env.initial_states[0] == 0.75
    assert learned_env.transition_function.T_dict[0][0][0] == 0.75
    assert learned_env.reward_function.R_dict[0][0] == 0.75
