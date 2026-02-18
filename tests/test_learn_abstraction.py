from verigym.abstraction.learn_abstraction import create_abstraction

from verigym.environments.generativeenv import GenerativeEnv

from utils import (
    make_original_env,
)


def test_create_abstraction():
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    _abstracted_env = create_abstraction(
        original_env=generative_env,
        exploration_strategy="random",
        num_steps=NUM_STEPS,
        bin_edges_per_dim=BIN_EDGES_PER_DIM
    )
