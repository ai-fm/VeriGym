import numpy as np

from verigym.abstraction.learn_abstraction import create_abstraction, learn_model_by_discretization

from verigym.environments.generativeenv import GenerativeEnv
from verigym.abstraction.discretization import generate_box_bins

from utils import (
    make_original_env,
)


def test_create_abstraction():
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    _abstracted_env = create_abstraction(
        original_env=generative_env,
        exploration_policy="random",
        num_steps=NUM_STEPS,
        bin_edges_per_dim=BIN_EDGES_PER_DIM
    )

def test_learn_model_by_discretization():
    env, NUM_STEPS, BIN_EDGES_PER_DIM = make_original_env()
    generative_env = GenerativeEnv.from_gymnasium(env)
    
    action_kwargs = {'bin_func':np.linspace, 'n_samples': 10}
    state_kwards = action_kwargs
    learn_model_by_discretization(
        original_env=generative_env,
        exploration_strategy=None,
        action_discretization_strategy=generate_box_bins,
        state_discretization_strategy=generate_box_bins,
        num_steps=10,
        action_kwargs=action_kwargs,
        state_kwargs=state_kwards,
        multithreading=False)