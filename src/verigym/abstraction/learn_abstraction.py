import logging
from typing import Literal
from math import prod

import gymnasium as gym
import numpy as np

from verigym.environments import ExplicitEnv, VeriGymEnv, GenerativeEnv
from verigym.abstraction.learn_transitions import learn_transition_function
from verigym.abstraction.learn_rewards import learn_reward_function
from verigym.abstraction.gym_utils.transform_observation import (
    DiscretizeBoxObservation,
)
from verigym.abstraction.discretization import (
    # centered_pow_bin,
    generate_box_bins,
)
from verigym.abstraction.utils import factored_to_index

logger = logging.getLogger(__name__)


def create_abstraction(
    original_env: VeriGymEnv,
    exploration_strategy: Literal["random", "sb3 policy"],
    num_steps: int,
    bin_edges_per_dim: int | list[int],
    verbose: bool = False,
) -> ExplicitEnv:
    """
    Creates an abstraction from a VeriGymEnv by discretizing the state space. Returns an `ExplicitEnv`.
    Action space is currently not abstracted.


    Parameters
    ----------
    original_env : VeriGymEnv
        The environment / model to be abstracted.
    exploration_strategy : Literal[&quot;random&quot;, &quot;sb3 policy&quot;]
        The strategy of interacting with the `original_env` (e.g. a random policy).
    num_steps : int
        Number of steps to take within the environment.
    bin_edges_per_dim : int | list[int]
        Number of discretization bins per feature dimension of the state space.
    verbose : bool, optional
        Whether to be verbose, by default False.

    Returns
    -------
    ExplicitEnv
        The abstracted model.
    """
    assert isinstance(original_env, gym.Env), (
        f"original_env is type {type(original_env)} and does not inherit from gym.Env"
    )

    # discretize space
    bin_edges = generate_box_bins(
        original_env.observation_space, np.linspace, bin_edges_per_dim
    )
    logger.info(f"bin_edges: {bin_edges}")
    logger.info(f"num states: {prod([len(dimension) + 1 for dimension in bin_edges])}")
    discretized_env = DiscretizeBoxObservation(
        original_env, bin_edges=bin_edges, use_box_space=True
    )

    # discretize actions
    # TODO: Currently we assume that the action space is already discrete and starts at 0. We should add a wrapper to discretize the action space if this is not the case. For now, we just check that the action space is compatible and warn if it isn't.
    assert isinstance(original_env.action_space, gym.spaces.Discrete), (
        f"Currently only Discrete action spaces are supported but found {original_env.action_space}"
    )
    logger.warning(
        "Currently only Discrete action spaces are supported, so no discretization is applied to the action space."
    )
    if original_env.action_space.start != 0:
        logger.warning(
            f"Action space starts at {original_env.action_space.start} instead of 0. This might cause issues with the current implementation as we expect actions to be integers starting from 0."
        )

    # Convert into VeriGym compatible object
    generative_env = GenerativeEnv.from_gymnasium(discretized_env)

    dataset = generative_env.simulate(policy=exploration_strategy, n_steps=num_steps)

    # convert states to indices
    dataset_indices = []
    for trajectory in dataset:
        trajectory_indices = []
        for s, a, r, s_next in trajectory:
            s_index = factored_to_index(bin_edges, s)
            assert s_index >= 1, f"s_index should be >= 1 but got {s_index}"
            s_next_index = factored_to_index(bin_edges, s_next)
            trajectory_indices.append((s_index, a, r, s_next_index))
        dataset_indices.append(trajectory_indices)

    # approximate the transition function
    n_actions = original_env.action_space.n
    n_states = prod([len(dimension) for dimension in bin_edges])
    T = learn_transition_function(
        dataset=dataset_indices, n_states=n_states, n_actions=n_actions
    )

    # approximate the reward function
    R = learn_reward_function(
        dataset=dataset_indices, n_states=n_states, n_actions=n_actions
    )

    # Construct the abstracted ExplicitEnv
    abstracted_env = ExplicitEnv(
        nr_states=n_states,
        nr_actions=n_actions,
        nr_rewards=1,  # TODO rename + compatability for multi objective gym envs
        initial_state_distr={0: 1.0},  # TODO
        transition_function=T,
        reward_function=R,
        abstraction_map=None,  # TODO
        original_env=original_env,
        render_mode=None,
    )

    return abstracted_env
