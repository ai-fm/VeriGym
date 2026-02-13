import logging
from collections import defaultdict

from numpy.typing import NDArray
import numpy as np

from verigym.environments.reward_func import RewardFunction

logger = logging.getLogger(__name__)


def learn_reward_function(
    dataset: list[tuple[NDArray, NDArray, NDArray]], n_states: int, n_actions: int
) -> RewardFunction:
    """
    Learn the transition function T using a frequentist approach.

    We expect the observations and actions in the dataset (tuples of state, action, next_state) to be integers that can be used as indices.
    This can be achieved by using the `DiscretizeBoxObservation` wrapper with argument `use_box_space=False` for gym environments with continuous spaces and then using the `factored_to_index` function.

    Uses hashmaps (dictionaries) as a proxy for sparse matrices to store the count table.
    """

    R_dict = defaultdict(lambda: defaultdict(lambda: list()))

    # TODO: Could this be sped up? Vectorization? Not sure. But it also isn't too slow at the moment.
    for trajectory in dataset:
        for s, a, r, s_next in trajectory:
            s, a, r = s.item(), a.item(), r.item()
            R_dict[s][a].append(r)

    # Take mean reward for each state-action pair
    for s in R_dict:
        for a in R_dict[s]:
            R_dict[s][a] = np.mean(R_dict[s][a])

    return RewardFunction(n_states, n_actions, R_dict)
