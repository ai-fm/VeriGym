import numpy as np
from verigym.environments.verigymenv import VeriGymEnv
from verigym.environments.explicitenv import ExplicitEnv

import gymnasium.spaces

from typing import Callable

from numpy.typing import NDArray

from verigym.environments.verigymenv import VeriGymEnv


def identity_map(x):
    return x


class AbstractionMap:
    def __init__(
        self,
        forward_map: Callable[[NDArray], int],
        backward_map: Callable[[int], NDArray] = None,
    ):
        self.forward_map = forward_map
        self.backward_map = backward_map
        self.has_backward_map = self.backward_map is not None


class IdentityAbstractionMap(AbstractionMap):
    def __init__(self):
        super().__init__(identity_map, identity_map)


class AbstractionMapper:
    """
    Provides a mapping between abstract and original spaces.

    TODO: Currently (in the MVP) only set up to support the mapping of continuous Gym states (numpy arrays) <-> abstract states (discrete integers).
    """

    def __init__(
        self,
        original_env: VeriGymEnv,
        abstract_env: ExplicitEnv,
        state_abstraction_map: AbstractionMap,
        action_abstraction_map: AbstractionMap = IdentityAbstractionMap(),
    ):
        self.original_env = original_env
        self.abstract_env = abstract_env

        self.state_abstraction_map = state_abstraction_map
        self.action_abstraction_map = action_abstraction_map

        self.nr_abstract_states = abstract_env.nr_states
        self.nr_abstract_actions = abstract_env.nr_actions
        self.nr_abstract_rewards = abstract_env.nr_rewards

        self.action_mask = abstract_env.action_mask

    def abstract_to_original_state(self, abs_state: int) -> NDArray:
        """
        Maps an abstract state to a set/range of original states.

        Parameters
        ----------
        abs_state : object
            A state in self.abstract_env
        Returns
        ----------
        orig_states : object
            A set/range of states in self.original_env
        """
        if self.state_abstraction_map.has_backward_map:
            return self.state_abstraction_map.backward_map(abs_state)
        else:
            raise ValueError(
                "Cannot map abstract state to original state without a backward map in the state abstraction."
            )

    def original_to_abstract_state(self, orig_state: NDArray) -> int:
        """
        Maps an original state to an abstract state.

        Parameters
        ----------
        orig_state : object
            A state in self.original_env

        Returns
        -------
        abs_state : object
            A state in self.abstract_env
        """
        abs_state = self.state_abstraction_map.forward_map(orig_state)
        if isinstance(abs_state, np.ndarray):
            assert abs_state.size == 1
            abs_state = abs_state.item()

        if isinstance(abs_state, (np.int64, int)):
            return abs_state
        else:
            raise ValueError("?")

    def original_to_abstract_action(self, orig_action: NDArray) -> int:
        """
        Maps an action in self.original_env to an action in self.abstract_env

        Parameters
        ----------
        orig_action : object
            An action in the original environment.

        Returns
        -------
        abs_action : object
            A set/range of actions in the abstract environment.
        """
        return self.action_abstraction_map.forward_map(orig_action)

    def abstract_to_original_action(self, abs_action: int) -> NDArray:
        """
        Maps an action in self.abstract_env to a set/range of actions in self.original_env

        Parameters
        ----------
        abs_action : object
            An action in the abstract environment

        Returns
        -------
        orig_actions : object
            A set/range of actions in the original environment
        """
        if self.action_abstraction_map.has_backward_map:
            return self.action_abstraction_map.backward_map(abs_action)
        else:
            raise ValueError(
                "Cannot map abstract action to original action without a backward map in the action abstraction."
            )
