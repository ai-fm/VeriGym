import numpy as np

from verigym.utils.utils import identity_map

from typing import Callable

from numpy.typing import NDArray


class AbstractionMap:
    """
    This class consists of functions mapping between continuous (numpy) and abstract (integer) spaces.
    As such, it can be used for mapping both state and action spaces.
    While a forward map is required, it is not always possible (or obvious how) to define a backward map.
    """

    def __init__(
        self,
        forward_map: Callable[[NDArray], int],
        backward_map: Callable[[int], NDArray] = None,
    ):
        """Constructor

        Parameters
        ----------
        forward_map : Callable[[NDArray], int]
            A callable function representing a forward mapping of continuous spaces (in numpy format) to abstract (integer) spaces
        backward_map : Callable[[int], NDArray], optional
            A callable function representing a backward mapping of abstract (integer) spaces to continuous spaces (in numpy format) if available, by default None
        """
        self.forward_map = forward_map
        self.backward_map = backward_map
        self.has_backward_map = self.backward_map is not None


class IdentityAbstractionMap(AbstractionMap):
    def __init__(self):
        super().__init__(identity_map, identity_map)


class AbstractionMapper:
    def __init__(
        self,
        state_abstraction_map: AbstractionMap = IdentityAbstractionMap(),
        action_abstraction_map: AbstractionMap = IdentityAbstractionMap(),
    ):
        """
        Provides a mapping between abstract and original spaces.

        TODO: Currently (in the MVP) only set up to support the mapping of continuous Gym states (numpy arrays) <-> abstract states (discrete integers).

        Parameters
        ----------
        state_abstraction_map : AbstractionMap, optional
            Mapping between original and abstract states, by default IdentityAbstractionMap()
        action_abstraction_map : AbstractionMap, optional
            Mapping between original and abstract actions, by default IdentityAbstractionMap()
        """

        self.state_abstraction_map = state_abstraction_map
        self.action_abstraction_map = action_abstraction_map

    def abstract_to_original_state(self, abs_state: int) -> NDArray:
        """
        Maps an abstract state to a (set/range of) original state(s).

        Parameters
        ----------
        abs_state : int
            A state in the abstracted environment
        Returns
        ----------
        orig_states : NDArray FIXME: should this be a gym.spaces.Box?
            A (set/range of) state(s) in the original environment
        """
        if self.state_abstraction_map.has_backward_map:
            return self.state_abstraction_map.backward_map(abs_state)
        else:
            raise ValueError(
                "Cannot map abstract state to original state without a backward map in the state abstraction."
            )

    def original_to_abstract_state(self, orig_state: NDArray) -> int:
        """
        Maps an original (continuous) state to an abstract (discrete) state.

        Parameters
        ----------
        orig_state : NDArray
            A state in the original environment.

        Returns
        -------
        abs_state : int
            A state in the abstracted environment.
        """
        abs_state = self.state_abstraction_map.forward_map(orig_state)
        if isinstance(abs_state, np.ndarray):
            if abs_state.size == 1:
                abs_state = abs_state.item()

        return abs_state

    def original_to_abstract_action(self, orig_action: NDArray) -> int:
        """
        Maps an action to an action

        Parameters
        ----------
        orig_action : NDArray
            An action in the original environment.

        Returns
        -------
        abs_action : int
            An action in the abstract environment.
        """
        return self.action_abstraction_map.forward_map(orig_action)

    def abstract_to_original_action(self, abs_action: int) -> NDArray:
        """
        Maps an action to a(n) (set/range of) action(s)

        Parameters
        ----------
        abs_action : int
            An action in the abstract environment

        Returns
        -------
        orig_actions : NDArray FIXME: should this be a gym.spaces.Box?
            A(n) (set/range of) action(s) in the original environment
        """
        if self.action_abstraction_map.has_backward_map:
            return self.action_abstraction_map.backward_map(abs_action)
        else:
            raise ValueError(
                "Cannot map abstract action to original action without a backward map in the action abstraction."
            )
