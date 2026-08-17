from typing import Callable

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from verigym.utils.utils import identity_map
from verigym.abstraction.gym_utils.spaces import get_n_elements_of_space, DummySpace



class AbstractionMap:
    """
    This class consists of functions mapping between original and abstract spaces (for example continuous and discrete).
    As such, it can be used for mapping both state and action spaces.
    While a forward map is required, it is not always possible (or obvious how) to define a backward map.
    
    Note: Use `DummySpace` may be used for passing `original_space` or `abstract_space` if necessary. Use sparingly.
    
    Attributes
    ----------
    - `original_space` holds the `gym.spaces` class corresponding to the original environment's space type.
    - `abstract_space` holds the `gym.spaces` class corresponding to the abstract environment's space type.
    - `forward_map` holds the function that takes an input from `original_space` and maps it to -> `abstract_space`
    - `backward_map` holds the function that takes an input from `abstrac_space` and maps it to -> `original_space`
    - `original_n_elements` holds the information about how many elements `original_space` contains. Should either be `int` or `verigym.abstraction.gym_utils.spaces.infty`.
    - `abstract_n_elements` holds the information about how many elements `abstract_space` contains. Should either be `int` or `verigym.abstraction.gym_utils.spaces.infty`.
    - `from_continuous_space` holds the information whether the orginal space is a continuous space. 
      (Note: Only when `original_space` is `DummySpace`, this attribute will equal `None`.)
    """
    
    original_space: gym.spaces.Space
    abstract_space: gym.spaces.Space
    forward_map: Callable
    backward_map: Callable
    original_n_elements: int | float | None
    abstract_n_elements: int | float | None
    from_continuous_space: bool | None

    def __init__(
        self,
        forward_map: Callable[[NDArray], int],
        backward_map: Callable[[int], NDArray] = None,
        original_space: gym.spaces.Space = None,
        abstract_space: gym.spaces.Space = None,
    ):
        """
        A map from an original to abstract space. Either state- or action-space mapping.
        This class consists of functions mapping between original and abstract spaces (for example continuous and discrete).
        As such, it can be used for mapping either state and action spaces, do not confuse with `AbstractionMapper` that 
        holds the combined mapping for state- and action space.
        While a forward map is required, it is not always possible (or obvious how) to define a backward map.

        Parameters
        ----------
        forward_map : Callable[[NDArray], int]
            A callable function representing a forward mapping of continuous spaces (in numpy format) to abstract (integer) spaces
        from_continuous_space: bool
            A boolean flag to indicate whether the original space was continuous or discrete (= finite, countable)
        backward_map : Callable[[int], NDArray], optional
            A callable function representing a backward mapping of abstract (integer) spaces to continuous spaces (in numpy format) if available, by default None
        """
        self.forward_map = forward_map
        self.backward_map = backward_map
        self.has_backward_map = self.backward_map is not None
        
        assert original_space is not None, "original_space should not be None."
        assert abstract_space is not None, "abstract_space should not be None."
        
        self.original_space = original_space
        self.original_n_elements = get_n_elements_of_space(original_space) if original_space is not None else None
        
        self.abstract_space = abstract_space
        self.abstract_n_elements = get_n_elements_of_space(abstract_space) if abstract_space is not None else None
        
        if isinstance(self.original_space, gym.spaces.Box):
            self.from_continuous_space = True
        else:
            self.from_continuous_space = False
        


class IdentityAbstractionMap(AbstractionMap):
    """
    This class can be used for an identity mapping. This means any input will be returned without being changed.
    """
    
    def __init__(self):
        """
        This class can be used for an identity mapping. This means any input will be returned without being changed.
        """
        super().__init__(
            forward_map = identity_map,
            backward_map = identity_map,
            original_space = DummySpace(),
            abstract_space = DummySpace(),
            )


class AbstractionMapper:
    """
    This class requires a docstring!
    """
    
    _state_abstraction_map: AbstractionMap
    _action_abstraction_map: AbstractionMap
    from_continuous_states: bool | None
    from_continuous_actions: bool | None
    original_n_states: int | float | None
    original_n_actions: int | float | None
    abstract_n_states: int | float | None
    abstract_n_actions: int | float | None
    
    def __init__(
        self,
        state_abstraction_map: AbstractionMap = IdentityAbstractionMap(),
        action_abstraction_map: AbstractionMap = IdentityAbstractionMap(),
    ):
        """
        Provides a mapping between abstract and original spaces.

        Parameters
        ----------
        state_abstraction_map : AbstractionMap, optional
            Mapping between original and abstract states, by default IdentityAbstractionMap()
        action_abstraction_map : AbstractionMap, optional
            Mapping between original and abstract actions, by default IdentityAbstractionMap()
        """

        self._state_abstraction_map = state_abstraction_map
        self._action_abstraction_map = action_abstraction_map

        self.from_continuous_states = self._state_abstraction_map.from_continuous_space
        self.from_continuous_actions = self._action_abstraction_map.from_continuous_space
        
        self.original_n_states = state_abstraction_map.original_n_elements
        self.original_n_actions = action_abstraction_map.original_n_elements
        self.abstract_n_states = state_abstraction_map.abstract_n_elements
        self.abstract_n_actions = action_abstraction_map.abstract_n_elements

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
        if self._state_abstraction_map.has_backward_map:
            return self._state_abstraction_map.backward_map(abs_state)
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
        abs_state = self._state_abstraction_map.forward_map(orig_state)
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
        abs_action = self._action_abstraction_map.forward_map(orig_action)
        if isinstance(abs_action, np.ndarray): # TODO I do not like this check. Only necessary as we are not consistent with when actions/states are NDArrays or tuples (Joshua) Issue #116
            if abs_action.size == 1:
                abs_action = abs_action.item()

        return abs_action

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
        if self._action_abstraction_map.has_backward_map:
            return self._action_abstraction_map.backward_map(abs_action)
        else:
            raise ValueError(
                "Cannot map abstract action to original action without a backward map in the action abstraction."
            )
