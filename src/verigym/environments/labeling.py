from typing import Callable

from verigym.utils.utils import check_sat_label
from verigym.abstraction.abstractionmapper import AbstractionMapper, BackwardKind

class StateLabel:
    def __init__(self, 
                 name: str, 
                 predicate: Callable
                 ):
        self.name = name
        self.predicate = predicate
    
    def __call__(self, state) -> bool:
        return self.predicate(state)
    

class StateLabeler:
    def __init__(self, 
                 labels: set[StateLabel]
                 ):
        self.labels = set() if labels is None else labels
        self.is_abstract = False

    def get_labels_of_state(self, state):
        return {
            label.name
            for label in self.labels if label(state)
        }

    def add_state_label(self, label: StateLabel):
        self.labels.add(label)

    def get_labels(self):
        return {label.name for label in self.labels}


class AbstractStateLabeler:
    def __init__(self, original_labeler: StateLabeler, abstraction_mapper: AbstractionMapper):
        """
        TODO Add docstring @julemarie
        
        Parameters
        ----------
        original_labeler : StateLabeler
            The state label manager of the original environment
        abstraction_mapper : AbstractionMapper
            Provides a mapping between original and abstract spaces.
            For abstractions from discrete state spaces, expects the abstraction_mapper's backward map to return a set of states.
            For abstractions from continuous state spaces, expects original states characterized as a 2d list/array
            where list[0] = lower bound state valuations per dimension, and list[1] = upper bound state valuations per dimension.

        Example
        -------
        ```
        abstract_state_idx = 0
        abstraction_mapper.abstract_to_original_state(abstract_state_idx)
        # discrete:
        [0, 1, 2]
        # continuous:
        [[-1, 1], [-2, 2]]
        ```
        """
        self.original_labeler = original_labeler
        self.abstraction_mapper = abstraction_mapper
        self.labels = original_labeler.labels
        self.is_abstract = True

        self.overapproximate_labels = set()
        self.underapproximate_labels = set()
    
    def get_labels_of_abstract_state_overapproximate(self, abstract_state):
        """
        Returns state labels for an abstract state, based on the state labels of the contained original states.
        Overapproximate = if any of the original states contained in the abstract state has the label, the abstract state gets the label.
        
        Parameters
        ----------
        abstract_state : int
            index of the abstract state to find labels for.
        Returns
        -------
        labels: set
            set of overapproximated state labels for the given abstract state.
        """
        original_states = self.abstraction_mapper.abstract_to_original_state(abstract_state)
        labels = set()
        kind = self.abstraction_mapper.state_backward_kind
        if kind is BackwardKind.SET:
            # a finite collection of original states: look at each one directly
            for s in original_states:
                for label in self.original_labeler.get_labels_of_state(s):
                    labels.add(label)
        elif kind in (BackwardKind.INTERVAL, BackwardKind.POINT):
            # a region of original states, given by its lower and upper bounds and
            # a single point is a region whose corners coincide.
            lb, ub = (original_states, original_states) if kind is BackwardKind.POINT else (
                original_states[0], original_states[1]
            )
            all_labels = self.original_labeler.labels
            for label in all_labels:
                res = check_sat_label(lb, ub, label, check_not=False)
                if res:
                    labels.add(label.name)
        else:
            # UNKNOWN
            raise ValueError(
                "Cannot compute labels for abstract state "
                f"{abstract_state!r}: the state abstraction map's backward_kind "
                "is UNKNOWN (undeclared backward semantics)."
            )

        return labels


    def get_labels_of_abstract_state_underapproximate(self, abstract_state):
        """
        Returns state labels for an abstract state, based on the state labels of the contained original states.
        Underapproximate = if all original states in the abstract state have the label, the abstract state gets the label.
        
        Parameters
        ----------
        abstract_state : int
            index of the abstract state to find labels for.
        Returns
        -------
        labels: set
            set of underapproximated state labels for the given abstract state.
        """
        original_states = self.abstraction_mapper.abstract_to_original_state(abstract_state)
        labels = set()
        kind = self.abstraction_mapper.state_backward_kind
        if kind is BackwardKind.SET:
            # a finite collection of original states: look at each one directly
            all_labels = self.original_labeler.get_labels()
            for s in original_states:
                orig_labels = self.original_labeler.get_labels_of_state(s)
                all_labels = all_labels.intersection(orig_labels)
            for label in all_labels:
                labels.add(label)
        elif kind in (BackwardKind.INTERVAL, BackwardKind.POINT):
            # a region of original states, given by its lower and upper limits and
            # aA single point is a region whose corners coincide.
            lb, ub = (original_states, original_states) if kind is BackwardKind.POINT else (
                original_states[0], original_states[1]
            )
            all_labels = self.original_labeler.labels
            for label in all_labels:
                res = check_sat_label(lb, ub, label, check_not=True)
                if not res:
                    # no counter example, holds for all labels
                    labels.add(label.name)
        else:
            # UNKNOWN:
            raise ValueError(
                "Cannot compute labels for abstract state "
                f"{abstract_state!r}: the state abstraction map's backward_kind "
                "is UNKNOWN (undeclared backward semantics)."
            )

        return labels
    