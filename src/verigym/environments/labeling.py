from typing import Callable

from verigym.utils.utils import check_sat_label
from verigym.abstraction.abstractionmapper import AbstractionMapper
import re
import z3

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
        self.original_labeler = self._init_state_labeler(original_labeler)
        self.abstraction_mapper = abstraction_mapper

        self.labels = self.original_labeler.labels
        self.is_abstract = True

    def _init_state_labeler(self, original_labeler):
        state_labeler = StateLabeler(set([]))

        for label in original_labeler.labels:
            state_labeler.add_state_label(label)

            def negate_predicate(pred_eval):
                if isinstance(pred_eval, z3.BoolRef):
                    return z3.Not(pred_eval)
                else:
                    return not pred_eval
            
            # initialize negated labels
            not_label = StateLabel(f"not_{label.name}",
                                    lambda s, pred=label.predicate: negate_predicate(pred(s))
                                   )
            state_labeler.add_state_label(not_label)

        return state_labeler

    def parse_property(self, propertystr: str):
        """
        Change occurrences of `!"label"` in a property string to `"not_label"`
        for correct abstract label over- and underapproximation.
        """
        return re.sub(
            r'!\s*"([^"]+)"',
            lambda m: f'"not_{m.group(1)}"',
            propertystr
        )
    
    def get_labels_of_abstract_state_exist(self, abstract_state):
        """
        Returns state labels for an abstract state, based on the state labels of the contained original states.
        This function applies the exist operator, i.e., it overapproximates:
        If any of the original states contained in the abstract state has the label, the abstract state gets the label.
        
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
        if not self.abstraction_mapper.from_continuous_states: # discrete
            for s in original_states:
                for label in self.original_labeler.get_labels_of_state(s):
                    labels.add(label)
        else: # continuous
            lb = original_states[0]
            ub = original_states[1]
            all_labels = self.original_labeler.labels
            for label in all_labels:
                res = check_sat_label(lb, ub, label, check_not=False)
                if res:
                    labels.add(label.name)

        return labels


    def get_labels_of_abstract_state_forall(self, abstract_state):
        """
        Returns state labels for an abstract state, based on the state labels of the contained original states.
        This function applies the forall operator, i.e., it underapproximates:
        Only if all original states in the abstract state have the label, the abstract state gets the label.
        
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
        if not self.abstraction_mapper.from_continuous_states: # discrete
            all_labels = self.original_labeler.get_labels()
            for s in original_states:
                orig_labels = self.original_labeler.get_labels_of_state(s)
                all_labels = all_labels.intersection(orig_labels)
            for label in all_labels:
                labels.add(label)
        else: # continuous
            lb = original_states[0]
            ub = original_states[1]
            for label in self.labels:
                res = check_sat_label(lb, ub, label, check_not=True)
                if not res:
                    print(label, res)
                    # no counter example, holds for all labels
                    labels.add(label.name)

        return labels
    