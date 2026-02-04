from verigym.environments.verigymenv import VeriGymEnv
from verigym.environments.explicitenv import ExplicitEnv


class AbstractionMap:
    """
    Provides a mapping between abstract and original spaces.
    """

    def __init__(
        self,
        original_env: VeriGymEnv,
        abstract_env: ExplicitEnv,
        abstraction_map: object,
    ):

        self.original_env = original_env
        self.abstract_env = abstract_env

        self.abstraction_map = abstraction_map

        self.nr_abstract_states = ...
        self.nr_abstract_actions = ...
        self.nr_abstract_rewards = ...

        self.action_mask = ...

    def abstract_to_original_state(abs_state):
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
        ...  # TODO

    def original_to_abstract_state(orig_state):
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
        ...  # TODO

    def original_to_abstract_action(orig_action):
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
        ...  # TODO

    def abstract_to_original_action(abs_action):
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
        ...  # TODO
