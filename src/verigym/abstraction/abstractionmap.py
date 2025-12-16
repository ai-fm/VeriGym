from verigym.environments.verigymenv import VeriGymEnv
from verigym.environments.explicitenv import ExplicitEnv


class AbstractionMap:
    def __init__(self,
                 original_env: VeriGymEnv,
                 abstract_env: ExplicitEnv,
                 abstraction_map: object,
                 ):
        
        self.original_env = original_env
        self.abstract_env = abstract_env

        self.abstraction_map = abstraction_map

    def abstract_to_original_state(orig_state):
        ... # TODO

    def original_to_abstract_state(abs_state):
        ... # TODO

    def original_to_abstract_action(orig_state):
        ... # TODO

    def abstract_to_original_action(abs_action):
        ... # TODO