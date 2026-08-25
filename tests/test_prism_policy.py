# Note: policy files where generated using the command:
# for .txt: prism resource-gathering.pm -const B=200,GOLD_TO_COLLECT=15,GEM_TO_COLLECT=15 -pf 'Pmax=? [F "success"]' -exportstrat testout.txt:states=false
# for .tra: prism resource-gathering.pm -const B=200,GOLD_TO_COLLECT=15,GEM_TO_COLLECT=15 -pf 'Pmax=? [F "success"]' -exportstrat testout.tra:states=false
# MDP file and prism are not included, only output policy files, to avoid dependency here.
# resource-gathering with that sice has size 24064
from verigym.frameworks.prism.prismpolicy import PrismPolicy
from verigym.abstraction.abstractionmapper import AbstractionMapper, AbstractionMap

import gymnasium as gym

def test_action_list_policies():
    list_policy_path = "tests/prism_policies/testout.txt"
    action_map = {
        "right": 0,
        "left": 1,
        "up": 2,
        "down": 3
    }
    
    state_space = gym.spaces.Discrete(24064)
    action_space = gym.spaces.Discrete(4)
    abstraction_mapper = AbstractionMapper.initialize_identity_mapper(state_space, action_space)

    PrismPolicy(
        policy=list_policy_path,
        action_map=action_map,
        abstraction_mapper=abstraction_mapper
    )

def test_tra_policies():
    tra_policy_path = "tests/prism_policies/testout.tra"

    action_map = {
        "right": 0,
        "left": 1,
        "up": 2,
        "down": 3
    }
    
    state_space = gym.spaces.Discrete(24064)
    action_space = gym.spaces.Discrete(4)
    abstraction_mapper = AbstractionMapper.initialize_identity_mapper(state_space, action_space)
    

    PrismPolicy(
        policy=tra_policy_path,
        action_map=action_map,
        abstraction_mapper=abstraction_mapper
    )