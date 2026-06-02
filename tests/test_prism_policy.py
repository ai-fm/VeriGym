# Note: policy files where generated using the command:
# for .txt: prism resource-gathering.pm -const B=200,GOLD_TO_COLLECT=15,GEM_TO_COLLECT=15 -pf 'Pmax=? [F "success"]' -exportstrat testout.txt:states=false
# for .tra: prism resource-gathering.pm -const B=200,GOLD_TO_COLLECT=15,GEM_TO_COLLECT=15 -pf 'Pmax=? [F "success"]' -exportstrat testout.tra:states=false
# MDP file and prism are not included, only output policy files, to avoid dependency here.
# resource-gathering with that sice has size 24064
from verigym.frameworks.prism.prismpolicy import PrismPolicy
from verigym.abstraction.abstractionmapper import AbstractionMapper

def test_action_list_policies():
    list_policy_path = "tests/prism_policies/testout.txt"
    action_map = {
        "right": 0,
        "left": 1,
        "up": 2,
        "down": 3
    }

    PrismPolicy(
        policy=list_policy_path,
        action_map=action_map,
        abstraction_mapper=AbstractionMapper()
    )

def test_tra_policies():
    tra_policy_path = "tests/prism_policies/testout.tra"

    action_map = {
        "right": 0,
        "left": 1,
        "up": 2,
        "down": 3
    }

    PrismPolicy(
        policy=tra_policy_path,
        action_map=action_map,
        abstraction_mapper=AbstractionMapper()
    )