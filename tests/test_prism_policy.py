# Note: policy files where generated using the command:
# for .txt: prism resource-gathering.pm -const B=200,GOLD_TO_COLLECT=15,GEM_TO_COLLECT=15 -pf 'Pmax=? [F "success"]' -exportstrat testout.txt:states=false
# for .tra: prism resource-gathering.pm -const B=200,GOLD_TO_COLLECT=15,GEM_TO_COLLECT=15 -pf 'Pmax=? [F "success"]' -exportstrat testout.tra:states=false
# MDP file and prism are not included, only output policy files, to avoid dependency here.
# resource-gathering with that sice has size 24064
import numpy as np
import pytest
import gymnasium as gym

from verigym.frameworks.prism.prismpolicy import PrismPolicy
from verigym.abstraction.abstractionmapper import AbstractionMapper, linspace_map

LIST_POLICY_PATH = "tests/prism_policies/testout.txt"
TRA_POLICY_PATH = "tests/prism_policies/testout.tra"
N_STATES = 24064
ACTION_MAP = {
    "right": 0,
    "left": 1,
    "top": 2,
    "down": 3
}

# (state, expected action index), one per label; identical in both policy files, and each
# chosen so that reading the neighbouring state (an off-by-one) gives a different label
KNOWN_STATES = [(2575, 0), (3600, 1), (2576, 2), (5647, 3)]


@pytest.fixture
def identity_mapper():
    state_space = gym.spaces.Discrete(N_STATES)
    action_space = gym.spaces.Discrete(4)
    return AbstractionMapper.initialize_identity_mapper(state_space, action_space)


def test_action_list_policies(identity_mapper):
    # With action map
    PrismPolicy(
        policy_path=LIST_POLICY_PATH,
        abstraction_mapper=identity_mapper,
        action_map=ACTION_MAP,
    )

    # Without action map
    PrismPolicy(
        policy_path=LIST_POLICY_PATH,
        abstraction_mapper=identity_mapper
    )


def test_tra_policies(identity_mapper):
    PrismPolicy(
        policy_path=TRA_POLICY_PATH,
        abstraction_mapper=identity_mapper,
        action_map=ACTION_MAP,
    )


@pytest.mark.parametrize("path", [LIST_POLICY_PATH, TRA_POLICY_PATH])
@pytest.mark.parametrize(["state", "expected"], KNOWN_STATES)
def test_reads_correct_action(identity_mapper, path, state, expected):
    """Each label (right/left/top/down) is read and mapped to the right action."""
    policy = PrismPolicy(policy_path=path, abstraction_mapper=identity_mapper, action_map=ACTION_MAP)
    assert policy.get_action(state) == expected


def test_txt_and_tra_agree(identity_mapper):
    """Both export formats of the same strategy give the same action in every state they share."""
    txt = PrismPolicy(policy_path=LIST_POLICY_PATH, abstraction_mapper=identity_mapper, action_map=ACTION_MAP)
    tra = PrismPolicy(policy_path=TRA_POLICY_PATH, abstraction_mapper=identity_mapper, action_map=ACTION_MAP)
    shared = txt.policy.keys() & tra.policy.keys()
    assert len(shared) > 20000
    for s in shared:
        assert txt.get_action(s) == tra.get_action(s), f"state {s}"


@pytest.mark.parametrize("path", [LIST_POLICY_PATH, TRA_POLICY_PATH])
def test_action_is_valid_and_hashable(identity_mapper, path):
    """Regression for `enum_to_abstract` returning `array([a])` for `Discrete` spaces."""
    policy = PrismPolicy(policy_path=path, abstraction_mapper=identity_mapper, action_map=ACTION_MAP)
    action = policy.get_action(1)
    assert gym.spaces.Discrete(4).contains(action)
    hash(action)


def test_missing_state_warns_and_returns_default(identity_mapper):
    policy = PrismPolicy(policy_path=LIST_POLICY_PATH, abstraction_mapper=identity_mapper, action_map=ACTION_MAP)
    missing = next(s for s in range(N_STATES) if s not in policy.policy)
    with pytest.warns(UserWarning, match="no action"):
        action = policy.get_action(missing)
    assert action == 0


def test_numeric_labels_without_action_map(tmp_path):
    """Without an `action_map`, labels are parsed as action indices."""
    path = tmp_path / "policy.tra"
    path.write_text("3 3\n0 1 1 2\n1 2 1 0\n2 0 1 3\n")
    mapper = AbstractionMapper.initialize_identity_mapper(gym.spaces.Discrete(3), gym.spaces.Discrete(4))
    policy = PrismPolicy(policy_path=str(path), abstraction_mapper=mapper)
    assert [policy.get_action(s) for s in range(3)] == [2, 0, 3]


def test_non_identity_mapper(tmp_path):
    """Regression for double mapping: with a binned 2-D state space, `get_action` must look up
    the enum of the observation's bin, not re-bin the bin indices."""
    state_map = linspace_map(gym.spaces.Box(low=np.zeros(2), high=np.ones(2)), 2)  # 4 abstract states
    action_map = AbstractionMapper.initialize_identity_mapper(gym.spaces.Discrete(1), gym.spaces.Discrete(2))._action_abstraction_map
    mapper = AbstractionMapper(state_map, action_map)

    # policy: action = enum % 2, so every state's action is checkable
    path = tmp_path / "policy.tra"
    path.write_text("4 4\n" + "".join(f"{s} {s} 1 {s % 2}\n" for s in range(4)))
    policy = PrismPolicy(policy_path=str(path), abstraction_mapper=mapper)

    points = [np.array(p, dtype=np.float32) for p in [(.25, .25), (.25, .75), (.75, .25), (.75, .75)]]
    enums = [mapper.original_to_abstract_state_enum(p) for p in points]
    assert sorted(enums) == [0, 1, 2, 3]  # the points cover every abstract state
    for p, e in zip(points, enums):
        assert policy.get_action(p) == e % 2
