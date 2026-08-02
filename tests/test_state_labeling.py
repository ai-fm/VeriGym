import gymnasium as gym
import numpy as np
from math import prod
import functools
import stormpy

from verigym.environments.labeling import StateLabel, AbstractStateLabeler
from verigym.environments.generativeenv import GenerativeEnv
from verigym.abstraction.abstractionmapper import AbstractionMap, AbstractionMapper
from verigym.abstraction.learn_abstraction import CachedDiscretizer, learn_abstraction, normalize_aggregated_counts
from verigym.abstraction.learn_abstraction import forward_mapping
# from verigym.environments.transition_func import TransitionFunction
# from verigym.environments.reward_func import RewardFunction
from verigym.policy.policy import RandomizedPolicy
from verigym.abstraction.discretization import generate_box_bins
from verigym.abstraction.gym_utils.mapping import sample_to_discrete
from verigym.abstraction.utils import factored_to_index, index_to_factored
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp

from verigym.environments.explicitenv import ExplicitEnv

def test_discrete_generative_state_labeling():
    env = gym.make("FrozenLake-v1", map_name="8x8")
    env = GenerativeEnv.from_gymnasium(env)

    #"8x8": [
    #    "SFFFFFFF",    00000000
    #    "FFFFFFFF",    00020000
    #    "FFFHFFFF",    00212200
    #    "FFFFFHFF",    00022120
    #    "FFFHFFFF",    02212220
    #    "FHHFFFHF",    21122212
    #    "FHFFHFHF",    21221212
    #    "FFFHFFFG",    02212020
    #]
    label_should_be_true = [0,0,0,0,0,0,0,0,
                            0,0,0,2,0,0,0,0,
                            0,0,2,1,2,2,0,0,
                            0,0,0,2,2,1,2,0,
                            0,2,2,1,2,2,2,0,
                            2,1,1,2,2,2,1,2,
                            2,1,2,2,1,2,1,2,
                            0,2,2,1,2,0,2,0]

    near_hole = StateLabel("near_hole",
                               lambda s: any([env.unwrapped.desc[int(s/8)][s%8] == b"H", # is hole
                                              env.unwrapped.desc[max(0, int(s/8)-1)][s%8] == b"H", # hole on top
                                              env.unwrapped.desc[min(7, int(s/8)+1)][s%8] == b"H", # hole below
                                              env.unwrapped.desc[int(s/8)][min(7, (s%8)+1)]== b"H", # hole to right
                                              env.unwrapped.desc[int(s/8)][max(0, (s%8)-1)]== b"H", # hole to left
                                              ])
                               )

    assert not env.has_state_labels()
    env.add_state_label(near_hole)
    assert env.has_state_labels()

    for s in range(64):
        n_label_in_s = len(env.get_labels_of_state(s))
        if n_label_in_s > 0:
            assert label_should_be_true[s] > 0


def test_discrete_frameworkenv_state_labeling():
    ... # TODO add test when abstraction from explicit env is implemented

def test_continuous_generative_state_labeling():
    env = gym.make("LunarLander-v3")
    env = GenerativeEnv.from_gymnasium(env)
    d = 0.75
    away_from_pad = StateLabel("unsafe",
                               lambda s:
                               s[0] < -d or s[0] > d)
    
    assert not env.has_state_labels()
    env.add_state_label(away_from_pad)
    assert env.has_state_labels()
    
    env.reset()
    for _ in range(1000):
        s = env.observation_space.sample()
        if s[0] > d or s[0] < -d:
            assert away_from_pad(s)
        else:
            assert not away_from_pad(s)


def test_underapproximation_discrete():
    # under approximate = abstract state gets the label if all concrete states in it have the label.
    env = gym.make("FrozenLake-v1", map_name="8x8")
    env = GenerativeEnv.from_gymnasium(env)

    near_hole = StateLabel("near_hole",
                               lambda s: any([env.unwrapped.desc[int(s/8)][s%8] == b"H", # is hole
                                              env.unwrapped.desc[max(0, int(s/8)-1)][s%8] == b"H", # hole on top
                                              env.unwrapped.desc[min(7, int(s/8)+1)][s%8] == b"H", # hole below
                                              env.unwrapped.desc[int(s/8)][min(7, (s%8)+1)]== b"H", # hole to right
                                              env.unwrapped.desc[int(s/8)][max(0, (s%8)-1)]== b"H", # hole to left
                                              ])
                               )

    env.add_state_label(near_hole)
    n_abstract = 16

    partition = {i: set([(i*4) + j for j in range(4)]) for i in range(n_abstract)}
    inv_partition = {}
    for key, val in partition.items():
        for s in val:
            inv_partition[s] = key

    abstraction_map = AbstractionMap(
        forward_map = lambda s: inv_partition[s],
        backward_map= lambda s: partition[s]
    )
    abstraction_mapper = AbstractionMapper(state_abstraction_map=abstraction_map)
    abstract_state_labeler = AbstractStateLabeler(env.state_labeler, abstraction_mapper)
    gold_truth_underapproximate = {s: set() for s in range(n_abstract)}
    gold_truth_underapproximate[10] = {"near_hole"}
    gold_truth_underapproximate[11] = {"near_hole"}
    gold_truth_underapproximate[12] = {"near_hole"}
    gold_truth_underapproximate[13] = {"near_hole"}

    #label_should_be_true = [0: 0,0,0,0,    False
    #                        1: 0,0,0,0,    False
    #                        2: 0,0,0,2,    False
    #                        3: 0,0,0,0,    False
    #                        4: 0,0,2,1,    False
    #                        5: 2,2,0,0,    False
    #                        6: 0,0,0,2,    False
    #                        7: 2,1,2,0,    False
    #                        8: 0,2,2,1,    False
    #                        9: 2,2,2,0,    False
    #                        10: 2,1,1,2,   True
    #                        11: 2,2,1,2,   True
    #                        12: 2,1,2,2,   True
    #                        13: 1,2,1,2,   True
    #                        14: 0,2,2,1,   False
    #                        15: 2,0,2,0]   False

    for s in range(16):
        assert abstract_state_labeler.get_labels_of_abstract_state_underapproximate(s) == gold_truth_underapproximate[s], f"not the same labels for state {s}. {[abstract_state_labeler.get_labels_of_abstract_state_underapproximate(i) for i in range(n_abstract)]}"
    

def test_overapproximation_discrete():
    # over approximate = abstract state gets the label if any state in it has the label
    env = gym.make("FrozenLake-v1", map_name="8x8")
    env = GenerativeEnv.from_gymnasium(env)

    near_hole = StateLabel("near_hole",
                               lambda s: any([env.unwrapped.desc[int(s/8)][s%8] == b"H", # is hole
                                              env.unwrapped.desc[max(0, int(s/8)-1)][s%8] == b"H", # hole on top
                                              env.unwrapped.desc[min(7, int(s/8)+1)][s%8] == b"H", # hole below
                                              env.unwrapped.desc[int(s/8)][min(7, (s%8)+1)]== b"H", # hole to right
                                              env.unwrapped.desc[int(s/8)][max(0, (s%8)-1)]== b"H", # hole to left
                                              ])
                               )

    env.add_state_label(near_hole)
    n_abstract = 16

    partition = {i: set([(i*4) + j for j in range(4)]) for i in range(n_abstract)}
    inv_partition = {}
    for key, val in partition.items():
        for s in val:
            inv_partition[s] = key

    abstraction_map = AbstractionMap(
        forward_map = lambda s: inv_partition[s],
        backward_map= lambda s: partition[s]
    )
    abstraction_mapper = AbstractionMapper(state_abstraction_map=abstraction_map)
    abstract_state_labeler = AbstractStateLabeler(env.state_labeler, abstraction_mapper)
    gold_truth_overapproximate = {s: {"near_hole"} for s in range(n_abstract)}
    gold_truth_overapproximate[0] = set()
    gold_truth_overapproximate[1] = set()
    gold_truth_overapproximate[3] = set()

    #label_should_be_true = [0: 0,0,0,0,    False
    #                        1: 0,0,0,0,    False
    #                        2: 0,0,0,2,    True
    #                        3: 0,0,0,0,    False
    #                        4: 0,0,2,1,    True
    #                        5: 2,2,0,0,    True
    #                        6: 0,0,0,2,    True
    #                        7: 2,1,2,0,    True
    #                        8: 0,2,2,1,    True
    #                        9: 2,2,2,0,    True
    #                        10: 2,1,1,2,   True
    #                        11: 2,2,1,2,   True
    #                        12: 2,1,2,2,   True
    #                        13: 1,2,1,2,   True
    #                        14: 0,2,2,1,   True
    #                        15: 2,0,2,0]   True

    for s in range(16):
        assert abstract_state_labeler.get_labels_of_abstract_state_overapproximate(s) == gold_truth_overapproximate[s], f"not the same labels for state {s}. {[abstract_state_labeler.get_labels_of_abstract_state_underapproximate(i) for i in range(n_abstract)]}"
    

def get_continuous_setup():
    env = gym.make("LunarLander-v3", continuous=False)
    env = GenerativeEnv.from_gymnasium(env)
    d = 1.0
    away_from_pad = StateLabel("unsafe",
                               lambda s:
                               (s[0] <= -d) | (s[0] >= d))
    
    env.add_state_label(away_from_pad)

    bin_edges_per_dim = np.array([6, 6, 1, 1, 1, 1, 1, 1], dtype=int)
    bin_edges = generate_box_bins(
        env.observation_space, np.linspace, bin_edges_per_dim
    )
    bin_step_sizes = abs(env.observation_space.high - env.observation_space.low) / [max(i-1, 1) for i in bin_edges_per_dim]

    exploration_policy = RandomizedPolicy(env)
    dataset = env.simulate(
        policy=exploration_policy, n_steps=int(1e5), verbose=True
    )
    f = functools.partial(sample_to_discrete, bin_edges=bin_edges, return_idx=True)
    discretizer = CachedDiscretizer(
        functools.partial(factored_to_index, bin_edges=bin_edges)
    )

    state_abstraction_map = AbstractionMap(
        forward_map=functools.partial(forward_mapping, to_int=discretizer.discretize, to_bins=f),
        backward_map=lambda idx: [index_to_factored(idx, bin_edges), index_to_factored(idx, bin_edges) + bin_step_sizes]
    )
    abstraction_mapper = AbstractionMapper(state_abstraction_map)
    n_actions = env.action_space.n
    n_states = prod([len(dimension) for dimension in bin_edges])
    T_dict, R_dict, P_tot, state_distr = learn_abstraction(dataset=dataset,
                                     n_states=n_states,
                                     n_actions=n_actions,
                                     abstraction_mapper=abstraction_mapper,
                                     multithreading=False)
    T, R, S_init = normalize_aggregated_counts(T_dict, R_dict, P_tot, state_distr, n_states, n_actions)
    explicit_env = ExplicitEnv(
        nr_states=n_states,
        nr_actions=n_actions,
        nr_rewards=1,
        initial_state_distr=S_init,
        transition_function=T,
        reward_function=R,
        abstraction_map=abstraction_mapper,
        original_env=env,
        render_mode=None
    )

    abstraction_map = explicit_env.abstraction_map
    assert isinstance(abstraction_map, AbstractionMapper)

    abstract_state_labeler = AbstractStateLabeler(env.state_labeler,
                                                  abstraction_map)
    explicit_env.state_labeler = abstract_state_labeler
    return explicit_env

def test_underapproximation_continuous():
    # under approximate = abstract state gets the label if all concrete states in it have the label.
    explicit_env = get_continuous_setup()

    ground_truth_labels = {s: set() for s in range(explicit_env.nr_states)}
    unsafe_idcs = [0,1,2,3,4,5, 24,25,26,27,28,29, 30,31,32,33,34,35]
    for idx in unsafe_idcs:
        ground_truth_labels[idx] = {"unsafe"}

    for abstract_state in range(explicit_env.nr_states):
        assert explicit_env.state_labeler.get_labels_of_abstract_state_underapproximate(abstract_state) == ground_truth_labels[abstract_state]

def test_overapproximation_continuous():
    # over approximate = abstract state gets the label if any state in it has the label
    explicit_env = get_continuous_setup()

    ground_truth_labels = {s: {"unsafe"} for s in range(explicit_env.nr_states)}
    safe_idcs = [12,13,14,15,16,17,]
    for idx in safe_idcs:
        ground_truth_labels[idx] = set()

    for abstract_state in range(explicit_env.nr_states):
        assert explicit_env.state_labeler.get_labels_of_abstract_state_overapproximate(abstract_state) == ground_truth_labels[abstract_state]

def test_modelcheck_label():
    explicit_env = get_continuous_setup()
    prop_overapproximate = stormpy.parse_properties('Pmin=? [F "unsafe"]')[0]
    prop_underapproximate = stormpy.parse_properties('Pmax=? [F "unsafe"]')[0]
    mdp_over = build_stormpy_mdp(explicit_env, overapproximate=True)
    mdp_under = build_stormpy_mdp(explicit_env, overapproximate=False)
    stormpy.check_model_sparse(mdp_over, prop_overapproximate)
    stormpy.check_model_sparse(mdp_under, prop_underapproximate)
