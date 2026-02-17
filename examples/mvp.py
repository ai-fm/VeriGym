import stormpy
import gymnasium as gym
import numpy as np

import verigym
from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation
# import verigym.environments.explicitenv
# import verigym.environments.verigymenv
import verigym.abstraction
import verigym.frameworks.stormpy
import verigym.frameworks.stormpy.stormpypolicy
import verigym.abstraction.learn_abstraction
import verigym.abstraction.abstractionmapper
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp

def get_mean_reward_from_trajectories(trajectories):
    rewards = []
    for trajectory in trajectories_abstracted:
        rewards += list(map(lambda tup: tup[2], trajectory))
    # rewards.append(list(map(lambda tup: tup[2], trajectory)))
    return np.mean(rewards)

# Load the gym env
gym_env = gym.make("CartPole-v1")
gym_env = ReplaceInfObservation(
    gym_env, neg_inf=-10, pos_inf=10
)  # TODO shold the tool infer this?


# Create a VeriGymEnv from gym env
# generative_model = verigym.environments.generativeenv.GenerativeEnv(gym_env)
generative_model = verigym.environments.generativeenv.GenerativeEnv.from_gymnasium(
    gym_env
)
del gym_env
# verigym.environments.verigymenv.from_gym()

# Create abstraction
abstracted_model = verigym.abstraction.learn_abstraction.create_abstraction(  # TODO add different discretisation functions as arguments
    original_env=generative_model,
    bin_edges_per_dim=5,  # Discretization: dim 1 has 10 bins, dim 2 has 5 bins, ...
    exploration_strategy="random",  # alternatively any verigym.Policy object
    num_steps=int(1e4),
)
print(abstracted_model is verigym.ExplicitEnv)  # returns True


# TODO: I/O
# save the abstracted model and free memory
# abstracted_model.to_drn("path/to/abstracted_model.drn")
# del abstracted_model

# abstracted_model.save_abstraction("myfile")

# load the abstracted model again
# abstracted_model = verigym.from_drn("path/to/abstracted_model.drn")

# abstracted_models.load_abstraction("myfile")


stormpy_mdp = build_stormpy_mdp(abstracted_model)
print(stormpy_mdp)
# exit()

# compute policy using storm -- OUTSIDE of Verigym

# stormpy_model = abstracted_model.to_storm_model()
# stormpy_policy = stormpy.compute_some_policy(stormpy_model)  # this is just placeholder
gamma = 0.95
prop = stormpy.parse_properties_without_context("Rmax=? [Cdiscount=0.95]")[0]
result = stormpy.model_checking(stormpy_mdp, prop, extract_scheduler=True)
sched = result.scheduler

# convert into VeriGym policy
verigym_policy = verigym.frameworks.stormpy.stormpypolicy.StormpyPolicy(
    sched, abstracted_model.abstraction_map
)
verigym_policy_on_abstracted = verigym.frameworks.stormpy.stormpypolicy.StormpyPolicy(
    sched, abstraction_mapper=verigym.abstraction.abstractionmapper.AbstractionMapper()
)
# verigym_policy = verigym.policy.from_stormpy(sched, abstracted_model.abstraction_map) # alternative

# verify the policy: (1) policy performance on orignal model
trajectories_original = generative_model.simulate(
    policy=verigym_policy,
    n_steps=int(10e4),
)
rewards_original = get_mean_reward_from_trajectories(trajectories_original)

# verify the policy: (2) policy performance on abstracted model
trajectories_abstracted = abstracted_model.simulate(
    policy=verigym_policy_on_abstracted,
    n_steps=int(10e4),
)
rewards_abstracted = get_mean_reward_from_trajectories(trajectories_abstracted)

print(f"{rewards_original = }\n{rewards_abstracted = }")

# TODO Evaluate abstraction quality
# results = verigym.compare_models(
#     original_model = gym_env,
#     abstracted_model = abstracted_model,
#     n_steps = 10e6
# )
