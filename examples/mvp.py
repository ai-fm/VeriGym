import stormpy
import gymnasium as gym
import numpy as np

import verigym
from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp
from verigym.frameworks.stormpy.stormpypolicy import StormpyPolicy


def get_average_episode_length(trajectories):
    return np.mean([len(traj) for traj in trajectories])

def get_mean_reward_from_trajectories(trajectories):
    rewards = []
    for trajectory in trajectories:
        rewards += list(map(lambda tup: tup[2], trajectory))
    # rewards.append(list(map(lambda tup: tup[2], trajectory)))
    return float(np.mean(rewards))


# Load the gym env
gym_env = gym.make("CartPole-v1")
gym_env = ReplaceInfObservation(
    gym_env, neg_inf=-10, pos_inf=10
)  # TODO shold the tool infer this?

# Create a VeriGymEnv from gym env
generative_model = verigym.GenerativeEnv.from_gymnasium(gym_env)
del gym_env

# Create abstraction
abstracted_model = verigym.create_abstraction(  # TODO add different discretisation functions as arguments
    original_env=generative_model,
    bin_edges_per_dim=5,  # Discretization: dim 1 has 10 bins, dim 2 has 5 bins, ...
    exploration_strategy="random",  # alternatively any verigym.Policy object
    num_steps=int(1e6),
)
print("Finishing creating the abstraction.")
print(isinstance(abstracted_model, verigym.ExplicitEnv))  # returns True

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

# compute policy using storm -- OUTSIDE of Verigym
gamma = 0.99
prop = stormpy.parse_properties(f"Rmax=? [Cdiscount={gamma}]")[0]
result = stormpy.check_model_sparse(stormpy_mdp, prop, extract_scheduler=True)
value_vector = [result.at(state.id) for state in stormpy_mdp.states]
scheduler = result.scheduler
# convert into VeriGym policy
verigym_policy = StormpyPolicy(scheduler, abstracted_model.abstraction_map)

verigym_policy_on_abstracted = StormpyPolicy(
    scheduler, abstraction_mapper=verigym.AbstractionMapper()
)

# Uncomment to render during testing of the policy
# generative_model.unwrapped.render_mode = "human"

# verify the policy: (1) policy performance on orignal model
trajectories_original = generative_model.simulate(
    policy=verigym_policy, n_steps=int(10e3)
)
rewards_original = get_mean_reward_from_trajectories(trajectories_original)

# verify the policy: (2) policy performance on abstracted model
trajectories_abstracted = abstracted_model.simulate(
    policy=verigym_policy_on_abstracted,
    n_steps=int(10e3),
)
rewards_abstracted = get_mean_reward_from_trajectories(trajectories_abstracted)

print(f"{get_average_episode_length(trajectories_original)=}")
print(f"{get_average_episode_length(trajectories_abstracted)=}")
print(f"{rewards_original = }\n{rewards_abstracted = }")

# TODO Evaluate abstraction quality
# results = verigym.compare_models(
#     original_model = gym_env,
#     abstracted_model = abstracted_model,
#     n_steps = 10e6
# )
