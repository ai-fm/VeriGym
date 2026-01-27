# Use Cases

Let's look at some use cases with the `VeriGym` tool.


# From generative model to explicit model
- We have access to a `gym` environment that is defined as a dynamical system.
- The model is *generative*, as it can give you the next state given a state and an action.
- The model is *not explicit*, i.e. we do not know the transition/reward function in matrix form. 
- We want to obtain the transition/reward function in matrix form and learn it via simulations (=interacting with it).

```Python
import custom_gym_env
import verigym

# Load the gym env
gym_env = custom_gym_env.create()

# Create a VeriGymEnv from gym env
generative_model = verigym.VeriGymEnv(gym_env)

# Create abstraction
abstracted_model = verigym.learn_abstraction(
    model = generative_model,
    n_bins_per_dim = [10,5,10],    # Dim 1 has 10 bins, dim 2 has 5 bins, ...
    exploration_type = "uniform",  # alternatively any verigym.Policy object
    n_interactions = 10e6
)
print(type(abstracted_model) == verigym.ExplicitEnv)  # returns True

# Evaluate abstraction quality
results = verigym.compare_models(
    original_model = generative_model,
    abstracted_model = abstracted_model,
    n_steps = 10e6
)

# save the abstracted model and free memory
abstracted_model.to_drn("path/to/abstracted_model.drn")
del abstracted_model

# load the abstracted model again
abstracted_model = verigym.from_drn("path/to/abstracted_model.drn")

# compute policy using storm -- OUTSIDE of Verigym
import stormpy
storm_model = abstracted_model.to_storm_model()
storm_policy = stormpy.compute_some_policy(storm_model) # this is just placeholder 

# convert into VeriGym policy
verigym_policy = verigym.Policy(storm_policy)

# verify the policy (1) policy performance on orignal model 
trajectories_original = generative_model.simulate(  # This does not have to be a class method
    policy = verigym_policy,
    n_steps = 10e6,
)
rewards_original = trajectories_original["rewards"].mean()

# verify the policy (2) policy performance on abstracted model 
trajectories_abstracted = abstracted_model.simulate(
    policy = verigym_policy,
    n_steps = 10e6,
)
rewards_abstracted = trajectories_abstracted["rewards"].mean()

print(f"{rewards_original = }\n{rewards_abstracted = }")

```