# Model Learning

`verigym`'s model learning is a core component to obtain explicit model representations from simulator environments.
It builds on the abstraction mapping, see the [abstraction mapping guide](abstraction_mapping.md).
This guide provides an overview of the two central steps of model learning:
1. **model exploration**, for which `verigym` implements different exploration policies, and
2. **model representation**, for which we currently support MDP and IMDP representations.


## A Minimal Example

Model learning takes a simulator environment, an abstraction mapper ([see tutorial](abstraction_mapping.md)), and an exploration policy, and returns an explicit abstract model.

```python
import gymnasium as gym
import verigym
from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.policy.qvalue import EntropyLearningPolicy

# load a gymnasium environment
gym_env = gym.make("MountainCarContinuous-v0)
env = verigym.GenerativeEnv.from_gymnasium(gym_env)

# define a mapping to the an abstract space: here we discretize state and action space evenly
abstraction_mapper = linspace_mapper(env=env, n_bins_states=10, n_bins_actions=5)

# learn an explicit abstract model through interacting with the original model
abstract_env = verigym.create_abstraction(
    original_env=env,
    abstraction_mapper=abstraction_mapper,
    exploration_policy=EntropyLearningPolicy(env, abstraction_mapper),
    num_steps=10_000,
    n_iterations=5,
)
assert isinstance(abstract_env, verigym.ExplicitEnv)
```

**Workflows with focus on model learning**: 
learning an abstract model of a `gymnasium` environment and use model checking for computing a policy ({doc}`here <../notebooks/workflow_comparing_abstraction_policies>`),
comparing different exploration policies for model learning ({doc}`here <../notebooks/workflow_comparing_exploration_policies>`) and
iteratively learning finer abstractions ({doc}`here <../notebooks/workflow_abstraction_refinement>`).


The sections below explain the exploration policy and the type of the learned model.


An abstract (explicit) model is learned through interactions with the original model (could be a simulator via `gymnasium` environment).
This means we have a dataset over trajectories that we use to learn the explicit model.
For that we require the abstract states to be enumerable (see *enum* in [abstraction mapping guide](abstraction_mapping.md)).
This allows us to iterate through the dataset of trajectories in the *original* representation and do the following:
- get the abstract state and action representation
- add the new transition count to the `TransitionFunction`
- add the reward to the `RewardFunction`
- at the end of the dataset for each state-action tuple, normalize the `TransitionFunction` counts and average the rewards in the `RewardFunction`.

See [Interval MDPs](#interval-mdps) for going beyond MDPs.


### Terminal state handling (`terminal` and `truncated`)

In the abstraction we add a terminal state.
The interactions with via the `gymnasium` API returns the `truncated` and `terminal` flag.
Whenever the `terminal` flag is raised, we add a transition from the corresponding state-action tuple to the terminal state.

The `truncated` flag signifies that the episode was interrupted, but not due to a visit of a terminal state. 
Therefore, we document the state transition and reward from the previous state-action as done during any other time step.


## Interleaving Abstractions and Exploration in the Original Model

The dataset of interactions with the original model is created by following a behavioral policy.
We provide the following policies out of the box:
- The *randomized policy* is a simple baseline policy that chooses actions uniformly at random.
- The *Q-value policy* samples actions from a softmax over Q-values estimated on the model learned so far, and takes a uniformly random action with a fixed probability `epsilon` (default `0.0`).
- The *active learning policy* implements active learning-based exploration based on [Araya-López et al. 2012](https://ewrl.wordpress.com/wp-content/uploads/2011/08/ewrl2011_submission_17.pdf).
- The *entropy learning policy* implements entropy-based exploration based on [Hazan et al. 2019](https://proceedings.mlr.press/v97/hazan19a.html).

```python
from verigym.policy.randomized import RandomizedPolicy
from verigym.policy.qvalue import (
    QValuePolicy, ActiveLearningPolicy, EntropyLearningPolicy
)

random_policy = RandomizedPolicy(env)  # needs no abstraction_mapper
q_val_policy = QValuePolicy(env, abstraction_mapper, epsilon=0.1)
active_policy = ActiveLearningPolicy(env, abstraction_mapper, epsilon=0.1)
entropy_policy = EntropyLearningPolicy(env, abstraction_mapper)
```

Further, note that it is also possible to use any of the framework-specific policies for exploration: for instance, we can use a trained RL policy, such as PPO, as an expert for exploration by loading it as an `SB3Policy`.

See {doc}`../notebooks/workflow_comparing_exploration_policies` for a comparison of exploration policies.


### Updating Policies During Learning (Interleaving)

While the `RandomizedPolicy` remains the same throughout the model learning process, other policies (i.e. `EntropyLearningPolicy`) may be updated throughout the learning process.

We can set the argument `n_iterations` for determining how many interleaving breaks we take for the policy to be updated.
At each iteration the exploration policy calls `PolicyClass.update_for_abstraction_refinement()` to update itself.

The total number of steps is then `n_iterations` x `num_steps`.



### Adding custom exploration policies

To implement further exploration policies, create a custom class that inherits from `PolicyClass` and implement the method `_action_from_policy` for action sampling. Policies that adapt during learning also override `update_for_abstraction_refinement`, which enables changes to the policy when used in an abstraction refinement loop and returns the updated policy (by default, it returns the policy unchanged).

```python
class CustomPolicy(PolicyClass):
    def __init__(self, policy, abstraction_mapper, seed=None):
        ...
    def _action_from_policy(self, obs):
        ...
    def update_for_abstraction_refinement(self, dataset, T_counts, P_tot, R_cumsum, S_init_counts):
        ...
        return self
```


## Interval MDPs

`verigym` supports learning MDP or IMDP representations.
IMDPs provide (confidence) intervals on transition probabilities and rewards.
Users can specify the type of model they learn through boolean parameters in the `create_abstraction` method:
- `intervals`: if `True`, the returned environment will be of type `IntervalAbstractedExplicitEnv`, which has an interval transition function and an interval reward function in addition to the standard point estimates. If `False`, the environment will be an `AbstractedExplicitEnv` (an MDP).
- `assume_iid`: if `True`, we assume the underlying data to be IID, otherwise not. This is only relevant if `intervals=True`. As this is generally difficult to infer from given data, we leave this specification to the user.

When learning MDP representations, we use point estimates to derive transition and reward functions.
When learning IMDPs, we implement Azuma-Hoeffding Martingale intervals for non-IID data and Clopper-Pearson Binomial intervals for IID data.

An example using the right parameters

```python
import verigym

assume_iid = False
use_intervals = True

abstract_env = verigym.create_abstraction(
    original_env=env,
    abstraction_mapper=abstraction_mapper,
    exploration_policy=policy,
    num_steps=num_steps,
    intervals=use_intervals,
    assume_iid=assume_iid
)
```

Checkout *extension: IMDP learning instead of MDP learning* in {doc}`../notebooks/workflow_comparing_abstraction_policies`.