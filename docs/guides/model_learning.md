# Model Learning

Next to the **abstraction mapping** that we provided a guide-through in `guides/abstraction_mapping.md`, a core component to obtain explicit model representations from simulated environments is **model learning**.
This guide provides an overview over the two central steps of model learning:
1. **model exploration**, for which `VeriGym` implements different exploration policies, and
2. **model representation**, for which we currently support MDP and IMDP representations.

## Model exploration

We provide an overview of the types of exploration policies implemented in `VeriGym` `v.0.1.0`.
All implemented policy types inherit from the `PolicyClass` class.
- The *randomized policy* is a simple baseline policy that explores randomly.
- The *Q-value policy* chooses the best action based on current state value estimates, and can incorporate some random exploration using a fixed threshold `epsilon`.
- The *active learning policy* implements active learning-based exploration based on [Araya-López et al. 2012](https://ewrl.wordpress.com/wp-content/uploads/2011/08/ewrl2011_submission_17.pdf)
- The *entropy learning policy* implements entropy-based exploration based on [Hazan et al. 2019](https://proceedings.mlr.press/v97/hazan19a.html)

```python
from verigym.policy.randomized import RandomizedPolicy
from verigym.policy.qvalue import (
    QValuePolicy, ActiveLearningPolicy, EntropyLearningPolicy
)

random_policy = RandomizedPolicy(env)
q_val_policy = QValuePolicy(env, abstraction_mapper, epsilon=0.1)
active_policy = ActiveLearningPolicy(env, abstraction_mapper, epsilon=0.1)
entropy_policy = EntropyLearningPolicy(env, abstraction_mapper)
```

Further, note that it is also possible to use any of the framework-specific policies for exploration: for instance, we can use a trained RL policy as an expert for exploration by loading it as a `SB3Policy`.

### Adding custom exploration policies

To implement further exploration policies, create a custom class that inherits from `PolicyClass` and implement the two methods `_action_from_policy` for action sampling, and `update_for_abstraction_refinement`, which enables changes to the policy when used in an abstraction refinement loop.

```python
class CustomPolicy(PolicyClass):
    def __init__(self, policy, abstraction_mapper, seed=None)
        ...
    def _action_from_policy(self, obs):
        ...
    def update_for_abstraction_refinement(self, **kwargs):
        ...
```

## Model representation

`VeriGym` supports learning MDP or IMDP representations. 
IMDPs provide (confidence) intervals on transition probabilities and reward function.
Users can specify the type of model they learn through boolean parameters in the `create_abstraction` method:
- `intervals`: if `True`, the returned environment will be of type `IntervalAbstractedExplicitEnv`, which has an interval transition function and an interval reward function in addition to the standard point estimates. If false, the environment will be an `AbstractedExplicitEnv`.
- `assume_iid`: if `True`, we assume the underlying data to be IID, otherwise not. As this is generally difficult to infer from given data, we leave this specification to the user.

When learning MDP representations, we use point-estimates to derive transition and reward functions.

When learning IMDPs, we implement Azuma-Hoeffding Martingale intervals for non-IID data and Clopper-Pearson Binomial intervals for IID data.

Finally, all components come together in the `create_abstraction` method.
```python
import verigym

verigym.create_abstraction(
    original_env=env,
    abstraction_mapper=abstraction_mapper,
    exploration_policy=policy,
    num_steps=num_steps,
    intervals=use_intervals,
    assume_iid=assume_iid
)
```


## Where to find examples?

Check out the following workflows:
- `workflow_comparing_exploration_policies.ipynb` for a comparison of exploration policies.
- `Extension: IMDP learning instead of MDP learning` in the `workflow_comparing_abstraction_policies.ipynb` notebook.