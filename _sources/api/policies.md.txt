# Policies

Policies map observations of an environment to actions.
All policies inherit from `PolicyClass`, which uses an `AbstractionMapper` to translate between the environment and the space the policy was computed for, e.g., to run a policy computed on an abstract model in the original environment.
*Exploration policies* collect data for model learning (see the [model learning guide](../guides/model_learning.md)).
*Policies from other frameworks* wrap policies computed by `stormpy`, PRISM, or Stable-Baselines3 (see [importing policies](../guides/import_export.md#importing-policies)).
`evaluate_policy` estimates the average return of a policy in an environment.

## Base class

```{eval-rst}
.. autoclass:: verigym.PolicyClass
   :members:
```

## Exploration policies

```{eval-rst}
.. autoclass:: verigym.policy.randomized.RandomizedPolicy
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: verigym.policy.qvalue.QValuePolicy
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: verigym.policy.qvalue.ActiveLearningPolicy
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: verigym.policy.qvalue.EntropyLearningPolicy
   :members:
   :show-inheritance:
```

## Policies from other frameworks

```{eval-rst}
.. autoclass:: verigym.StormpyPolicy
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: verigym.frameworks.prism.prismpolicy.PrismPolicy
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: verigym.frameworks.stable_baselines3.policy.SB3Policy
   :members:
   :show-inheritance:
```

## Evaluation

```{eval-rst}
.. autofunction:: verigym.policy.utils.evaluate_policy
```
