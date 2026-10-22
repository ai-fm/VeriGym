# Policies

:::{admonition} TODO: add text
:class: warning
Introduction to this page.
:::

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
