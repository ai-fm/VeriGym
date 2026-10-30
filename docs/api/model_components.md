# Model Components

The building blocks of explicit environments (see [Environments](environments.md)).
Transition and reward functions are stored as nested dicts over integer state and action indices, with interval variants for IMDPs, and the initial state distribution is stored sparsely.
State labels assign names to states that satisfy a predicate, e.g., `"goal"` or `"unsafe"`, so that model checking properties can refer to them.
Formatters convert models from external frameworks into this representation, see [Extending `VeriGym`](../guides/import_export.md#extending-verigym) in the import and export guide.

## Transition functions

```{eval-rst}
.. autoclass:: verigym.TransitionFunction
   :members:
```

```{eval-rst}
.. autoclass:: verigym.environments.transition_func.IntervalTransitionFunction
   :members:
   :show-inheritance:
```

## Reward functions

```{eval-rst}
.. autoclass:: verigym.RewardFunction
   :members:
```

```{eval-rst}
.. autoclass:: verigym.environments.reward_func.IntervalRewardFunction
   :members:
   :show-inheritance:
```

## Initial state distribution

```{eval-rst}
.. autoclass:: verigym.environments.state_distributions.StateDistribution
   :members:
```

## State labels

:::{admonition} TODO: add text
:class: warning
Introduction to this section.
:::

```{eval-rst}
.. autoclass:: verigym.environments.labeling.StateLabel
   :members:
```

```{eval-rst}
.. autoclass:: verigym.environments.labeling.StateLabeler
   :members:
```

```{eval-rst}
.. autoclass:: verigym.environments.labeling.AbstractStateLabeler
   :members:
```

## Formatters

:::{admonition} TODO: add text
:class: warning
Introduction to this section.
:::

```{eval-rst}
.. autoclass:: verigym.environments.formatter.ExplicitFormatter
   :members:
```

```{eval-rst}
.. autoclass:: verigym.frameworks.stormpy.formatter.StormpyFormatter
   :members:
   :show-inheritance:
```
