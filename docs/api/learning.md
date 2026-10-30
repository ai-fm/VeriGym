# Model Learning

Functions for learning explicit models from a simulator.
`create_abstraction` is the main entry point: it explores the original environment with an exploration policy (see [Policies](policies.md)) and returns the learned MDP or IMDP.
`abstraction_refinement` repeats this with finer and finer abstraction mappers until a stopping criterion is met, see the {doc}`abstraction refinement workflow <../notebooks/workflow_abstraction_refinement>`.
The remaining functions are the building blocks of `create_abstraction`, e.g., for learning from your own data.
*Feature selection* removes state features before learning, see the {doc}`feature selection workflow <../notebooks/workflow_featurereduction>`.
For an introduction, see the [model learning guide](../guides/model_learning.md).

## Learning abstractions

```{eval-rst}
.. autofunction:: verigym.create_abstraction
```

```{eval-rst}
.. autofunction:: verigym.abstraction.abstraction_refinement.abstraction_refinement
```

```{eval-rst}
.. autofunction:: verigym.abstraction.learn_abstraction.learn_abstraction
```

```{eval-rst}
.. autofunction:: verigym.abstraction.learn_abstraction.collect_data_from_trajectories
```

```{eval-rst}
.. autofunction:: verigym.abstraction.learn_abstraction.normalize_aggregated_counts
```

```{eval-rst}
.. autofunction:: verigym.abstraction.learn_abstraction.get_interval_transition_reward
```

## Feature selection

```{eval-rst}
.. autofunction:: verigym.abstraction.feature_selection.state_feature_selection
```

```{eval-rst}
.. autoclass:: verigym.abstraction.feature_selection.ReduceFeaturesWrapper
   :members:
```

```{eval-rst}
.. autoclass:: verigym.abstraction.feature_selection.BinFeaturesWrapper
   :members:
```
