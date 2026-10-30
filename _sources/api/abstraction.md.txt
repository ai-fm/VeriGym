# Abstraction

An abstraction maps the (often continuous) state and action spaces of an environment onto abstract spaces.
A discrete abstraction is what allows `verigym` to learn an explicit MDP from a simulator.
An `AbstractionMapper` combines one `AbstractionMap` for the states and one for the actions.
The functions under *Building maps* create mappers from common discretizations, e.g., `linspace_mapper` for evenly spaced bins.
*Discretization* contains the underlying binning, and *Gymnasium wrappers and spaces* contains helpers to prepare environments, e.g., `ReplaceInfObservation` for observation spaces with infinite bounds.
For an a detailed guide, see the [abstraction mapping guide](../guides/abstraction_mapping.md).

## Abstraction maps

```{eval-rst}
.. autoclass:: verigym.AbstractionMapper
   :members:
```

```{eval-rst}
.. autoclass:: verigym.abstraction.abstractionmapper.AbstractionMap
   :members:
```

```{eval-rst}
.. autoclass:: verigym.abstraction.abstractionmapper.BackwardKind
   :members:
```

## Building maps

```{eval-rst}
.. autofunction:: verigym.abstraction.abstractionmapper.linspace_mapper
```

```{eval-rst}
.. autofunction:: verigym.abstraction.abstractionmapper.linspace_map
```

```{eval-rst}
.. autofunction:: verigym.abstraction.abstractionmapper.pow_map
```

```{eval-rst}
.. autofunction:: verigym.abstraction.abstractionmapper.binned_mapper
```

```{eval-rst}
.. autofunction:: verigym.abstraction.abstractionmapper.binned_map
```

```{eval-rst}
.. autofunction:: verigym.abstraction.abstractionmapper.bin_edges_map
```

```{eval-rst}
.. autofunction:: verigym.abstraction.abstractionmapper.validate_for_abstraction
```

```{eval-rst}
.. autofunction:: verigym.abstraction.abstractionmapper.enumeration_of_space
```

```{eval-rst}
.. autofunction:: verigym.abstraction.abstractionmapper.nvec_of_space
```

## Discretization

```{eval-rst}
.. autoclass:: verigym.abstraction.discretization.BinEdges
   :members:
```

```{eval-rst}
.. autofunction:: verigym.abstraction.discretization.generate_box_bins
```

```{eval-rst}
.. autofunction:: verigym.abstraction.discretization.centered_pow_bin
```

```{eval-rst}
.. autofunction:: verigym.abstraction.discretization.nvec_from_samples
```

## Gymnasium wrappers and spaces

```{eval-rst}
.. autoclass:: verigym.abstraction.gym_utils.transform_observation.ReplaceInfObservation
   :members:
```

```{eval-rst}
.. autoclass:: verigym.abstraction.gym_utils.transform_observation.DiscretizeBoxObservation
   :members:
```

```{eval-rst}
.. autoclass:: verigym.abstraction.gym_utils.transform_action.DiscretizeBoxAction
   :members:
```

```{eval-rst}
.. autofunction:: verigym.abstraction.gym_utils.spaces.is_bounded_space
```

```{eval-rst}
.. autofunction:: verigym.abstraction.gym_utils.spaces.get_n_elements_of_space
```
