# Environments

All `verigym` environments follow the `gymnasium` API and inherit from `VeriGymEnv`.
*Generative environments* wrap simulators, e.g., a `gymnasium` environment (`GenerativeEnv.from_gymnasium`) or a PRISM program (`GenerativeEnv.from_prism`).
*Explicit environments* store the full transition and reward functions (see [Model Components](model_components.md)), so they can be model checked:
an `AbstractedExplicitEnv` is learned from a simulator (see the [model learning guide](../guides/model_learning.md)), and a `FrameworkExplicitEnv` is loaded from an external tool, e.g., from `stormpy` or a `.umb` file (see the [import and export guide](../guides/import_export.md)).
*Interval environments* are the interval MDP (IMDP) counterparts of the explicit environments.

## Base class

```{eval-rst}
.. autoclass:: verigym.VeriGymEnv
   :members:
```

## Generative environments

```{eval-rst}
.. autoclass:: verigym.GenerativeEnv
   :members:
```

```{eval-rst}
.. autoclass:: verigym.environments.generativeenv.SymbolicGenerativeEnv
   :members:
   :show-inheritance:
```

## Explicit environments

```{eval-rst}
.. autoclass:: verigym.environments.base_explicitenv.BaseExplicitEnv
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: verigym.AbstractedExplicitEnv
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: verigym.FrameworkExplicitEnv
   :members:
   :show-inheritance:
```

## Interval environments

```{eval-rst}
.. autoclass:: verigym.environments.interval_explicitenv.IntervalEnvMixin
   :members:
```

```{eval-rst}
.. autoclass:: verigym.environments.interval_explicitenv.IntervalAbstractedExplicitEnv
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: verigym.environments.frameworkexplicitenv.IntervalFrameworkExplicitEnv
   :members:
   :show-inheritance:
```
