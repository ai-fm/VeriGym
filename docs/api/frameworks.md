# Frameworks and Model Checking

This module connects to external frameworks, like `stormpy` for model checking. We include methods for *building (I)MDPs from explicit environments*, convenience functions for *model checking* and *policy synthesis*, as well as wrappers to apply *shielding* to environments.

## Building stormpy models

```{eval-rst}
.. autofunction:: verigym.frameworks.stormpy.stormpy_utils.build_stormpy_mdp
```

```{eval-rst}
.. autofunction:: verigym.frameworks.stormpy.stormpy_utils.build_stormpy_imdp
```

```{eval-rst}
.. autofunction:: verigym.frameworks.stormpy.stormpy_utils.build_stormpy_dtmc
```

```{eval-rst}
.. autofunction:: verigym.frameworks.stormpy.stormpy_utils.load_stormpy_model
```

## Model checking

```{eval-rst}
.. autofunction:: verigym.supplements.modelchecking.get_policy_from_stormpy
```

```{eval-rst}
.. autofunction:: verigym.supplements.modelchecking.check_policy_value_in_stormpy
```

## Shielding

```{eval-rst}
.. autoclass:: verigym.supplements.shield_wrapper.AbstractPostShieldWrapper
   :members:
   :show-inheritance:
```
