# Importing and Exporting of Models and Policies

The `verigym.io` module contains functionality for importing and exporting models to and from external tools.
For importing policies from other frameworks see [here](#importing-policies).

## Overview

**Models**

| Format / framework | Import into `VeriGym` | Export from `VeriGym`  | Notes |
|---|---|---|---|
| `gymnasium` environment | `GenerativeEnv.from_gymnasium(env)` | – | Simulator only, no explicit model |
| PRISM program (`.prism`, `.pm`, `.nm`) | `GenerativeEnv.from_prism(path)` | – | `verigym` uses `stormpy` simulator under the hood → the full MDP is not built |
| `stormpy` Python object | `FrameworkExplicitEnv.from_stormpy(model)` | `export_to_stormpy_mdp(env)`, `export_to_stormpy_imdp(env)` | MDP/IMDP. |
| `.umb` file | `load_from_umb(path)` | `export_to_umb(env, path)` | MDP/IMDP. Initial distribution is loaded as uniform over the initial states.  |
| `.drn` file | `load_from_drn(path)` | `export_to_drn(env, path)`| Flat states only; initial distribution is loaded as uniform |

**Policies**

| Framework | `verigym` Class | Input | Notes |
|---|---|---|---|
| Storm (`stormpy`) | `StormpyPolicy` | `stormpy.storage.Scheduler` and the MDP | Memoryless, deterministic |
| PRISM | `PrismPolicy` | `.tra` file | Memoryless, deterministic, exported with `states=false` |
| Stable-Baselines3 | `SB3Policy` | Any object with a `predict` method (e.g. a trained `PPO`) | – |


## Importing Models

The `verigym.io` module allows importing of models from the following formats:
- any `gym.Env` environment following the `gymnasium` API.
- instantiated Python objects holding MDPs from Stormpy.
- `.umb`, the [Unified Markov Binary format](https://github.com/pmc-tools/umbi/), a file format for explicit models, which enables compatibility between several state-of-the-art model checking tools, including PRISM, Storm, and Modest.
- `.drn`, a format for explicit models, which is compatible with the model checkers PRISM and Storm.
- `.prism`, `.pm` and `.nm` file formats following the PRISM language.


### Loading `gymnasium` Environments

We can also load environments from **datastructures with pythonic interfaces**, specifically generative environments from `gymnasium` environments, and explicit environments from `stormpy` MDPs and IMDPs.

We can load any simulation environment that follows the `gymnasium` API.
A conversion to a `GenerativeEnv` object is sometimes required for the environment to be used within the `verigym` framework.

```python
import gymnasium as gym
from verigym.environments.generativeenv import GenerativeEnv

env = gym.make("CartPole-v1")
gen_env = GenerativeEnv.from_gymnasium(env)
```

### Importing from `stormpy` objects

Loading of instantiated `stormpy` objects (MDPs or IMDPs) into the `verigym` framework is possible using the same function:

```python
from verigym.environments.frameworkexplicitenv import (
    FrameworkExplicitEnv, IntervalFrameworkExplicitEnv
)

env = FrameworkExplicitEnv.from_stormpy(mdp, flat=True)
assert isinstance(env, FrameworkExplicitEnv)

env_i = FrameworkExplicitEnv.from_stormpy(imdp, flat=True)
assert isinstance(env_i, IntervalFrameworkExplicitEnv)
```

### Loading Models from files

We can load models stored as `.umb` and `.drn` files using `load_from_umb()` and `load_from_drn()` respectively.
The resulting object class will either be a `FrameworkExplicitEnv` or `IntervalFrameworkExplicitEnv`.

Note: the argument `flat` is a boolean flag that indicates whether to load states as integers or in a factored representation (see the [abstraction mapping guide](abstraction_mapping.md) for more details). `.drn` files do not store state valuations, so they are always loaded with flat states.

```python
from verigym.io.importer import load_from_umb, load_from_drn

env_umb = load_from_umb(in_file="path/to/model.umb", flat=True)

env_drn = load_from_drn(in_file="path/to/model.drn", flat=True)
```

:::{admonition} Interval MDPs
:class: warning
As of `verigym` version 0.1.0, IMDPs may only be loaded from `.umb`.
:::


We can load **PRISM programs** (`.prism`, `.pm`, and `.nm`) for simulation. 
The model is loaded in a generative representation which does not require constructing the full underlying MDP in memory.
The `verigym` class of such a model is a `SymbolicGenerativeEnv`.
Internally, the simulation is handled using `stormpy`.

```python
from verigym.environments.generativeenv import GenerativeEnv, SymbolicGenerativeEnv

sim_env = GenerativeEnv.from_prism("path/to/model.prism")
assert isinstance(sim_env, SymbolicGenerativeEnv)
```

## Importing policies

`verigym` currently supports loading policies computed in `stormpy`, `PRISM`, and `stable-baselines3`. 

### Policies from `stormpy`

We may load a `stormpy.storage.Scheduler` into `verigym` by using the `StormpyPolicy` class.

```python
import stormpy
from verigym.frameworks.stormpy.stormpypolicy import StormpyPolicy

assert isinstance(scheduler, stormpy.storage.Scheduler)
stormpy_policy = StormpyPolicy(scheduler, abstraction_mapper, mdp, label_to_action)
```

:::{admonition} `stormpy` schedulers
:class: warning
`StormpyPolicy` currently assumes memoryless deterministic policies.
:::

The `StormpyPolicy` class may be instantiated with the `label_to_action` argument, which maps choice labels of the `mdp` to actions in the `env`. If it is left to `None`, these are derived internally using the function `_unwrap_scheduler`.


### Policies from PRISM

Policies computed in `PRISM` can be loaded from `.tra` files.
```python
from verigym.frameworks.prism.prismpolicy import PrismPolicy

prism_policy = PrismPolicy("path/to/policy.tra", abstraction_mapper, action_map)
```

:::{admonition} `PRISM` policies
:class: warning
`PrismPolicy` currently assumes memoryless deterministic policies and that the policy was generated in PRISM with option `states=false`. This assumption will be relaxed in future updates.
:::

### Policies from Stable-Baselines3

The `SB3Policy` class in `verigym` can load any policy produced by `stable_baselines3` which matches the `PolicyPredictor` class.

```python
from stable_baselines3.common.type_aliases import PolicyPredictor
from verigym.frameworks.stable_baselines3.policy import SB3Policy

assert isinstance(policy, PolicyPredictor)
sb3_policy = SB3Policy(policy, env, abstraction_mapper, deterministic=True)
```


## Exporting explicit models from `VeriGym`

We can export MDPs or IMDPs from explicit environment representations in `VeriGym` (i.e., instances of sub-classes of `BaseExplicitEnv`) to several output formats, specifically `stormpy` MDPs and IMDPs, and the `.drn` and `.umb` file formats, which can be read by external model checkers, such as Storm, PRISM, and Modest.

```python
from verigym.io.exporter import (
    export_to_stormpy_mdp, export_to_stormpy_imdp, export_to_drn, export_to_umb
)

# returns a stormpy.storage.SparseMdp
mdp = export_to_stormpy_mdp(env, overapproximate=True)

# returns a stormpy.storage.SparseIntervalMdp
imdp = export_to_stormpy_imdp(env, overapproximate=True, use_reward_uncertainty=False)

# writes an MDP directly to a .drn file
export_to_drn(env, "path/to/outfile.drn")

# writes an (I)MDP directly to a .umb file
export_to_umb(env, "path/to/outfile.umb", intervals=None, overapproximate=True)
```

## Extending `VeriGym`

### Adding custom model frameworks

`VeriGym` can easily be extended with further model types.
Here is a short pseudo-code overview of how to add further frameworks.

To import from a **new generative/simulator-based framework**, add a new classmethod to `GenerativeEnv`.

```python
# in src/verigym/environments/generativeenv.py
class GenerativeEnv(VeriGymEnv):
    ...
    @classmethod
    def from_new_framework(cls, simulator):
        instance = cls.__new__(cls)
        ...
        GenerativeEnv.__init__(instance)
        return instance
```

To import from **new explicit model formats**, add a custom formatter and add a new classmethod to `FrameworkExplicitEnv`.
The formatter converts information about the explicit model type into `verigym`-compatible form.
For reference, check out the `StormpyFormatter`!

```python
# in src/verigym/frameworks/customframework/formatter.py
class CustomFormatter(ExplicitFormatter):
    def __init__(self, model):
        ...

    def _convert_transition_matrix(self, transition_matrix) -> TransitionFunction:
        ...

    def convert_reward_matrix(self, reward_models) -> RewardFunction:
        ...

# in src/verigym/environments/frameworkexplicitenv.py
class FrameworkExplicitEnv(BaseExplicitEnv):
    ...
    @classmethod
    def from_new_framework(cls, model, flat: bool = True, render_mode: str | None = None):
        formatter = CustomFormatter(model)
        instance = cls(model=model, formatter=formatter, flat=flat, render_mode=render_mode)
        return instance
```

### Adding custom policy frameworks

To **load policies from new frameworks** currently not supported by `verigym`, add a new class inheriting from `PolicyClass` in `src/verigym/frameworks/customframework/policy.py`.
For reference, check out the implemented policy classes in that file.

```python
class CustomFrameworkPolicy(PolicyClass):
    def __init__(self, policy: Any, abstraction_mapper: AbstractionMapper):
        ...

    def _action_from_policy(self, obs):
        ...
```

### Adding custom exports

To export to different data types or file formats, add an export function following the example of functions in `src/verigym/io/exporter.py`.

```python
def export_to_datatype(env: BaseExplicitEnv, overapproximate: bool = True):
    ...
```
