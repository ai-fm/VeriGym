# Import and Export

`VeriGym` can import from and export to several model types and languages.
Further, `VeriGym` also allows import of policies that were derived with external solvers.
Here we provide an overview of the included imports and exports as of `v.0.1.0`.

The `io` module contains functionality for import and export to external tools, and some datastructures from other pythonic frameworks can directly be imported through subclasses of `VeriGymEnv` and `PolicyClass`.

## Importing environments

We first consider several ways of loading **environments** from different inputs, including **external files** and **pythonic objects**.

To **connect to external file formats**, `VeriGym` has convenience funtions in the `io` module that read explicitly specified models from the `.umb` and `.drn` file-formats.
`VeriGym` loads models specified in either format into a `FrameworkExplicitEnv`, and allows loading to `IntervalExplicitEnv` for `.umb`.
- `.drn` is a format compatible with the PRISM model checker and can also be read by stormpy.
- `.umb`, the Unified Markov Binary format, is a recent file format that enables compatibility with several state-of-the-art model checking tools, including PRISM, Storm, and Modest (for statistical model checking).

Both functions take two input parameters: (1) `in_file`, the path to the model file and (2) `flat`, a boolean parameter that indicates whether to load states as integers or factored representations.

```python
from verigym.io.importer import load_from_umb, load_from_drn

umb_fw_env = load_from_umb(in_file="path/to/model.umb", flat=True)

drn_fw_env = load_from_drn(in_file="path/to/model.drn", flat=True) 
```

:::{admonition} Interval MDPs
:class: warning
As of version `1.14.0`, `stormpy` does not support loading IMDPs from `.drn` files. IMDPs can be loaded from `.umb`.
:::

A further option is to load files with symbolic model descriptions in `PRISM` language (files with `.prism`, `.pm`, and `.nm` endings) for simulation. 
This leads to a generative representation which does not require constructing the full underlying MDP.
The output type of the environment is a `SymbolicGenerativeEnv`.
Internally, the simulation is handled using `stormpy`.

```python 
from verigym.environments.generativeenv import GenerativeEnv

sim_env = GenerativeEnv.from_prism("path/to/model.prism")
# type(sim_env) == SymbolicGenerativeEnv
```

We can also load environments from **datastructures with pythonic interfaces**, specifically generative environments from `gymnasium` environments, and explicit environments from `stormpy` MDPs and IMDPs.

```python
import gymnasium as gym
from verigym.environments.generativeenv import GenerativeEnv
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv

env = gym.make("CartPole-v1")
gen_env = GenerativeEnv.from_gymnasium(env)

fw_env = FrameworkExplicitEnv.from_stormpy(mdp, flat=True)
# type(fw_env) == FrameworkExplicitEnv
fw_i_env = FrameworkExplicitEnv.from_stormpy(imdp, flat=True)
# type(fw_i_env) == IntervalFrameworkExplicitEnv
```

### Adding custom frameworks

`VeriGym` can easily be extended with further model types.
Here is a short pseudo-code overview of how to add further frameworks.

To import from further **generative/simulator-based frameworks**, add a new classmethod to `GenerativeEnv`.

```python
# in src/verigym/environments/generativeenv.py
class GenerativeEnv(VeriGymEnv):
    ...
    @classmethod
    def from_framework(cls, simulator):
        instance = cls.__new__(cls)
        ...
        GenerativeEnv.__init__(instance)
        return instance
```

To import from further **explicit model formats**, add a custom formatter and add a new classmethod to `FrameworkExplicitEnv`.
The formatter converts information about the explict model type into `VeriGym`-compatible form.
For reference, check out the `StormpyFormatter`!

```python
# in src/frameworks/customframework/formatter.py
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
    def from_framework(cls, model, flat: bool = True, render_mode: str | None = None):
        formatter = CustomFormatter(model)
        instance = cls(model=model, formatter=formatter, flat=flat, render_mode=render_mode)
        return instance

```

## Importing externally computed policies

`VeriGym` currently supports loading policies computed in `PRISM`, `stormpy`, and `stable-baselines3`. 

Policies computed in `stormpy` are given by an object `scheduler` of type `stormpy.storage.Scheduler`.
They can be initialized with an optional parameter `label_to_action`, that maps choice labels of the `mdp` to actions in the `env`. If it is left to `None`, these are derived internally using the function `_unwrap_scheduler`.
```python
from verigym.frameworks.stormpy.stormpypolicy import StormpyPolicy

stormpy_policy = StormpyPolicy(scheduler, abstraction_mapper, mdp, label_to_action)
```

:::{admonition} `stormpy` schedulers
:class: warning
Our implementation currently assumes memoryless deterministic policies.
:::

Policies computed in `PRISM` can be loaded from `.tra` files. 
```python
from verigym.frameworks.prism.prismpolicy import PrismPolicy

prism_policy = PrismPolicy("path/to/policy.tra", abstraction_mapper, action_map)
```

:::{admonition} `PRISM` policies
:class: warning
Our implementation currently assumes memoryless deterministic policies.
Further, our implementation assumes that the policy was generated in PRISM with option `states=false`. This assumption will be relieved in future updates.
:::

The `SB3Policy` class can load any opject that implements a `predict` method such as the classes `BaseAlgorithm` or `BasePolicy` from the `stable-baselines3`library.
Policies can be queried deterministically (taking the most likely action) or sampled from the policy's action distribution. 

```python
from verigym.frameworks.stable_baselines3.policy import SB3Policy

sb3_policy = SB3Policy(policy, env, abstraction_mapper, deterministic)
```

### Adding custom frameworks

To **load policies computed by external frameworks**, add a new sub-class of `PolicyClass` in `src/frameworks/customframework/policy.py`.
For reference, check out the implemented policy classes above!

```python
class CustomFrameworkPolicy(PolicyClass):
    def __init__(self, policy: Any, abstraction_mapper: AbstractionMapper):
        ...

    def _action_from_policy(self, obs):
        ...
```

# Exporting explicit models from `VeriGym`

We can export MDPs or IMDPs from explicit environment representations in `VeriGym` (i.e., instances of sub-classes of `BaseExplicitEnv`) to several output formats, specifically `stormpy` MDPs and IMDPs, and the `.drn` and `.umb` file formats, which can be read by external model checkers, such as Storm, PRISM, and Modest.

```python
export_to_stormpy_mdp(env, overapproximate) 
# returns a stormpy.storage.SparseMdp

export_to_stormpy_imdp(env, overapproximate, use_reward_uncertainty) 
# returns a stormpy.storage.SparseIntervalMdp

export_to_drn(env, "path/to/outfile.drn")
# writes an MDP directly to a .drn file

export_to_umb(env, "path/to/outfile.umb", intervals, overapproximate)
# writes an (I)MDP directly to a .umb file
```

### Adding custom exports

To export to different data types or file formats, add an export function to `src/verigym/io/exporter.py`-

```python
def export_to_datatype(env: BaseExplicitEnv, overapproximate: bool = True):
    ...
```