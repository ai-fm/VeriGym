# Quickstart

The quickest way to start working with `VeriGym` is to implement a simple pipeline from a continuous simulator environment to a model checkable MDP.

## A step-by-step guide: continuous environment &rarr; model checked result &rarr; evaluation

**Step 1:** Loading a continuous `Gymnasium` environment into `VeriGym`.

```python
import gymnasium as gym
import verigym

veri_env = verigym.GenerativeEnv.from_gymnasium(
    gym.make("MountainCarContinuous-v0")
)
```

**Step 2:** Defining an abstraction mapping.

We use a `linspace_mapper` and divide each state space dimension into 25 bins, and each action space dimension into 10 bins.
The `linspace_mapper` divides each dimension into equally spaced bins.

```python
from verigym.abstractions.abstractionmapper import linspace_mapper

abstraction_mapper = linspace_mapper(
    env=veri_env,
    n_bins_states=25,
    n_bins_actions=10
)
```

**Step 3:** Model learning.

We sample the continuous environment for 10.000 steps with an entropy-based exploration policy. 
Then, we learn the model by counting and averaging the observed transitions and rewards.

```python
from verigym.policy.qvalue import EntropyLearningPolicy

abstracted_env = verigym.create_abstraction(
    original_env=veri_env,
    abstraction_mapper=abstraction_mapper,
    exploration_policy=EntropyLearningPolicy(veri_env, abstraction_mapper),
    num_steps=int(1e4),
)
```

**Step 4:** Model checking (external).

We can export the MDP from our learned abstracted environment and model check it using `stormpy`. 
We check for maximal cumulative discounted reward with a discount factor of `0.995`.

```python
import stormpy
from verigym.frameworks.stormpy.stormpy_utils import build_stormpy_mdp

stormpy_mdp = build_stormpy_mdp(abstracted_env)
prop = stormpy.parse_properties(f"Rmax=?[Cdiscount=0.995]")[0]
result = stormpy.check_model_sparse(stormpy_mdp, prop, extract_scheduler=True)
```

**Step 5:** Policy evaluation.

We can import the model checked result into verigym as a `StormpyPolicy` object and then simulate on the original environment with this policy.


```python
verigym_policy = verigym.StormpyPolicy(result.scheduler, abstracted_env.abstraction_map, stormpy_mdp)

trajectories = veri_env.simulate(
    policy=verigym_policy,
    n_steps=int(10e4),
    seed = SEED
)
```

## More details, extensions, and other ways to use `VeriGym`

**Take a look at our workflows!**
Specifically, an extended version of this quickstart example is implemented in `workflow_comparing_abstraction_policies.ipynb`.

Further **workflows** include pipelines for:
- using an abstraction-refinement loop
- applying feature reduction for Deep Reinforcement Learning
- implementing a simple post-shield for a continuous environment
- connecting to external frameworks (`POMDPS.js`)

Our **guides** also include walk-throughs through central `VeriGym` modules:
- import and export from/to different datastructures and file types
- abstraction mapping
- model learning