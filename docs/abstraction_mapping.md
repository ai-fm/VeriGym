# Abstraction Mapping

A simulator usually has a **continuous** state space: infinitely many states, so there is no
explicit MDP to reason about. An *abstraction* maps that space onto an **abstract
space**. The abstract space may be continuous or discrete.
If the abstract space is discrete and the abstraction map maps every original state to exactly one abstract state, VeriGym can sample the simulator and build an explicit MDP from it.

## The short version

```python
import gymnasium as gym
from verigym.abstraction.abstractionmapper import linspace_mapper
from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation

env = gym.make("CartPole-v1")
env = ReplaceInfObservation(env, neg_inf=-10, pos_inf=10)   # see the note below

mapper = linspace_mapper(env, n_bins_states=5, n_bins_actions=2)

observation, _ = env.reset(seed=42)
mapper.original_to_abstract_state_enum(observation)   # -> a single int, e.g. 312
```

That `mapper` is the single object `create_abstraction` needs:

```python
abstracted_env = verigym.create_abstraction(
    original_env=verigym.GenerativeEnv.from_gymnasium(env),
    abstraction_mapper=mapper,
    exploration_policy=RandomizedPolicy(...),
    num_steps=1000,
)
```

!!! warning "Unbounded spaces"
    You cannot cut an infinite interval into bins. `CartPole-v1` has `±inf` observation
    dimensions, so wrap it first:
    `env = ReplaceInfObservation(env, neg_inf=-10, pos_inf=10)`.

## The three objects

| object | what it is |
|---|---|
| `BinEdges` | *where* a space is cut — one array of bin edges per dimension |
| `AbstractionMap` | the mapping for **one** space (states *or* actions) |
| `AbstractionMapper` | the pair of maps for a whole environment (states **and** actions) |

A mapping does not have to be based on `BinEdges`. 
An `AbstractionMap` is just a forward callable, an (optional) backward
callable, and the abstract space they connect.
Binning is one way to produce such a mapping as is clustering, tile coding or some irregular partition.

## Going backwards (abstract → original)

An abstraction may throw information away, and so the backward may not be able to return your original sample —
only something representing the abstract state. *What* it returns is declared
by `backward_kind`:

| `backward_kind` | returns | use it for |
|---|---|---|
| `"point"` | one representative sample | deploying a policy |
| `"interval"` | lower and upper bound, shape `(2, *space.shape)` | model checking and labeling — "does `x > 2` hold in abstract state 17?" |
| `"set"` | an iterable of samples | a finite, explicit set of originals |
| `"unknown"` | — | no backward map was given; consumers raise rather than guess |

This has to be declared because it cannot be inferred: for a 2-D `Box` original space, a point, an interval and a
2-element set are all `(2, 2)` float arrays.

A backward map is **optional**. `create_abstraction` does not need one — but labeling and policy
deployment do.

## Discretizing via `BinEdges`

`BinEdges` are a useful tool when the abstract map should represent the discretization of a continuous original space.
Through defining the `BinEdge` we obtain the positions of the bins.
We achieve this via helper functions such as `generate_box_bins` mentioned further below.

**The four representations in `BinEdges`**  
When creating abstractions through the use of `BinEdges`, it is important to distinguish the four representations:

| symbol | name | domain | example | belongs to |
|---|---|---|---|---|
| **O** | `orig` | anything (for example $\mathbb{R}^d$) |  `[0.31, 7.42]` | the original space |
| **V** | `value` | countable (discretized $\mathbb{R}^d$) | `[0.25, 7.50]` | the sample snapped onto its bin edges (of 0.25 increments) |
| **I** | `idx` | $\mathbb{N}^d$ | `[1, 3]` | one bin index per dimension|
| **E** | `enum` |$\mathbb{N}^1$ | `8` | a single flat integer |

The example assumes a 2-dimensional continuous space ($\mathbb{R}^2$) as the original space. 
The example and its abstract space representations are visualized below indicating the conversion functions from one representation to another.

<div style="margin: 1.5rem 0; overflow-x: auto;">
<svg viewBox="0 48 930 175" style="width:100%; max-width: 760px; display:block; margin: 0 auto;" role="img" aria-label="Diagram of the four BinEdges representations O, V, I, E, chained by forward and backward conversions, with example values 0.31 7.42 for O, 0.25 7.50 for V, 1 3 for I, and 8 for E">
  <defs>
    <marker id="be-arrow-fwd" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--md-accent-fg-color)"/>
    </marker>
    <marker id="be-arrow-bwd" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--md-default-fg-color--light)"/>
    </marker>
  </defs>

  <!-- forward arrows (top) -->
  <g font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" font-size="10.5" fill="var(--md-accent-fg-color)">
    <line x1="174" y1="100" x2="276" y2="100" stroke="var(--md-accent-fg-color)" stroke-width="1.5" marker-end="url(#be-arrow-fwd)"/>
    <text x="225" y="90" text-anchor="middle">orig_to_value</text>
    <line x1="414" y1="100" x2="516" y2="100" stroke="var(--md-accent-fg-color)" stroke-width="1.5" marker-end="url(#be-arrow-fwd)"/>
    <text x="465" y="90" text-anchor="middle">value_to_idx</text>
    <line x1="654" y1="100" x2="756" y2="100" stroke="var(--md-accent-fg-color)" stroke-width="1.5" marker-end="url(#be-arrow-fwd)"/>
    <text x="705" y="90" text-anchor="middle">idx_to_enum</text>
  </g>

  <!-- backward arrows (bottom) -->
  <g font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" font-size="10.5" fill="var(--md-default-fg-color--light)">
    <line x1="276" y1="180" x2="174" y2="180" stroke="var(--md-default-fg-color--light)" stroke-width="1.5" marker-end="url(#be-arrow-bwd)"/>
    <text x="225" y="198" text-anchor="middle">value_to_orig</text>
    <line x1="516" y1="180" x2="414" y2="180" stroke="var(--md-default-fg-color--light)" stroke-width="1.5" marker-end="url(#be-arrow-bwd)"/>
    <text x="465" y="198" text-anchor="middle">idx_to_value</text>
    <line x1="756" y1="180" x2="654" y2="180" stroke="var(--md-default-fg-color--light)" stroke-width="1.5" marker-end="url(#be-arrow-bwd)"/>
    <text x="705" y="198" text-anchor="middle">enum_to_idx</text>
  </g>

  <!-- O -->
  <g>
    <rect x="40" y="60" width="130" height="150" rx="12" fill="var(--md-code-bg-color)" stroke="var(--md-default-fg-color--lighter)" stroke-width="1.5"/>
    <text x="105" y="100" text-anchor="middle" font-size="32" font-weight="700" fill="var(--md-accent-fg-color)">O</text>
    <text x="105" y="120" text-anchor="middle" font-size="12" font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" fill="var(--md-default-fg-color)">orig</text>
    <line x1="54" y1="132" x2="156" y2="132" stroke="var(--md-default-fg-color--lighter)"/>
    <path d="M86,142 L81,142 L81,176 L86,176" fill="none" stroke="var(--md-default-fg-color--lighter)" stroke-width="1.5"/>
    <path d="M124,142 L129,142 L129,176 L124,176" fill="none" stroke="var(--md-default-fg-color--lighter)" stroke-width="1.5"/>
    <text x="105" y="152" text-anchor="middle" font-size="14" font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" fill="var(--md-default-fg-color)">0.31</text>
    <text x="105" y="168" text-anchor="middle" font-size="14" font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" fill="var(--md-default-fg-color)">7.42</text>
    <text x="105" y="194" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color--light)">original space</text>
  </g>

  <!-- V -->
  <g>
    <rect x="280" y="60" width="130" height="150" rx="12" fill="var(--md-code-bg-color)" stroke="var(--md-default-fg-color--lighter)" stroke-width="1.5"/>
    <text x="345" y="100" text-anchor="middle" font-size="32" font-weight="700" fill="var(--md-accent-fg-color)">V</text>
    <text x="345" y="120" text-anchor="middle" font-size="12" font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" fill="var(--md-default-fg-color)">value</text>
    <line x1="294" y1="132" x2="396" y2="132" stroke="var(--md-default-fg-color--lighter)"/>
    <path d="M326,142 L321,142 L321,176 L326,176" fill="none" stroke="var(--md-default-fg-color--lighter)" stroke-width="1.5"/>
    <path d="M364,142 L369,142 L369,176 L364,176" fill="none" stroke="var(--md-default-fg-color--lighter)" stroke-width="1.5"/>
    <text x="345" y="152" text-anchor="middle" font-size="14" font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" fill="var(--md-default-fg-color)">0.25</text>
    <text x="345" y="168" text-anchor="middle" font-size="14" font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" fill="var(--md-default-fg-color)">7.50</text>
    <text x="345" y="194" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color--light)">snapped to bins</text>
  </g>

  <!-- I -->
  <g>
    <rect x="520" y="60" width="130" height="150" rx="12" fill="var(--md-code-bg-color)" stroke="var(--md-default-fg-color--lighter)" stroke-width="1.5"/>
    <text x="585" y="100" text-anchor="middle" font-size="32" font-weight="700" fill="var(--md-accent-fg-color)">I</text>
    <text x="585" y="120" text-anchor="middle" font-size="12" font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" fill="var(--md-default-fg-color)">idx</text>
    <line x1="534" y1="132" x2="636" y2="132" stroke="var(--md-default-fg-color--lighter)"/>
    <path d="M566,142 L561,142 L561,176 L566,176" fill="none" stroke="var(--md-default-fg-color--lighter)" stroke-width="1.5"/>
    <path d="M604,142 L609,142 L609,176 L604,176" fill="none" stroke="var(--md-default-fg-color--lighter)" stroke-width="1.5"/>
    <text x="585" y="152" text-anchor="middle" font-size="14" font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" fill="var(--md-default-fg-color)">1</text>
    <text x="585" y="168" text-anchor="middle" font-size="14" font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" fill="var(--md-default-fg-color)">3</text>
    <text x="585" y="194" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color--light)">per-dim index</text>
  </g>

  <!-- E -->
  <g>
    <rect x="760" y="60" width="130" height="150" rx="12" fill="var(--md-code-bg-color)" stroke="var(--md-default-fg-color--lighter)" stroke-width="1.5"/>
    <text x="825" y="100" text-anchor="middle" font-size="32" font-weight="700" fill="var(--md-accent-fg-color)">E</text>
    <text x="825" y="120" text-anchor="middle" font-size="12" font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" fill="var(--md-default-fg-color)">enum</text>
    <line x1="774" y1="132" x2="876" y2="132" stroke="var(--md-default-fg-color--lighter)"/>
    <text x="825" y="163" text-anchor="middle" font-size="14" font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" fill="var(--md-default-fg-color)">8</text>
    <text x="825" y="194" text-anchor="middle" font-size="10.5" fill="var(--md-default-fg-color--light)">flat integer</text>
  </g>
</svg>
</div>

Not all conversion functions are displayed in the visualization. 
`BinEdges` provides a conversions between any two of the representations, named `BinEdges.<from>_to_<to>`:

```python
# one direction
BinEdges.orig_to_idx(sample)     # O -> I
BinEdges.orig_to_enum(sample)    # O -> E
BinEdges.orig_to_value(sample)   # O -> V
BinEdges.value_to_idx(value)     # V -> I
BinEdges.value_to_enum(value)    # V -> E
BinEdges.idx_to_enum(index)      # I -> E
# backward direction
BinEdges.enum_to_idx(8)          # E -> I
BinEdges.enum_to_value(8)        # E -> V
BinEdges.enum_to_orig(8)         # E -> O
BinEdges.idx_to_value(index)     # I -> V
BinEdges.idx_to_orig(index)      # I -> O
BinEdges.value_to_orig(index)    # V -> O
```

## Factored Index (`I`) vs enumerated (`E`)

A `BinEdge`'s map's `orig_to_idx` returns the **factored** index `I`, because that is what generalises to
abstractions that are not binnings. But `create_abstraction` uses abstract states as dictionary
keys and array indices (`T_counts[s][a][s_next]`), and a numpy array is not hashable — so it needs
the single `int` `E`.

Two accessors give you that, and they differ only in what they *take as input*:

- `abstract_to_enum(i)` — takes an already-abstract sample `I`
- `original_to_enum(x)` — takes an original sample `O` and does both steps

!!! warning "Future versions will not use `AbstractionMap.forward_map`"
    Instead they will use `AbstractionMap.original_to_abstract`.

!!! note "Enumeration is explicit, never assumed"
    Flattening `I` into `E` assumes the abstract space is a rectangular grid (`gym.spaces.Box`). 
    But an `AbstractionMap`
    has **no default enumeration**: it is either given one or it has none, and `AbstractionMap.is_enumerable`
    tells you which. The built-in convenience function provide the enumeration for you; if you build a map by hand over a
    rectangular space, pass `enumeration_of_space(abstract_space)`.

The function `validate_for_abstraction(mapper)` checks all of this once, up front, with a readable error —
instead of failing deep inside a worker process.


## Choosing where to cut the space

`BinEdges` places a set of cut points along a dimension; every sample is assigned the index of the
bin it falls into. The two outer bins are **open-ended**, meaning that they catch everything below the first
edge, or above the last edge:

<div style="margin: 1.5rem 0; overflow-x: auto;">
<svg viewBox="0 38 900 95" style="width:100%; max-width: 700px; display:block; margin: 0 auto;" role="img" aria-label="Number line with four bin edges at -3, -1, 1, and 3, creating five bins indexed 0 to 4. Bin 0, below -3, and bin 4, above 3, are open-ended and extend past the edges of the visible line.">
  <defs>
    <marker id="numline-arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--md-default-fg-color--light)"/>
    </marker>
    <linearGradient id="numline-fade-left" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="var(--md-code-bg-color)" stop-opacity="1"/>
      <stop offset="1" stop-color="var(--md-code-bg-color)" stop-opacity="1"/>
    </linearGradient>
    <linearGradient id="numline-fade-right" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="var(--md-code-bg-color)" stop-opacity="1"/>
      <stop offset="1" stop-color="var(--md-code-bg-color)" stop-opacity="1"/>
    </linearGradient>
  </defs>

  <!-- bin bands -->
  <rect x="50" y="78" width="160" height="24" rx="4" fill="url(#numline-fade-left)"/>
  <rect x="370" y="78" width="160" height="24" rx="4" fill="var(--md-code-bg-color)"/>
  <rect x="690" y="78" width="160" height="24" rx="4" fill="url(#numline-fade-right)"/>

  <!-- number line, open-ended (arrow) on the right -->
  <line x1="50" y1="90" x2="850" y2="90" stroke="var(--md-default-fg-color--light)" stroke-width="1.5" marker-end="url(#numline-arrow)"/>

  <!-- bin edges (ticks) -->
  <g stroke="var(--md-default-fg-color)" stroke-width="1.5">
    <line x1="210" y1="74" x2="210" y2="106"/>
    <line x1="370" y1="74" x2="370" y2="106"/>
    <line x1="530" y1="74" x2="530" y2="106"/>
    <line x1="690" y1="74" x2="690" y2="106"/>
  </g>

  <!-- edge values -->
  <g font-size="14" font-family="var(--md-code-font-family, ui-monospace, SFMono-Regular, Menlo, monospace)" fill="var(--md-default-fg-color)" text-anchor="middle">
    <text x="210" y="120">-3</text>
    <text x="370" y="120">-1</text>
    <text x="530" y="120">1</text>
    <text x="690" y="120">3</text>
  </g>

  <!-- bin indices -->
  <g font-size="16" font-weight="700" fill="var(--md-accent-fg-color)" text-anchor="middle">
    <text x="130" y="52">0</text>
    <text x="290" y="52">1</text>
    <text x="450" y="52">2</text>
    <text x="610" y="52">3</text>
    <text x="770" y="52">4</text>
  </g>

  <!-- row labels -->
  <g font-size="18" font-style="italic" fill="var(--md-default-fg-color--light)" text-anchor="end">
    <text x="80" y="52">bin index</text>
    <text x="80" y="120">value</text>
  </g>
</svg>
</div>

`generate_box_bins(space, bin_func, n_bins)` accepts *any* callable where `bin_func` matches
`(low, high, n_bins) -> array`, so adding a new discretization type is a one-line change.
We provide some defaults:

- `np.linspace` — equidistant
- `centered_pow_bin` — narrow bins in the middle, wide at the borders (e.g. a pole angle near 0)
- your own function — must return `n_bins` ascending edges from `low` to `high`
- data-driven — a closure returning quantiles of observed samples, so visited regions get finer bins

`n_bins` may be an array, giving each dimension its own resolution.

!!! warning "adjust/explanation to newest `BinEdge` update"
    There was a new PR, we need to update the explanation for that.

!!! warning "data-driven"
    Data-driven will soon be added.

## When no abstraction is needed

Some environments are already discrete (`Taxi-v3`). Use the identity map, which returns every input
unchanged:

```python
mapper = AbstractionMapper.initialize_identity_mapper(
    state_space=gym.spaces.Discrete(500),
    action_space=gym.spaces.Discrete(6),
)
```

`AbstractionMap.initialize_identity_map(space)` does the same for a single space — useful when
states are continuous but actions are already discrete.

## Where to go next

The notebook `examples/tutorial_abstractionmap.ipynb` walks through all of this with runnable code,
including custom and data-driven bin functions and a full `CartPole-v1` example.
