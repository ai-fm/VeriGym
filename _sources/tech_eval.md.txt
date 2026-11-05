# Technical Evaluation

We provide a set of experiments for the technical evaluation of `verigym`.

- {doc}`notebooks/tech_eval_discretization_overhead`: Compares how fast VeriGym and Gymnasium discretize observations, depending on the number of dimensions and the data type.
- {doc}`notebooks/tech_eval_mdp_memory`: Compares the memory use of VeriGym's explicit model data structures with `stormpy` and dense `numpy` arrays on PRISM benchmarks.
- {doc}`notebooks/tech_eval_runtimes`: Breaks down how long each step of learning an abstraction takes, and how this grows with the number of bins and the exploration budget.

```{toctree}
:hidden:
:glob:

notebooks/tech_eval_*
```
