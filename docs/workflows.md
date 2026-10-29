# Workflows

:::{admonition} TODO: add text
:class: warning
Introduction to the workflows.
:::

- {doc}`notebooks/workflow_abstraction_refinement`: Learns finer and finer abstractions of CartPole until the policy computed by the model checker performs well enough on the original environment.
- {doc}`notebooks/workflow_compare_rl_model_checking`: Trains a DQN agent on a PRISM model and compares it with the optimal policy computed by model checking.
- {doc}`notebooks/workflow_comparing_abstraction_policies`: Learns an abstract MDP of CartPole, computes a policy for it with `stormpy`, and evaluates that policy on the abstract and the original environment.
- {doc}`notebooks/workflow_comparing_exploration_policies`: Compares the exploration policies built into VeriGym for model learning on MountainCar, and shows how to write your own.
- {doc}`notebooks/workflow_external_tools`: Exports a learned abstraction, solves it with a Julia solver (`MCTS.jl`), and evaluates the resulting policy in VeriGym.
- {doc}`notebooks/workflow_featurereduction`: Removes state features from the MuJoCo Ant environment and compares RL training on the full and the reduced observations.
- {doc}`notebooks/workflow_shielding`: Builds a safety shield for CartPole from model checking results and compares PPO agents trained with and without it.

```{toctree}
:hidden:
:glob:

notebooks/workflow_*
```
