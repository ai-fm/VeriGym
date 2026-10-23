# Artifact Evaluation for TACAS 2027 Paper 1441

**Title:** `VeriGym`: From Deep Reinforcement Learning Simulators to Verifiable Models and Back

**Paper number:** 1441

**Find the paper file:** `path/to/paper`

**Availability:** 
- Our artifact is permanently available on [Zenodo]() with DOI: . 
- The updated repository is publicly available on [GitHub](https://github.com/ai-fm/VeriGym).

**Badges:** We apply for all three badges: Available, Functional, and Reusable.

**Tested Platforms:** ... 

**Documentation:**
- Our public documentation is available [online](https://ai-fm.github.io/VeriGym/).
- For offline use within the artifact, a compiled version is included in `path/to/docs/frontpage`.


# Running the Artifact (Docker)

...

# Smoke Test Evaluation

**Estimated time:**

**Estimated resources:**

Images are stored in the following folders:
- `artifact_out/smoke_reference/` contains pre-generated figures from the paper/expected outputs.
- `artifact_out/smoke_eval/` is where figures generated during evaluation are stored

The full list of expected figures for the **smoke test** is:
- `path..`

# Full Evaluation

**Estimated time:**

**Estimated resources:**

## Functional Badge: Replicating the Plots from the Paper

Images are stored in the following folders:

- `artifact_out/full_reference/` contains pre-generated figures from the paper/expected outputs.
- `artifact_out/full_eval/` is where figures generated during evaluation are stored

The full list of expected figures from the **workflows** chapter is:
- `workflow1_figure4a.pdf`
- `workflow1_figure4b.pdf`
- `workflow1_appendix_figure14a.pdf`
- `workflow1_appendix_figure14b.pdf`
- `workflow1b_figure12a.pdf`
- `workflow1b_figure12b.pdf`
- `workflow1d_figure13.pdf`
- `workflow2_figure6a.pdf`
- `workflow2_figure6b.pdf`
- `workflow2_figure6c.pdf`
- `workflow3_figure8.pdf`

The full list of expected figures from the **workflows** chapter is:
- `tech_eval_q1_figure9a.pdf`
- `tech_eval_q1_figure15a.pdf`
- `tech_eval_q1_figure15b.pdf`
- `tech_eval_q1_figure16a.pdf`
- `tech_eval_q2_figure9b.pdf`
- `tech_eval_q2_figure16b.pdf`
- `tech_eval_q3_figure10.pdf`
- `tech_eval_q4_figure11.pdf`



### Note on the technical evaluation results
> Exact results of the technical evaluation are hardware dependent and may deviate for reviewers. Visible trends however should stay the same.  


> We used the following **technical setup for paper results**: Ubuntu 22.04.5 LTS benchmark server with Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz, NVIDIA GeForce RTX 3090 (24 GB VRAM) GPU and 256 GB (8 x 32GB DDR4-3200 RAM). We used Python 3.12.15 and `VeriGym`v1.0.0.

## Reusable Badge: 

### Documentation, examples, API

Our publicly available documentation contains examples, workflows, tutorials, and API reference at [DOCS](https://ai-fm.github.io/VeriGym/index.html).

### Installing `VeriGym``

`VeriGym` is installable from **PyPi**:

```
uv pip install verigym
# or
pip install verigym
```

Further installation instructions for development are included in our documentation for [Installation](https://ai-fm.github.io/VeriGym/installation.html)

### Example usages

We include two types of examples:
1. **Guides** showcase the different capabilities of one part of `VeriGym`. For example, check out the guide for [Abstraction Mapping](https://ai-fm.github.io/VeriGym/guides/abstraction_mapping.html)
2. **Workflows** are full pipeline executions of different tasks, which we also report in our paper. You can find them in our documentation on [Workflows](https://ai-fm.github.io/VeriGym/workflows.html), or as executable Jupyter notebooks in our repository/this artifact in `VeriGym/examples/workflow_*.ipynb`


### Extending `VeriGym` and experiments

`VeriGym` experiments can easily be run with further benchmarks:
```
# gymnasium environments
gym_env = gym.make(GYM_NAME)
verigym.GenerativeEnv.from_gymnasium(gym_env)

# stormpy MDPs
mdp = verigym.load_stormpy_model(PRISM_FILE_PATH, constants, propertystr)
verigym.FrameworkExplicitEnv.from_stormpy(mdp)

# simulating PRISM programs
verigym.GenerativeEnv.from_prism(PRISM_FILE_PATH)
```

> **Note:** `VeriGym` v1.0.0 only supports models with 1 reward function.

`VeriGym`'s modular design allows easy extensions to **functionality**:
- inherit from `PolicyClass` to add further model exploration techniques
- define arbitrary input functions to `AbstractionMapper` for further space abstractions
- add new external frameworks using `ExplicitFormatter`, `PolicyClass`, and adding constructors via `VeriGymEnv.from_framework`

For the **easiest start to custom experiments in `VeriGym`**, we recommend modifying the existing workflow notebooks.

The [**Contribution Guide**](https://ai-fm.github.io/VeriGym/contribution.html) provides information for developers who would like to contribute to `VeriGym`.
