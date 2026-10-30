# Artifact Evaluation for TACAS 2027 Paper 1441

- **Title:** `VeriGym`: From Deep Reinforcement Learning Simulators to Verifiable Models and Back
- **Paper number:** 1441
- **Paper file:** `TACAS27_paper__1441.pdf`
- **Badges:** We apply for all three badges: Available, Functional, Reusable.

## Availability

- Our artifact is permanently available on [Zenodo](), DOI: 
- Our maintained repository is publicly available online on [GitHub](https://github.com/ai-fm/VeriGym).
- You can find our publicly available documentation [online](https://ai-fm.github.io/VeriGym/) (recommended for re-use), or compiled as part of the Zenodo artifact (open with `Verigym/docs/index.html`).

## Setting up and running the Artifact

The artifact uploaded to Zenodo includes pre-built **Docker images**.

(OPTIONAL:) If you would like to rebuild the container from scratch, please check the instructions for running [`VeriGym`with Docker](#verigym-in-docker).

To load and prepare for evaluation, execute the commands below.

**Download the artifact from Zenodo**, from [here]().
Unzip and open a terminal in the artifact folder.

**Load the image according to your system architecture.** 
```bash
docker load -i verigym-arm64.tar.gz     # on Apple Silicon and other ARM machines
```
Or,
```bash
docker load -i verigym-amd64.tar.gz     # on Linux with AMD/Intel
```

If needed, check the troubleshooting section in [VeriGym in docker](#verigym-in-docker).

**Create directories** to store the results on your machine:

```bash
mkdir -p results/smoke
mkdir -p results/full
```
Make sure you have read/write permissions for the folders.

For more technical information (including tested platforms), see the section on [technical details](#further-technical-information).

## Running the Smoke Test

- **estimated time**: <10 minutes on a standard laptop
- **required resources**: min. 4GB

Don't forget to create a local directory to store results (see above):
```bash
mkdir -p results/smoke
```

**Run this command:**

```
docker run --rm --network none -v "$PWD/results/smoke:/results:z" localhost/verigym repro --quick
```

This will run a short version of the scripts. We mount the `results` folder in the container to `./results/smoke`. The container will be deleted upon exit.

### Expected Outputs
**Console:**

> **Note:**
> Exact runtimes may differ per machine and per run.
> 10/10 notebooks should finish successfully, indicated by ✅. Failure is indicated by ❌.

```
Running 10 notebook scripts (QUICK=1); logs in docs/notebooks/output/logs/
[1/10] workflow_external_tools ... ✅ 0m10s
[2/10] workflow_comparing_abstraction_policies ... ✅ 0m40s
[3/10] workflow_shielding ... ✅ 0m04s
[4/10] workflow_comparing_exploration_policies ... ✅ 0m03s
[5/10] tech_eval_discretization_overhead ... ✅ 0m18s
[6/10] tech_eval_mdp_memory ... ✅ 0m15s
[7/10] workflow_compare_rl_model_checking ... ✅ 0m37s
[8/10] workflow_abstraction_refinement ... ✅ 0m03s
[9/10] workflow_featurereduction ... ✅ 0m41s
[10/10] tech_eval_runtimes ... ✅ 0m13s
```

**Figures:** 

Reference figures are stored in the artifact folder `reference_outputs/smoke`. 

Results for the **technical evaluation** are stored in `results/smoke/tech_eval/`, with subfolders:
- `discretization_overhead/` (expect 5 figures): 
    - (1) `break_even_steps.pdf`, (2) `compile_cost_datatype.pdf`, (3) `discretization_overhead_env_step.pdf`, (4) `discretization_times.pdf`, (5) `time_BinEdges_conversion_functions.pdf`
- `memory_eval/`(expect 6 figures): 
    - (1)`memory_eval_1a_transition_func.pdf`, (2) `memory_eval_1b_size_environment.pdf`, (3) `memory_eval_2a_transition_func.pdf`, (4) `memory_eval_2b_environment_size.pdf`, (5) `memory_eval_3b_environment_size.pdf`, (6)`memory_eval_3b_transition_func.pdf`
- `runtimes/quick/` (expect 3 figures):
    - (1) `policy_vs_explore_budget.pdf`, (2) `runtime_breakdown_share.pdf`, (3) `runtime_breakdown.pdf`

Results for *workflows* are stored in `results/smoke/workflows/figures/`, with subfolders:
- `abstraction_refinement/` (expect 1 figure): 
    - (1)`workflow_abstraction_refinement_cartpole.pdf`
- `compare_files/` (expect 3 figures)
    - (1) `comparing_abstraction_policies_with_abstracted_ppo.pdf`, (2) `comparing_abstraction_policies.pdf`, (3) `ppo_training_curve.pdf`
- `ext_tools_files/`(expect 1 figure)
    - (1) `mcts_vs_model_checking.pdf`
- `feature_reduction/` (expect 3 figures):
    - (1) `eval_rewards.pdf`, (2) `sac_training_ep_lengths.pdf`, (3) `sac_training_rewards.pdf`
- `rl_mc_files/` (expect 2 figures):
    - (1) `dqn_training_curves.pdf`, (2) `dqn_vs_pmc_rewards.pdf`
- `shield_files/`(expect 3 figures):
    - (1) `workflow_shield_reward.pdf`, (2) `workflow_shield_safety_violations.pdf`, (3) `workflow_shield_training_curve.pdf`

**Further outputs:** 
The smoke test generates additional files, mainly trained/exported model files, in the following folders:
- `results/smoke/logs`: redirected notebook outputs
- `results/smoke/workflows`: trained DRL models and training logs from SB3, and model exports.

### When is the smoke test successful?
1. All notebooks finish successfully, indicated by ✅.
2. All expected figures are generated.


## Running the Full Evaluation - Functional Badge

- **estimated time:** ca. 9h on a standard laptop
- **required resources:** min. 12GB of memory (16GB recommended)

Don't forget to create a local directory to store results (see above):
```bash
mkdir -p results/full
```

**Run this command:**

```
docker run --rm --network none -v "$PWD/results/full:/results:z" localhost/verigym repro --full
```

This will run the full version of the scripts used for the paper's results. We mount the `results` folder in the container to `./results/full`. The container will be deleted when it exits.

### Expected Outputs
**Console:**

> **Note:**
> Exact runtimes may differ per machine and per run.
> 10/10 notebooks should finish successfully, indicated by ✅. Failure is indicated by ❌.

```
Running 10 notebook scripts (QUICK=0); logs in docs/notebooks/output/logs/
[1/10] workflow_external_tools ... ✅ 0m24s
[2/10] workflow_comparing_abstraction_policies ... ✅ 9m58s
[3/10] workflow_shielding ... ✅ 19m42s
[4/10] workflow_comparing_exploration_policies ... ✅ 0m36s
[5/10] tech_eval_discretization_overhead ... ✅ 2m35s
[6/10] tech_eval_mdp_memory ... ✅ 4m18s
[7/10] workflow_compare_rl_model_checking ... ✅ 33m15s
[8/10] workflow_abstraction_refinement ... ✅ 34m04s
[9/10] workflow_featurereduction ... ✅ 292m17s
[10/10] tech_eval_runtimes ... ✅ 150m47s
```
**Figures:** 

To check for reproducibility, please compare the **figures listed below** with the **corresponding reference figures** that are stored with the same path structure in `reference_outputs/full/` in the artifact.


Results for the **technical evaluation** are stored in `results/full/tech_eval/`, with subfolders:
- `discretization_overhead/` (expect 5 figures): 
    - (1) `break_even_steps.pdf`, (2) `compile_cost_datatype.pdf`, (3) `discretization_overhead_env_step.pdf`, (4) `discretization_times.pdf`, (5) `time_BinEdges_conversion_functions.pdf`
- `memory_eval/`(expect 6 figures): 
    - (1)`memory_eval_1a_transition_func.pdf`, (2) `memory_eval_1b_size_environment.pdf`, (3) `memory_eval_2a_transition_func.pdf`, (4) `memory_eval_2b_environment_size.pdf`, (5) `memory_eval_3b_environment_size.pdf`, (6)`memory_eval_3b_transition_func.pdf`
- `runtimes/` (expect 3 figures):
    - (1) `policy_vs_explore_budget.pdf`, (2) `runtime_breakdown_share.pdf`, (3) `runtime_breakdown.pdf`

> **Exact results of the technical evaluation are hardware dependent and may deviate for reviewers.** Visible trends, however, should stay the same.  
> We used the following **technical setup for paper results**: Ubuntu 22.04.5 LTS benchmark server with Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz, NVIDIA GeForce RTX 3090 (24 GB VRAM) GPU and 256 GB (8 x 32GB DDR4-3200 RAM). We used Python 3.12.15 and `VeriGym` v0.1.0.

Results for *workflows* are stored in `results/full/workflows/figures/`, with subfolders:
- `abstraction_refinement/` (expect 1 figure): 
    - (1)`workflow_abstraction_refinement_cartpole.pdf`
- `compare_files/` (expect 3 figures)
    - (1) `comparing_abstraction_policies_with_abstracted_ppo.pdf`, (2) `comparing_abstraction_policies.pdf`, (3) `ppo_training_curve.pdf`
- `ext_tools_files/`(expect 1 figures)
    - (1) `mcts_vs_model_checking.pdf`
- `feature_reduction/` (expect 3 figures):
    - (1) `eval_rewards.pdf`, (2) `sac_training_ep_lengths.pdf`, (3) `sac_training_rewards.pdf`
- `rl_mc_files/` (expect 2 figures):
    - (1) `dqn_training_curves.pdf`, (2) `dqn_vs_pmc_rewards.pdf`
- `shield_files/`(expect 3 figures):
    - (1) `workflow_shield_reward.pdf`, (2) `workflow_shield_safety_violations.pdf`, (3) `workflow_shield_training_curve.pdf`

**Matching output and paper figures:**

> ⚠️ **Attention:**
> We noticed seeding discrepancies between the paper and artifact submission. We have now fixed all seeds. 
> Consequently, while artifact *reference plots partially deviate from paper figures*, we have now ensured *reproducibility: artifact plots are the same across multiple runs*.
> Importantly, *all trends and findings reported in the paper remain the same*.
> We will update the paper figures with the artifact reference plots in the final paper version.

Check below whether the generated figure matches the paper/reference output.
- A ✅ in the **same as paper** column signifies that output figures of the artifact evaluation will be identical to the ones in the paper.
- A ✅ in the **same as in reference output** column signifies that the output figures of the artifact evaluation will be identical to the ones in the reference folder on Zenodo `VeriGym/reference_outputs/`
A (✅) means almost identical after seeding.

| Paper Figure  | Path | same as paper | same as in reference output |
| ------------- |------|--------------------|-----------------------|
| Figure 4.a    | `results/full/workflows/figures/compare_files/comparing_abstraction_policies.pdf` | (✅)  | ✅
| Figure 4.b    | `results/full/workflows/figures/rl_mc_files/dqn_vs_pmc_rewards.pdf` | (✅) | ✅
| Figure 6.a    | `results/full/workflows/figures/shield_files/workflow_shield_training_curve.pdf` |  | ✅
| Figure 6.b    | `results/full/workflows/figures/shield_files/workflow_shield_reward.pdf` |  | ✅
| Figure 6.c    | `results/full/workflows/figures/shield_files/workflow_shield_safety_violations.pdf` |  | ✅
| Figure 8      | `results/full/workflows/figures/abstraction_refinement/workflow_abstraction_refinement_cartpole.pdf` |  | ✅
| Figure 9.a    | `results/full/tech_eval/discretization_overhead/discretization_times.pdf` | (✅) | ✅
| Figure 9.b    | `results/full/tech_eval/memory_eval/memory_eval_3b_environment_size.pdf` | ✅ | ✅
| Figure 10     | `results/full/tech_eval/runtimes/runtime_breakdown.pdf` | ✅ | ✅
| Figure 11     | `results/full/tech_eval/runtimes/policy_vs_explore_budget.pdf` | ✅ | ✅
| Figure 12.a   | `results/full/workflows/figures/feature_reduction/sac_training_rewards.pdf` |  | ✅
| Figure 12.b   | `results/full/workflows/figures/feature_reduction/eval_rewards.pdf` | (✅) | ✅
| Figure 13     | `results/full/workflows/figures/ext_tools_files/mcts_vs_model_checking.pdf` | ✅ | ✅
| Figure 14.a   | `results/full/workflows/figures/compare_files/ppo_training_curves.pdf` | ✅ | ✅
| Figure 14.b   | `results/full/workflows/figures/rl_mc_files/dqn_training_curves.pdf` |  | ✅
| Figure 15.a   | `results/full/tech_eval/discretization_overhead/discretization_overhead_env_step.pdf` | ✅ | ✅
| Figure 15.b   | `results/full/tech_eval/discretization_overhead/break_even_steps.pdf`| ✅ | ✅
| Figure 16.a   | `results/full/tech_eval/discretization_overhead/compile_cost_datatype` |  | 
| Figure 16.b   | `results/full/tech_eval/memory_eval/memory_eval_2a_transition_func.pdf` | ✅ | ✅

**Further outputs:** 
The evaluation generates additional files, largely trained/exported model files, in the following folders:
- `results/full/logs`: redirected notebook outputs
- `results/full/workflows`: trained DRL models and training logs from SB3, and model exports.

### When is the functional badge check a success?
1. All notebooks finish successfully, indicated by ✅.
2. All expected figures are generated.
3. The generated plots coincide with/are similar to the reference plots (see Note on seeding above).

## Reusable Badge

**Documentation, examples, and API:** 
Next to the artifact, the `VeriGym` repository is publicly available on [GitHub](https://github.com/ai-fm/VeriGym). 
The documentation is also available [online](https://ai-fm.github.io/VeriGym/) and contains examples (workflows and guides) and the full API reference.

- Our maintained repository is publicly available on [GitHub](https://github.com/ai-fm/VeriGym).
- Documentation is available as compiled HTML files in the Zenodo artifact in `VeriGym/docs/`.
- Documentation is also publicly available [online](https://ai-fm.github.io/VeriGym/).

**Installing the `VeriGym` Python library:** `VeryGym` is installable from **PyPi**:

```
uv venv # create a new virtual environment
uv pip install verigym
```
or
```
pip install verigym
```

Further installation instructions are included in our documentation for [Installation](docs/installation.md)

**Example usages:**

- Check the **Quickstart guide** (in `VeriGym/docs/quickstart.md` or [online](https://ai-fm.github.io/VeriGym/quickstart.html)) for an easy start to programming with `VeriGym`.
- The documentation includes **guides** that showcase the range of capabilities for different parts of `VeriGym`, including *Abstraction Mapping* (in `VeriGym/docs/guides/abstraction_mapping.md` or [online](https://ai-fm.github.io/VeriGym/guides/abstraction_mapping.html)), *Model Learning* (in `VeriGym/docs/guides/model_learning.md` or [online](https://ai-fm.github.io/VeriGym/guides/model_learning.html)), and *importing/exporting* from different frameworks (in `VeriGym/docs/guides/import_export.md` or [online](https://ai-fm.github.io/VeriGym/guides/import_export.html)) from different frameworks.
- The **workflows** contain full pipelines for different tasks, which we report in the paper. Check them out in the workflows section (`VeriGym/docs/workflows.md` or [online](https://ai-fm.github.io/VeriGym/workflows.html)), or run the executable Python notebooks in `VeriGym/docs/workflow_*.py`.

**Using additional benchmarks:**

The included experiments can easily be run with further benchmarks, including `Gymnasium` environments, `stormpy` MDPs, and `PRISM` programs. Check out the guide on *importing/exporting* from different frameworks.

**Adding functionality to the codebase:**

`VeriGym`'s modular design allows easy extensions to **functionality**:
- add further model exploration techniques by inheriting from `VeriGym`; see the guide on *importing/exporting*.
- define your own state space abstraction; see the guide on *Abstraction Mapping*.
- add new external frameworks. For an example, check the workflow on connecting with `POMDPS.js` (`VeriGym/docs/workflow_external_tools.py` or [online](https://ai-fm.github.io/VeriGym/notebooks/workflow_external_tools.html)).

**Contribute to `VeriGym`:**

The online [**Contribution Guide**](https://ai-fm.github.io/VeriGym/contribution.html) provides information for developers who would like to contribute to `VeriGym`.

## Further technical information

### Tested Platforms

We ran and passed full evaluation in Docker (and Podman) on the following platforms:
- MacBook Pro 2023, 18GB RAM, macOS Golden Gate 27.0.1
- Debian 12 Bookworm, CPU AMD Ryzen Threadripper PRO 5965WX 24-Cores, RAM 8x64 GiB Samsung DDR4-3200, GPU Nvidia Geforce RTX 4090
- Mac Studio with Apple M1 Ultra, 128 Unified RAM, running Colima for Docker with 16GB.
- Fedora 43 with Podman 5.8.8

### Network access

The artifact does not require internet access: you can load and run the provided Docker image without internet access. Running the container, reproducing results, and reusing `VeriGym` also don't require internet access.

Only rebuilding the Docker image or installing `VeriGym` with `uv` or `pip` requires internet access.

### VeriGym in Docker

For information on manually building Docker containers, check the *Installation guide* (`VeriGym/docs/installation.md` or [online](https://ai-fm.github.io/VeriGym/installation.html))

**Using the Docker image for development**

The commands below using the `./docker/run-X.sh` signature are meant for development: they mount your code at /workspace, so the container runs your local code (and the image also works as a devcontainer image).

Use `./docker/run-docker.sh test` to run the tests, `./docker/run-docker.sh repro [--quick | --full]` to run the notebooks (`docs/notebooks/run_notebooks.sh`, quick by default; the executed notebooks are moved to `docs/notebooks/output/executed`, or with `run_notebooks.sh --inplace` over the notebooks themselves), and `./docker/run-docker.sh shell` for a shell.

Prefix a command in `./docker/run-docker.sh` with `OFFLINE=1`, e.g., `OFFLINE=1 ./docker/run-docker.sh test` to run the command without network access. 

With these commands and the provided Docker image, you can modify the notebooks/scripts and/or add functionality to `VeriGym` and run/test it within the artifact without internet access.

**Explaining the run command**

```sh
docker run 
    --rm                            # Container gets deleted after exit.
    --network none                  # Blocks network access.
    -v "$PWD/results:/results"      # Mount the output folder in the container with a volume to a local folder
    localhost/verigym               # Specify image name
    repro --full                    # command to execute in container - can be `repro` (`--quick` by default, or `--full`), `test` (with args passed) and `shell`.
```

**What happens when running the evaluation in Docker?**

- `repro` with `--full` or `--quick` executes the shell script in `docs/notebooks/run_notebooks.sh`. The script runs all Jupyter notebooks for workflows and technical evaluation in `docs/notebooks` and writes results to `results`. We mount the `results` directory inside the container to `results/full` or `results/smoke` which we set up depending on whether we pass `--full` or `--quick` to the run command, respectively. 

**Docker troubleshooting**

The image runs as UID 1000. Docker Desktop (macOS) works for any user. With rootless Podman, add `--userns=keep-id:uid=1000,gid=1000` to any `podman run` command. 

In some cases, it may help (e.g., for non-SELinux users) to remove `:z` from the mount command. E.g., use `-v "$PWD/results/smoke:/results"` and `-v "$PWD/results/full:/results"`.

If Docker returns any `Permission Denied` errors on Linux, try adding `--users 0:0` (emulating root in Docker) to the `docker run` command. Alternatively, try `--users 1000:1000`.

With Docker Engine on Linux and a UID other than 1000, make the results directory writable first (`chmod a+rwx results`).

On Linux, Docker must work without `sudo`: follow Docker's [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) (add your user to the `docker` group; optionally start Docker at boot). Running the commands with `sudo` is not advised and not explicitly supported.

If you experience out-of-memory issues with Docker Desktop, open the application, go to `settings > resources`, and increase the memory limit to the specified requirements.
