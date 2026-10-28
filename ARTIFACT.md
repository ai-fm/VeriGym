# Artifact Evaluation for TACAS 2027 Paper 1441

- **Title:** `VeriGym`: From Deep Reinforcement Learning Simulators to Verifiable Models and Back
- **Paper number:** 1441
- **Paper file:** `TACAS27_paper__1441.pdf`
- **Badges:** We apply for all three badges: Available, Functional, Reusable.

## Availability

- Our artifact is permanently available on [Zenodo](), DOI: 
- Our maintained repository is publicly available on [GitHub](https://github.com/ai-fm/VeriGym).
- Documentation is available as compiled html files in [docs](`docs/index.html`).
- Documentation is also publicly available [online](https://ai-fm.github.io/VeriGym/).

## Setting up and running the Artifact

We deliver our artifact in **Docker**. 
The artifact includes pre-built docker containers.
If you would like to re-build the container from scratch, please check the instructions for running [`VeriGym`with docker](#verigym-in-docker).
To load and prepare for evaluation, execute the commands below.

**Load the image according to your system architecture.** 
```bash
docker load -i verigym-arm64.tar.gz     # on Apple Silicon and other ARM machines
                                        # OR
docker load -i verigym-amd64.tar.gz     # on Linux
```

**Run the tests** *(optional)***:** 
- `--no-cov` disables coverage checks. Enabling leads so longer run time. `-n 2`uses two threads for efficiency.
- Expected duration with `--no-cov`: < 90sec, without: < 140sec

```bash 
docker run —rm —network none localhost/verigym test —-no-cov -n 2
```

**Create directories** to store the results on your machine:

```bash
mkdir -p results
mkdir -p results/smoke
```

## Running the Smoke Test

- **estimated time**: <10min on a standard laptop
- **estimated resources**: <2GB

**Run this command:**

```
docker run --rm --network none -v "$PWD/results:/results:z" localhost/verigym repro --quick
```

### Expected Outputs
**Console:**

```
Running 10 notebooks (QUICK=1); logs in docs/notebooks/output/logs/
[1/10] workflow_external_tools ... ✅ 0m12s
[2/10] workflow_comparing_abstraction_policies ... ✅ 0m18s
[3/10] workflow_shielding ... ✅ 0m07s
[4/10] workflow_abstraction_refinement ... ✅ 0m16s
[5/10] workflow_compare_rl_model_checking ... ✅ 0m36s
[6/10] workflow_comparing_exploration_policies ... ✅ 0m06s
[7/10] workflow_featurereduction ... ✅ 0m34s
[8/10] tech_eval_discretization_overhead ... ✅ 0m18s
[9/10] tech_eval_mdp_memory ... ✅ 0m15s
[10/10] tech_eval_runtimes ... ✅ 0m14s
```

> **Note:**
> Exact runtimes may differ per machine and per run.
> 10/10 notebooks should finish successully, indicated by ✅. Failure is indicated by ❌.

**Figures:** 

Results for the **technical evaluation** are stored in `results/smoke/tech_eval/`, with subfolders:
- `discretization_overhead/` (expect 5 figures): 
    - (1) `break_even_steps.pdf`, (2) `compile_cost_datatype.pdf`, (3) `discretization_overhead_env_step.pdf`, (4) `discretization_times.pdf`, (5) `time_BinEdges_conversion_functions.pdf`
- `memory_eval/`(expect 6 figures): 
    - (1)`memory_eval_1a_transition_func.pdf`, (2) `memory_eval_1b_size_environment.pdf`, (3) `memory_eval_2a_transition_func.pdf`, (4) `memory_eval_2b_environment_size.pdf`, (5) `memory_eval_3b_environment_size.pdf`, (6)`memory_eval_3b_transition_func.pdf`
- `runtimes/` (expect 3 figures):
    - (1) `policy_vs_explore_budget.pdf`, (2) `runtime_breakdown_share.pdf`, (3) `runtime_breakdown.pdf`

Results for *workflows* are stored in `results/smoke/workflows/figures/*, with subfolders:
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
- `shield_files/`(expect 2 figures):
    - (1) `workflow_shield_reward.pdf`, (2) `workflow_shield_safety_violations.pdf`

**Further outputs:** 
The smoke test generates additional files, largely trained/exported model files in the following folders:
- `results/smoke/executed/`: executed notebooks with outputs
- `results/smoke/logs`: notebook outputs, re-directed from cmd
- `results/smoke/workflows`: trained DRL models ad training logs from SB3 and model exports.

### When is the smoke test successfull?
1. All notebooks finish successfully, indicated by ✅.
2. All expected figures are generated.


## Running the Full Evaluation - Functional Badge

- **estimated time:** ca. 9h on a standard laptop
- **estimated resources:** min. 12GB memory


**Run this command:**

```
docker run --rm --network none -v "$PWD/results:/results:z" localhost/verigym repro --full
```

### Expected Outputs
**Console:**

```
Running 10 notebooks (QUICK=0); logs in docs/notebooks/output/logs/
[1/10] workflow_external_tools ... ✅ 0m29s
[2/10] workflow_comparing_abstraction_policies ... ✅ 2m23s
[3/10] workflow_shielding ... ✅ 12m45s
[4/10] workflow_abstraction_refinement ... ✅ 40m48s
[5/10] workflow_compare_rl_model_checking ... ✅ 15m11s
[6/10] workflow_comparing_exploration_policies ... ✅ 0m42s
[7/10] workflow_featurereduction ... ✅ 295m33s
[8/10] tech_eval_discretization_overhead ... ✅ 2m34s
[9/10] tech_eval_mdp_memory ... ✅ 4m08s
[10/10] tech_eval_runtimes ... ✅ 151m41s
```

> **Note:**
> Exact runtimes may differ per machine and per run.
> 10/10 notebooks should finish successully, indicated by ✅. Failure is indicated by ❌.

**Figures:** 

To check for reproducibility, please compare the list of figures below with the corresponding reference figures that are stored with the same path structure in `docs/notebooks/output_paper_submission`.

> **Note:**
> We noticed oversights regarding seeding between paper and artifact submission. We have now fixed all seeds. 
> Consequently, while artifact *reference plots partially deviate from paper figures*, we have now ensured *reproducibility: artifact plots are the same across multiple runs*.
> Importantly, *all trends and findings reported in the paper remain the same*.
> We will update the paper figures with the artifact reference plots in the final paper version.


Results for the **technical evaluation** are stored in `results/tech_eval/`, with subfolders:
- `discretization_overhead/` (expect 5 figures): 
    - (1) `break_even_steps.pdf`, (2) `compile_cost_datatype.pdf`, (3) `discretization_overhead_env_step.pdf`, (4) `discretization_times.pdf`, (5) `time_BinEdges_conversion_functions.pdf`
- `memory_eval/`(expect 6 figures): 
    - (1)`memory_eval_1a_transition_func.pdf`, (2) `memory_eval_1b_size_environment.pdf`, (3) `memory_eval_2a_transition_func.pdf`, (4) `memory_eval_2b_environment_size.pdf`, (5) `memory_eval_3b_environment_size.pdf`, (6)`memory_eval_3b_transition_func.pdf`
- `runtimes/` (expect 3 figures):
    - (1) `policy_vs_explore_budget.pdf`, (2) `runtime_breakdown_share.pdf`, (3) `runtime_breakdown.pdf`

> **Exact results of the technical evaluation are hardware dependent and may deviate for reviewers.** Visible trends however should stay the same.  
> We used the following **technical setup for paper results**: Ubuntu 22.04.5 LTS benchmark server with Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz, NVIDIA GeForce RTX 3090 (24 GB VRAM) GPU and 256 GB (8 x 32GB DDR4-3200 RAM). We used Python 3.12.15 and `VeriGym`v1.0.0.

Results for *workflows* are stored in `results/workflows/figures/`, with subfolders:
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
- `shield_files/`(expect 2 figures):
    - (1) `workflow_shield_reward.pdf`, (2) `workflow_shield_safety_violations.pdf`

**Further outputs:** 
The evaluation generates additional files, largely trained/exported model files in the following folders:
- `results/executed/`: executed notebooks with outputs
- `results/logs`: notebook outputs, re-directed from cmd
- `results/workflows`: trained DRL models ad training logs from SB3 and model exports.

### When is the test for functional successfull?
1. All notebooks finish successfully, indicated by ✅.
2. All expected figures are generated.
3. The generated plots coincide with/are very similar to the reference plots.

## Reusable Badge

**Documentation, examples, and API:** 
Next to the artifact, the `VeriGym` repository is publicly available on [GitHub](https://github.com/ai-fm/VeriGym). 
The documentation is also available [online](https://ai-fm.github.io/VeriGym/) and contains examples (workflows and guides), and the full API reference.

- Our maintained repository is publicly available on [GitHub](https://github.com/ai-fm/VeriGym).
- Documentation is available as compiled html files in [docs](`docs/index.html`).
- Documentation is also publicly available [online](https://ai-fm.github.io/VeriGym/).

**Installing the `VeriGym` Python library:** `VeryGym` is installable from **PyPi**:

```
uv pip install verigym
# or
pip install verigym
```

**Example usages:**

- Check the [Quickstart](docs/quickstart.md) for an easy start to programming with `VeriGym`.
- The documentation includes *guides* that showcase the range of capabilities for different parts of `VeriGym`, including [Abstraction Mapping](docs/guides/abstraction_mapping.md), [Model Learning](docs/guides/model_learning.md), and [importing/exporting](docs/guides//import_export.md) from different frameworks.
- The *workflows* contain full pipelines for different tasks, which we report in the paper. Check out them out in the [Workflows](docs/workflows.md) section, or run them as executable jupyter notebooks included in the repository at `docs/notebooks/workflow_*.ipynb`

Further installation instructions for development are included in our documentation for [Installation](docs/installation.md)

**Using additional benchmarks:**

The included experiments can easily be run with further benchmarks:
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

**Adding functionality to the code base:**

`VeriGym`'s modular design allows easy extensions to **functionality**:
- inherit from `PolicyClass` to add further model exploration techniques
- define arbitrary input functions to `AbstractionMapper` for further space abstractions
- add new external frameworks using `ExplicitFormatter`, `PolicyClass`, and adding constructors via `VeriGymEnv.from_framework`

The [**Contribution Guide**](https://ai-fm.github.io/VeriGym/contribution.html) provides information for developers who would like to contribute to `VeriGym`.

## Further technical information

### VeriGym in Docker

**Building the container**

- The `docker/Dockerfile` builds an image for Linux or macOS (amd64, arm64) with Python 3.15, the locked dependencies from `uv.lock`.
- `./docker/run-docker.sh build` builds a `localhost/verigym:latest` Docker image. (`--prism` optionally installs Java and PRISM; `--julia` installs Julia `1.10.12` with the precompiled packages for the external-tools workflow, about 1 GB).
- Python, uv, and Julia versions can be set as build arguments.
- To create the archives, build and save the image natively on each architecture, with Podman or with Docker as UID 1000: `./docker/run-docker.sh build && ./docker/run-docker.sh save` writes `verigym-<arch>.tar.gz`. 
- A multi-platform image for a registry can be built with `docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile -t <registry>/verigym:<tag> --push .`, which emulates the other architecture and is slow.

> Use this command to re-build the docker container including a julia installation. Building may take a few minutes. 
> The command writes `verigym-<arch>.tar.gz`.
> After building, continue as described above.
>```bash
>./docker/run-docker.sh build --julia && ./docker/run-docker.sh save
>```

**Using a published image from GitHub's registry**

Run the following commands. This requires network access once. Docker or Podman picks the image for your architecture.

```
docker pull ghcr.io/ai-fm/verigym:<commit>
```

To run the image, replace `localhost/verigym` with `ghcr.io/ai-fm/verigym:<commit>` in the run command.

The commit an image was built from is in its labels. Find it by running

```
docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' localhost/verigym
```


**Explaining the run command**

```sh
docker run 
    --rm                            # Container gets deleted after exit.
    --network none                  # Removes network access.
    -v "$PWD/results:/results:z"    # Mount a volume to the local `results` folder
    localhost/verigym               # Specify image name
    repro --full                    # command to execute in container - can be `repro` (`--quick` by default, or `--full`), `test` (with args passed) and `shell`.
```

Prefix a command in `./docker/run-docker.sh` with `OFFLINE=1`, e.g., `OFFLINE=1 ./docker/run-docker.sh test` to run the command without network access. 


**What happens when running the evaluation in docker?**

- `tests`: runs pytests, and, if the container is built with julia, runs tests for julia.
- `repro` with `--full` or `--quick` executes the shell script in `docs/notebooks/run_notebooks.sh`. The script runs all jupyter notebooks for workflows and technical evaluation in `docs/notebooks` and writes results to `results` (or `results/smoke` with `--quick`). 

### Docker troubleshooting

The image runs as UID 1000. Docker Desktop (macOS) works for any user. With rootless Podman, add `--userns=keep-id:uid=1000,gid=1000` to any `podman run` command. 

With Docker Engine on Linux and a UID other than 1000, make the results directory writable first (`chmod a+rwx results`).

On Linux, Docker must work without `sudo`: follow Docker's [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) (add your user to the `docker` group; optionally start Docker at boot). Running the commands with `sudo` is not advised and not explicitly supported.

When receiving out-of-memory issues with Docker Desktop, open the application, go to `settings > resources` and increase memory limit to the specified requirements.

### Tested Platforms

We ran and passed full evaluation in docker on the following platforms:
- MacBook Pro 2023, 18GB RAM, macOS Golden Gate 27.0.1
- Debian 12 Bookworm, CPU AMD Ryzen Threadripper PRO 5965WX 24-Cores, RAM 8x64 GiB Samsung DDR4-3200, GPU Nvidia Geforce RTX 4090

### Network access

The artifact does not require internet access:  The provided docker container can be loaded or re-built without internet access. Executing the container, reproducing results, and re-using `VeriGym` also does not require internet access.

If reviewers would like to download a published image from GitHub (see above), the download does require internet access.