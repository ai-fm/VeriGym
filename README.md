<center>
<img src="https://raw.githubusercontent.com/ai-fm/VeriGym/main/docs/VeriGym_Logo.png" width=100>

<h1>VeriGym: From Deep Reinforcement Learning Simulators to Verifiable Models and Back</h1>
</center>

`VeriGym` is a modular and comprehensive open-source framework to bridge Reinforcement Learning (RL) and Probabilistic Model Checking (PMC) research.
`VeriGym` exposes an abstraction interface to convert simulator-based RL environments following the `Gymnasium` API into explicit model formats supported by state-of-the-art PMC and synthesis tools such as `Storm` and `PRISM`.
`VeriGym` maps solutions back to the original environment, e.g., for direct deployment or to guide RL training.

# Installation

`VeriGym` can be installed from PyPi:

```
uv pip install verigym
# or
pip install verigym
```

For further options, check out the documentation on [Installation](https://ai-fm.github.io/VeriGym/installation.html).

> **Supported**: Python 3.12–3.14 on Linux and macOS. Windows is not supported natively (stormpy has no Windows wheels), but works through [WSL](https://learn.microsoft.com/windows/wsl/).

# Structure of the repository

```
VeriGym
└───    docs    # available at https://ai-fm.github.io/VeriGym/
|   └───    ...
|   └───    tech_eval_files # technical evaluation of memory, runtime, etc.
|   └───    tutorials       # how-to-use guides on modules
|   └───    workflows       # end-to-end implementation of different tasks
└───    src/verigym
|   └───    abstraction     # space abstractions and model learning
|   └───    frameworks      # linking to external frameworks
|   └───    io              # loading and exporting several formats
|   └───    policy          # types of model exploration
|   └───    supplements     # convenience funtions
|   └───    utils           # utilities
└───    tests   # pytest for all modules
|   LICENSE.txt # VeriGym is open-source via the MIT License
|   ...
```

## Docker

### Building an image
The `Dockerfile` in the `docker` directory builds an image for Linux and macOS (amd64 and arm64) with Python 3.14, the locked dependencies from `uv.lock`, and Julia 1.11.5 with the packages for the external-tools workflow. The Python, uv, and Julia versions can be set as build arguments.

Use `./docker/run-docker.sh build` to build a `verigym:latest` Docker image (`--prism` optionally installs Java and PRISM).

### Running the container
Use `./docker/run-docker.sh test` to run the tests, `./docker/run-docker.sh repro` to run the example workflows (`examples/run_workflows.sh`), and `./docker/run-docker.sh shell` for a shell. The repository is mounted at `/workspace`, so the image also works as a devcontainer image. 


### Network access
Prefix a command in `./docker/run-docker.sh` with `OFFLINE=1`, e.g., `OFFLINE=1 ./docker/run-docker.sh test` to run the command without network access. The container can reproduce the paper without internet access, only building the image requires internet access. That means that with a given image, the artefact does not require internet access.


# Contribution Guide

Please check out the [contribution guide](https://github.com/ai-fm/VeriGym/blob/main/docs/contribution.md) for branch naming, testing, naming conventions, etc.

# License and Artifact

`VeriGym` is released under the **MIT License**.

`VeriGym` is under review at TACAS 2027 as a regular tool paper.

The permanent artifact for TACAS 2027 is available on Zenodo with DOI `TODO:add DOI`.
