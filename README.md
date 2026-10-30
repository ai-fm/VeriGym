<center>
<img src="https://raw.githubusercontent.com/ai-fm/VeriGym/main/docs/VeriGym_Logo.png" width=100>

<h1>VeriGym: From Deep Reinforcement Learning Simulators to Verifiable Models and Back</h1>
</center>


![Python 3.12](https://raw.githubusercontent.com/ai-fm/VeriGym/main/docs/badges/python-3.12.png)
![Python 3.13](https://raw.githubusercontent.com/ai-fm/VeriGym/main/docs/badges/python-3.13.png)
![Python 3.14](https://raw.githubusercontent.com/ai-fm/VeriGym/main/docs/badges/python-3.14.png)
![Python 3.15](https://raw.githubusercontent.com/ai-fm/VeriGym/main/docs/badges/python-3.15.png)


`verigym` is a modular and comprehensive open-source framework to bridge Reinforcement Learning (RL) and Probabilistic Model Checking (PMC) research.
`verigym` exposes an abstraction interface to convert simulator-based RL environments following the `Gymnasium` API into explicit model formats supported by state-of-the-art PMC and synthesis tools such as `Storm` and `PRISM`.
`verigym` maps solutions back to the original environment, e.g., for direct deployment or to guide RL training.

# Installation

Install `verigym` from [PyPI](https://pypi.org/project/verigym/) using [uv](https://docs.astral.sh/uv/) package manager:

```
uv pip install verigym
```

For further options, check out the documentation on [Installation](https://ai-fm.github.io/VeriGym/installation.html).

> **Supported**: Python 3.12–3.15 on Linux and macOS. Windows is not supported natively (stormpy has no Windows wheels), but works through [WSL](https://learn.microsoft.com/windows/wsl/).

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

# Docker

## Using Docker for VeriGym

### Building an image
The `Dockerfile` in the `docker` directory of the source code builds an image for Linux and macOS (amd64 and arm64) with Python 3.15, the locked dependencies from `uv.lock`, and optionally Julia 1.10.12 with the packages for the external-tools workflow. The Python, uv, and Julia versions can be set as build arguments.

Use `./docker/run-docker.sh build` to build a `localhost/verigym:latest` Docker image (`--prism` optionally installs Java and PRISM; `--julia` installs Julia with the precompiled packages for the external-tools workflow, about 1 GB).

To create the archives, build and save the image natively on each architecture, with Podman or with Docker as UID 1000: `./docker/run-docker.sh build && ./docker/run-docker.sh save` writes `verigym-<arch>.tar.gz`. 

A multi-platform image for a registry can be built with `docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile -t <registry>/verigym:<tag> --push .`, which emulates the other architecture and is slow.

### Running the container
Use `./docker/run-docker.sh test` to run the tests, `./docker/run-docker.sh repro [--quick | --full]` to run the workflows and technical evaluation as Python scripts (`docs/notebooks/run_notebooks.sh`, quick by default; one log per script goes to `docs/notebooks/output/logs`), and `./docker/run-docker.sh shell` for a shell. These commands are meant for development: they mount your checkout at `/workspace`, so the container runs your local code (and the image also works as a devcontainer image). 

Prefix a command in `./docker/run-docker.sh` with `OFFLINE=1`, e.g., `OFFLINE=1 ./docker/run-docker.sh test` to run the command without network access. 

To run the code built into an image, see [Artifact evaluation](#artifact-evaluation-prebuilt-image).


# Contribution Guide

Please check out the [contribution guide](https://github.com/ai-fm/VeriGym/blob/main/docs/contribution.md) for branch naming, testing, naming conventions, etc.

# License

`verigym` is released under the **MIT License**.
