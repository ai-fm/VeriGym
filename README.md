<center>
<img src="https://raw.githubusercontent.com/ai-fm/VeriGym/main/docs/VeriGym_Logo.png" width=100>

<h1>VeriGym: From Deep Reinforcement Learning Simulators to Verifiable Models and Back</h1>
</center>

`VeriGym` is a modular and comprehensive open-source framework to bridge Reinforcement Learning (RL) and Probabilistic Model Checking (PMC) research.
`VeriGym` exposes an abstraction interface to convert simulator-based RL environments following the `Gymnasium` API into explicit model formats supported by state-of-the-art PMC and synthesis tools such as `Storm` and `PRISM`.
`VeriGym` maps solutions back to the original environment, e.g., for direct deployment or to guide RL training.

# Installation

Install `VeriGym` from [PyPI](https://pypi.org/project/verigym/) using [uv](https://docs.astral.sh/uv/) package manager:

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

## Artifact evaluation (prebuilt image)
A prebuilt image contains and runs the code it was built from. Load the archive for your machine, then run the tests and the example workflows. Running these commands in the docker container doesn't require network access:

```sh
docker load -i verigym-amd64.tar.gz  # Intel/AMD; verigym-arm64.tar.gz on Apple Silicon and other ARM machines
docker run --rm --network none localhost/verigym test
mkdir -p results
docker run --rm --network none -v "$PWD/results:/results:z" localhost/verigym repro
```

`repro` copies the image's repository into `results/` and runs the example workflows there. The executed notebooks and figures appear in `results/examples` while they are written. Running it again with the same directory continues that run; use an empty directory for a fresh run (and one per image). Create `results/` before the run, so that Docker does not create it as root. `docker run --rm -it localhost/verigym` opens a shell in the image.

To use the image published to GitHub's registry instead, run `docker pull ghcr.io/ai-fm/verigym:<commit>` (needs network access once; Docker or Podman picks the image for your architecture) and replace `localhost/verigym` with `ghcr.io/ai-fm/verigym:<commit>` above. The commit an image was built from is in its labels: `docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' localhost/verigym`.

The image runs as UID 1000. Docker Desktop (macOS) works for any user. With rootless Podman, add `--userns=keep-id:uid=1000,gid=1000` to `podman run`; with Docker Engine on Linux and a UID other than 1000, make the results directory writable first (`chmod a+rwx results`).

## Building an image
The `Dockerfile` in the `docker` directory of the source code builds an image for Linux and macOS (amd64 and arm64) with Python 3.14, the locked dependencies from `uv.lock`, and Julia 1.11.5 with the packages for the external-tools workflow. The Python, uv, and Julia versions can be set as build arguments.

Use `./docker/run-docker.sh build` to build a `localhost/verigym:latest` Docker image (`--prism` optionally installs Java and PRISM).

To create the archives, build and save the image natively on each architecture, with Podman or with Docker as UID 1000: `./docker/run-docker.sh build && ./docker/run-docker.sh save` writes `verigym-<arch>.tar.gz`. 

A multi-platform image for a registry can be built with `docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile -t <registry>/verigym:<tag> --push .`, which emulates the other architecture and is slow.

### Running the container
Use `./docker/run-docker.sh test` to run the tests, `./docker/run-docker.sh repro` to run the example workflows (`examples/run_workflows.sh`), and `./docker/run-docker.sh shell` for a shell. These commands are meant for development: they mount your checkout at `/workspace`, so the container runs your local code (and the image also works as a devcontainer image). To run the code built into an image, see [Artifact evaluation](#artifact-evaluation-prebuilt-image).


### Network access
Prefix a command in `./docker/run-docker.sh` with `OFFLINE=1`, e.g., `OFFLINE=1 ./docker/run-docker.sh test` to run the command without network access. The container can reproduce the paper without internet access, only building the image requires internet access. That means that with a given image, the artefact does not require internet access.


# Contribution Guide

Please check out the [contribution guide](https://github.com/ai-fm/VeriGym/blob/main/docs/contribution.md) for branch naming, testing, naming conventions, etc.

# License and Artifact

`VeriGym` is released under the **MIT License**.

`VeriGym` is under review at TACAS 2027 as a regular tool paper.

The permanent artifact for TACAS 2027 is available on Zenodo with DOI `TODO:add DOI`.
