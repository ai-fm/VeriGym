<center>
<img src="https://raw.githubusercontent.com/ai-fm/VeriGym/main/docs/VeriGym_Logo.png" width=100>

<h1>VeriGym</h1>
</center>

This is **VeriGym**.

> [!WARNING]
> This README is currently undergoing construction.

# Installation

Install VeriGym from [PyPI](https://pypi.org/project/verigym/) using [uv](https://docs.astral.sh/uv/) package manager:

```sh
uv pip install verigym
# alternatively using pip
pip install verigym
```

Supported: Python 3.12–3.15 on Linux and macOS. Windows is supported through Windows Subsystem for Linux ([WSL](https://learn.microsoft.com/windows/wsl/)).

On Python 3.14 and newer, the Box2D environments (e.g. `LunarLander`) are not installed automatically, because Box2D has to be compiled for that Python version. To use them, install [SWIG](https://www.swig.org/) (`brew install swig` or `sudo apt install swig`), then run `pip install "gymnasium[box2d]"` with the activated venv (alternatively just run `uv add verigym`).

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
