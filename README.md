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
