<center>
<img src="https://raw.githubusercontent.com/ai-fm/VeriGym/main/docs/VeriGym_Logo.png" width=100>

<h1>VeriGym</h1>
</center>

This is **VeriGym**.

> [!WARNING]
> This README is currently undergoing construction.

# Installation

After `git clone` please run `uv sync` to create the virtual environment. [UV](https://docs.astral.sh/uv/) is a strongly recommended package manager, faster and better than `pip`.

Supported: Python 3.12–3.14 on Linux and macOS. Windows is not supported natively (stormpy has no Windows wheels), but works through [WSL](https://learn.microsoft.com/windows/wsl/).

## Docker

### Building an image
The `Dockerfile` in the `docker` directory builds an image for Linux and macOS (amd64 and arm64) with Python 3.14, the locked dependencies from `uv.lock`, and Julia 1.11.5 with the packages for the external-tools workflow. The Python, uv, and Julia versions can be set as build arguments.

Use `./docker/run-docker.sh build` to build a `localhost/verigym:latest` Docker image (`--prism` optionally installs Java and PRISM).

### Running the container
Use `./docker/run-docker.sh test` to run the tests, `./docker/run-docker.sh repro` to run the example workflows (`examples/run_workflows.sh`), and `./docker/run-docker.sh shell` for a shell. The repository is mounted at `/workspace`, so the image also works as a devcontainer image. 


### Network access
Prefix a command in `./docker/run-docker.sh` with `OFFLINE=1`, e.g., `OFFLINE=1 ./docker/run-docker.sh test` to run the command without network access. The container can reproduce the paper without internet access, only building the image requires internet access. That means that with a given image, the artefact does not require internet access.

### Artifact evaluation (prebuilt image)
With a prebuilt image, use the repository at the same commit as the image and do not run `build` (it needs network access and replaces the image):

```sh
./docker/run-docker.sh load verigym-amd64.tar.gz  # Intel/AMD; use verigym-arm64.tar.gz on Apple Silicon and other ARM machines
OFFLINE=1 ./docker/run-docker.sh test
OFFLINE=1 ./docker/run-docker.sh repro
```

The image runs as UID 1000. Docker Desktop (macOS) and Podman work for any user; with Docker Engine on Linux your UID must be 1000 (usually the first account), otherwise build the image yourself with `./docker/run-docker.sh build` (requires internet access).

To create the archives, build and save the image natively on each architecture, with Podman or with Docker as UID 1000: `./docker/run-docker.sh build && ./docker/run-docker.sh save` writes `verigym-<arch>.tar.gz`. 

A multi-platform image for a registry can be built with `docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile -t <registry>/verigym:<tag> --push .`, which emulates the other architecture and is slow.

Prebuilt archives for a commit can also be made on GitHub: **Actions → Docker image → Run workflow**, pick the branch, a flavour (`amd64`, `arm64`, or `multi-platform` for all three archives) and optionally PRISM, then download the artifacts from the run page. GitHub wraps each artifact in a zip, so unzip it first; artifacts expire after 90 days, so archive the files with the artifact (e.g. on Zenodo). `verigym-multi.oci.tar` holds both platforms and is meant for registries (`skopeo copy --all oci-archive:verigym-multi.oci.tar docker://<registry>/verigym:<tag>`); to load an image locally, use the per-architecture `.tar.gz` files.


# Contribution Guide

Please check out the [contribution guide](https://github.com/ai-fm/VeriGym/blob/main/docs/contribution.md) for branch naming, testing, naming conventions, etc.
