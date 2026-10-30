# Installation

## Install from PyPI

```sh
uv pip install verigym
# alternatively using pip
pip install verigym
```

Supported are Python 3.12.2 and newer on Linux and macOS, and Windows through [WSL](https://learn.microsoft.com/windows/wsl/).

!!! note "Box2D environments on Python 3.14 and newer"
    The Box2D environments (e.g. `LunarLander`) are not installed automatically on Python 3.14 and newer, because Box2D has to be compiled there. To use them, install [SWIG](https://www.swig.org/) (`brew install swig` or `sudo apt install swig`), then run `pip install "gymnasium[box2d]"`.

## Development install with `uv` (Recommended)

Clone the repository and open a shell inside its root folder.

:::{admonition} What is `uv`?
:class: tip
`uv` is a python package manager designed for speed and ease of use. It is a lot faster than `pip` and also packs many more features for organizing projects in workspaces that are inspired by the rust programming language. In fact, `uv` is so fast that you can afford to resolve dependencies every time you execute any code. Get `uv` [here](https://docs.astral.sh/uv/)
:::

`uv` will create a virtual environment for you so you can install the package with:

```sh
uv sync --all-extras
```

`uv` will infer the correct python version from the `pyproject.toml`. If you want to test against a different python version add the `--python 3.x` flag.

## Install using `pip` (not recommended)

VeriGym can be installed using pip directly from the repository.

First clone the repository and open a shell inside the root folder of the cloned repository.

Create a virtual environment (tested with python 3.12).

```sh
python3 -m venv venv
```

Activate the virtual environment

```sh
source venv/bin/activate
```

Install the package into the virtual environment

```sh
pip install .
```

:::{admonition} Editable Installs
:class: note
You can use the `-e` flag in order to create an 'editable' install. This way there is no need to reinstall the package after pulling the latest changes.
:::

## VeriGym in Docker

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

**Run the tests** *(optional)***:** 
- `--no-cov` disables coverage checks. Enabling leads so longer run time. `-n 2`uses two threads for efficiency.
- Expected duration with `--no-cov`: < 90sec, without: < 140sec

```bash 
docker run —rm —network none localhost/verigym test —-no-cov -n 2
```

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

### Docker troubleshooting

The image runs as UID 1000. Docker Desktop (macOS) works for any user. With rootless Podman, add `--userns=keep-id:uid=1000,gid=1000` to any `podman run` command. 

With Docker Engine on Linux and a UID other than 1000, make the results directory writable first (`chmod a+rwx results`).

On Linux, Docker must work without `sudo`: follow Docker's [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) (add your user to the `docker` group; optionally start Docker at boot). Running the commands with `sudo` is not advised and not explicitly supported.

When receiving out-of-memory issues with Docker Desktop, open the application, go to `settings > resources` and increase memory limit to the specified requirements.
