
## Install with `uv` (Recommended)

??? question "What is `uv`?"
    `uv` is a python package manager designed for speed and ease of use. It is a lot faster than `pip` and also packs many more features for organizing projects in workspaces that are inspired by the rust programming language. In fact, `uv` is so fast that you can afford to resolve dependencies every time you execute any code. Get `uv` [here](https://docs.astral.sh/uv/)

`uv` will create a virtual environment for you so you can install the package with:

```sh
uv sync --all-extras
```

`uv` will infer the correct python version from the `pyproject.toml`. If you want to test against a different python version add the `--python 3.x` flag.

## Install using `pip` (not recommended)

Omnigym is not yet available on PyPi but can be installed as a developer using pip.

First clone the repository and open a shell inside the root folder of the cloned repository.

Create a virtual environment (tested with python 3.12).

```sh
python3 -m venv venv
```

Activate the virtual environment

```sh
./venv/bin/activate
```

Install the package into the virtual environment

```py
pip install .
```

!!! note "Editable Installs"
    You can use the `-e` flag in order to create an 'editable' install. This way there is no need to reinstall the package after pulling the latest changes.

