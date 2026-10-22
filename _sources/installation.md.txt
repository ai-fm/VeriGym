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
