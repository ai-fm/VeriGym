<center>
<img src="https://raw.githubusercontent.com/ai-fm/VeriGym/main/docs/VeriGym_Logo.png" width=100>

<h1>VeriGym</h1>
</center>

This is **VeriGym**.

> [!WARNING]
> This README is currently undergoing construction.

# Installation

After `git clone` please run `uv sync` to create the virtual environment. [UV](https://docs.astral.sh/uv/) is a strongly recommended package manager, faster and better than `pip`.

Supported: Python 3.12–3.15 on Linux and macOS. Windows is supported through Windows Subsystem for Linux ([WSL](https://learn.microsoft.com/windows/wsl/)).

On Python 3.14 and newer, the Box2D environments (e.g. `LunarLander`) are not installed automatically, because Box2D has to be compiled for that Python version. To use them, install [SWIG](https://www.swig.org/) (`brew install swig` or `sudo apt install swig`), then run `pip install "gymnasium[box2d]"`.


# Contribution Guide

Please check out the [contribution guide](https://github.com/ai-fm/VeriGym/blob/main/docs/contribution.md) for branch naming, testing, naming conventions, etc.
