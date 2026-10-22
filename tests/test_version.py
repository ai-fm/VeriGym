import tomllib
from pathlib import Path

import verigym


def test_version_matches_pyproject():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert verigym.__version__ == pyproject["project"]["version"]
