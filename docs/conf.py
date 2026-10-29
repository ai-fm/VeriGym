"""Sphinx configuration for the VeriGym documentation.

Build locally with `uv run --group docs sphinx-build docs docs/_build/html`,
or with live reload: `uv run --group docs sphinx-autobuild docs docs/_build/html`.
"""

import sys
from pathlib import Path

import verigym

sys.path.insert(0, str(Path(__file__).parent / "_ext"))  # local helper modules, e.g. percent_script

project = "VeriGym"
author = "VeriGym contributors"
copyright = "2026, VeriGym contributors"
release = verigym.__version__

extensions = [
    "myst_nb",  # Markdown pages (MyST) and Jupyter notebooks
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # numpy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",  # "[source]" links in the API reference
    "sphinx_copybutton",
]

exclude_patterns = [
    "_build",
    "_ext",
    "**/.ipynb_checkpoints",
    "conf.py",
    # The workflows and technical evaluations are the .py scripts in notebooks/ (read with jupytext, see below),
    # not their helper modules, their outputs or the old .ipynb copies
    "notebooks/*_utils",
    "notebooks/output*",
    "notebooks/*.ipynb",
    "tutorials/*.py",  # the tutorials are the .ipynb files
]

# -- Markdown and notebooks --------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath", "tasklist"]
myst_heading_anchors = 3  # allows links like `page.md#some-heading`

# Notebooks are shown with their stored outputs. They are not executed, because
# the workflows need external tools (Julia, PRISM) and long training runs.
nb_execution_mode = "off"

# Python scripts with `# %%` cells (jupytext percent format) are read as notebooks
nb_custom_formats = {".py": "percent_script.read"}

# -- API reference -----------------------------------------------------------
autoclass_content = "both"  # class docstring followed by the __init__ docstring
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
toc_object_entries_show_parents = "hide"  # "simulate()" instead of "VeriGymEnv.simulate()"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# "Attributes" sections as fields, so an attribute does not clash with a method of the same name
napoleon_use_ivar = True
# Single backticks in docstrings (`VeriGymEnv`) link to the object if it is documented, otherwise render as code
default_role = "py:obj"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "gymnasium": ("https://gymnasium.farama.org", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = "VeriGym"
html_logo = "VeriGym_Logo.png"
html_theme_options = {
    "source_repository": "https://github.com/ai-fm/VeriGym/",
    "source_branch": "main",
    "source_directory": "docs/",
}


# Overview pages that list their pages themselves, and the prefix of the pages they list
_OVERVIEWS = {"workflows": "notebooks/workflow_", "tech_eval": "notebooks/tech_eval_"}


def _hide_overview_children_in_sidebar(app, pagename, templatename, context, doctree):
    """The workflows and technical evaluations are listed on their overview pages, not one by one in the sidebar."""
    tree = context.get("furo_navigation_tree")
    if not tree:
        return
    from bs4 import BeautifulSoup  # a dependency of furo

    soup = BeautifulSoup(tree, "html.parser")

    def overview_of(href):  # on the overview page itself, furo links to it with "#"
        target = pagename if href == "#" else href.split("#")[0].removesuffix(".html")
        return next((name for name in _OVERVIEWS if target.endswith(name)), None)

    # Collect first: removing an overview's children also removes links that select() already returned
    items = [(a.parent, overview) for a in soup.select("li > a.reference") if (overview := overview_of(a.get("href", "")))]
    for item, overview in items:
        for child in item.find_all(["ul", "input", "label"], recursive=False):
            child.decompose()
        item["class"] = [c for c in item.get("class", []) if c != "has-children"]
        if pagename.startswith(_OVERVIEWS[overview]):  # highlight the overview while reading one of its pages
            item["class"].append("current-page")
    context["furo_navigation_tree"] = str(soup)


def setup(app):
    # Runs after furo's own handler (priority 500), which computes the sidebar
    app.connect("html-page-context", _hide_overview_children_in_sidebar, priority=900)
