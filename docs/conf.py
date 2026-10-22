"""Sphinx configuration for the VeriGym documentation.

Build locally with `uv run --group docs sphinx-build docs docs/_build/html`,
or with live reload: `uv run --group docs sphinx-autobuild docs docs/_build/html`.
"""

import verigym

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

exclude_patterns = ["_build", "**/.ipynb_checkpoints"]

# -- Markdown and notebooks --------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath", "tasklist"]
myst_heading_anchors = 3  # allows links like `page.md#some-heading`

# Notebooks are shown with their stored outputs. They are not executed, because
# the workflows need external tools (Julia, PRISM) and long training runs.
nb_execution_mode = "off"

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


def _hide_workflows_in_sidebar(app, pagename, templatename, context, doctree):
    """The workflows are listed on their overview page, not one by one in the sidebar."""
    tree = context.get("furo_navigation_tree")
    if not tree:
        return
    from bs4 import BeautifulSoup  # a dependency of furo

    soup = BeautifulSoup(tree, "html.parser")

    def is_overview(href):  # on the overview page itself, furo links to it with "#"
        return href.split("#")[0].endswith("workflows.html") or (pagename == "workflows" and href == "#")

    items = [a.parent for a in soup.select("li > a.reference") if is_overview(a.get("href", ""))]
    for item in items:
        for child in item.find_all(["ul", "input", "label"], recursive=False):
            child.decompose()
        item["class"] = [c for c in item.get("class", []) if c != "has-children"]
        if pagename.startswith("workflows/"):  # highlight the overview while reading a workflow
            item["class"].append("current-page")
    context["furo_navigation_tree"] = str(soup)


def setup(app):
    # Runs after furo's own handler (priority 500), which computes the sidebar
    app.connect("html-page-context", _hide_workflows_in_sidebar, priority=900)
