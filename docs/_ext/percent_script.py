"""Reader that lets myst-nb show Python scripts with `# %%` cells (jupytext percent format) as notebooks."""

import jupytext


def read(text):
    """Read a percent-format script as a notebook."""
    nb = jupytext.reads(text, fmt="py:percent")
    # The scripts have no notebook metadata header: tell myst-nb to highlight the code cells as Python
    nb.metadata.setdefault("language_info", {"name": "python", "pygments_lexer": "ipython3"})
    return nb
