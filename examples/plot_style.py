"""Shared matplotlib style for the VeriGym examples.

Usage in a notebook in examples/:

    from plot_style import LIB_COLORS  # importing applies the style

Provides the following:
- sets the default colors (colorblind friendly) by loading the `verigym.mplstyle` file
- named colors via `LIB_COLORS['verigym']`
- list of markers
- default alpha value for fill areas of plots 

For notebooks in subfolders first need `sys.path.append("../..")` (one ".." per level).
"""

from pathlib import Path

import matplotlib.pyplot as plt

# load and apply the style file
STYLE_PATH = Path(__file__).with_name("verigym.mplstyle")
plt.style.use(STYLE_PATH)

# IBM colorblind-safe palette, https://davidmathlogic.com/colorblind/
BLUE = "#648FFF"
PURPLE = "#785EF0"
MAGENTA = "#DC267F"
ORANGE = "#FE6100"
YELLOW = "#FFB000"  # weak on white, use last and not for thin lines

# Same order as the color cycle in verigym.mplstyle (C0, C1, ...)
PALETTE = [BLUE, MAGENTA, ORANGE, PURPLE, YELLOW]

# Fixed color per library, so a library looks the same in every notebook.
# Variants of one library (e.g. "verigym (process + MDP)") share its color and
# differ by marker.
LIB_COLORS = {
    "verigym": BLUE,
    "gymnasium": ORANGE,
    "stormpy": MAGENTA,
    "numpy": PURPLE,
}

ANNOTATION_GREY = "#808080"  # arrows, notes and reference lines

MARKERS = ["o", "s", "D", "^", "v"]  # use in this order for marker-coded groups

AREA_ALPHA = 0.25  # fill_between bands (std, min/max, quartiles)
