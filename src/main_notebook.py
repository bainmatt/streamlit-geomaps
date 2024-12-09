# %%
# Configure

# Python interactive window help:
# https://code.visualstudio.com/docs/python/jupyter-support-py

# flake8: noqa: E403

import sys
from pathlib import Path

src_path = Path('..')
sys.path.append(str(src_path.resolve()))

from src.paths import get_path_to
from src.inspection import display
from src.stylesheet import customize_plots
from src.inspection import display, display2

customize_plots()
# %config InlineBackend.figure_format = 'svg'

# %%
# xx
