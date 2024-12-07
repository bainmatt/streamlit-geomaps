# %%
# Configure

# Python interactive window help:
# https://code.visualstudio.com/docs/python/jupyter-support-py

# flake8: noqa: E403

import re
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt

from pathlib import Path
from pandas.plotting import scatter_matrix

src_path = Path('..')
sys.path.append(str(src_path.resolve()))

from src.paths import get_path_to
from src.inspection import display
from src.stylesheet import customize_plots
from src.inspection import make_df, display, display2

customize_plots()
# %config InlineBackend.figure_format = 'svg'

# %%
# Load data
if 'data' not in locals():
    # data = pd.read_csv(
    #     get_path_to("data", "raw", "PBJ_Daily_Nurse_Staffing_Q1_2024.zip"),
    #     encoding='ISO-8859-1',
    #     low_memory=False
    # )
    pass
else:
    print("data loaded.")

# %%
