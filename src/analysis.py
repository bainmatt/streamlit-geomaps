"""
Utility functions for data wrangling, preprocessing, and analysis.
"""

import pandas as pd

from src.inspection import display

# z: str = 4

x = pd.DataFrame([1, 2, 3])
display("x", globs=globals())
