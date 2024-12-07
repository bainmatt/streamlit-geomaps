"""
Tools for visualizing matrix and dataframe operations.

Notes
-----
Adapted from [1]_ to support NumPy arrays, in addition to Pandas
data frames.

References
----------
.. [1] VanderPlas, J. (2016). Python data science handbook: Essential tools
       for working with data. O'Reilly Media, Inc.
----
"""

import numpy as np
import pandas as pd
from typing import Any, Iterable

from IPython.display import display as ipy_display, HTML


def make_df(cols: Iterable[Any], ind: Iterable[Any]) -> pd.DataFrame:
    """
    Generate a data frame with a simple structure for conducting tests.

    Parameters
    ----------
    cols : Iterable[Any]
        An iterable with items representing column names.
    ind : Iterable[Any]
        An iterable with items representing row names.

    Returns
    -------
    pd.DataFrame
        A pandas data frame of shape (len(ind), len(cols)).

    Examples
    --------
    >>> from src.inspection import make_df

    >>> import pandas as pd
    >>> make_df("ABC", [1,2,3])
        A   B   C
    1  A1  B1  C1
    2  A2  B2  C2
    3  A3  B3  C3

    >>> make_df([1,2,3], ("A", "B", "C"))
        1   2   3
    A  1A  2A  3A
    B  1B  2B  3B
    C  1C  2C  3C
    """

    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data=data, index=pd.Index(ind))


def display(
    *args,
    globs: dict[str, Any] | None = None,
    bold: bool = True
) -> None:
    """
    Display an informative representation of multiple objects side-by-side.

    Parameters
    ----------
    *args : tuple
        Tuple of expressions to evaluate and display.
    globs : dict[str, Any], default=None
        Global namespace, to give eval() access to nonlocals passed by name.
    bold : bool, default=True
        Option to enable/disable string styling.

    Notes
    -----
    To view a complete column view of a DataFrame:

        with pd.option_context('display.max_columns', None):
            display("{object}", globs=globals())

    Warnings
    --------
    This function uses `eval()` to render expressions it receives
    as strings. Access to variables in the global namespace is controlled
    by `globs`. Take care to only pass trusted expressions to the function.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from src.inspection import display, make_df

    Data frame example

    >>> df1 = make_df('AB', [1, 2]); df2 = make_df('AB', [3, 4])
    >>> display('df1', 'df2', 'pd.concat([df1, df2])', globs=globals(), bold=False)
    <BLANKLINE>
    df1
    --- (2, 2) ---
        A   B
    1  A1  B1
    2  A2  B2
    <BLANKLINE>
    <BLANKLINE>
    df2
    --- (2, 2) ---
        A   B
    3  A3  B3
    4  A4  B4
    <BLANKLINE>
    <BLANKLINE>
    pd.concat([df1, df2])
    --- (4, 2) ---
        A   B
    1  A1  B1
    2  A2  B2
    3  A3  B3
    4  A4  B4
    <BLANKLINE>
    <BLANKLINE>

    Matrix example

    >>> A = np.array([[1, 3], [2, 4]]); x = np.array([[0, 1]])
    >>> display("A", "x.T", "np.dot(A, x.T)", globs=globals(), bold=False)
    <BLANKLINE>
    A
    --- (2, 2) ---
    array([[1, 3],
           [2, 4]])
    <BLANKLINE>
    <BLANKLINE>
    x.T
    --- (2, 1) ---
    array([[0],
           [1]])
    <BLANKLINE>
    <BLANKLINE>
    np.dot(A, x.T)
    --- (2, 1) ---
    array([[3],
           [4]])
    <BLANKLINE>
    <BLANKLINE>
    """

    if globs is None:
        globs = {}

    output = ""
    for arg in args:
        name = '\033[1m' + arg + '\033[0m' if bold else arg
        value = np.round(eval(arg, globs), 2)
        shape = np.shape(value)
        output += f"\n{name}\n--- {repr(shape)} ---\n{repr(value)}\n\n"

    print(output)
    return None


def display2(
    *args,
    globs: dict[str, Any] | None = None,
    bold: bool = True,
    width: str = "400px"  # Fixed width for each block
) -> None:
    """
    Display an informative representation of multiple objects side-by-side in Jupyter.

    Parameters
    ----------
    *args : tuple
        Tuple of expressions to evaluate and display.
    globs : dict[str, Any], default=None
        Global namespace, to give eval() access to nonlocals passed by name.
    bold : bool, default=True
        Option to enable/disable string styling.
    width : str, default="400px"
        Fixed width for each displayed block in the Jupyter notebook.

    Warnings
    --------
    This function uses `eval()` to render expressions it receives
    as strings. Access to variables in the global namespace is controlled
    by `globs`. Take care to only pass trusted expressions to the function.
    """

    if globs is None:
        globs = {}

    outputs = []
    for arg in args:
        name = f"<b>{arg}</b>" if bold else arg
        value = np.round(eval(arg, globs), 2)
        shape = np.shape(value)
        content = (
            f"<div style='width:{width}; padding:10px; float:left;'>"
            f"<pre>{name}\n"
            f"--- {repr(shape)} ---\n"
            f"{repr(value)}</pre></div>"
        )
        # content = f"<div style='width:{width}; padding:10px; float:left;'><pre>{name}\n--- {repr(shape)} ---\n{repr(value)}</pre></div>"  # noqa: E501
        outputs.append(content)

    # Clearfix for layout
    clearfix = "<div style='clear: both;'></div>"

    # Display the HTML content in Jupyter
    html_output = ''.join(outputs) + clearfix
    ipy_display(HTML(html_output))

    return None


def main():
    # Comment out (2) to run all tests in script; (1) to run specific tests
    import doctest
    doctest.testmod(verbose=True)

    # from src.workflow import doctest_function
    # doctest_function(display, globs=globals())

    return None


if __name__ == "__main__":
    main()
