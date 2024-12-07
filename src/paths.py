"""
Routines for managing project I/O paths.
"""

import os
from pathlib import Path
from typing import Any, Callable, Literal

import doctest
from src.workflow import doctest_function
from dotenv import find_dotenv, load_dotenv


def get_path_to(
    dir: Literal["root", "src", "data", "output"] = "root",
    *suffixes: str
) -> Path:
    """
    Construct an absolute path to a project directory for saving and loading.

    Parameters
    ----------
    dir : Literal["root", "src", "data", "output"], default="root"
        The project directory to return the absolute path to.
    *suffixes : str
        Additional path segments to append to the base directory.

    Returns
    -------
    Path
        The absolute path to `dir` and any additional components.

    Examples
    --------
    >>> get_path_to("data")  # doctest: +ELLIPSIS
    PosixPath('.../data')
    >>> get_path_to("data", "raw", "housing_data.csv")  # doctest: +ELLIPSIS
    PosixPath('.../data/raw/housing_data.csv')
    """

    root_path = Path(
        os.getenv('ROOT', Path(__file__).resolve().parent)
    )
    path = root_path if dir == "root" else root_path / dir

    # Remove leading slash from the first suffix if it exists
    if suffixes and suffixes[0].startswith("/"):
        suffixes = (suffixes[0][1:], *suffixes[1:])

    return path.joinpath(*suffixes)


def main(to_test: list[Callable[..., Any]] | None = None):
    """
    Load environment variables, run doctests and checks, and orchestrate.

    Parameters
    ----------
    to_test : list[Callable[..., Any]] | None, default=None
        A list of test functions to run. If None, all doctests in the module are executed.
    """

    load_dotenv(find_dotenv())

    # TODO: include this logic in `workflow.doctest_function` itself
    if to_test == []:
        to_test = None

    if to_test:
        for test_object in to_test:
            try:
                doctest_function(test_object, globs=globals())
            except Exception as e:
                print(f"Error running test {test_object.__name__}: {e}")
    else:
        doctest.testmod(verbose=True)

    pass


if __name__ == "__main__":
    main([])
