"""
Tools for data input/output.
"""

import doctest
from typing import Any, Callable


def doctest_function(
    object: Callable[..., Any],
    globs: dict[str, Any],
    verbose=True
) -> None:
    """
    Run doctests for a specific function or class.

    Parameters
    ----------
    object : Callable[..., Any]
        Class, function, or other object with doctests to be run.
    globs : dict[str, Any]
        Global variables from module of interest.

    See Also
    --------
    run_doctests.run_doctest_suite :
        Simultaneously run all doctests across modules.
    """
    print('-------------------------------------------------------')
    finder = doctest.DocTestFinder(verbose=verbose, recurse=False)
    runner = doctest.DocTestRunner(verbose=verbose)
    for test in finder.find(obj=object, globs=globs):
        results = runner.run(test)
    print('-------------------------------------------------------')
    print(results)


def main():
    # Comment out (2) to run all tests in script; (1) to run specific tests
    import doctest
    doctest.testmod(verbose=True)

    # doctest_function(doctest_function, globs=globals())

    pass


if __name__ == "__main__":
    main()
