"""
Custom parameter validation routines.
"""

import pandas as pd
from functools import wraps
from inspect import signature
from typing import Any, Callable


def validate_args(func):
    """
    Decorator to validate types and values of common args to
    functions and methods.

    The parameters to validate and their corresponding validators are
    defined in the `VALIDATOR_LOOKUP` dictionary.
    Supply additional params for validators that check field existence.

    Returns
    -------
    decorated_function : function or method
        The decorated function.

    Raises
    ------
    AssertionError
        If any of the arguments do not meet validation criteria.

    References
    ----------
    .. [1] sklearn.utils.validation : Utilities for input validation in
           scikit-learn.
           https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/validation.py


    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from src.validate import validate_args

    >>> @validate_args
    ... def plot_precis(df, err_prob=0.94, metric_cols=None, param_cols=None):
    ...     print("validation succeeded")

    >>> try:
    ...     plot_precis(df=[1, 2, 3])
    ... except AssertionError as e:
    ...     print(f"AssertionError: {e}")
    AssertionError: df must be a pandas DataFrame, got list

    >>> df = pd.DataFrame({
    ...     "accuracy": np.random.uniform(0.7, 0.95, 10),
    ...     "rmse": np.random.uniform(0.1, 0.3, 10),
    ...     "hyperparameter_1": np.random.choice([0.1, 0.01, 0.001], 10),
    ...     "hyperparameter_2": np.random.choice([50, 100], 10)
    ... })
    >>> try:
    ...     plot_precis(
    ...         df,
    ...         param_cols=["hyperparameter_1", 2],
    ...         metric_cols=["accuracy"],
    ...     )
    ... except AssertionError as e:
    ...     print(f"AssertionError: {e}")
    AssertionError: all elements of param_cols must be strings, got ['str', 'int']

    >>> plot_precis(
    ...     df,
    ...     metric_cols="accuracy",
    ...     param_cols=["hyperparameter_1"],
    ... )
    validation succeeded
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract {'{arg}': {val}} pairs from function signature
        # (See reference [1] in the docs).
        func_sig = signature(func)

        # Map supplied *args/**kwargs to the function signature
        params = func_sig.bind(*args, **kwargs)
        params.apply_defaults()

        # Ignore self/cls and positional/keyword markers
        to_ignore = [
            p.name
            for p in func_sig.parameters.values()
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ] + ["self", "cls"]
        params = {
            k: v for k, v in params.arguments.items() if k not in to_ignore
        }

        # For each validated param supplied, call the corresponding validator.
        # Supply additional params for validators that check field existence.
        param_validator_lookup = {
            param: validator
            for validator, params in VALIDATOR_LOOKUP.items()
            for param in params
        }
        for (param, value) in params.items():
            if param in param_validator_lookup.keys():
                param_validator_lookup[param](value, param, params.get('df'))

        return func(*args, **kwargs)

    return wrapper


# -- Custom validators -------------------------------------------------------


def validate_df(var, name, *args):
    assert isinstance(var, pd.DataFrame), \
        f"{name} must be a pandas DataFrame, got {type(var).__name__}"


def validate_proportion(val, name, *args):
    assert isinstance(val, float) and 0 < val < 1, \
        f"{name} must be a float between 0 and 1, got " \
        f"{type(val).__name__} = {val}"


def validate_column_list(var, name, df=None):
    if var is not None:
        assert isinstance(var, (str, list)), \
            f"{name} must be a string, list of strings, or None"
        if isinstance(var, list):
            assert all(isinstance(c, str) for c in var), \
                f"all elements of {name} must be strings, got " \
                f"{[type(c).__name__ for c in var]}"
        if df is not None:
            if isinstance(var, str):
                var = [var]
            missing_cols = [col for col in var if col not in df.columns]

            formatted_cols = ", ".join([
                f"{col} ({dtype})"
                for col, dtype in zip(df.columns, df.dtypes)
            ])
            assert not missing_cols, \
                f"columns {missing_cols} not found in DataFrame, " \
                f"valid columns are [{formatted_cols}]"


# -- Define (validator, [parameters]) dictionary -----------------------------

VALIDATOR_LOOKUP: dict[Callable[..., Any], list[str]] = {
    validate_df: ['df'],
    validate_proportion: ['err_prob'],
    validate_column_list: ['metric_cols', 'param_cols', 'metric_col'],
}


def main():
    # Comment out (2) to run all tests in script; (1) to run specific tests
    import doctest
    doctest.testmod(verbose=True)

    # from src.workflow import doctest_function
    # doctest_function(DetrendAndDeseasonalize, globs=globals())

    pass


if __name__ == "__main__":
    main()
