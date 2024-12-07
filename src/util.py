"""
Utilities for data manipulation and preprocessing.
"""

import sqlite3
from typing import Optional, Tuple
from contextlib import contextmanager

# from src.paths import get_path_to

# -- I/O utilities -----------------------------------------------------------

# import pandas as pd
# print(pd.read_csv(
#     "data/raw/PBJ_Daily_Nurse_Staffing_Q1_2024.zip",
#     encoding='ISO-8859-1',
#     low_memory=False
# ))
# print(pd.read_csv(
#     get_path_to("data", "/raw/PBJ_Daily_Nurse_Staffing_Q1_2024.zip"),
#     encoding='ISO-8859-1',
#     low_memory=False
# ))

# -- SQL utilities -----------------------------------------------------------


@contextmanager
def db_connection(
    db_file: str,
    commit: bool = True
):
    """
    Context manager for running commands on a database connection.
    """
    conn = sqlite3.connect(db_file, timeout=10.0)
    cursor = conn.cursor()
    try:
        yield conn, cursor
        if commit:
            conn.commit()
    finally:
        cursor.close()
        conn.close()


class DBManager:
    def __init__(self, db_file: str = "testdatabase.db"):
        """
        Initialize the DatabaseManager with the specified database file.

        Parameters
        ----------
        db_file : str, default="testdatabase.db"
            The database file to connect to.

        Examples
        --------
        >>> import os
        >>> import sqlite3
        >>> import pandas as pd
        >>> from src.util import DBManager

        >>> db_file = "testdatabase.db"
        >>> db = DBManager(db_file)
        >>> conn, cursor = db.open()

        Create a test table

        >>> cursor.execute(
        ...     "CREATE TABLE IF NOT EXISTS test ("
        ...     "id INTEGER PRIMARY KEY, name TEXT);"
        ... )  # doctest: +ELLIPSIS
        <...>
        >>> cursor.execute(
        ...     "INSERT INTO test (name) VALUES ('Test Name');"
        ... )  # doctest: +ELLIPSIS
        <...>
        >>> db.save()

        Run a query

        >>> pd.read_sql(
        ...     "SELECT * FROM test LIMIT 5",
        ...     conn
        ... )
           id       name
        0   1  Test Name

        Drop the table and close the database

        >>> cursor.execute("DROP TABLE test")  # doctest: +ELLIPSIS
        <...>
        >>> db.close()
        >>> os.remove(db_file)
        """
        self.db_file = db_file
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

    def open(self) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
        """
        Create a database connection and cursor.

        Returns
        -------
        Tuple[sqlite3.Connection, sqlite3.Cursor]
            A tuple containing the database connection and cursor.
        """
        self.conn = sqlite3.connect(self.db_file, timeout=10.0)
        self.cursor = self.conn.cursor()
        return self.conn, self.cursor

    def save(self) -> None:
        """
        Save changes to the database on an active connection.
        """
        if self.conn:
            self.conn.commit()

    def close(self) -> None:
        """
        Close an open database connection and cursor.
        """
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def main():
    # Comment out (2) to run all tests in script; (1) to run specific tests
    import doctest
    doctest.testmod(verbose=True)

    # from src.workflow import doctest_function
    # doctest_function(DBManager, globs=globals())

    # One-off tests
    # import pandas as pd
    # pd.read_csv(
    #     "data/raw/PBJ_Daily_Nurse_Staffing_Q1_2024.zip",
    #     encoding='ISO-8859-1',
    #     low_memory=False
    # )

    # from pathlib import Path
    # print(Path(__file__))

    return None


if __name__ == "__main__":
    main()
