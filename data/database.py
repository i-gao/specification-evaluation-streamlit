from typing import Dict, List, Any, Tuple
import pandas as pd
from langchain_core.tools import tool
from data.actions import Action


class Database:
    def __init__(self, dfs: Dict[str, Tuple[str, pd.DataFrame, Dict[str, str]]]):
        """Initialize the Database with table information.

        Args:
            dfs: A dictionary mapping table names to tuples containing:
                - table description (str)
                - pandas DataFrame
                - dictionary mapping column names to column descriptions
        """
        self.table_names = list(dfs.keys())
        self.table_descriptions = {
            table_name: dfs[table_name][0] for table_name in self.table_names
        }
        self.tables = {
            table_name: dfs[table_name][1] for table_name in self.table_names
        }
        self.column_descriptions = {
            table_name: dfs[table_name][2] for table_name in self.table_names
        }

    def _list_tables(self) -> List[Dict[str, Any]]:
        """List all tables in the database with their metadata.

        Returns:
            List of dictionaries, where each dict contains:
                - table_name: Name of the table
                - description: Description of the table
                - num_rows: Number of rows in the table
        """
        return [
            {
                "table_name": table_name,
                "description": self.table_descriptions[table_name],
                "num_rows": len(self.tables[table_name]),
            }
            for table_name in self.table_names
        ]

    def _list_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """List columns and their values in the specified table.

        Args:
            table_name: Name of the table to inspect.

        Returns:
            List of dictionaries, where each dict contains:
                - column_name: Name of the column
                - description: Description of the column
                - values: List of unique values in the column
        """

        def _get_values(column_name: str) -> List[str]:
            values = self.tables[table_name][column_name]
            # if it's a numerical type, return summary stats
            if values.dtype in [int, float]:
                return {
                    "min": values.min(),
                    "max": values.max(),
                    "mean": values.mean(),
                    "std": values.std(),
                    "median": values.median(),
                    "mode": values.mode()[0],
                }
            values = list(values.astype(str).value_counts().index)
            return {
                "num_unique_values": len(values),
                "top_10_most_common_values": values[:10],
            }

        return [
            {
                "column_name": column_name,
                "description": self.column_descriptions[table_name][column_name],
                "values": _get_values(column_name),
            }
            for column_name in self.column_descriptions[table_name]
        ]

    def _get_all_column_values(self, table_name: str, column_name: str) -> List[Any]:
        """
        Get all unique values in a column.

        Args:
            table_name: Name of the table to inspect.
            column_name: Name of the column to inspect.

        Returns:
            List of unique values in the column.
        """
        return self.tables[table_name][column_name].unique()

    def _get_table(
        self,
        table_name: str,
        query: str = "",
        max_rows: int = 50,
        shuffle: bool = False,
    ) -> pd.DataFrame:
        """Get the first n rows of a table with optional filtering.

        Args:
            table_name: Name of the table to retrieve.
            query: Optional query string to filter rows in the dataframe.
                Use pandas query syntax.
            n: Number of rows to return (default: 5).

        Returns:
            pandas.DataFrame: The first n rows of the requested table data.
        """
        df = self.tables[table_name]
        if query != "":
            df = df.query(query)
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(max_rows)
        return df.to_dict(orient="records")

    def get_tools(self):
        """Get LangChain tools for database operations."""

        @tool(parse_docstring=True)
        def list_tables_in_database() -> List[Dict[str, Any]]:
            """
            For each table in the database, return the table name, description, and number of rows.
            """
            return self._list_tables()

        @tool(parse_docstring=True)
        def list_columns_in_table(table_name: str) -> List[Dict[str, Any]]:
            """
            For each column in the specified table, return the column name, description, and most common values.

            Args:
                table_name (str): Name of the table to inspect.
            """
            return self._list_columns(table_name)

        @tool(parse_docstring=True)
        def get_table(
            table_name: str, query: str = "", max_rows: int = 50, sample: bool = False
        ) -> pd.DataFrame:
            """
            Get max_rows rows of a table by name. Optionally filter the rows with a pandas query.

            Args:
                table_name (str): Name of the table to retrieve.
                query (str): Optional query string to filter rows in the dataframe.
                    Use pandas query syntax.
                max_rows (int): Maximum number of rows to return (default: 50).
                sample (bool): Whether to randomly sample the max_rows rows instead of head (default: False).
            """
            return self._get_table(table_name, query, max_rows, sample)

        @tool(parse_docstring=True)
        def get_all_column_values(table_name: str, column_name: str) -> List[Any]:
            """
            List all the unique values in a column.

            Args:
                table_name (str): Name of the table to inspect.
                column_name (str): Name of the column to inspect.
            """
            return self._get_all_column_values(table_name, column_name)

        return [
            Action(
                fn=list_tables_in_database,
                is_public=True,
                is_human=False,
                name="List tables in database",
            ),
            Action(
                fn=list_columns_in_table,
                is_public=True,
                is_human=False,
                name="List columns in table",
            ),
            Action(fn=get_table, is_public=True, is_human=False, name="Get table"),
            Action(
                fn=get_all_column_values,
                is_public=True,
                is_human=False,
                name="Get all column values",
            ),
        ]

    def stats(self):
        raise NotImplementedError("Not implemented")
