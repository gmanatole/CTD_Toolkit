
import duckdb
import pandas as pd


class SQLQuery:

    """
    Use duckdb to execute sql queries.
    Implemented to work on parquet file containing profile paths and associated timestamps, lat and lon
    """
    def __init__(self, database: str = ":memory:"):
        self.con = duckdb.connect(database)

    def register_dataframe(self, name: str, df: pd.DataFrame):
        self.con.register(name, df)

    def query(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def close(self):
        self.con.close()