import os
from typing import List, Optional

import pandas as pd
import polars as pl


class MovieLens1M:
    """https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset
    """
    RELEASE_DATE_COL = "release_date"
    VIDEO_RELEASE_DATE = "video_release_date"

    COL_DTYPES = {
        "item_id": "uint32[pyarrow]",
        "title": "utf8[pyarrow]",
        "genres": "utf8[pyarrow]"
    }

    @staticmethod
    def load_items(data_dir: str, item_columns: Optional[List[str]] = None):
        path_to_data = os.path.join(data_dir, "movies.dat")

        if item_columns is None:
            item_columns = list(MovieLens1M.COL_DTYPES.keys())
            column_indices = None
            dtypes = MovieLens1M.COL_DTYPES
        else:
            all_cols = list(MovieLens1M.COL_DTYPES.keys())
            column_indices = [all_cols.index(col) for col in item_columns]
            column_indices.sort()
            item_columns = [item_columns[index] for index in column_indices]
            dtypes = {col: MovieLens1M.COL_DTYPES[col] for col in item_columns}

        data = pd.read_csv(
            path_to_data,
            sep="::",
            names=item_columns,
            encoding="latin-1",
            usecols=column_indices,
            dtype=dtypes,
            dtype_backend="pyarrow"
        )

        data = pl.from_pandas(data)
        return data

    @ staticmethod
    def load_users(data_dir: str):
        path_to_data = os.path.join(data_dir, "users.dat")

        dtypes = {"user_id": "uint32[pyarrow]", "gender": "category", "age": "int8[pyarrow]",
                  "occupation": "int8[pyarrow]", "zip_code": "category"}

        data = pd.read_csv(
            path_to_data,
            sep="::",
            names=list(dtypes.keys()),
            encoding="utf-8",
            dtype=dtypes,
            dtype_backend="pyarrow"
        )

        return pl.from_pandas(data)

    @ staticmethod
    def load_interactions(data_dir: str, sort: bool = True):
        path_to_data = os.path.join(data_dir, "ratings.dat")

        ratings = pd.read_csv(
            path_to_data,
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="utf-8",
            dtype={"user_id": "uint32[pyarrow]", "item_id": "uint32[pyarrow]",
                   "rating": "float32[pyarrow]", "timestamp": "int32[pyarrow]"},
            dtype_backend="pyarrow"
        )

        ratings = pl.from_pandas(ratings)

        ratings = ratings.with_columns(pl.from_epoch(
            pl.col("timestamp"), time_unit="s"))

        if sort:
            ratings = ratings.sort(["user_id", "item_id", "timestamp"])

        return ratings
