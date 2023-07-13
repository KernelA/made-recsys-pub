import logging
from abc import ABC, abstractmethod
from typing import NamedTuple, Union

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.preprocessing import LabelEncoder


class KnnBatchInfo(NamedTuple):
    indices: np.ndarray
    distances: np.ndarray


class BaseRecommender(ABC):
    def __init__(self, user_encoder, item_encoder: LabelEncoder, user_id_col: str = "user_id", item_id_col: str = "item_id") -> None:
        super().__init__()
        self._logger = logging.getLogger("base_recommender")
        self._user_encoder = user_encoder
        self._item_encoder = item_encoder
        self._user_id_col = user_id_col
        self._item_id_col = item_id_col
        self._sim_item_id_col = "sim_item_id"
        self._sim_col = "distance"

    @abstractmethod
    def save_index(self, base_dir: str):
        pass

    @abstractmethod
    def load_index(self, base_dir: str):
        pass

    @abstractmethod
    def fit(self, user_item_interactions: sparse.csc_matrix, *args, **kwargs):
        pass

    @abstractmethod
    def _knn_query(self, vectors, k: int) -> KnnBatchInfo:
        pass

    @abstractmethod
    def _get_item_vectors(self, zero_based_indices: np.ndarray) -> np.ndarray:
        pass

    def most_similiar_items(self, item_ids: Union[pl.Series, pl.DataFrame], num_neighs: int) -> pl.DataFrame:
        assert num_neighs > 0

        if isinstance(item_ids, pl.Series):
            local_item_ids = item_ids.unique()
        else:
            local_item_ids = item_ids.get_column(self._item_id_col).unique()

        numpy_local_ids = local_item_ids.to_numpy()

        zero_based_indices = self._item_encoder.transform(numpy_local_ids)
        self._logger.info("Find %d neighbours for %d items", num_neighs, len(zero_based_indices))

        knn_info = self._knn_query(self._get_item_vectors(zero_based_indices), k=num_neighs + 1)
        k = knn_info.indices.shape[1]
        indices = self._item_encoder.inverse_transform(knn_info.indices.reshape(-1))

        most_sim_items = pl.DataFrame({
            self._item_id_col: np.repeat(numpy_local_ids, k),
            self._sim_item_id_col: indices,
            self._sim_col: knn_info.distances.reshape(-1)},
            schema={
            self._item_id_col: local_item_ids.dtype,
            self._sim_item_id_col: local_item_ids.dtype,
            self._sim_col: pl.Float32}).lazy()\
            .filter(pl.col(self._item_id_col) != pl.col(self._sim_item_id_col))\
            .with_columns(pl.col(self._sim_item_id_col).cumcount().over(self._item_id_col).alias("rank"))\
            .filter(pl.col("rank") < num_neighs)\
            .select(pl.all().exclude("rank")).collect()

        if isinstance(item_ids, pl.Series):
            return most_sim_items
        else:
            return item_ids.join(most_sim_items, on=self._item_id_col, how="inner")

    @abstractmethod
    def recommend(self, user_item_pair: pl.DataFrame, *args, **kwargs) -> pl.DataFrame:
        pass
