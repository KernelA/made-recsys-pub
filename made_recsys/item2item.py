import logging
import os
from typing import List, Optional, Union

import numba
import numpy as np
import polars as pl
from pynndescent import NNDescent
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from made_recsys.base_recommender import KnnBatchInfo

from .base_recommender import BaseRecommender
from .utils import load_pickle, save_pickle


class Item2ItemRecommender(BaseRecommender):
    def __init__(self,
                 metric: str,
                 seed: int,
                 user_encoder: LabelEncoder,
                 item_encoder: LabelEncoder,
                 low_memory: bool,
                 max_candidates: Optional[int] = None,
                 user_id_col: str = "user_id",
                 item_id_col: str = "item_id"):
        super().__init__(user_encoder=user_encoder, item_encoder=item_encoder,
                         user_id_col=user_id_col, item_id_col=item_id_col)
        self._metric = metric
        self._seed = seed
        self._index = None
        self._item_user_matrix = None
        self._max_candidates = max_candidates
        self._low_memory = low_memory

    def _data_name(self):
        return "data.npz"

    def _index_name(self):
        return "index.pickle"

    def save_index(self, path_to_dir: str):
        assert self._index is not None
        assert self._item_user_matrix is not None
        assert os.path.isdir(path_to_dir)
        sparse.save_npz(os.path.join(path_to_dir, self._data_name()), self._item_user_matrix)
        save_pickle(os.path.join(path_to_dir, self._index_name()), self._index)

    def load_index(self, path_to_dir: str):
        assert self._index is None, "Index already exists"
        assert self._item_user_matrix is None, "Index already exists"
        assert os.path.isdir(path_to_dir)
        self._item_user_matrix = sparse.load_npz(os.path.join(path_to_dir, self._data_name()))
        self._index = load_pickle(os.path.join(path_to_dir, self._index_name()))

    def fit(self, user_item_matrix: sparse.csc_matrix):
        assert sparse.isspmatrix_csc(user_item_matrix)
        self._item_user_matrix = user_item_matrix.T
        self._index = NNDescent(self._item_user_matrix.copy(), metric=self._metric,
                                max_candidates=self._max_candidates,
                                low_memory=self._low_memory,
                                random_state=self._seed,
                                parallel_batch_queries=True)

    def _get_item_vectors(self, zero_based_indices: np.ndarray) -> np.ndarray:
        assert self._item_user_matrix is not None, "You must fit first"
        return self._item_user_matrix[zero_based_indices, :]

    def _knn_query(self, vectors, k: int) -> KnnBatchInfo:
        indices, distances = self._index.query(vectors, k=k)
        return KnnBatchInfo(indices, distances)

    def recommend(self, user_item_pair: pl.DataFrame, num_rec_per_user: int, num_neighs: int) -> pl.DataFrame:
        self._logger.info("Begin most similiar item search")
        most_sim_items = self.most_similiar_items(user_item_pair, num_neighs)
        self._logger.info("Done")

        self._logger.info("Postprocessing")
        # remove existed interactions
        most_sim_items = most_sim_items.join(
            user_item_pair,
            left_on=[self._user_id_col, self._sim_item_id_col],
            right_on=[self._user_id_col, self._item_id_col],
            how="anti")

        num_sim_per_user = most_sim_items.lazy().select(
            pl.col(self._user_id_col),
            pl.col(self._sim_col).abs()).groupby(self._user_id_col).agg(
            [
                pl.sum(self._sim_col).alias("denum")
            ]
        ).collect()

        recommendations = most_sim_items.lazy().join(
            num_sim_per_user.lazy(), on=self._user_id_col
        ).select(
            pl.col(self._user_id_col),
            pl.col(self._sim_item_id_col),
            (pl.col(self._sim_col) / pl.col("denum")).alias("score")).sort(
            self._user_id_col, "score", descending=[False, True]).with_columns(
                (pl.col(self._sim_item_id_col).cumcount() + 1).over(self._user_id_col).alias("rank")
        ).select(
                pl.col(self._user_id_col),
                pl.col(self._sim_item_id_col).alias(self._item_id_col),
                pl.col("rank")
        ).filter(pl.col("rank") <= num_rec_per_user).collect()

        self._logger.info("Done")

        return recommendations
