import os
from typing import Callable, Optional, Union

import faiss
import numba
import numpy as np
import polars as pl
from scipy import linalg, sparse
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm, trange

from .base_recommender import BaseRecommender, KnnBatchInfo
from .matrix_ops import interactions_to_csc_matrix


@numba.jit(signature_or_function=numba.void(
    numba.float32[:, :],
    numba.float32[:, :],
    numba.int32[:],
    numba.int32[:],
    numba.float32[:],
    numba.float32[:]
),
    nopython=True,
    parallel=True)
def _compute_loss_without_reg(user_features, item_features, csc_indptr, csc_row_indices, csc_values, loss_per_item):
    for item_num in numba.prange(item_features.shape[0]):
        start_row = csc_indptr[item_num]
        end_row = csc_indptr[item_num + 1]
        loss_value = 0.0

        for user_num in range(user_features.shape[0]):
            if start_row < end_row and user_num == csc_row_indices[start_row]:
                item_loss = np.dot(user_features[user_num, :], item_features[item_num, :])
                item_loss = csc_values[start_row] - item_loss
                item_loss *= item_loss
                loss_value += item_loss
                start_row += 1

        loss_per_item[item_num] = loss_value


@numba.jit(nopython=True,
           signature_or_function=numba.int32[:](
               numba.int32[:],
               numba.int32[:]
           )
           )
def set1_diff_assume_unique(x, y):
    sorted_x_indices = np.argsort(x)
    sorted_y_indices = np.argsort(y)

    start_x = 0
    start_y = 0
    end_x = len(sorted_x_indices)
    end_y = len(sorted_y_indices)

    select_indices = []

    while start_x < end_x and start_y < end_y:
        if x[sorted_x_indices[start_x]] == y[sorted_y_indices[start_y]]:
            start_x += 1
            start_y += 1
        elif x[sorted_x_indices[start_x]] < y[sorted_y_indices[start_y]]:
            select_indices.append(start_x)
            start_x += 1
        else:
            start_y += 1

    while start_x < end_x:
        select_indices.append(start_x)
        start_x += 1

    return x[np.array(select_indices, dtype=np.int32)]


@numba.jit(nopython=True, parallel=True,
           signature_or_function=numba.int32[:, :](
               numba.int32[:],
               numba.int32,
               numba.int32[:],
               numba.int32[:],
               numba.float32[:, :],
               numba.float32[:, :],
               numba.int32[:]
           )
           )
def _batch_recommend(
    user_ids,
    num_rec,
    pos_item_features_bound_per_user,
    pos_item_feature_indices,
    all_user_features,
    all_item_features,
    all_item_zero_ids
):
    recommended_item_ids = np.full(fill_value=-1, shape=(len(user_ids), num_rec), dtype=np.int32)

    for i in numba.prange(len(user_ids)):
        start_bound = pos_item_features_bound_per_user[i]
        end_bound = pos_item_features_bound_per_user[i + 1]
        pos_item_ids = pos_item_feature_indices[start_bound: end_bound]
        user_features = all_user_features[user_ids[i]]
        item_ids = set1_diff_assume_unique(all_item_zero_ids, pos_item_ids)
        all_item_features_except_positive = all_item_features[item_ids, :]
        scores_per_user = (user_features @ all_item_features_except_positive.T).reshape(-1)
        candidate_item_ids = np.flip(np.argsort(scores_per_user))
        recommended_item_ids[i] = item_ids[candidate_item_ids[:num_rec]]

    return recommended_item_ids


@numba.jit(nopython=True, parallel=True, debug=False,
           signature_or_function=numba.void(
               numba.float32,
               numba.float32[:, :],
               numba.float32[:, :],
               numba.int32[:],
               numba.int32[:],
               numba.float32[:]
           ))
def _update_als_features(l2_reg, update_features, fixed_features, fixed_update_ratings_indptr, fixed_update_ratings_rowindices, fixed_update_ratings_data):
    identity = np.eye(update_features.shape[1], update_features.shape[1], dtype=np.float32)

    for update_index in numba.prange(update_features.shape[0]):
        start_index = fixed_update_ratings_indptr[update_index]
        end_index = fixed_update_ratings_indptr[update_index + 1]

        left_matrix = l2_reg * identity
        target = np.zeros((update_features.shape[1],), dtype=np.float32)

        while start_index < end_index:
            fixed_index = fixed_update_ratings_rowindices[start_index]
            left_matrix += np.dot(fixed_features[fixed_index][:, np.newaxis],
                                  fixed_features[fixed_index][np.newaxis, :])
            target += fixed_update_ratings_data[start_index] * fixed_features[fixed_index]
            start_index += 1

        update_features[update_index] = np.linalg.solve(
            left_matrix, target.reshape(-1, 1)).reshape(-1)


@numba.jit(nopython=True, parallel=False, debug=False,
           signature_or_function=numba.void(
               numba.int32,
               numba.float32,
               numba.float32[:, :],
               numba.float32[:, :],
               numba.int32[:],
               numba.int32[:],
               numba.float32[:],
               numba.int32[:],
               numba.int32[:],
               numba.float32[:]
           ))
def _als_iterations(
    num_iters,
    l2_reg,
    user_features,
    item_features,
    item_user_ratings_ind_ptr,
    item_user_ratings_rowindices,
    item_user_ratings_data,
    user_item_ratings_ind_ptr,
    user_item_ratings_rowindices,
    user_item_ratings_data,
):
    for i in range(num_iters):
        _update_als_features(
            l2_reg,
            user_features,
            item_features,
            item_user_ratings_ind_ptr,
            item_user_ratings_rowindices,
            item_user_ratings_data
        )

        _update_als_features(
            l2_reg,
            item_features,
            user_features,
            user_item_ratings_ind_ptr,
            user_item_ratings_rowindices,
            user_item_ratings_data
        )


class ALS(BaseRecommender):
    def __init__(self,
                 latent_size: int,
                 l2_reg: float,
                 num_iterations: int,
                 seed: int,
                 user_encoder: LabelEncoder,
                 item_encoder: LabelEncoder,
                 user_id_col: str = "user_id",
                 item_id_col: str = "item_id"):
        assert latent_size > 0
        assert l2_reg > 0
        assert num_iterations > 0
        assert seed > 0
        super().__init__(user_encoder=user_encoder, item_encoder=item_encoder,
                         user_id_col=user_id_col, item_id_col=item_id_col)
        self._latent_size = latent_size
        self._l2_reg = np.float32(l2_reg)
        self._num_iters = num_iterations
        self.user_features = None
        self.item_features = None
        self._seed = seed
        self.final_loss = 0.0
        self._user_item_interactions = None
        self._most_sim_index = None

    def _compute_loss(self, user_features: np.ndarray, item_features: np.ndarray, user_item_ratings: sparse.csc_matrix, loss_values: np.ndarray):
        _compute_loss_without_reg(
            user_features,
            item_features,
            user_item_ratings.indptr,
            user_item_ratings.indices,
            user_item_ratings.data,
            loss_values)

        return loss_values.sum()

    def _knn_query(self, vectors, k: int) -> KnnBatchInfo:
        assert self._most_sim_index is not None, "You need fit first"
        norm_vectors = vectors.copy()
        faiss.normalize_L2(norm_vectors)
        cosine_sim, indices = self._most_sim_index.search(norm_vectors, k)
        return KnnBatchInfo(indices, 1 - cosine_sim)

    def _get_item_vectors(self, zero_based_indices: np.ndarray):
        assert self.item_features is not None
        features = self.item_features[zero_based_indices, :]
        return features

    def _data_name(self):
        return "data.npz"

    def _index_name(self):
        return "index.pickle"

    def save_index(self, base_dir: str):
        assert self._most_sim_index is not None
        assert self._user_item_interactions is not None
        assert os.path.isdir(base_dir)
        sparse.save_npz(os.path.join(base_dir, self._data_name()), self._user_item_interactions)
        faiss.write_index(self._most_sim_index, os.path.join(base_dir, self._index_name()))

    def load_index(self, base_dir: str):
        assert self._most_sim_index is None, "Index already exists"
        assert self._user_item_interactions is None, "Index already exists"
        assert os.path.isdir(base_dir)
        self._user_item_interactions = sparse.load_npz(os.path.join(base_dir, self._data_name()))
        faiss.read_index(os.path.join(base_dir, self._index_name()))

    def fit(self, user_item_interactions: sparse.csc_matrix,
            loss_step_report: Optional[int] = None,
            loss_callback: Optional[Callable[[int, float], None]] = None,
            progress: bool = False):
        assert sparse.isspmatrix_csc(
            user_item_interactions), "An input matrix muts be in the CSC format"
        num_users = user_item_interactions.shape[0]
        num_items = user_item_interactions.shape[1]

        gen = np.random.default_rng(self._seed)
        self.user_features = np.ascontiguousarray(gen.normal(0, 1, size=(num_users,
                                                                         self._latent_size)).astype(np.float32))
        self.item_features = np.ascontiguousarray(gen.normal(0, 1, size=(num_items,
                                                                         self._latent_size)).astype(np.float32))
        item_user_ratings = user_item_interactions.T.tocsc()
        loss_values = np.zeros((num_items,), dtype=np.float32)
        num_iters = self._num_iters

        if loss_step_report is None:
            loss_step_report = num_iters

        progress = trange(num_iters, disable=not progress, mininterval=1)
        prev_loss = None

        current_iter = 0

        with progress:
            while current_iter < num_iters:
                local_iters = min(loss_step_report, num_iters - current_iter)
                _als_iterations(
                    local_iters,
                    np.float32(self._l2_reg),
                    self.user_features,
                    self.item_features,
                    item_user_ratings.indptr,
                    item_user_ratings.indices,
                    item_user_ratings.data,
                    user_item_interactions.indptr,
                    user_item_interactions.indices,
                    user_item_interactions.data
                )
                current_iter += local_iters
                progress.update(local_iters)

                if loss_callback is not None:
                    loss_value = self._compute_loss(
                        self.user_features, self.item_features, user_item_interactions, loss_values)
                    loss_callback(current_iter, loss_value)

                    progress_info = {"loss": loss_value}
                    if prev_loss is not None:
                        progress_info["loss_decreasing"] = prev_loss - loss_value

                    progress.set_postfix(progress_info)
                    prev_loss = loss_value

        self.final_loss = self._compute_loss(
            self.user_features, self.item_features, user_item_interactions, loss_values)

        self._user_item_interactions = user_item_interactions
        self._most_sim_index = faiss.IndexFlatIP(self.item_features.shape[1])
        item_features = self.item_features.copy()
        faiss.normalize_L2(item_features)
        self._most_sim_index.add(item_features)

    def _recommend(self,
                   user_item_interactions: pl.DataFrame,
                   user_features: Optional[np.ndarray],
                   item_features: np.ndarray,
                   num_rec: int,
                   all_item_zero_based_ids: np.ndarray):
        inter_with_unknown_users = user_item_interactions
        pos_zero_item_ids = self._item_encoder.transform(
            inter_with_unknown_users.get_column(self._item_id_col).to_numpy()).astype(np.int32)

        num_pos_items_per_user = inter_with_unknown_users.groupby(self._user_id_col).count()
        item_features_index_bounds_per_user = num_pos_items_per_user.get_column("count")
        last_value = item_features_index_bounds_per_user[-1]
        item_features_index_bounds_per_user = item_features_index_bounds_per_user.shift_and_fill(
            0, periods=1).append(pl.Series([last_value], dtype=item_features_index_bounds_per_user.dtype)).cumsum().to_numpy().astype(np.int32)

        del last_value

        user_encoder = self._user_encoder

        if user_features is None:
            user_encoder = LabelEncoder()
            zero_based_user_ids = user_encoder.fit_transform(inter_with_unknown_users.get_column(
                self._user_id_col).unique().to_numpy()).astype(np.int32)

            assert len(item_features_index_bounds_per_user) == len(zero_based_user_ids) + 1

            # item_user_ratings
            item_user_ratings = interactions_to_csc_matrix(
                inter_with_unknown_users, user_encoder, self._item_encoder).T.tocsc()

            unknown_user_features = np.zeros(
                (len(zero_based_user_ids), self._latent_size), dtype=np.float32)

            _update_als_features(
                np.float32(self._l2_reg),
                unknown_user_features,
                self.item_features,
                item_user_ratings.indptr,
                item_user_ratings.indices,
                item_user_ratings.data
            )

            zero_based_candidate_items = _batch_recommend(
                zero_based_user_ids,
                np.int32(num_rec),
                item_features_index_bounds_per_user,
                pos_zero_item_ids,
                unknown_user_features,
                item_features,
                all_item_zero_based_ids
            )
        else:
            zero_based_user_ids = user_encoder.transform(inter_with_unknown_users.get_column(
                self._user_id_col).unique().to_numpy()).astype(np.int32)

            assert len(item_features_index_bounds_per_user) == len(zero_based_user_ids) + 1

            zero_based_candidate_items = _batch_recommend(
                zero_based_user_ids,
                np.int32(num_rec),
                item_features_index_bounds_per_user,
                pos_zero_item_ids,
                self.user_features,
                self.item_features,
                all_item_zero_based_ids
            )

        recommended_item_ids_per_user = self._item_encoder.inverse_transform(
            zero_based_candidate_items.reshape(-1))

        rec = pl.DataFrame(
            {
                self._user_id_col: np.repeat(user_encoder.inverse_transform(zero_based_user_ids), num_rec),
                self._item_id_col: recommended_item_ids_per_user
            },
            schema={name: value for name, value in user_item_interactions.schema.items(
            ) if name in (self._user_id_col, self._item_id_col)}
        )

        return rec

    def recommend(self, user_item_pair: pl.DataFrame, num_rec_per_user: int) -> pl.DataFrame:
        assert self.item_features is not None
        assert self.user_features is not None
        all_user_ids = user_item_pair.get_column(self._user_id_col).unique()

        known_mask = all_user_ids.is_in(self._user_encoder.classes_)
        unknown_user_ids = all_user_ids.filter(~known_mask)
        known_user_ids = all_user_ids.filter(known_mask)

        rec_for_unknown_users = pl.DataFrame()
        rec_for_known_users = pl.DataFrame()
        all_item_ids = np.arange(self.item_features.shape[0], dtype=np.int32)

        if not unknown_user_ids.is_empty():
            inter_with_unknown_users = user_item_pair.filter(
                pl.col(self._user_id_col).is_in(unknown_user_ids)
            )

            rec_for_unknown_users = self._recommend(
                inter_with_unknown_users,
                None,
                self.item_features,
                num_rec_per_user,
                all_item_ids
            )

            del inter_with_unknown_users

        if not known_user_ids.is_empty():
            inter_with_known_users = user_item_pair.filter(
                pl.col(self._user_id_col).is_in(known_user_ids)
            )

            rec_for_known_users = self._recommend(
                inter_with_known_users,
                self.user_features,
                self.item_features,
                num_rec_per_user,
                all_item_ids
            )

        rec = rec_for_known_users.vstack(rec_for_unknown_users)

        return rec.with_columns((pl.col(self._item_id_col).cumcount() + 1).over(self._user_id_col).alias("rank")).filter(pl.col("rank") <= num_rec_per_user)
