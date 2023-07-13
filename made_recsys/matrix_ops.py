from typing import Dict, Optional, Type, Union

import numpy as np
import polars as pl
from scipy import linalg, sparse
from sklearn.preprocessing import LabelEncoder


def eig_values(matrix):
    return linalg.eigh(matrix, eigvals_only=True)


def _to_sparse_matrix(matrix_class: Union[Type[sparse.csc_matrix], Type[sparse.csr_matrix]],
                      interactions: pl.DataFrame,
                      users_mapping: LabelEncoder,
                      items_mapping: LabelEncoder,
                      user_col: str = "user_id",
                      item_col: str = "item_id",
                      weight_col: Optional[str] = None):

    if weight_col is None:
        weights = np.ones(len(interactions), dtype=np.float32)
    else:
        weights = interactions.select(weight_col).to_series().to_numpy().astype(np.float32)

    interaction_matrix = matrix_class(
        (
            weights,
            (
                users_mapping.transform(interactions.select(
                    pl.col(user_col)).to_series().to_numpy()),
                items_mapping.transform(interactions.select(
                    pl.col(item_col)).to_series().to_numpy())
            )
        )
    )

    return interaction_matrix


def interactions_to_csr_matrix(interactions: pl.DataFrame,
                               users_mapping: LabelEncoder,
                               items_mapping: LabelEncoder,
                               user_col: str = 'user_id',
                               item_col: str = 'item_id',
                               weight_col=None) -> sparse.csr_matrix:
    return _to_sparse_matrix(sparse.csr_matrix, interactions, users_mapping, items_mapping, user_col=user_col, item_col=item_col, weight_col=weight_col)


def interactions_to_csc_matrix(interactions: pl.DataFrame,
                               users_mapping: LabelEncoder,
                               items_mapping: LabelEncoder,
                               user_col: str = 'user_id',
                               item_col: str = 'item_id',
                               weight_col=None) -> sparse.csc_matrix:
    return _to_sparse_matrix(sparse.csc_matrix, interactions, users_mapping, items_mapping, user_col=user_col, item_col=item_col, weight_col=weight_col)
