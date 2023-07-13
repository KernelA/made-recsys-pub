import numpy as np
import polars as pl
import pytest
from scipy import sparse

from made_recsys.als import ALS
from made_recsys.matrix_ops import eig_values

from .utils import check_recommendations, check_sim_results


@pytest.mark.parametrize("k", [2, 3, 4])
def test_als(als_user_item_matrix, k, user_item_hist_dataframe, user_encoder, item_encoder):
    als = ALS(k, 10, 3 * k, 123, user_encoder=user_encoder,
              item_encoder=item_encoder)
    # target_loss = np.cumsum(eig_values(als_user_item_matrix.T @
    #                         als_user_item_matrix))[als_user_item_matrix.shape[1] - k - 1]

    als.fit(sparse.csc_matrix(als_user_item_matrix))
    # assert pytest.approx(target_loss, abs=1e-1) == als.final_loss

    item_ids = pl.Series(np.arange(als_user_item_matrix.shape[1], dtype=np.uint32))
    num_neighs = 2
    most_sim_items = als.most_similiar_items(item_ids, num_neighs)

    check_sim_results(
        most_sim_items, als_user_item_matrix.shape[1], num_neighs, als._sim_col, als._sim_item_id_col)

    num_rec = k
    rec = als.recommend(user_item_hist_dataframe, num_rec)
    check_recommendations(rec, user_item_hist_dataframe.get_column("user_id").unique(), num_rec)
