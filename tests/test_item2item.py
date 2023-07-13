import numpy as np
import polars as pl
import pytest
from scipy import sparse

from made_recsys.item2item import Item2ItemRecommender

from .utils import check_recommendations, check_sim_results


@pytest.mark.parametrize("num_neighs", [2, 3])
def test_fit(num_neighs, als_user_item_matrix, user_item_hist_dataframe, user_encoder, item_encoder):
    user_item = sparse.csc_matrix(als_user_item_matrix)
    item2item_rec = Item2ItemRecommender(
        metric="cosine", seed=12, user_encoder=user_encoder, item_encoder=item_encoder, low_memory=False)
    item2item_rec.fit(user_item)

    item_ids = pl.Series(np.arange(als_user_item_matrix.shape[1], dtype=np.uint32))
    most_sim_items = item2item_rec.most_similiar_items(item_ids, num_neighs)

    check_sim_results(
        most_sim_items, als_user_item_matrix.shape[1], num_neighs, item2item_rec._sim_col, item2item_rec._sim_item_id_col)

    item2item_rec = item2item_rec.recommend(user_item_hist_dataframe, num_neighs, num_neighs)
    check_recommendations(item2item_rec, user_item_hist_dataframe.get_column(
        "user_id").unique(), num_neighs)
