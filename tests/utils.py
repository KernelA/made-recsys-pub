import polars as pl


def check_sim_results(most_sim_items: pl.DataFrame, num_items: int, num_neighs: int, sim_col: str, sim_item_col: str):
    assert len(most_sim_items) == num_neighs * num_items
    assert most_sim_items.filter(pl.col("item_id") == pl.col(sim_item_col)).is_empty()
    assert most_sim_items.frame_equal(most_sim_items.sort(["item_id", sim_col]))


def check_recommendations(rec: pl.DataFrame, user_ids, num_rec: int):
    assert set(rec.get_column("user_id").unique()) == set(user_ids)
    assert rec.groupby("user_id").count().get_column("count").min() == num_rec
    assert "rank" in rec.columns
