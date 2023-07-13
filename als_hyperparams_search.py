import logging
import pathlib

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig

from made_recsys.als import ALS
from made_recsys.log_set import init_logging
from made_recsys.matrix_ops import interactions_to_csc_matrix
from made_recsys.metrics import compute_metrics
from made_recsys.utils import load_pickle, save_json


@hydra.main(config_path="configs", config_name="als", version_base="1.3")
def main(config):
    current_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    train_dir = pathlib.Path(config.train_dir)
    train_interactions = pl.read_parquet(train_dir / "train.parquet")
    user_encoder = load_pickle(str(train_dir / "user_encoder.pickle"))
    item_encoder = load_pickle(str(train_dir / "item_encoder.pickle"))

    algo: ALS = hydra.utils.instantiate(
        config.algo, user_encoder=user_encoder, item_encoder=item_encoder)

    user_item_train_inter = interactions_to_csc_matrix(
        train_interactions, user_encoder, item_encoder, weight_col="rating")

    del train_interactions

    algo.fit(user_item_train_inter)

    del user_item_train_inter

    # dump_dir = current_dir / config.dump_dir
    # dump_dir.mkdir(exist_ok=True, parents=True)
    # algo.save_index(str(dump_dir))

    test_dir = pathlib.Path(config.test_dir)
    test_histories = pl.read_parquet(test_dir / "test_hist.parquet")
    true_recs = pl.read_parquet(test_dir / "test_valid.parquet")

    num_recs_per_user = true_recs.groupby("user_id").count().get_column("count").max()

    logging.info("Num recs per user: %d", num_recs_per_user)
    recommendations = algo.recommend(test_histories, num_rec_per_user=num_recs_per_user)

    metric_dir = current_dir / "metrics"
    metric_dir.mkdir(exist_ok=True, parents=True)

    metrics = compute_metrics(true_recs, recommendations, max_k=num_recs_per_user)
    metrics.write_parquet(metric_dir / "metrics.parquet")

    map_value = metrics.filter(pl.col("name") == "MAP").get_column("value").max()
    save_json(metric_dir / "map.json",
              {"map": map_value})

    return map_value


if __name__ == "__main__":
    init_logging("./log_settings.yaml")
    main()
