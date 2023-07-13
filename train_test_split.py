import json
import logging
import pathlib

import hydra
import polars as pl
from sklearn.preprocessing import LabelEncoder

from made_recsys.load_data import MovieLens1M
from made_recsys.log_set import init_logging
from made_recsys.utils import save_json, save_pickle


def save_data(data: pl.DataFrame, base_dir: pathlib.Path, subdir: str, name: str):
    data_dir = base_dir / subdir
    data_dir.mkdir(exist_ok=True, parents=True)
    data.write_parquet(data_dir / f"{name}.parquet")


@hydra.main(config_path="configs", config_name="train_test_split", version_base="1.3")
def main(config):
    data_dir = pathlib.Path(config.dump_dir)
    data = pl.read_parquet(data_dir / "interactions.parquet")

    train_ratio = config.train_ratio

    train_user_ids = data.get_column("user_id").unique().sample(
        fraction=train_ratio,
        seed=config.seed,
        with_replacement=False)

    test_user_ids = data.filter(
        ~pl.col("user_id").is_in(train_user_ids)).get_column("user_id").unique()

    num_cold_users = test_user_ids.filter(~test_user_ids.is_in(train_user_ids)).len()

    split_info = {"num_cold_users": num_cold_users}

    test_item_ids = data.filter(pl.col("user_id").is_in(
        test_user_ids)).get_column("item_id").unique()

    train_item_ids = data.filter(pl.col("user_id").is_in(
        train_user_ids)).get_column("item_id").unique()

    cold_item_ids = test_item_ids.filter(~test_item_ids.is_in(train_item_ids))
    num_col_items = cold_item_ids.len()

    split_info["num_cold_items"] = num_col_items

    logging.info("Split info: %s\nTotal users: %d\nTotal items: %d", split_info, train_user_ids.len() + test_user_ids.len(),
                 len(test_item_ids) + len(train_item_ids))

    del test_item_ids
    del train_item_ids

    train_data = data.filter(pl.col("user_id").is_in(train_user_ids))

    data_dir = data_dir.parent

    mapping_dir = data_dir / "train"

    user_encoder = LabelEncoder()
    user_encoder.fit(train_data.get_column("user_id").unique().to_numpy())
    item_encoder = LabelEncoder()
    item_encoder.fit(train_data.get_column("item_id").unique().to_numpy())

    save_data(train_data, data_dir, "train", "train")

    save_pickle(str(mapping_dir / "user_encoder.pickle"), user_encoder)
    save_pickle(str(mapping_dir / "item_encoder.pickle"), item_encoder)

    del train_data

    test_data = data.filter(pl.col("user_id").is_in(test_user_ids) &
                            ~pl.col("item_id").is_in(cold_item_ids))

    test_valid = test_data.groupby("user_id").apply(
        lambda group: group.top_k(k=config.predict_top_n, by="timestamp"))
    test_hist = test_data.join(test_valid, on=["user_id", "item_id"], how="anti")

    save_data(test_hist, data_dir, "test", "test_hist")
    save_data(test_valid, data_dir, "test", "test_valid")
    save_json(data_dir / "train" / "split_info.json", split_info)


if __name__ == "__main__":
    init_logging(log_config="./log_settings.yaml")
    main()
