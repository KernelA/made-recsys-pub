import pathlib

import hydra

from made_recsys.load_data import MovieLens1M


@hydra.main(config_path="configs", config_name="convert", version_base="1.3")
def main(config):
    data_dir = pathlib.Path(config.data_dir)
    dump_dir = pathlib.Path(config.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    all_names = MovieLens1M.load_items(str(data_dir))
    all_names.write_parquet(dump_dir / "items.parquet")
    del all_names

    users = MovieLens1M.load_users(str(data_dir))
    users.write_parquet(dump_dir / "users.parquet")
    del users

    interactions = MovieLens1M.load_interactions(str(data_dir))
    interactions.write_parquet(dump_dir / "interactions.parquet")


if __name__ == "__main__":
    main()
