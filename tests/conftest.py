import numpy as np
import polars as pl
import pytest

from .identity_encoder import IdentityEncoder

MAX_USERS = 10
MAX_ITEMS = 11


@pytest.fixture(scope="session")
def numpy_gen():
    return np.random.default_rng(2234)


@pytest.fixture(scope="session")
def user_item_hist_dataframe():
    return pl.DataFrame(
        {"user_id": [1, 1, 0, MAX_USERS + 1],
         "item_id": [0, 1, 1, 0]
         }, schema={"user_id": pl.UInt16, "item_id": pl.UInt16}
    )


@ pytest.fixture(scope="session")
def als_user_item_matrix(numpy_gen: np.random.Generator):
    return numpy_gen.integers(0, 2, size=(MAX_USERS, MAX_ITEMS)).astype(np.float32)


@pytest.fixture(scope="function")
def user_encoder(als_user_item_matrix):
    encoder = IdentityEncoder()
    encoder.fit(np.arange(als_user_item_matrix.shape[0]))
    return encoder


@pytest.fixture(scope="function")
def item_encoder(als_user_item_matrix):
    encoder = IdentityEncoder()
    encoder.fit(np.arange(als_user_item_matrix.shape[1]))
    return encoder
