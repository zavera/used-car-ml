# Copyright (c) 2026 Callisto Tech — see LICENSE

import pandas as pd
import pytest

from feature_store.store import FeatureStore


@pytest.fixture
def store(tmp_path):
    return FeatureStore(
        store_path=str(tmp_path / "features.parquet"),
        reference_path=str(tmp_path / "reference.parquet"),
    )


def _df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"car_age": range(n), "price": [10000] * n})


def test_append_creates_store(store):
    store.append(_df(5))
    loaded = store.load_current()
    assert len(loaded) == 5


def test_append_accumulates(store):
    store.append(_df(5))
    store.append(_df(3))
    assert len(store.load_current()) == 8


def test_snapshot_as_reference(store):
    store.append(_df(10))
    store.snapshot_as_reference()
    ref = store.load_reference()
    assert len(ref) == 10


def test_new_row_count(store):
    store.append(_df(10))
    store.snapshot_as_reference()
    store.append(_df(4))
    assert store.new_row_count() == 4


def test_load_reference_raises_when_missing(store):
    with pytest.raises(FileNotFoundError):
        store.load_reference()


def test_replace_overwrites(store):
    store.append(_df(20))
    store.replace(_df(5))
    assert len(store.load_current()) == 5
