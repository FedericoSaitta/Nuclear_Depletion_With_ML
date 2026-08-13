"""Unit tests for the data pipeline: pure numpy, no torch, milliseconds.

These cover the transforms that decide what a "sample" is. They are the cheapest
tests in the suite and they guard the assumptions everything downstream inherits
— run boundaries, split fractions, and the invertibility of scaling.
"""

import numpy as np
import pytest

from nuclear_surrogates.datamodule import data_scalers, dataset_helper

# ── run structure ────────────────────────────────────────────────────────────


def test_detect_run_length_finds_the_first_time_reset():
    import polars as pl

    df = pl.DataFrame({"time_days": [0.0, 10.0, 20.0, 0.0, 10.0, 20.0]})
    assert dataset_helper.detect_run_length(df) == 3


def test_detect_run_length_of_a_single_run_is_its_length():
    import polars as pl

    df = pl.DataFrame({"time_days": [0.0, 10.0, 20.0, 30.0]})
    assert dataset_helper.detect_run_length(df) == 4


def test_pairs_never_cross_run_boundaries():
    """(t -> t+1) samples must stop at each run's end.

    If they did not, the last step of one run would be paired with the first
    step of the next — a physically meaningless transition that the model would
    happily learn.
    """
    time = np.array([0.0, 10.0, 20.0, 0.0, 10.0, 20.0])  # two runs of 3 steps
    inputs = np.arange(12, dtype=float).reshape(6, 2)
    targets = np.arange(6, dtype=float).reshape(6, 1)

    X, y = dataset_helper.create_timeseries_targets(
        inputs, targets, time, {"a": 0, "b": 1}, {"c": 0}, delta_conc=False
    )

    assert len(X) == 4  # 2 valid pairs per run, not 5
    np.testing.assert_array_equal(X[0], inputs[0])
    np.testing.assert_array_equal(y[0], targets[1])
    np.testing.assert_array_equal(X[1], inputs[1])
    np.testing.assert_array_equal(y[1], targets[2])
    # Row 2 ends run one and must never appear as an input.
    np.testing.assert_array_equal(X[2], inputs[3])
    np.testing.assert_array_equal(y[2], targets[4])


def test_delta_mode_returns_differences():
    time = np.array([0.0, 10.0, 20.0])
    inputs = np.arange(6, dtype=float).reshape(3, 2)
    targets = np.array([[1.0], [4.0], [9.0]])

    _, y = dataset_helper.create_timeseries_targets(
        inputs, targets, time, {"a": 0, "b": 1}, {"c": 0}, delta_conc=True
    )
    np.testing.assert_allclose(y.ravel(), [3.0, 5.0])


# ── splitting ────────────────────────────────────────────────────────────────


def _split_fixture(n_runs=10, steps=100, n_features=3):
    X = np.arange(n_runs * steps * n_features, dtype=float).reshape(-1, n_features)
    Y = np.arange(n_runs * steps, dtype=float).reshape(-1, 1)
    return X, Y


def test_split_fractions_are_by_whole_runs():
    X, Y = _split_fixture()
    X_tr, X_va, X_te, *_ = dataset_helper.timeseries_train_val_test_split(
        X, Y, 0.8, 0.1, 0.1, steps_per_run=100, rng=np.random.default_rng(0)
    )
    assert len(X_tr) == 800 and len(X_va) == 100 and len(X_te) == 100


def test_split_rejects_fractions_that_do_not_sum_to_one():
    X, Y = _split_fixture()
    with pytest.raises(ValueError, match="sum to 1.0"):
        dataset_helper.timeseries_train_val_test_split(X, Y, 0.8, 0.8, 0.8)


def test_same_seed_gives_the_same_training_order():
    X, Y = _split_fixture()
    kwargs = {
        "train_frac": 0.8,
        "val_frac": 0.1,
        "test_frac": 0.1,
        "steps_per_run": 100,
    }
    a = dataset_helper.timeseries_train_val_test_split(
        X, Y, rng=np.random.default_rng(7), **kwargs
    )
    b = dataset_helper.timeseries_train_val_test_split(
        X, Y, rng=np.random.default_rng(7), **kwargs
    )
    np.testing.assert_array_equal(a[0], b[0])
    assert a[6]["train_shuffle_order"] == b[6]["train_shuffle_order"]


def test_different_seeds_give_different_training_orders():
    """Guards against a seed that is accepted and then ignored."""
    X, Y = _split_fixture()
    kwargs = {
        "train_frac": 0.8,
        "val_frac": 0.1,
        "test_frac": 0.1,
        "steps_per_run": 100,
    }
    a = dataset_helper.timeseries_train_val_test_split(
        X, Y, rng=np.random.default_rng(1), **kwargs
    )
    b = dataset_helper.timeseries_train_val_test_split(
        X, Y, rng=np.random.default_rng(2), **kwargs
    )
    assert a[6]["train_shuffle_order"] != b[6]["train_shuffle_order"]


def test_shuffle_keeps_timesteps_ordered_within_a_run():
    """Runs may be reordered; the steps inside one may not."""
    X, Y = _split_fixture(n_runs=10, steps=100, n_features=1)
    X_tr, *_ = dataset_helper.timeseries_train_val_test_split(
        X, Y, 0.8, 0.1, 0.1, steps_per_run=100, rng=np.random.default_rng(3)
    )
    for run in range(8):
        block = X_tr[run * 100 : (run + 1) * 100, 0]
        assert np.all(np.diff(block) > 0), "timestep order was scrambled"


def test_split_fractions_default_when_config_is_silent():
    """A config written before `dataset.split` existed must behave as it did."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"dataset": {"fraction_of_data": 1.0}})
    assert dataset_helper.split_fractions(cfg, default=(0.8, 0.1, 0.1)) == (
        0.8,
        0.1,
        0.1,
    )
    assert dataset_helper.split_fractions(cfg, default=(0.6, 0.2, 0.2)) == (
        0.6,
        0.2,
        0.2,
    )


def test_split_fractions_read_from_config():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {"dataset": {"split": {"train": 0.7, "val": 0.2, "test": 0.1}}}
    )
    assert dataset_helper.split_fractions(cfg, default=(0.8, 0.1, 0.1)) == (
        0.7,
        0.2,
        0.1,
    )


def test_split_fractions_reject_a_config_that_does_not_sum_to_one():
    """Silently renormalising would quietly change the size of the test set."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {"dataset": {"split": {"train": 0.8, "val": 0.2, "test": 0.2}}}
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        dataset_helper.split_fractions(cfg, default=(0.8, 0.1, 0.1))


def test_split_info_partitions_every_run_exactly_once():
    X, Y = _split_fixture()
    *_, info = dataset_helper.timeseries_train_val_test_split(
        X, Y, 0.8, 0.1, 0.1, steps_per_run=100, rng=np.random.default_rng(0)
    )
    allocated = info["train"] + info["val"] + info["test"]
    assert sorted(allocated) == list(range(info["n_runs"]))


# ── scaling ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kind", ["minmax", "standard", "robust", "maxabs", "quantile", "none"]
)
def test_inverse_transform_round_trips(kind):
    from nuclear_surrogates.datamodule.preprocessor import ColumnWiseScaler

    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 3)) * [1.0, 1e-6, 1e3]

    scaler = ColumnWiseScaler(
        ["a", "b", "c"], [data_scalers.get_scaler(kind) for _ in range(3)]
    ).fit(data)

    restored = scaler.inverse_transform(scaler.transform(data))
    np.testing.assert_allclose(restored, data, rtol=1e-6, atol=1e-9)


def test_unknown_scaler_raises():
    """A typo'd scaler name must fail, not silently train on unscaled data.

    `get_scaler` used to log an error and hand back a NoOpScaler, so
    `minmaxx` produced a model whose reported metrics looked ordinary.
    """
    with pytest.raises(ValueError, match="Unknown scaler"):
        data_scalers.get_scaler("minmaxx")


def test_remove_empty_columns_drops_all_zero_columns():
    """A hazard for small fixtures: an isotope that is zero for every row of a
    10-run slice is dropped, and the later column selection then raises."""
    import polars as pl

    df = pl.DataFrame({"a": [1.0, 2.0], "zeros": [0.0, 0.0], "b": [3.0, 4.0]})
    assert dataset_helper.remove_empty_columns(df).columns == ["a", "b"]


def test_mini_fixture_keeps_every_configured_column():
    """The committed fixture must survive the empty-column filter, or every
    test that uses it fails in a confusing place."""
    import os

    import polars as pl
    from golden_setup import MINI_H5

    if not os.path.exists(MINI_H5):
        pytest.skip("mini fixture absent")

    df = dataset_helper.read_h5_file(MINI_H5)
    kept = dataset_helper.remove_empty_columns(df)
    for column in (
        "power_W_g",
        "U238",
        "U239",
        "Np239",
        "Pu239",
        "Pu240",
        "Pu241",
        "Pu242",
    ):
        assert column in kept.columns, f"{column} was dropped from the fixture"
    assert isinstance(kept, pl.DataFrame)
