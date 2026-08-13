"""The Preprocessor is now the other half of every published model.

If it round-trips wrongly, a served model silently works in a different unit
system than the one it was trained in — the failure is invisible because both
paths "work". So these tests are deliberately paranoid.
"""

import json

import numpy as np
import pytest

from nuclear_surrogates.datamodule import data_scalers
from nuclear_surrogates.datamodule.preprocessor import (
    SCHEMA_VERSION,
    ColumnWiseScaler,
    Preprocessor,
)

KINDS = ["minmax", "standard", "robust", "maxabs", "quantile", "power", "none"]


def _make(kind, n_inputs=2, n_targets=3, seed=0, magnitudes=None):
    rng = np.random.default_rng(seed)
    # Modest magnitudes by default. PowerTransformer's yeo-johnson genuinely
    # overflows on multi-decade data, which is a property of the transform, not
    # of serialisation — the wide-dynamic-range case is exercised separately
    # for the linear scalers below.
    in_mag, tgt_mag = magnitudes or ([1.0, 10.0], [1.0, 0.5, 2.0])
    X = rng.random((256, n_inputs)) * in_mag[:n_inputs]
    Y = rng.random((256, n_targets)) * tgt_mag[:n_targets] + 1e-3

    inputs = {f"in{i}": data_scalers.get_scaler(kind) for i in range(n_inputs)}
    targets = {f"t{i}": data_scalers.get_scaler(kind) for i in range(n_targets)}
    col_map = {f"in{i}": i for i in range(n_inputs)}
    tgt_map = {f"t{i}": i for i in range(n_targets)}

    pre = Preprocessor.fit(
        inputs, targets, X, Y, col_map, tgt_map, t_days=990.0, t_days_data_span=990.0
    )
    return pre, X, Y


@pytest.mark.parametrize("kind", KINDS)
def test_json_round_trip_is_exact(tmp_path, kind):
    pre, X, Y = _make(kind)
    pre.save(tmp_path, write_joblib=False)
    reloaded = Preprocessor.load(tmp_path)

    np.testing.assert_allclose(
        reloaded.input_scaler.transform(X), pre.input_scaler.transform(X), atol=1e-12
    )
    np.testing.assert_allclose(
        reloaded.target_scaler.transform(Y), pre.target_scaler.transform(Y), atol=1e-12
    )
    np.testing.assert_allclose(
        reloaded.inverse_targets(pre.target_scaler.transform(Y)), Y, rtol=1e-6
    )


@pytest.mark.parametrize("kind", ["minmax", "standard", "robust", "maxabs"])
def test_round_trip_survives_realistic_dynamic_range(tmp_path, kind):
    """Concentrations here span from ~1e-2 (U238) down to ~1e-9 (Pu242 early in
    burnup). A float-formatting slip in the JSON would show up as a relative
    error on the small columns, which is exactly where it matters."""
    pre, X, Y = _make(kind, magnitudes=([1.0, 1e3], [1e-2, 1e-6, 1e-9]))
    pre.save(tmp_path, write_joblib=False)
    reloaded = Preprocessor.load(tmp_path)

    scaled = pre.transform_targets(Y)
    np.testing.assert_allclose(reloaded.transform_targets(Y), scaled, rtol=1e-12)
    np.testing.assert_allclose(reloaded.inverse_targets(scaled), Y, rtol=1e-9)


@pytest.mark.parametrize("kind", KINDS)
def test_json_and_joblib_agree(tmp_path, kind):
    """The joblib is a convenience copy; if it ever disagrees with the JSON,
    the JSON wins and this test is how we find out."""
    joblib = pytest.importorskip("joblib")
    pre, X, _ = _make(kind)
    pre.save(tmp_path, write_joblib=True)

    from_json = Preprocessor.load(tmp_path)
    from_pickle = joblib.load(tmp_path / "preprocessor.joblib")
    np.testing.assert_allclose(
        from_json.input_scaler.transform(X),
        from_pickle.input_scaler.transform(X),
        atol=1e-12,
    )


def test_load_never_fits(tmp_path):
    """A loaded preprocessor must be inert: transforming new data with a
    different distribution must not move its parameters."""
    pre, X, _ = _make("minmax")
    pre.save(tmp_path, write_joblib=False)
    loaded = Preprocessor.load(tmp_path)

    before = loaded.input_scaler.scalers[0].data_max_.copy()
    wildly_different = X * 1000 + 500
    scaled = loaded.input_scaler.transform(wildly_different)

    np.testing.assert_array_equal(loaded.input_scaler.scalers[0].data_max_, before)
    # And the out-of-range data should scale outside [0, 1], not be re-normalised.
    assert scaled.max() > 1.0


def test_t_days_survives_the_round_trip(tmp_path):
    pre, _, _ = _make("minmax")
    pre.save(tmp_path, write_joblib=False)
    loaded = Preprocessor.load(tmp_path)
    assert loaded.t_days == 990.0
    assert loaded.t_days_data_span == 990.0


def test_column_order_is_preserved(tmp_path):
    """Column order is the contract between the scalers and the network's
    input layer; a silent reorder would corrupt every prediction."""
    pre, _, _ = _make("minmax", n_inputs=2, n_targets=3)
    pre.save(tmp_path, write_joblib=False)
    loaded = Preprocessor.load(tmp_path)
    assert loaded.input_names == ["in0", "in1"]
    assert loaded.target_names == ["t0", "t1", "t2"]


def test_future_schema_version_is_refused(tmp_path):
    pre, _, _ = _make("minmax")
    pre.save(tmp_path, write_joblib=False)
    path = tmp_path / "preprocessor.json"
    payload = json.loads(path.read_text())
    payload["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="schema version"):
        Preprocessor.load(path)


def test_save_verification_catches_a_missing_attribute(tmp_path, monkeypatch):
    """The registry is hand-maintained, so `save` re-reads what it wrote.

    Simulate a forgotten attribute and assert the failure happens at write
    time — not silently, on load, months later.
    """
    from nuclear_surrogates.datamodule import preprocessor as mod

    pre, _, _ = _make("minmax")
    monkeypatch.setitem(mod._FITTED_ATTRS, "MinMaxScaler", ("data_min_",))

    with pytest.raises((AssertionError, AttributeError, ValueError)):
        pre.save(tmp_path, write_joblib=False)


def test_column_wise_scaler_rejects_wrong_width():
    scaler = ColumnWiseScaler(["a", "b"], [data_scalers.get_scaler("minmax")] * 2)
    scaler.fit(np.ones((10, 2)))
    with pytest.raises(ValueError, match="2 columns"):
        scaler.transform(np.ones((10, 3)))


def test_matches_the_column_transformer_it_replaces():
    """ColumnWiseScaler stands in for sklearn's ColumnTransformer. Equivalence
    is what kept the existing golden values byte-identical through the swap."""
    from sklearn.compose import ColumnTransformer

    rng = np.random.default_rng(0)
    data = rng.random((128, 3)) * [1.0, 1e-3, 1e6]

    reference = ColumnTransformer(
        [
            (name, data_scalers.get_scaler("minmax"), [i])
            for i, name in enumerate("abc")
        ],
        sparse_threshold=0,
    ).fit(data)
    ours = ColumnWiseScaler(
        ["a", "b", "c"], [data_scalers.get_scaler("minmax") for _ in range(3)]
    ).fit(data)

    np.testing.assert_array_equal(reference.transform(data), ours.transform(data))

    # ColumnTransformer has no inverse_transform, so invert column by column —
    # which is precisely the indirection ColumnWiseScaler removes.
    scaled = reference.transform(data)
    reference_inverse = np.hstack(
        [
            fitted.inverse_transform(scaled[:, [i]])
            for i, (_, fitted, _) in enumerate(reference.transformers_)
        ]
    )
    np.testing.assert_allclose(
        reference_inverse, ours.inverse_transform(ours.transform(data)), atol=1e-12
    )
