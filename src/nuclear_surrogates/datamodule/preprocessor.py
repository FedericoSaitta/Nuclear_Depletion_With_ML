"""Fitted preprocessing state, saved next to the weights.

A checkpoint on its own is half a model; the other half is the fitted scalers.
Persisting them is what lets a published checkpoint be served or evaluated
without the training dataset, and what stops inference from re-fitting on a
different set than training used.

`save` writes `preprocessor.json` and `preprocessor.joblib`. **The JSON is the
source of truth**: scikit-learn does not guarantee a pickled estimator unpickles
across minor versions, and unpickling is arbitrary code execution. The joblib is
a convenience copy that `tests/test_preprocessor.py` checks against it.

Both the ``t_days`` a run actually used and the data's true span are recorded,
because historical runs hardcoded 1000 days against 990 days of data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import sklearn
from loguru import logger

import nuclear_surrogates.datamodule.data_scalers as data_scalers

SCHEMA_VERSION = 1

# Fitted attributes that fully determine `transform` / `inverse_transform`, per
# estimator. Constructor defaults are reproduced by `data_scalers.get_scaler`,
# so only fitted state needs to travel. `n_features_in_` is always stored.
_FITTED_ATTRS = {
    "MinMaxScaler": ("min_", "scale_", "data_min_", "data_max_", "data_range_"),
    "StandardScaler": ("mean_", "var_", "scale_"),
    "RobustScaler": ("center_", "scale_"),
    "MaxAbsScaler": ("scale_", "max_abs_"),
    "QuantileTransformer": ("quantiles_", "references_"),
    "PowerTransformer": ("lambdas_",),
    "Normalizer": (),
    "NoOpScaler": (),
}

# Estimator class name -> the config keyword `data_scalers.get_scaler` accepts.
_KIND_TO_CONFIG_NAME = {
    "MinMaxScaler": "minmax",
    "StandardScaler": "standard",
    "RobustScaler": "robust",
    "MaxAbsScaler": "maxabs",
    "QuantileTransformer": "quantile",
    "PowerTransformer": "power",
    "Normalizer": "normalizer",
    "NoOpScaler": "none",
}


def _jsonify(value):
    """Convert numpy scalars/arrays to JSON-native types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _dump_scaler(scaler) -> dict:
    """Serialise one fitted single-column scaler to plain JSON types."""
    kind = type(scaler).__name__
    if kind not in _FITTED_ATTRS:
        raise NotImplementedError(
            f"{kind} has no entry in _FITTED_ATTRS, so it cannot be serialised. "
            f"Add the attributes its transform() depends on."
        )

    params = {
        attr: _jsonify(getattr(scaler, attr))
        for attr in _FITTED_ATTRS[kind]
        if hasattr(scaler, attr)
    }
    if hasattr(scaler, "n_features_in_"):
        params["n_features_in_"] = int(scaler.n_features_in_)

    # PowerTransformer(standardize=True) applies a nested StandardScaler inside
    # transform(), so its state has to travel too or the output is wrong.
    if kind == "PowerTransformer" and getattr(scaler, "_scaler", None) is not None:
        params["_scaler"] = _dump_scaler(scaler._scaler)

    return {"type": kind, "params": params}


def _load_scaler(spec: dict):
    """Rebuild a fitted scaler from `_dump_scaler` output."""
    kind = spec["type"]
    if kind not in _KIND_TO_CONFIG_NAME:
        raise NotImplementedError(f"Cannot reconstruct unknown scaler type {kind!r}")

    scaler = data_scalers.get_scaler(_KIND_TO_CONFIG_NAME[kind])
    for attr, value in spec["params"].items():
        if attr == "_scaler":
            scaler._scaler = _load_scaler(value)
        elif attr == "n_features_in_":
            scaler.n_features_in_ = int(value)
        else:
            setattr(scaler, attr, np.asarray(value, dtype=np.float64))
    return scaler


def _clamp_quantiles(scaler, n_samples: int) -> None:
    """Cap a QuantileTransformer's `n_quantiles` at the number of rows it sees.

    sklearn does exactly this internally and warns while doing it, so every fit
    on a small array — the test fixtures, or any `fraction_of_data` run — emits
    a UserWarning about a decision it already made for us. Setting the parameter
    up front is the same arithmetic silently: `quantiles_` and `references_`,
    the only fitted state that is serialised, come out identical.

    Non-quantile scalers have no such parameter and are left alone.
    """
    n_quantiles = getattr(scaler, "n_quantiles", None)
    if n_quantiles is not None and n_quantiles > n_samples:
        scaler.set_params(n_quantiles=max(1, n_samples))


class ColumnWiseScaler:
    """Per-column fitted scalers applied as a single array transform.

    Replaces sklearn's `ColumnTransformer`, which cannot invert itself and whose
    private fitted state (`transformers_`, `output_indices_`, ...) would have to
    be hand-restored from JSON. Every transformer here covers exactly one
    column, so a plain list is both simpler and exactly equivalent.
    """

    def __init__(self, names, scalers):
        if len(names) != len(scalers):
            raise ValueError(f"{len(names)} column names but {len(scalers)} scalers")
        self.names = list(names)
        self.scalers = list(scalers)

    def fit(self, X):
        X = np.asarray(X)
        self._check_width(X)
        for i, scaler in enumerate(self.scalers):
            _clamp_quantiles(scaler, X.shape[0])
            scaler.fit(X[:, i : i + 1])
        return self

    def transform(self, X):
        X = np.asarray(X)
        self._check_width(X)
        cols = [s.transform(X[:, i : i + 1]) for i, s in enumerate(self.scalers)]
        return np.hstack(cols)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X)
        self._check_width(X)
        cols = [
            s.inverse_transform(X[:, i : i + 1]) for i, s in enumerate(self.scalers)
        ]
        return np.hstack(cols)

    def _check_width(self, X):
        if X.ndim != 2 or X.shape[1] != len(self.scalers):
            raise ValueError(
                f"expected a 2-D array with {len(self.scalers)} columns "
                f"{self.names}, got shape {X.shape}"
            )

    def __repr__(self):
        kinds = [type(s).__name__ for s in self.scalers]
        return f"ColumnWiseScaler({dict(zip(self.names, kinds, strict=False))})"


@dataclass
class Preprocessor:
    """Everything needed to turn raw concentrations into model units and back."""

    input_scaler: ColumnWiseScaler
    target_scaler: ColumnWiseScaler
    t_days: float
    t_days_data_span: float | None = None

    @property
    def input_names(self):
        return list(self.input_scaler.names)

    @property
    def target_names(self):
        return list(self.target_scaler.names)

    # ── the transforms, named for what they mean ────────────────────────────

    def transform_inputs(self, X):
        return self.input_scaler.transform(X)

    def transform_targets(self, Y):
        return self.target_scaler.transform(Y)

    def inverse_inputs(self, X):
        return self.input_scaler.inverse_transform(X)

    def inverse_targets(self, Y):
        """Model units -> atom/b-cm. The one every evaluation path needs."""
        return self.target_scaler.inverse_transform(Y)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        input_scaler_dict,
        target_scaler_dict,
        input_data,
        target_data,
        col_index_map,
        target_index_map,
        t_days,
        t_days_data_span=None,
    ):
        """Fit both scalers on *already split* training data.

        Callers must pass the training split only — this class does not know
        about splits and will happily fit on whatever it is given.
        """
        pre = cls(
            input_scaler=_build(input_scaler_dict, col_index_map),
            target_scaler=_build(target_scaler_dict, target_index_map),
            t_days=float(t_days),
            t_days_data_span=(
                None if t_days_data_span is None else float(t_days_data_span)
            ),
        )
        pre.input_scaler.fit(input_data)
        pre.target_scaler.fit(target_data)
        return pre

    # ── persistence ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "sklearn_version": sklearn.__version__,
            "t_days": self.t_days,
            "t_days_data_span": self.t_days_data_span,
            "inputs": _dump_side(self.input_scaler),
            "targets": _dump_side(self.target_scaler),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Preprocessor:
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"preprocessor schema version {version!r} is not supported "
                f"(this build reads version {SCHEMA_VERSION})"
            )
        saved = payload.get("sklearn_version")
        if saved != sklearn.__version__:
            logger.warning(
                f"preprocessor was written by scikit-learn {saved}, loading under "
                f"{sklearn.__version__} — parameters are version-independent, but "
                f"verify if transform results look wrong"
            )
        return cls(
            input_scaler=_load_side(payload["inputs"]),
            target_scaler=_load_side(payload["targets"]),
            t_days=float(payload["t_days"]),
            t_days_data_span=payload.get("t_days_data_span"),
        )

    def save(self, directory, verify=True, write_joblib=True) -> str:
        """Write `preprocessor.json` (+ optionally `.joblib`) into *directory*.

        With *verify*, the JSON is immediately reloaded and checked to transform
        a probe identically — so a gap in `_FITTED_ATTRS` fails here, at write
        time, rather than silently producing wrong predictions on load.

        *write_joblib* is off for committed test fixtures: a pickle in a public
        repo that CI loads is a supply-chain surface, and the JSON is already
        the source of truth.
        """
        os.makedirs(directory, exist_ok=True)
        json_path = os.path.join(directory, "preprocessor.json")
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        if write_joblib:
            try:
                import joblib

                joblib.dump(self, os.path.join(directory, "preprocessor.joblib"))
            except ImportError:  # rides along with scikit-learn, but not required
                logger.warning("joblib unavailable — wrote preprocessor.json only")

        if verify:
            self._verify_roundtrip(json_path)

        logger.info(f"Wrote preprocessor to {json_path}")
        return json_path

    @classmethod
    def load(cls, path) -> Preprocessor:
        """Load from a `preprocessor.json`, or from a directory holding one.

        Never fits. This is the whole point: the scalers a model is served with
        are the ones it was trained with. See `require_fitted_scalers`, which
        is what stops there being any alternative.
        """
        if os.path.isdir(path):
            path = os.path.join(path, "preprocessor.json")
        with open(path) as f:
            payload = json.load(f)
        logger.info(f"Loaded preprocessor from {path}")
        return cls.from_dict(payload)

    def _verify_roundtrip(self, json_path, atol=1e-12):
        reloaded = Preprocessor.load(json_path)
        rng = np.random.default_rng(0)
        for original, copy_, label in (
            (self.input_scaler, reloaded.input_scaler, "inputs"),
            (self.target_scaler, reloaded.target_scaler, "targets"),
        ):
            probe = rng.random((8, len(original.scalers)))
            np.testing.assert_allclose(
                original.transform(probe),
                copy_.transform(probe),
                atol=atol,
                err_msg=(
                    f"{label}: preprocessor.json does not round-trip. A scaler's "
                    f"fitted state is missing from _FITTED_ATTRS."
                ),
            )


def require_fitted_scalers(preprocessor_path) -> None:
    """Refuse to serve a model without the scalers it was trained with.

    There is deliberately no fallback. Re-fitting on whatever data happens to
    be to hand produces scalers the checkpoint never saw, and every number
    computed through them is then wrong by however much the two fits disagree —
    silently, and by an amount nothing downstream can detect. Inference used to
    do exactly that whenever this path was unset, behind a `logger.warning`
    nobody reads in a 500-line log.

    A checkpoint without its scalers is half a model. Half a model does not
    run.
    """
    if not preprocessor_path:
        raise SystemExit(
            "Inference needs the fitted scalers, and dataset.preprocessor_path "
            "is not set.\n\n"
            "A checkpoint is half a model; the other half is the scalers it "
            "was trained with. There is no fallback — re-fitting them on other "
            "data would silently change every number this run produces.\n\n"
            "Run from a bundle:\n"
            "    nucml --bundle <results/.../model-bundle> --data <file>\n\n"
            "For a checkpoint that predates bundles, build one once with "
            "nucml-package, on a machine that still has the training dataset."
        )


def _build(scaler_dict, index_map) -> ColumnWiseScaler:
    """Order the configured scalers by their column index."""
    ordered = sorted(index_map.items(), key=lambda kv: kv[1])
    names, scalers = [], []
    for col_name, _ in ordered:
        if col_name not in scaler_dict:
            raise ValueError(
                f"Column '{col_name}' has no scaler. Available: {list(scaler_dict)}"
            )
        names.append(col_name)
        scalers.append(scaler_dict[col_name])
    return ColumnWiseScaler(names, scalers)


def _dump_side(scaler: ColumnWiseScaler) -> dict:
    return {
        "order": list(scaler.names),
        "scalers": {
            name: _dump_scaler(s)
            for name, s in zip(scaler.names, scaler.scalers, strict=False)
        },
    }


def _load_side(payload: dict) -> ColumnWiseScaler:
    order = list(payload["order"])
    return ColumnWiseScaler(order, [_load_scaler(payload["scalers"][n]) for n in order])
