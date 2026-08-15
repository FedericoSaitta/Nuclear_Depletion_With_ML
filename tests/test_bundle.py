"""The bundle round trip: what `write_bundle` writes, `read_bundle` must load.

`test_training.py` covers the end-to-end serving path. These are the reader's
own contract — what it accepts, what it refuses, and what it rewrites — plus the
CLI wiring that reaches it.
"""

import json
import os

import pytest
from omegaconf import OmegaConf

from nuclear_surrogates.bundle import (
    BUNDLE_VERSION,
    CONFIG_NAME,
    METADATA_NAME,
    PREPROCESSOR_NAME,
    REQUIRED_NAMES,
    WEIGHTS_NAME,
    read_bundle,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MINI_H5 = os.path.join(REPO, "tests", "fixtures", "mini_casl_10runs.h5")


@pytest.fixture
def bundle(tmp_path):
    """A minimal bundle written by hand.

    Deliberately not produced by a training run: this file is testing the
    reader against the documented layout, so the layout is spelled out here
    rather than inherited from whatever the writer currently does.
    """
    path = tmp_path / "model-bundle"
    path.mkdir()
    (path / WEIGHTS_NAME).write_bytes(b"not a real checkpoint")
    (path / PREPROCESSOR_NAME).write_text("{}")
    OmegaConf.save(
        OmegaConf.create(
            {
                "dataset": {
                    "path_to_data": "/cluster/home/someone/datasets/train.h5",
                    "fraction_of_data": 0.1,
                },
                "model": {"name": "demo"},
                "runtime": {
                    "mode": "train",
                    "model": "NODE",
                    "ckp_path": "/cluster/home/someone/results/best.ckpt",
                    "output_dir": "/cluster/home/someone/results",
                    "seed": 42,
                },
            }
        ),
        path / CONFIG_NAME,
    )
    (path / METADATA_NAME).write_text(
        json.dumps(
            {
                "bundle_version": BUNDLE_VERSION,
                "model_name": "demo",
                "model_kind": "NODE",
                "git_sha": "0123456789abcdef",
            }
        )
    )
    return path


def test_read_bundle_points_the_config_at_the_bundles_own_files(bundle):
    """A bundle copied off the cluster must stop referring to the cluster.

    The config it carries was written on the machine that trained the model, so
    every absolute path in it is wrong somewhere else. The three paths that name
    bundle contents are rewritten to the bundle; the training dataset, which is
    not in the bundle, is cleared rather than left dangling.
    """
    cfg, metadata = read_bundle(bundle)

    assert cfg.runtime.mode == "inference"
    assert cfg.runtime.ckp_path == str(bundle / WEIGHTS_NAME)
    assert cfg.dataset.preprocessor_path == str(bundle / PREPROCESSOR_NAME)
    assert cfg.dataset.path_to_data == ""
    assert metadata["model_kind"] == "NODE"


def test_read_bundle_applies_data_out_and_overrides(bundle, tmp_path):
    cfg, _ = read_bundle(
        bundle,
        inference_data=MINI_H5,
        output_dir=tmp_path / "served",
        overrides=["runtime.device=cuda", "dataset.fraction_of_data=1.0"],
    )
    assert cfg.dataset.path_to_inference_data == os.path.abspath(MINI_H5)
    assert cfg.runtime.output_dir == str((tmp_path / "served").resolve())
    assert cfg.runtime.device == "cuda"
    assert cfg.dataset.fraction_of_data == 1.0


def test_overrides_win_over_the_bundles_own_paths(bundle, tmp_path):
    """Last word to the user — including over the paths read_bundle just set.

    Swapping in a different checkpoint against a bundle's scalers is a
    legitimate thing to want; it should not require editing the bundle.
    """
    other = tmp_path / "other.ckpt"
    other.write_bytes(b"")
    cfg, _ = read_bundle(bundle, overrides=[f"runtime.ckp_path={other}"])
    assert cfg.runtime.ckp_path == str(other)


@pytest.mark.parametrize("missing", REQUIRED_NAMES)
def test_an_incomplete_directory_is_not_a_bundle(bundle, missing):
    """Name what is absent. A bundle missing its scalers is the dangerous case:
    it would otherwise load and serve against the wrong ones."""
    os.remove(bundle / missing)
    with pytest.raises(SystemExit, match=missing):
        read_bundle(bundle)


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="No bundle directory"):
        read_bundle(tmp_path / "nope")


def test_a_future_bundle_version_is_refused(bundle):
    """Mirrors Preprocessor.from_dict's schema check: refuse to guess at a
    layout this build does not know."""
    meta = json.loads((bundle / METADATA_NAME).read_text())
    meta["bundle_version"] = BUNDLE_VERSION + 1
    (bundle / METADATA_NAME).write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="Upgrade nuclear_surrogates"):
        read_bundle(bundle)


def test_metadata_is_strict_json(tmp_path):
    """`json.dump` writes `Infinity` for a metric a run never logged, which
    Python reads back and nothing else does. A never-recorded metric is null.
    """
    from nuclear_surrogates.bundle import write_bundle
    from nuclear_surrogates.datamodule.preprocessor import Preprocessor

    class _StubPreprocessor(Preprocessor):
        def __init__(self):
            pass

        def save(self, directory, **kwargs):
            with open(os.path.join(directory, PREPROCESSOR_NAME), "w") as f:
                f.write("{}")

    out = write_bundle(
        out_dir=str(tmp_path / "b"),
        cfg=OmegaConf.create(
            {"dataset": {}, "model": {"name": "x"}, "runtime": {"seed": 1}}
        ),
        preprocessor=_StubPreprocessor(),
        metrics={"val_r2": -float("inf"), "val_loss": 0.5, "val_mae": float("inf")},
        hash_dataset=False,
    )

    raw = (tmp_path / "b" / METADATA_NAME).read_text()
    assert "Infinity" not in raw, "metadata.json is not parseable outside Python"
    metrics = json.loads(raw)["metrics"]
    assert metrics == {"val_r2": None, "val_loss": 0.5, "val_mae": None}
    assert out == str(tmp_path / "b")


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_bundle_needs_something_to_evaluate(bundle):
    from nuclear_surrogates.main import _build_parser, build_config

    args = _build_parser().parse_args(["--bundle", str(bundle)])
    with pytest.raises(SystemExit, match="--data"):
        build_config(args)


def test_cli_rejects_config_with_data_or_out(tmp_path):
    """--data/--out are bundle flags. Silently ignoring them with --config
    would leave a user believing they had redirected a run."""
    from nuclear_surrogates.main import _build_parser, build_config

    args = _build_parser().parse_args(
        ["--config", os.path.join(REPO, "configs", "smoke_dnn.yaml"), "--data", MINI_H5]
    )
    with pytest.raises(SystemExit, match="apply to --bundle"):
        build_config(args)


def test_cli_bundle_and_config_are_mutually_exclusive(bundle):
    from nuclear_surrogates.main import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--config", "x.yaml", "--bundle", str(bundle)])
