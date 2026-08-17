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
                "model": {"name": "demo", "kind": "NODE"},
                # A real config.resolved.yaml records the training machine's
                # runtime block. read_bundle must drop it: the cluster's device
                # and output directory are not this machine's.
                "runtime": {
                    "device": "cuda",
                    "num_workers": 12,
                    "output_dir": "/cluster/home/someone/results",
                    "seed": 7,
                },
            }
        ),
        path / CONFIG_NAME,
    )
    (path / METADATA_NAME).write_text(
        json.dumps(
            {
                "model_name": "demo",
                "model_kind": "NODE",
                "git_sha": "0123456789abcdef",
            }
        )
    )
    return path


def test_read_bundle_addresses_the_bundles_own_files(bundle):
    """A bundle copied off the cluster must stop referring to the cluster.

    The config it carries was written on the machine that trained the model, so
    every absolute path in it is wrong somewhere else. The weights and scalers
    come back as bundle-relative paths; the training dataset, which is not in
    the bundle, is cleared rather than left dangling.
    """
    cfg, loaded = read_bundle(bundle)

    assert loaded.weights == bundle / WEIGHTS_NAME
    assert loaded.preprocessor == bundle / PREPROCESSOR_NAME
    assert cfg.dataset.preprocessor_path == str(bundle / PREPROCESSOR_NAME)
    assert cfg.dataset.path_to_data == ""
    assert loaded.metadata["model_kind"] == "NODE"


def test_read_bundle_drops_the_training_machines_runtime(bundle):
    """Everything about the machine is the new command line's to decide.

    Inheriting `device: cuda` from a cluster run would make a laptop run fail
    for a reason that has nothing to do with what the user asked for.
    """
    cfg, loaded = read_bundle(bundle)

    assert "runtime" not in cfg, "the recorded runtime block must not survive"
    # The seed is the exception, and only because `plots` rebuilds the recorded
    # run's split from it.
    assert loaded.seed == 7


def test_read_bundle_keeps_the_model_description(bundle):
    """What a bundle is *for*: the architecture the weights load into."""
    cfg, _ = read_bundle(bundle)

    assert cfg.model.kind == "NODE"
    assert cfg.model.name == "demo"
    assert cfg.dataset.fraction_of_data == 0.1


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


def test_a_bundle_carries_no_absolute_paths(tmp_path):
    """A bundle is meant to be published, so it must not ship the author's
    filesystem.

    An absolute path says nothing useful about the file — the dataset is
    identified by its SHA-256 — while saying quite a lot about the machine that
    wrote it. Both the metadata and the resolved config are checked, because the
    config is the one a reader actually opens.
    """
    from nuclear_surrogates.bundle import write_bundle
    from nuclear_surrogates.datamodule.preprocessor import Preprocessor

    class _StubPreprocessor(Preprocessor):
        def __init__(self):
            pass

        def save(self, directory, **kwargs):
            with open(os.path.join(directory, PREPROCESSOR_NAME), "w") as f:
                f.write("{}")

    cfg = OmegaConf.create(
        {
            "dataset": {
                "path_to_data": "/cluster/scratch/someone/datasets/train.h5",
                "path_to_inference_data": r"C:\Users\someone\datasets\new.h5",
                "preprocessor_path": "/cluster/scratch/someone/preprocessor.json",
                "fraction_of_data": 0.1,
            },
            "model": {"name": "demo", "kind": "NODE"},
            "runtime": {"seed": 3, "output_dir": r"C:\Users\someone\results"},
        }
    )
    out = tmp_path / "b"
    write_bundle(
        out_dir=str(out),
        cfg=cfg,
        preprocessor=_StubPreprocessor(),
        ckpt_path=r"C:\Users\someone\results\demo\best-demo-epoch=07.ckpt",
        hash_dataset=False,
    )

    for name in (METADATA_NAME, CONFIG_NAME):
        text = (out / name).read_text()
        assert "someone" not in text, f"{name} leaks the author's home directory"
        assert "/cluster/" not in text and "C:\\" not in text, f"{name} leaks a path"

    metadata = json.loads((out / METADATA_NAME).read_text())
    # The filename survives — it is the part that identifies the artefact.
    assert metadata["dataset"]["name"] == "train.h5"
    assert metadata["source_checkpoint"] == "best-demo-epoch=07.ckpt"

    saved = OmegaConf.load(out / CONFIG_NAME)
    assert saved.dataset.path_to_data == "train.h5"
    assert saved.dataset.path_to_inference_data == "new.h5"
    assert saved.dataset.preprocessor_path == "preprocessor.json"

    # The live config the run is still using must not have been touched.
    assert cfg.dataset.path_to_data == "/cluster/scratch/someone/datasets/train.h5"
    assert cfg.runtime.output_dir == r"C:\Users\someone\results"


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
            {
                "dataset": {},
                "model": {"name": "x", "kind": "DNN"},
                "runtime": {"seed": 1},
            }
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

SMOKE_DNN = os.path.join(REPO, "configs", "smoke_dnn.yaml")


@pytest.mark.parametrize("verb", ["train", "finetune", "infer", "plots"])
def test_every_verb_requires_a_source_and_data(verb, tmp_path):
    """Argparse enforces the legal combinations, so no mode can be reached
    half-configured. This used to be three hand-written SystemExit guards."""
    from nuclear_surrogates.main import _build_parser

    source = "--config" if verb == "train" else "--bundle"
    value = SMOKE_DNN if verb == "train" else str(tmp_path)

    for argv in ([verb], [verb, source, value], [verb, "--data", MINI_H5]):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(argv)

    # Both together parse.
    args = _build_parser().parse_args([verb, source, value, "--data", MINI_H5])
    assert args.command == verb


def test_train_builds_a_runtime_block_from_the_flags(tmp_path):
    """The `runtime` section exists at run time but lives in no YAML."""
    from nuclear_surrogates.main import _build_parser, build_config

    args = _build_parser().parse_args(
        [
            "train",
            "--config",
            SMOKE_DNN,
            "--data",
            MINI_H5,
            "--out",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--workers",
            "3",
            "--seed",
            "11",
            "--no-plots",
        ]
    )
    cfg, loaded = build_config(args)

    assert loaded is None
    assert cfg.dataset.path_to_data == str(os.path.abspath(MINI_H5))
    assert cfg.runtime.output_dir == str((tmp_path / "out").resolve())
    assert cfg.runtime.device == "cpu"
    assert cfg.runtime.num_workers == 3
    assert cfg.runtime.seed == 11
    assert cfg.runtime.plots is False


def test_infer_takes_the_bundles_seed_and_never_the_training_path(bundle, tmp_path):
    """`--data` on `infer` is the file to evaluate, never the training file."""
    from nuclear_surrogates.main import _build_parser, build_config

    args = _build_parser().parse_args(
        ["infer", "--bundle", str(bundle), "--data", MINI_H5, "--out", str(tmp_path)]
    )
    cfg, loaded = build_config(args)

    assert cfg.dataset.path_to_inference_data == str(os.path.abspath(MINI_H5))
    assert cfg.dataset.path_to_data == "", "infer must not be given the training file"
    assert (
        cfg.runtime.seed == loaded.seed == 7
    ), "an unflagged seed comes from the bundle"


def test_overrides_win_over_everything_the_cli_set(bundle, tmp_path):
    """Last word to the user — including over the runtime block just built."""
    from nuclear_surrogates.main import _build_parser, build_config

    args = _build_parser().parse_args(
        [
            "infer",
            "--bundle",
            str(bundle),
            "--data",
            MINI_H5,
            "--out",
            str(tmp_path),
            "--device",
            "cpu",
            "runtime.device=cuda",
            "dataset.fraction_of_data=1.0",
        ]
    )
    cfg, _ = build_config(args)

    assert cfg.runtime.device == "cuda"
    assert cfg.dataset.fraction_of_data == 1.0
