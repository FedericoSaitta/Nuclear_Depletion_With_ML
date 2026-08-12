"""Regression tests for training itself.

Everything here runs on the committed 10-run fixture, on CPU, into tmp_path — no
datasets, no GPU, seconds per test.

Two tiers, deliberately separated:

* **Contract and determinism** (unmarked): no pinned numbers, so they never go
  stale across a torch, BLAS or OS change and never need regenerating. These are
  the higher-value tier — `test_same_seed_gives_the_same_split` is what stops
  the unseeded-split class of bug (AUDIT.md §A1) from returning.
* **Pinned trajectory** (`@pytest.mark.golden_train`): catches numerical drift a
  contract test cannot see, but is only reproducible on a fixed runner image, so
  CI runs it in its own Linux-only job.

`trainer.fit` is driven directly rather than through `modes.train_and_test`,
which chains an expensive `trainer.test` (Jacobians, permutation importance,
dozens of figures) that these tests do not need.
"""

import json
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = {
    "DNN": os.path.join(REPO, "configs", "smoke_dnn.yaml"),
    "NODE": os.path.join(REPO, "configs", "smoke_node.yaml"),
}
GOLDEN_TRAIN = os.path.join(REPO, "tests", "fixtures", "golden_train.json")


def load_cfg(kind, tmp_path, **overrides):
    """Smoke config, redirected so a test writes nothing outside tmp_path."""
    from nuclear_surrogates.utils.paths import resolve_config_paths

    path = CONFIGS[kind]
    cfg = OmegaConf.load(path)
    resolve_config_paths(cfg, path)
    cfg.runtime.output_dir = str(tmp_path / "results")
    cfg.runtime.model_database = str(tmp_path / "experiments.db")
    cfg.runtime.device = "cpu"
    cfg.runtime.num_workers = 0
    cfg.runtime.plots = False
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


def build(cfg):
    """Seed, then construct (model, datamodule) exactly as main.py does."""
    import lightning as L

    from nuclear_surrogates.main import MODELS

    L.seed_everything(cfg.runtime.seed, workers=True)
    model_cls, dm_cls = MODELS[cfg.runtime.model]
    return model_cls(cfg), dm_cls(cfg)


def epoch_losses(model):
    """Per-epoch training losses.

    NODE_Model calls the list `_train_losses` and DNN_Model calls it
    `train_losses`; accessing either here rather than renaming one keeps this
    change out of code it does not otherwise touch.
    """
    losses = getattr(model, "_train_losses", None)
    if losses is None:
        losses = model.train_losses
    return [float(x) for x in losses]


def train(cfg, max_epochs=None):
    """Fit and return (model, datamodule, trainer)."""
    import lightning as L

    model, dm = build(cfg)
    trainer = L.Trainer(
        max_epochs=max_epochs or cfg.train.num_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=cfg.train.grad_clip,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(model, datamodule=dm)
    return model, dm, trainer


# ── determinism ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_same_seed_gives_the_same_split(kind, tmp_path):
    """The guard against AUDIT.md §A1.

    The published NODE's test set was drawn by an unseeded permutation, so the
    runs it was evaluated on are unknowable and ~60% of them had been trained
    on. That must never be possible again.
    """
    a, b = (
        _setup_only(load_cfg(kind, tmp_path / str(i), **{"runtime.seed": 5}))
        for i in range(2)
    )
    assert a["train"] == b["train"]
    assert a["val"] == b["val"]
    assert a["test"] == b["test"]


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_different_seeds_give_different_splits(kind, tmp_path):
    """Guards the opposite failure: a seed that is read and then ignored."""
    a = _setup_only(load_cfg(kind, tmp_path / "a", **{"runtime.seed": 1}))
    b = _setup_only(load_cfg(kind, tmp_path / "b", **{"runtime.seed": 2}))
    key = "train_shuffle_order" if "train_shuffle_order" in a else "train"
    assert a[key] != b[key]


def _setup_only(cfg):
    _, dm = build(cfg)
    dm.setup(stage="fit")
    return dm.split_info


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_same_seed_gives_the_same_initial_weights(kind, tmp_path):
    first, _ = build(load_cfg(kind, tmp_path / "a", **{"runtime.seed": 3}))
    second, _ = build(load_cfg(kind, tmp_path / "b", **{"runtime.seed": 3}))
    for (name, p), (_, q) in zip(
        first.state_dict().items(), second.state_dict().items()
    ):
        torch.testing.assert_close(p, q, msg=f"{name} differs at equal seed")


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_different_seeds_give_different_initial_weights(kind, tmp_path):
    first, _ = build(load_cfg(kind, tmp_path / "a", **{"runtime.seed": 3}))
    second, _ = build(load_cfg(kind, tmp_path / "b", **{"runtime.seed": 4}))
    assert any(
        not torch.equal(p, q)
        for p, q in zip(first.state_dict().values(), second.state_dict().values())
    )


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_two_runs_at_one_seed_produce_the_same_losses(kind, tmp_path):
    """End-to-end determinism: same seed, same data, same numbers.

    No golden file involved, so this stays valid across dependency upgrades —
    it only asserts that the pipeline agrees with itself.
    """
    losses = []
    for i in range(2):
        model, _, _ = train(load_cfg(kind, tmp_path / str(i), **{"runtime.seed": 0}))
        losses.append(epoch_losses(model))

    np.testing.assert_allclose(losses[0], losses[1], rtol=1e-6, atol=1e-9)


# ── leakage and contract ─────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_scalers_are_fit_on_the_training_split_only(kind, tmp_path, monkeypatch):
    """Blow up the held-out runs; not one scaler parameter may move.

    A direct check on the property that decides whether a reported metric is
    honest. The runs to perturb come from the run's own `split_info`, because
    the two models partition differently — DNN sequentially, NODE by a seeded
    permutation — so "the held-out runs" is not a fixed slice of the file.
    """
    import polars as pl

    from nuclear_surrogates.datamodule import dataset_helper

    def scaler_params(perturb_runs=()):
        real_read = dataset_helper.read_data

        def patched(path, fraction, **kwargs):
            df, run_length, time_array = real_read(path, fraction, **kwargs)
            if len(perturb_runs):
                factor = np.ones(df.shape[0])
                for run in perturb_runs:
                    factor[run * run_length : (run + 1) * run_length] = 1000.0
                multiplier = pl.Series("factor", factor)
                df = df.with_columns(
                    [
                        (pl.col(c) * multiplier).alias(c)
                        for c in df.columns
                        if c != "time_days"
                    ]
                )
            return df, run_length, time_array

        monkeypatch.setattr(dataset_helper, "read_data", patched)
        _, dm = build(load_cfg(kind, tmp_path / f"p{len(perturb_runs)}"))
        dm.setup(stage="fit")
        monkeypatch.undo()

        params = [
            np.asarray(
                getattr(s, "data_max_", None) if hasattr(s, "data_max_") else s.scale_
            )
            .copy()
            .ravel()
            for s in dm.preprocessor.target_scaler.scalers
        ]
        return params, dm.split_info

    baseline, split = scaler_params()
    held_out = list(split["val"]) + list(split["test"])
    assert held_out, "no held-out runs — the fixture is too small to test leakage"

    perturbed, _ = scaler_params(perturb_runs=held_out)
    for column, (a, b) in enumerate(zip(baseline, perturbed)):
        np.testing.assert_allclose(
            a,
            b,
            rtol=1e-9,
            err_msg=(
                f"target column {column}: held-out runs {held_out} changed the "
                f"fitted scaler, so validation/test data leaked into training"
            ),
        )

    # Positive control: perturbing TRAINING runs must move the parameters.
    # Without this, a test that silently stopped perturbing anything would still
    # pass and we would believe we had checked for leakage.
    moved, _ = scaler_params(perturb_runs=list(split["train"]))
    assert any(
        not np.allclose(a, b) for a, b in zip(baseline, moved)
    ), "perturbing the training runs changed nothing — the test is not testing anything"


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_loss_decreases(kind, tmp_path):
    model, _, _ = train(load_cfg(kind, tmp_path, **{"train.num_epochs": 4}))
    losses = epoch_losses(model)
    assert np.isfinite(losses).all(), f"non-finite training loss: {losses}"
    assert losses[-1] < losses[0], f"loss did not improve: {losses}"


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_checkpoint_round_trips(kind, tmp_path):
    """Saving and reloading must reproduce predictions exactly.

    This is what makes a published checkpoint meaningful at all.
    """
    from nuclear_surrogates.main import MODELS
    from nuclear_surrogates.models.modes import load_checkpoint_into_model

    cfg = load_cfg(kind, tmp_path)
    model, dm, trainer = train(cfg)

    ckpt = tmp_path / "roundtrip.ckpt"
    trainer.save_checkpoint(ckpt)

    model_cls, _ = MODELS[kind]
    restored = load_checkpoint_into_model(model_cls(cfg), str(ckpt))

    for (name, a), (_, b) in zip(
        model.state_dict().items(), restored.state_dict().items()
    ):
        torch.testing.assert_close(a, b, msg=f"{name} changed across save/load")


def test_run_writes_split_indices_and_a_bundle(tmp_path):
    """A finished run must leave enough behind to be audited later."""
    from nuclear_surrogates.models import modes

    cfg = load_cfg("DNN", tmp_path, **{"train.num_epochs": 1})
    _, dm = build(cfg)
    result_dir = modes.result_dir(cfg)

    dm.setup(stage="fit")
    assert os.path.exists(os.path.join(result_dir, "split_indices.json"))

    bundle = modes._write_run_bundle(dm, cfg, result_dir, ckpt_path=None)
    assert bundle is not None
    for name in ("preprocessor.json", "config.resolved.yaml", "metadata.json"):
        assert os.path.exists(os.path.join(bundle, name)), f"bundle lacks {name}"

    with open(os.path.join(bundle, "metadata.json")) as f:
        metadata = json.load(f)
    assert metadata["seed"] == cfg.runtime.seed
    assert metadata["libraries"]["torch"]


def test_dnn_inference_mode_fails_loudly(tmp_path):
    """The DNN has no inference branch. It must raise rather than quietly
    evaluate the training file's own test split (AUDIT.md §A3)."""
    _, dm = build(load_cfg("DNN", tmp_path))
    dm.inference_mode = True
    with pytest.raises(NotImplementedError, match="DNN inference mode"):
        dm.setup(stage="test")


# ── pinned trajectory ────────────────────────────────────────────────────────


@pytest.mark.golden_train
@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_loss_trajectory_unchanged(kind, tmp_path):
    """Pinned per-epoch training loss.

    Reproducible only on a fixed runner image and library set — CI runs this in
    a Linux-only job, and a mismatch on another platform is expected rather than
    a bug. Regenerate with `python tests/make_golden_train.py` and say why.
    """
    if not os.path.exists(GOLDEN_TRAIN):
        pytest.skip("golden_train.json not generated — run tests/make_golden_train.py")

    with open(GOLDEN_TRAIN) as f:
        golden = json.load(f)
    if kind not in golden:
        pytest.skip(f"no pinned trajectory for {kind}")

    torch.set_num_threads(1)  # thread count changes float reduction order
    model, _, _ = train(load_cfg(kind, tmp_path, **{"runtime.seed": 0}))
    losses = epoch_losses(model)

    np.testing.assert_allclose(
        losses,
        golden[kind]["train_losses"],
        rtol=1e-3,
        err_msg=(
            f"{kind} training loss trajectory moved. If this is a deliberate "
            f"change, regenerate tests/fixtures/golden_train.json; if it is a "
            f"platform difference, this job should only run on Linux CI."
        ),
    )
