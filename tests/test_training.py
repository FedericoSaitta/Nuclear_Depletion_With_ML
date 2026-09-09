"""Regression tests for training itself.

Everything here runs on the committed 10-run fixture, on CPU, into tmp_path — no
datasets, no GPU, seconds per test.

Two tiers, deliberately separated:

* **Contract and determinism** (unmarked): no pinned numbers, so they never go
  stale across a torch, BLAS or OS change and never need regenerating. These are
  the higher-value tier — `test_same_seed_gives_the_same_split` is what stops
  the unseeded-split class of bug from returning.
* **Pinned trajectory** (`@pytest.mark.golden_train`): catches numerical drift a
  contract test cannot see, but is only reproducible on a fixed runner image, so
  CI runs it in its own Linux-only job.

`trainer.fit` is driven directly rather than through `modes.train`, which chains
an expensive `trainer.test` (Jacobians, permutation importance, dozens of
figures) that these tests do not need.
"""

import csv
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
MINI_H5 = os.path.join(REPO, "tests", "fixtures", "mini_casl_10runs.h5")
GOLDEN_TRAIN = os.path.join(REPO, "tests", "fixtures", "golden_train.json")


def load_cfg(kind, tmp_path, **overrides):
    """A smoke config plus the runtime block `main.py` would synthesise.

    Configs carry no paths and no `runtime` section any more, so what the CLI
    would build from its flags is built here instead — redirected into tmp_path
    so a test writes nothing outside it.

    `seed: 0` is what the smoke configs used to declare, and
    `fixtures/golden_train.json` pins a loss trajectory produced at it.
    """
    cfg = OmegaConf.load(CONFIGS[kind])
    cfg.dataset.path_to_data = MINI_H5
    cfg.runtime = {
        "output_dir": str(tmp_path / "results"),
        "device": "cpu",
        "num_workers": 0,
        "analyses": False,
        "seed": 0,
    }
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


def build(cfg):
    """Seed, then construct (model, datamodule) exactly as main.py does."""
    import lightning as L

    from nuclear_surrogates.main import MODELS

    L.seed_everything(cfg.runtime.seed, workers=True)
    model_cls, dm_cls = MODELS[cfg.model.kind]
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
    """The guard against an unseeded split.

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


def test_both_models_hold_out_the_same_runs_under_random_by_run(tmp_path):
    """The head-to-head comparison has to be paired, not merely matched.

    The two models used to partition differently — the DNN sequentially, the
    NODE by a seeded permutation — so at one seed, on one dataset, with
    identical fractions, they were still scored on disjoint sets of runs. Under
    `random_by_run` both deal out `dataset_helper.run_permutation`, so every
    split must agree run for run. If this fails, the paper's DNN and NODE
    columns are describing different test sets again.
    """
    splits = {
        kind: _setup_only(
            load_cfg(
                kind,
                tmp_path / kind,
                **{"runtime.seed": 5, "dataset.split.strategy": "random_by_run"},
            )
        )
        for kind in ("DNN", "NODE")
    }

    assert splits["DNN"]["strategy"] == "random_by_run"
    assert splits["NODE"]["strategy"] == "random_by_run"
    assert splits["DNN"]["n_runs"] == splits["NODE"]["n_runs"]
    for part in ("train", "val", "test"):
        assert splits["DNN"][part] == splits["NODE"][part], part


def test_both_models_evaluate_the_same_100_points(tmp_path):
    """The two models' ground truth must be the same numbers on the same grid.

    The DNN's targets sit one step ahead of its inputs, so its trajectories
    naturally run c(1)…c(N) — days 10-1000 — while the NODE drops the file's
    last row and reports c(0)…c(N-1), days 0-990. The two were therefore scored
    on windows offset by a full 10-day step, which gave `metrics.mare` different
    denominators and misaligned every error-growth curve.
    `evaluation.align_to_initial` removes the offset; this checks that it did,
    by reconstructing each model's truth the way its own evaluation path does
    and comparing them channel by channel.
    """
    from nuclear_surrogates import evaluation
    from nuclear_surrogates.datamodule.dataset_helper import ordered_names

    overrides = {"runtime.seed": 0, "dataset.split.strategy": "random_by_run"}
    dnn = _fitted_datamodule(load_cfg("DNN", tmp_path / "dnn", **overrides))
    node = _fitted_datamodule(load_cfg("NODE", tmp_path / "node", **overrides))

    names = ordered_names(dnn.target_index_map)
    assert names == ordered_names(node.target_index_map)

    # DNN: inverse-transform the delta targets, integrate from the true c(0)
    # carried in each run's first input row — `_build_trajectories` in miniature.
    X, Y = (t.numpy() for t in dnn.test_dataset.tensors)
    steps = dnn.samples_per_run
    n_runs = len(X) // steps
    deltas = dnn.target_scaler.inverse_transform(Y)
    first_rows = dnn.input_scaler.inverse_transform(X[::steps])
    dnn_true = np.stack(
        [
            evaluation.integrate_deltas(
                deltas[:, i].reshape(n_runs, steps),
                first_rows[:, dnn.col_index_map[name]],
            )
            for i, name in enumerate(names)
        ],
        axis=-1,
    )

    # NODE: the target half of each trajectory, unscaled.
    trajectories = node.test_dataset.tensors[0].numpy()
    targets = trajectories[:, :, node.n_input_features :]
    runs, node_steps, n_target = targets.shape
    node_true = node.target_scaler.inverse_transform(
        targets.reshape(-1, n_target)
    ).reshape(runs, node_steps, n_target)

    assert dnn_true.shape == node_true.shape
    for i, name in enumerate(names):
        a, b = dnn_true[:, :, i], node_true[:, :, i]
        # float32 throughout, and the DNN's route there is a 100-term cumsum.
        assert np.max(np.abs(a - b)) <= 1e-5 * np.max(np.abs(b)), name


def _fitted_datamodule(cfg):
    _, dm = build(cfg)
    dm.setup(stage="fit")
    return dm


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_same_seed_gives_the_same_initial_weights(kind, tmp_path):
    first, _ = build(load_cfg(kind, tmp_path / "a", **{"runtime.seed": 3}))
    second, _ = build(load_cfg(kind, tmp_path / "b", **{"runtime.seed": 3}))
    for (name, p), (_, q) in zip(
        first.state_dict().items(), second.state_dict().items(), strict=False
    ):
        torch.testing.assert_close(p, q, msg=f"{name} differs at equal seed")


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_different_seeds_give_different_initial_weights(kind, tmp_path):
    first, _ = build(load_cfg(kind, tmp_path / "a", **{"runtime.seed": 3}))
    second, _ = build(load_cfg(kind, tmp_path / "b", **{"runtime.seed": 4}))
    assert any(
        not torch.equal(p, q)
        for p, q in zip(
            first.state_dict().values(), second.state_dict().values(), strict=False
        )
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
    for column, (a, b) in enumerate(zip(baseline, perturbed, strict=False)):
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
        not np.allclose(a, b) for a, b in zip(baseline, moved, strict=False)
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
        model.state_dict().items(), restored.state_dict().items(), strict=False
    ):
        torch.testing.assert_close(a, b, msg=f"{name} changed across save/load")


def test_run_writes_split_indices_and_a_bundle(tmp_path):
    """A finished run must leave enough behind to be audited later."""
    from nuclear_surrogates.models import modes

    cfg = load_cfg("DNN", tmp_path, **{"train.num_epochs": 1})
    _, dm = build(cfg)
    result_dir = modes.result_dir(cfg)

    dm.setup(stage="fit")
    bundle = modes._write_run_bundle(dm, cfg, result_dir, ckpt_path=None)
    assert bundle is not None
    for name in (
        "preprocessor.json",
        "config.resolved.yaml",
        "metadata.json",
        "split_indices.json",
    ):
        assert os.path.exists(os.path.join(bundle, name)), f"bundle lacks {name}"

    with open(os.path.join(bundle, "metadata.json")) as f:
        metadata = json.load(f)
    assert metadata["seed"] == cfg.runtime.seed
    assert metadata["libraries"]["torch"]


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_inference_without_fitted_scalers_is_refused(kind, tmp_path):
    """A checkpoint without its scalers is half a model, and half a model does
    not run.

    Inference used to refit on `path_to_data` whenever `preprocessor_path` was
    absent, behind a warning. There is now no fallback at all: evaluating a
    checkpoint against scalers it was never trained with is worse than not
    running, because the resulting numbers look fine.
    """
    cfg = load_cfg(kind, tmp_path)
    cfg.dataset.path_to_inference_data = cfg.dataset.path_to_data
    _, dm = build(cfg)
    dm.inference_mode = True
    with pytest.raises(SystemExit, match="preprocessor_path"):
        dm.setup(stage="test")


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_inference_from_a_bundle_never_opens_the_training_file(kind, tmp_path):
    """The point of a bundle: weights and scalers travel without the dataset.

    Trains a tiny model, bundles it, then evaluates that bundle on a different
    file and asserts the only file opened is the inference one. `test_golden`
    pins the same property for the frozen NODE; this covers both models on a
    bundle built end to end by the code under test.
    """
    import nuclear_surrogates.datamodule.dataset_helper as data_help
    from nuclear_surrogates.bundle import read_bundle
    from nuclear_surrogates.main import MODELS
    from nuclear_surrogates.models import modes

    cfg = load_cfg(kind, tmp_path, **{"train.num_epochs": 1})
    _, dm, trainer = train(cfg)

    ckpt = tmp_path / "trained.ckpt"
    trainer.save_checkpoint(ckpt)
    bundle = modes._write_run_bundle(dm, cfg, modes.result_dir(cfg), str(ckpt))

    inference_cfg, loaded = read_bundle(bundle)
    inference_cfg.dataset.path_to_inference_data = cfg.dataset.path_to_data
    inference_cfg.runtime = {**cfg.runtime, "output_dir": str(tmp_path / "served")}

    assert loaded.metadata["model_kind"] == kind
    assert not inference_cfg.dataset.path_to_data, "training path must be cleared"
    assert loaded.weights.parent == loaded.path
    assert str(loaded.path) == os.path.abspath(bundle)

    opened = []
    real_read = data_help.read_data

    def spy(path, *args, **kwargs):
        opened.append(path)
        return real_read(path, *args, **kwargs)

    data_help.read_data = spy
    try:
        model_cls, dm_cls = MODELS[kind]
        served = dm_cls(inference_cfg)
        served.inference_mode = True
        served.setup(stage="test")
    finally:
        data_help.read_data = real_read

    # Only one fixture exists, so the two paths name the same file — but the
    # refit branch reads twice (training file, then inference file), so
    # "exactly one open" is still what distinguishes the two paths.
    assert opened == [inference_cfg.dataset.path_to_inference_data], (
        f"inference opened {opened} — a bundle must not reach for the "
        f"training dataset"
    )
    assert len(served.test_dataset) > 0
    assert len(served.train_dataset) == 0, "inference must not build a train split"


# ── plots ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_plots_replays_the_runs_own_test_split(kind, tmp_path):
    """Redrawing a finished run's figures must use that run's own test runs.

    The split is rebuilt from the seed rather than replayed from the recorded
    indices, so this pins that the rebuild lands on exactly what the bundle
    recorded — otherwise the figures would be of a different subset while
    still being labelled as the published run's.
    """
    import json as _json

    from nuclear_surrogates.bundle import read_bundle
    from nuclear_surrogates.main import MODELS
    from nuclear_surrogates.models import modes

    cfg = load_cfg(kind, tmp_path, **{"train.num_epochs": 1})
    _, dm, trainer = train(cfg)

    ckpt = tmp_path / "trained.ckpt"
    trainer.save_checkpoint(ckpt)
    bundle = modes._write_run_bundle(dm, cfg, modes.result_dir(cfg), str(ckpt))

    replay_cfg, loaded = read_bundle(bundle)
    replay_cfg.dataset.path_to_data = cfg.dataset.path_to_data
    replay_cfg.runtime = {**cfg.runtime, "output_dir": str(tmp_path / "redrawn")}

    # The seed is the one runtime value a bundle carries forward, because the
    # split this test checks is a function of it.
    assert loaded.seed == cfg.runtime.seed

    _, dm_cls = MODELS[kind]
    replayed = dm_cls(replay_cfg)
    replayed.setup(stage="fit")

    with open(os.path.join(bundle, "split_indices.json")) as f:
        recorded = _json.load(f)
    for key in ("train", "val", "test"):
        assert list(replayed.split_info[key]) == list(
            recorded[key]
        ), f"{key} split drifted from the bundle's record"

    # And the replay must use the run's scalers, not a fresh fit of its own.
    modes._verify_split_matches_bundle(replayed, loaded)
    assert replayed.preprocessor.to_dict() == dm.preprocessor.to_dict()


def test_plots_refuses_a_split_that_does_not_match(tmp_path):
    """A mismatch means these are not the published run's runs. Fail, don't draw."""
    from pathlib import Path

    from nuclear_surrogates.bundle import Bundle
    from nuclear_surrogates.models import modes

    cfg = load_cfg("NODE", tmp_path, **{"train.num_epochs": 1})
    _, dm = build(cfg)
    dm.setup(stage="fit")
    bundle = modes._write_run_bundle(dm, cfg, modes.result_dir(cfg), ckpt_path=None)

    # Built directly rather than via read_bundle: this run wrote no weights, and
    # only the directory matters for locating split_indices.json.
    loaded = Bundle(Path(bundle), None, None, {}, cfg.runtime.seed)
    dm.split_info = {**dm.split_info, "test": [999]}
    with pytest.raises(SystemExit, match="split_indices.json"):
        modes._verify_split_matches_bundle(dm, loaded)


def test_plots_needs_the_training_dataset(tmp_path):
    """It carves the test split out of the training file, so it cannot run
    without it — unlike `infer`, which needs only the bundle."""
    from nuclear_surrogates.main import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["plots", "--bundle", str(tmp_path)])


# ── evaluation epoch ─────────────────────────────────────────────────────────


def test_dnn_test_epoch_reports_and_plots_every_target(tmp_path):
    """Drive `DNN_Model.on_test_epoch_end` end to end.

    The rest of this module stops at `trainer.fit`, so without this nothing
    exercises the autoregressive rollout, the delta -> absolute conversion or
    any of the evaluation figures. `smoke_dnn.yaml` sets `target_delta_conc`,
    which is the branch that carries the conversion.
    """
    import lightning as L

    cfg = load_cfg("DNN", tmp_path, **{"train.num_epochs": 1, "runtime.analyses": True})
    model, dm, _ = train(cfg)

    trainer = L.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    reported = trainer.test(model, datamodule=dm)[0]

    for key in (
        "Mean Absolute Error (avg)",
        "Root Mean Squared Error (avg)",
        "R-squared coefficient (avg)",
    ):
        assert np.isfinite(reported[key]), f"{key} is not finite: {reported[key]}"

    result_dir = os.path.join(str(tmp_path / "results"), cfg.model.name)
    for target in cfg.dataset.targets:
        for metric in ("MARE_TeacherForcing", "MARE_Autoregressive"):
            value = reported[f"{target}/{metric}"]
            assert np.isfinite(value), f"{target}/{metric} is not finite: {value}"

        target_dir = os.path.join(result_dir, target)
        for figure in (
            "predictions_vs_actual.png",
            "residuals_combined.png",
            "residuals_combined_loglog.png",
            f"{target}_prediction_comparison.png",
            f"{target}_MAE_growth_linear.png",
            f"{target}_MALE_growth_linear.png",
            # The DNN draws trajectories too now. It used to compute the arrays
            # and throw them away, which left the head-to-head against the NODE
            # without comparable figures.
            "test_traj_1.png",
            "test_all_trajectories.png",
            "r2_score_importance.png",
        ):
            assert os.path.exists(
                os.path.join(target_dir, figure)
            ), f"{target}: {figure} was not written"

        # The log-scaled twins are gone: MALE is already Mean Absolute *Log*
        # Error, so a log axis on it plotted log-of-log, and nothing referenced
        # the MAE one.
        for gone in (f"{target}_MAE_growth_log.png", f"{target}_MALE_growth_log.png"):
            assert not os.path.exists(
                os.path.join(target_dir, gone)
            ), f"{target}: {gone} should no longer be written"

        # The numbers behind the importance bar charts, not just the picture.
        # The NODE has always written its per-step sweep out as a CSV; the DNN's
        # permutation sweep used to reach a PNG and nowhere else.
        csv_path = os.path.join(target_dir, "permutation_importance.csv")
        assert os.path.exists(csv_path), f"{target}: importance CSV was not written"
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(cfg.dataset.inputs), (
            f"{target}: expected one importance row per input feature, got "
            f"{len(rows)}"
        )
        assert {row["Feature"] for row in rows} == set(cfg.dataset.inputs)
        for row in rows:
            assert row["Type"] in ("Forcing", "State")
            for metric in ("r2", "mse"):
                assert np.isfinite(float(row[f"{metric}_importance_mean"]))
                assert np.isfinite(float(row[f"{metric}_importance_pct"]))

    table = os.path.join(result_dir, "permutation_importance.md")
    assert os.path.exists(table), "the importance markdown table was not written"
    with open(table, encoding="utf-8") as f:
        written = f.read()
    for target in cfg.dataset.targets:
        assert f"`{target}`" in written, f"{target} is missing from {table}"


@pytest.mark.parametrize("kind", ["DNN", "NODE"])
def test_no_analyses_writes_metrics_but_no_figures(kind, tmp_path):
    """`--no-analyses` is the flag a hyperparameter sweep wants.

    It has to suppress every figure *and* the expensive post-hoc passes —
    permutation importance, Jacobians, the depletion matrix — while still
    producing the metrics that decide whether the trial was any good.
    """
    import lightning as L

    cfg = load_cfg(kind, tmp_path, **{"train.num_epochs": 1})
    assert cfg.runtime.analyses is False
    model, dm, _ = train(cfg)

    L.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    ).test(model, datamodule=dm)

    results = tmp_path / "results"
    written = sorted(p.name for p in results.rglob("*.png"))
    # The loss curve is the one exception, and the rule behind it is
    # regenerability: `nucml plots` never runs `fit`, so this figure cannot be
    # recreated from the bundle the way every other one can.
    assert written == [
        "training_loss_log.png"
    ], f"{kind}: --no-analyses should leave only the loss curve, got {written[:6]}"
    for pattern in ("stepwise_importance.*", "permutation_importance.*"):
        assert not list(
            results.rglob(pattern)
        ), f"{kind}: {pattern} is analysis output and must be skipped too"

    metrics_file = results / cfg.model.name / "test_metrics.json"
    payload = json.loads(metrics_file.read_text())
    assert len(payload["per_target"]) == len(cfg.dataset.targets)
    assert np.isfinite(payload["mae_avg"])


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
