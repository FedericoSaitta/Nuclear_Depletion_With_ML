"""Golden regression tests: refactors must not move published numbers.

These run anywhere, including CI. They need only the committed fixtures — the
frozen checkpoint, the 10-run mini dataset, and the fitted scalers distilled out
of the training file by `make_golden.py`. Nothing here opens the 542 MB training
dataset; that dependency is what used to make this whole module skip.

Tolerances (rationale in AUDIT.md §D): CPU float32 dopri5 trajectories get
atol=1e-6/rtol=1e-5 — enough headroom to survive operation reordering from a
refactor, tight enough to catch a real change. Scalar metrics get rtol=1e-4.
Loosen these only deliberately, never to make a red test green.
"""

import json
import os

import numpy as np
import pytest

from golden_setup import FIX, build_inference_cfg, fixtures_present, matrix_probes
from golden_setup import run_inference as _run_inference

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(not fixtures_present(), reason="golden fixtures are missing"),
]


@pytest.fixture(scope="module")
def node_setup(frozen_node_run):
    model, _, preds, _ = frozen_node_run
    return model, preds


def test_node_trajectories_unchanged(node_setup):
    _, pred = node_setup
    golden = np.load(os.path.join(FIX, "golden_node_preds.npy"))
    np.testing.assert_allclose(pred, golden, rtol=1e-5, atol=1e-6)


def test_node_metrics_unchanged(node_setup):
    """Recompute the paper metrics from fresh predictions vs frozen ground truth
    and compare against the golden metric values."""
    from nuclear_surrogates.utils import metrics

    _, pred = node_setup
    with open(os.path.join(FIX, "golden_node_metrics.json")) as f:
        golden = json.load(f)
    truth = np.load(os.path.join(FIX, "golden_node_trues.npy"))
    flat_p, flat_t = pred.reshape(-1, 7), truth.reshape(-1, 7)
    np.testing.assert_allclose(metrics.mae(flat_t, flat_p), golden["mae"], rtol=1e-4)
    np.testing.assert_allclose(metrics.r2(flat_t, flat_p), golden["r2"], rtol=1e-4)
    np.testing.assert_allclose(
        [metrics.mare(flat_t[:, i], flat_p[:, i]) for i in range(7)],
        golden["mare"],
        rtol=1e-4,
    )


def test_depletion_matrix_unchanged(node_setup):
    model, _ = node_setup
    np.testing.assert_allclose(
        matrix_probes(model),
        np.load(os.path.join(FIX, "golden_matrix_A.npy")),
        rtol=1e-5,
        atol=1e-7,
    )


def test_inference_does_not_read_the_training_dataset(monkeypatch, tmp_path):
    """The bundle decoupling, asserted rather than assumed.

    This is the property that keeps the suite alive in CI: if someone
    reintroduces a read of `path_to_data` in inference mode, every other test
    here would still pass on a developer machine and silently start skipping
    everywhere else. So fail loudly instead.
    """
    import nuclear_surrogates.datamodule.dataset_helper as data_help

    cfg = build_inference_cfg(output_dir=tmp_path)
    opened = []
    real_read_data = data_help.read_data

    def spy(file_path, *args, **kwargs):
        opened.append(file_path)
        return real_read_data(file_path, *args, **kwargs)

    monkeypatch.setattr(data_help, "read_data", spy)
    _run_inference(cfg)

    assert len(opened) == 1, f"expected only the inference file, opened {opened}"
    assert os.path.basename(opened[0]) == "mini_casl_10runs.h5"
