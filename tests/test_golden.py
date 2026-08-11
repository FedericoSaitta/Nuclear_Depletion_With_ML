"""Golden regression tests: refactors must not move published numbers.

Skips cleanly when the fixtures or the full training dataset are absent, so CI
stays green on machines without them.
"""

import json
import os

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIX = os.path.join(REPO, "tests", "fixtures")
CONFIG = os.path.join(REPO, "configs", "main_config.yaml")
CKPT = os.path.join(FIX, "best-matrix_ode_7x7_breeding_chain-epoch=2367.ckpt")

need = [
    "mini_casl_10runs.h5",
    "golden_node_preds.npy",
    "golden_node_trues.npy",
    "golden_node_metrics.json",
    "golden_matrix_A.npy",
    "best-matrix_ode_7x7_breeding_chain-epoch=2367.ckpt",
]
# The datamodule fits its scalers on the full training file before it ever looks
# at the inference file, so the goldens are only reproducible where that file is.
TRAIN_H5 = os.path.join(REPO, "datasets", "casl_3305_runs_inter.h5")
pytestmark = pytest.mark.skipif(
    not (
        all(os.path.exists(os.path.join(FIX, f)) for f in need)
        and os.path.exists(TRAIN_H5)
    ),
    reason="golden fixtures or the full training dataset are not present",
)


@pytest.fixture(scope="module")
def node_setup():
    import lightning as L
    from omegaconf import OmegaConf

    import nuclear_surrogates.datamodule.neural_ode_datamodule as node_dm
    from nuclear_surrogates.models.modes import load_checkpoint_into_model
    from nuclear_surrogates.models.neural_ode import NODE_Model
    from nuclear_surrogates.utils.paths import resolve_config_paths

    cfg = OmegaConf.load(CONFIG)
    resolve_config_paths(cfg, CONFIG)
    cfg.runtime.update(mode="inference", device="cpu", num_workers=0)
    cfg.dataset.path_to_data = TRAIN_H5
    cfg.dataset.path_to_inference_data = os.path.join(FIX, "mini_casl_10runs.h5")

    dm = node_dm.NODE_Datamodule(cfg)
    dm.inference_mode = True
    model = load_checkpoint_into_model(NODE_Model(cfg), CKPT)
    trainer = L.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
    preds = trainer.predict(model, datamodule=dm)
    return model, torch.cat([p["pred"] for p in preds]).numpy()


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
    torch.manual_seed(0)
    y_probe = torch.rand(3, 7)
    f_probe = torch.rand(3, 1)
    A = model.func._build_matrix(f_probe, y_probe).detach().numpy()
    np.testing.assert_allclose(
        A, np.load(os.path.join(FIX, "golden_matrix_A.npy")), rtol=1e-5, atol=1e-7
    )
