"""Regenerate the golden fixtures.

Run from anywhere:  uv run --extra ml python tests/make_golden.py

Only run this deliberately: it overwrites the reference values that
tests/test_golden.py compares against, so it erases the safety net if used to
"fix" a failing test. Record why in the commit message when you do.
"""

import hashlib
import json
import os

import h5py
import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf

import nuclear_surrogates.datamodule.neural_ode_datamodule as node_dm
from nuclear_surrogates.models.modes import load_checkpoint_into_model
from nuclear_surrogates.models.neural_ode import NODE_Model
from nuclear_surrogates.utils import metrics
from nuclear_surrogates.utils.paths import resolve_config_paths

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIX = os.path.join(REPO, "tests", "fixtures")
CONFIG = os.path.join(REPO, "configs", "main_config.yaml")
TRAIN_H5 = os.path.join(REPO, "datasets", "casl_3305_runs_inter.h5")
CKPT = os.path.join(FIX, "best-matrix_ode_7x7_breeding_chain-epoch=2367.ckpt")
RUNS, ROWS_PER_RUN = 10, 101


def slice_mini_h5(src=TRAIN_H5, dst=os.path.join(FIX, "mini_casl_10runs.h5")):
    with h5py.File(src) as f, h5py.File(dst, "w") as g:
        n = RUNS * ROWS_PER_RUN
        g["numeric_data"] = f["numeric_data"][:n]
        for k in ("numeric_columns", "all_columns"):
            g[k] = f[k][:]
    with open(src, "rb") as f:
        print("full-file sha256:", hashlib.sha256(f.read()).hexdigest())


def golden_node(ckpt=CKPT):
    cfg = OmegaConf.load(CONFIG)
    resolve_config_paths(cfg, CONFIG)
    cfg.runtime.update(mode="inference", device="cpu", num_workers=0)
    cfg.dataset.path_to_data = TRAIN_H5
    cfg.dataset.path_to_inference_data = os.path.join(FIX, "mini_casl_10runs.h5")

    dm = node_dm.NODE_Datamodule(cfg)
    dm.inference_mode = True
    model = load_checkpoint_into_model(NODE_Model(cfg), ckpt)
    trainer = L.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
    preds = trainer.predict(model, datamodule=dm)
    pred_arr = torch.cat([p["pred"] for p in preds]).numpy()
    true_arr = torch.cat([p["true"] for p in preds]).numpy()
    np.save(os.path.join(FIX, "golden_node_preds.npy"), pred_arr)
    np.save(os.path.join(FIX, "golden_node_trues.npy"), true_arr)

    flat_p, flat_t = pred_arr.reshape(-1, 7), true_arr.reshape(-1, 7)
    with open(os.path.join(FIX, "golden_node_metrics.json"), "w") as f:
        json.dump(
            {
                "r2": metrics.r2(flat_t, flat_p).tolist(),
                "mae": metrics.mae(flat_t, flat_p).tolist(),
                "mare": [metrics.mare(flat_t[:, i], flat_p[:, i]) for i in range(7)],
            },
            f,
            indent=2,
        )

    # Matrix probes — protects the depletion-matrix figure through refactors
    torch.manual_seed(0)
    y_probe = torch.rand(3, 7)
    f_probe = torch.rand(3, 1)
    A = model.func._build_matrix(f_probe, y_probe)
    np.save(os.path.join(FIX, "golden_matrix_A.npy"), A.detach().numpy())


if __name__ == "__main__":
    slice_mini_h5()
    golden_node()
