"""Generate golden fixtures. Run once from ML/: uv run python ../tests/make_golden.py"""
import json, hashlib, h5py, numpy as np, torch
from omegaconf import OmegaConf
import lightning as L

import ML.datamodule.neural_ode_datamodule as node_dm
from ML.models.neural_ode import NODE_Model
from ML.models.modes import load_checkpoint_into_model
from ML.utils import metrics

FIX = "../tests/fixtures"
RUNS, ROWS_PER_RUN = 10, 101

def slice_mini_h5(src="data/casl_3305_runs_inter.h5", dst=f"{FIX}/mini_casl_10runs.h5"):
    with h5py.File(src) as f, h5py.File(dst, "w") as g:
        n = RUNS * ROWS_PER_RUN
        g["numeric_data"] = f["numeric_data"][:n]
        for k in ("numeric_columns", "all_columns"):
            g[k] = f[k][:]
    print("full-file sha256:", hashlib.sha256(open(src, "rb").read()).hexdigest())

def golden_node(ckpt="../tests/fixtures/best-matrix_ode_7x7_breeding_chain-epoch=2367.ckpt"):
    cfg = OmegaConf.load("main_config.yaml")
    cfg.runtime.update(mode="inference", device="cpu", num_workers=0)
    cfg.dataset.path_to_inference_data = f"{FIX}/mini_casl_10runs.h5"
    dm = node_dm.NODE_Datamodule(cfg); dm.inference_mode = True
    model = load_checkpoint_into_model(NODE_Model(cfg), ckpt, save_fixed=False)
    trainer = L.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
    preds = trainer.predict(model, datamodule=dm)
    pred_arr = torch.cat([p["pred"] for p in preds]).numpy()
    true_arr = torch.cat([p["true"] for p in preds]).numpy()
    np.save(f"{FIX}/golden_node_preds.npy", pred_arr)
    np.save(f"{FIX}/golden_node_trues.npy", true_arr)
    flat_p, flat_t = pred_arr.reshape(-1, 7), true_arr.reshape(-1, 7)
    json.dump({
        "r2":   metrics.r2(flat_t, flat_p).tolist(),
        "mae":  metrics.mae(flat_t, flat_p).tolist(),
        "mare": [metrics.mare(flat_t[:, i], flat_p[:, i]) for i in range(7)],
    }, open(f"{FIX}/golden_node_metrics.json", "w"), indent=2)

    # Matrix probes — protects the depletion-matrix figure through refactors
    torch.manual_seed(0)
    y_probe = torch.rand(3, 7); f_probe = torch.rand(3, 1)
    A = model.func._build_matrix(f_probe, y_probe)
    np.save(f"{FIX}/golden_matrix_A.npy", A.detach().numpy())

if __name__ == "__main__":
    slice_mini_h5(); golden_node()
