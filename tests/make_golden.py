"""Regenerate the golden fixtures.

Run from anywhere:  uv run --extra ml python tests/make_golden.py

Requires `datasets/casl_3305_runs_inter.h5` — this script is the ONLY thing that
does. It distils the fitted scalers out of that 542 MB file into a ~3 KB
`preprocessor.json`, which is what lets the tests themselves run anywhere,
including CI.

Only run this deliberately: it overwrites the reference values that
tests/test_golden.py compares against, so it erases the safety net if used to
"fix" a failing test. Record why in the commit message when you do.
"""

import hashlib
import json
import os

import h5py
import numpy as np
from golden_setup import (
    CKPT,
    FIX,
    MINI_H5,
    PREPROCESSOR,
    TRAIN_H5,
    build_inference_cfg,
    matrix_probes,
    run_inference,
)

RUNS, ROWS_PER_RUN = 10, 101


def slice_mini_h5(src=TRAIN_H5, dst=MINI_H5):
    with h5py.File(src) as f, h5py.File(dst, "w") as g:
        n = RUNS * ROWS_PER_RUN
        g["numeric_data"] = f["numeric_data"][:n]
        for k in ("numeric_columns", "all_columns"):
            g[k] = f[k][:]

    digest = hashlib.sha256()
    with open(src, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    print("source sha256:", digest.hexdigest())


def freeze_preprocessor():
    """Fit the scalers the historical inference path used, and save them.

    Fitting is on `path_to_data` under the config's `fraction_of_data`, which is
    exactly what inference mode did before the bundle existed — so freezing
    these parameters keeps every existing golden value byte-identical while
    removing the dependency on the training file.

    The fit lives here rather than in the datamodule because production code no
    longer refits scalers at serve time: a model is served with the scalers it
    was trained with or it does not run. This script is the one place that
    *produces* such a file, so it does the fit itself, with the parameters the
    committed goldens were generated under held fixed.
    """
    import nuclear_surrogates.datamodule.neural_ode_datamodule as node_dm
    from nuclear_surrogates.datamodule.preprocessor import Preprocessor

    cfg = build_inference_cfg()
    cfg.dataset.path_to_data = TRAIN_H5
    dm = node_dm.NODE_Datamodule(cfg)

    all_columns = list(dm.inputs.keys()) + [k for k in dm.target if k not in dm.inputs]
    traj = dm._read_trajectories(TRAIN_H5, dm.fraction_of_data, all_columns)

    Preprocessor.fit(
        dm.inputs,
        dm.target,
        traj.input_flat,
        traj.target_flat,
        traj.col_index_map,
        traj.target_index_map,
        t_days=node_dm.DEFAULT_TRAINING_T_DAYS,
        t_days_data_span=node_dm._data_span(traj),
    ).save(FIX, write_joblib=False)
    print(f"wrote {PREPROCESSOR}")


def golden_node():
    """Generate the goldens through the same path the tests will use."""
    cfg = build_inference_cfg(preprocessor_path=PREPROCESSOR)
    model, _, preds, trues = run_inference(cfg)

    np.save(os.path.join(FIX, "golden_node_preds.npy"), preds)
    np.save(os.path.join(FIX, "golden_node_trues.npy"), trues)

    from nuclear_surrogates.utils import metrics

    flat_p, flat_t = preds.reshape(-1, 7), trues.reshape(-1, 7)
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
    np.save(os.path.join(FIX, "golden_matrix_A.npy"), matrix_probes(model))


def golden_eval():
    """Pin the evaluation outputs that become paper figures."""
    import torch

    from nuclear_surrogates import evaluation

    cfg = build_inference_cfg(preprocessor_path=PREPROCESSOR)
    model, dm, ar_scaled, trues_scaled = run_inference(cfg)

    with torch.no_grad():
        tf_scaled = model._teacher_forced_predictions(
            dm.test_trajs[:, :, : dm.n_input_features].numpy(), trues_scaled
        )
    np.save(os.path.join(FIX, "golden_node_tf_preds.npy"), tf_scaled)

    pre = dm.preprocessor
    n_targets = trues_scaled.shape[-1]

    def unscale(a):
        return pre.inverse_targets(a.reshape(-1, n_targets)).reshape(a.shape)

    trues, ar, tf = unscale(trues_scaled), unscale(ar_scaled), unscale(tf_scaled)
    names = pre.target_names

    curves = evaluation.error_growth_curves(trues, ar, tf)
    comparison = evaluation.mare_comparison(trues, ar, tf)

    payload = {
        "error_growth": {
            name: {
                key: curves[i][key].tolist()
                for key in ("avg_ar_mae", "avg_tf_mae", "avg_ar_male", "avg_tf_male")
            }
            for i, name in enumerate(names)
        },
        "mare": {name: comparison[i] for i, name in enumerate(names)},
    }
    with open(os.path.join(FIX, "golden_node_eval.json"), "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    if not os.path.exists(CKPT):
        raise SystemExit(f"Missing frozen checkpoint: {CKPT}")
    slice_mini_h5()
    freeze_preprocessor()
    golden_node()
    golden_eval()
    print("\nRegenerated. Review `git diff tests/fixtures/` before committing.")
