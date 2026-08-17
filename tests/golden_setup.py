"""Shared construction of the frozen-NODE inference run.

`test_golden.py` and `make_golden.py` must build *exactly* the same thing, or
the tests compare against goldens that pin something else. They used to
duplicate this block verbatim, which is one edit away from silent drift.
"""

import os

import lightning as L
import torch
from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIX = os.path.join(REPO, "tests", "fixtures")
# Frozen on purpose: a working config is free to change its solver tolerances,
# and that silently moves every golden trajectory.
CONFIG = os.path.join(FIX, "golden_node_eval.yaml")
CKPT = os.path.join(FIX, "best-matrix_ode_7x7_breeding_chain-epoch=2367.ckpt")
MINI_H5 = os.path.join(FIX, "mini_casl_10runs.h5")
PREPROCESSOR = os.path.join(FIX, "preprocessor.json")

# Only `make_golden.py` needs this: the fixture preprocessor is derived from it
# once, and committed, so the tests never touch it again.
TRAIN_H5 = os.path.join(REPO, "datasets", "casl_3305_runs_inter.h5")

# Fixtures the golden tests require. The full training dataset is deliberately
# NOT in this list — decoupling from it is the point of the bundle.
REQUIRED_FIXTURES = (
    "golden_node_eval.yaml",
    "mini_casl_10runs.h5",
    "preprocessor.json",
    "golden_node_preds.npy",
    "golden_node_trues.npy",
    "golden_node_metrics.json",
    "golden_matrix_A.npy",
    "best-matrix_ode_7x7_breeding_chain-epoch=2367.ckpt",
)


# Tolerances for comparing against the committed fixtures.
#
# TRAJECTORY_* is deliberately looser than the rest. dopri5 chooses its steps
# adaptively from an error estimate, so a last-bit arithmetic difference can tip
# it into a different step sequence and the whole trajectory inherits that. The
# fixtures were generated with the CUDA build of torch on Windows; CI installs
# the CPU wheel on Linux, a different oneDNN/MKL, and the two disagree by up to
# ~2e-3 relative. That is float noise amplified by step selection, not a change
# in behaviour — confirmed by the fact that the depletion-matrix probes, which
# involve no solver at all, still agree to 1e-5, and the aggregate metrics to
# 1e-3.
#
# It is still a real anchor: every genuine regression seen so far moved
# trajectories by >= 2e-2, four times this bound.
TRAJECTORY_RTOL, TRAJECTORY_ATOL = 5e-3, 1e-6

# Averaging 7000 samples cancels most of that scatter, so the metrics pin five
# times tighter than the trajectories they are computed from — but not
# arbitrarily tighter, and the three platforms do not agree equally. The Linux
# CPU wheel reproduces the fixtures bit for bit; the windows-latest runner, a
# different CPU to the machine that generated them, moves per-isotope MAE and
# MARE by up to 1.1e-4 relative. This bound clears that by ~9x while still
# sitting 20x below the smallest genuine regression observed.
METRIC_RTOL = 1e-3
# No ODE solve, so no adaptive amplification — this one stays tight.
MATRIX_RTOL, MATRIX_ATOL = 1e-5, 1e-7


def fixtures_present() -> bool:
    return all(os.path.exists(os.path.join(FIX, f)) for f in REQUIRED_FIXTURES)


def build_inference_cfg(preprocessor_path=PREPROCESSOR, output_dir=None):
    """Config for evaluating the frozen NODE on the 10-run mini fixture.

    The fitted scalers are loaded from *preprocessor_path* and the training
    file is never opened — there is no other way to run inference, by design.
    `make_golden.freeze_preprocessor` is what produces that file.

    Every path is set here, absolute. The frozen fixture config is never asked
    to supply one, which is why it survives untouched across the move to a
    path-free config schema — and why the goldens it pins cannot move with it.
    """
    cfg = OmegaConf.load(CONFIG)
    # The frozen config carries no runtime block — a config describes a model.
    # 42 is the seed the committed goldens were generated under; it reaches the
    # NODE datamodule's run permutation, so it is load-bearing, not decoration.
    cfg.runtime = {
        "device": "cpu",
        "num_workers": 0,
        "analyses": False,
        "seed": 42,
        "output_dir": str(output_dir) if output_dir is not None else "results",
    }
    cfg.dataset.path_to_inference_data = MINI_H5
    cfg.dataset.preprocessor_path = preprocessor_path
    # Empty, not absent: the datamodules read it by attribute even in inference
    # mode, where nothing opens it. `read_bundle` blanks it the same way, so a
    # bundle can never reopen the file it was trained on.
    cfg.dataset.path_to_data = ""
    return cfg


def run_inference(cfg):
    """Return (model, datamodule, preds, trues) for a frozen-checkpoint run."""
    import nuclear_surrogates.datamodule.neural_ode_datamodule as node_dm
    from nuclear_surrogates.models.modes import load_checkpoint_into_model
    from nuclear_surrogates.models.neural_ode import NODE_Model

    torch.manual_seed(0)
    dm = node_dm.NODE_Datamodule(cfg)
    dm.inference_mode = True
    model = load_checkpoint_into_model(NODE_Model(cfg), CKPT)
    trainer = L.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
    batches = trainer.predict(model, datamodule=dm)

    preds = torch.cat([b["pred"] for b in batches]).numpy()
    trues = torch.cat([b["true"] for b in batches]).numpy()
    return model, dm, preds, trues


def matrix_probes(model):
    """Fixed (state, forcing) probes of the learned depletion matrix.

    Seeded here rather than at the call site so the generator and the test
    cannot drift apart.
    """
    torch.manual_seed(0)
    y_probe = torch.rand(3, 7)
    f_probe = torch.rand(3, 1)
    return model.func.build_matrix(f_probe, y_probe).detach().numpy()
