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
CONFIG = os.path.join(REPO, "configs", "main_config.yaml")
CKPT = os.path.join(FIX, "best-matrix_ode_7x7_breeding_chain-epoch=2367.ckpt")
MINI_H5 = os.path.join(FIX, "mini_casl_10runs.h5")
PREPROCESSOR = os.path.join(FIX, "preprocessor.json")

# Only `make_golden.py` needs this: the fixture preprocessor is derived from it
# once, and committed, so the tests never touch it again.
TRAIN_H5 = os.path.join(REPO, "datasets", "casl_3305_runs_inter.h5")

# Fixtures the golden tests require. The full training dataset is deliberately
# NOT in this list — decoupling from it is the point of the bundle.
REQUIRED_FIXTURES = (
    "mini_casl_10runs.h5",
    "preprocessor.json",
    "golden_node_preds.npy",
    "golden_node_trues.npy",
    "golden_node_metrics.json",
    "golden_matrix_A.npy",
    "best-matrix_ode_7x7_breeding_chain-epoch=2367.ckpt",
)


def fixtures_present() -> bool:
    return all(os.path.exists(os.path.join(FIX, f)) for f in REQUIRED_FIXTURES)


def build_inference_cfg(preprocessor_path=PREPROCESSOR, train_h5=None, output_dir=None):
    """Config for evaluating the frozen NODE on the 10-run mini fixture.

    With *preprocessor_path* the fitted scalers are loaded and the training file
    is never opened. Passing ``preprocessor_path=None`` plus *train_h5* selects
    the historical path that re-fits them — which is how the fixture
    preprocessor itself is generated.
    """
    from nuclear_surrogates.utils.paths import resolve_config_paths

    cfg = OmegaConf.load(CONFIG)
    resolve_config_paths(cfg, CONFIG)
    cfg.runtime.update(mode="inference", device="cpu", num_workers=0, plots=False)
    if output_dir is not None:
        cfg.runtime.output_dir = str(output_dir)
    cfg.dataset.path_to_inference_data = MINI_H5

    if preprocessor_path:
        cfg.dataset.preprocessor_path = preprocessor_path
    else:
        cfg.dataset.path_to_data = train_h5 or TRAIN_H5
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
    return model.func._build_matrix(f_probe, y_probe).detach().numpy()
