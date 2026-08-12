"""Every config in configs/ must actually be runnable.

These are cheap, import-light checks that catch the failure mode this repo has
hit repeatedly: a config that looks complete, is committed, and then dies on
first use because a key the code dereferences is absent — or worse, *runs* with
a silently substituted default.

Two of the traps here are the silent fallbacks AUDIT.md flags: `get_scaler`
returns a NoOpScaler for an unrecognised name and `get_activation` returns ReLU,
each logging an error and carrying on. A typo in a scaler name therefore trains
a model on unscaled data and reports it as a result.
"""

import os

import pytest
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(REPO, "configs")

CONFIG_FILES = sorted(
    f for f in os.listdir(CONFIG_DIR) if f.endswith((".yaml", ".yml"))
)

# Keys dereferenced unconditionally by main.py, modes.py and both datamodules.
REQUIRED = [
    ("dataset", "path_to_data"),
    ("dataset", "fraction_of_data"),
    ("dataset", "inputs"),
    ("dataset", "targets"),
    ("model", "name"),
    ("model", "layers"),
    ("model", "activation"),
    ("model", "output_activation"),
    ("model", "dropout_probability"),
    ("model", "residual_connections"),
    ("train", "loss"),
    ("train", "learning_rate"),
    ("train", "weight_decay"),
    ("train", "lr_scheduler_patience"),
    ("train", "num_epochs"),
    ("train", "grad_clip"),  # modes.py:_build_trainer, no default
    ("train", "drop_last"),
    ("runtime", "mode"),
    ("runtime", "model"),  # main.py:MODELS[...], no default
    ("runtime", "device"),
    ("runtime", "seed"),
    ("runtime", "num_workers"),
    ("runtime", "output_dir"),  # else results land CWD-relative
    ("runtime", "model_database"),
]

VALID_SCALERS = {
    "minmax",
    "standard",
    "robust",
    "maxabs",
    "normalizer",
    "quantile",
    "power",
    "none",
}
VALID_ACTIVATIONS = {
    "relu",
    "tanh",
    "sigmoid",
    "leaky_relu",
    "elu",
    "gelu",
    "selu",
    "softplus",
    "none",
}
VALID_LOSSES = {"mse", "mae", "huber", "smooth_l1"}
VALID_MODES = {"train", "train_from_ckp", "inference"}
VALID_MODELS = {"DNN", "NODE"}


def load(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def test_configs_directory_is_not_empty():
    assert CONFIG_FILES, "no configs found — has configs/ moved?"


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_required_keys_present(name):
    cfg = load(name)
    missing = [
        f"{section}.{key}"
        for section, key in REQUIRED
        if not isinstance(cfg.get(section), dict) or key not in cfg[section]
    ]
    assert not missing, f"{name} is missing {missing} — it would crash on first run"


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_enumerated_values_are_recognised(name):
    """Guards the silent fallbacks: an unknown scaler or activation does not
    raise, it substitutes a default and logs. That must not reach a config."""
    cfg = load(name)

    assert cfg["runtime"]["mode"] in VALID_MODES
    assert cfg["runtime"]["model"] in VALID_MODELS
    assert str(cfg["train"]["loss"]).lower() in VALID_LOSSES
    assert str(cfg["model"]["activation"]).lower() in VALID_ACTIVATIONS
    assert str(cfg["model"]["output_activation"]).lower() in VALID_ACTIVATIONS

    for side in ("inputs", "targets"):
        for column, scaler in cfg["dataset"][side].items():
            assert str(scaler).lower() in VALID_SCALERS, (
                f"{name}: {side}.{column} uses scaler {scaler!r}, which "
                f"get_scaler does not recognise — it would silently become a "
                f"NoOpScaler and the column would go unscaled"
            )


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_runtime_values_are_usable(name):
    cfg = load(name)

    workers = cfg["runtime"]["num_workers"]
    assert (
        isinstance(workers, int) and workers >= 0
    ), f"{name}: num_workers={workers!r}; DataLoader rejects negative values"
    assert cfg["runtime"]["device"] in {"cpu", "cuda", "auto", "gpu", "mps"}
    assert isinstance(cfg["runtime"]["seed"], int)
    assert str(cfg["dataset"]["path_to_data"]).endswith((".h5", ".csv"))

    frac = cfg["dataset"]["fraction_of_data"]
    assert 0 < frac <= 1.0, f"{name}: fraction_of_data={frac}"


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_node_configs_carry_solver_settings(name):
    cfg = load(name)
    if cfg["runtime"]["model"] != "NODE":
        pytest.skip("DNN config")

    for key in ("solver", "rtol", "atol", "step_size"):
        assert key in cfg["train"], f"{name}: NODE needs train.{key}"

    if cfg["model"].get("matrix_ode"):
        entries = cfg["model"].get("matrix_zero_entries")
        assert entries, f"{name}: matrix_ode is on but matrix_zero_entries is empty"
        n_targets = len(cfg["dataset"]["targets"])
        for i, j in entries:
            assert 0 <= i < n_targets and 0 <= j < n_targets, (
                f"{name}: matrix_zero_entries has ({i},{j}) but there are only "
                f"{n_targets} targets"
            )


@pytest.mark.parametrize("name", [f for f in CONFIG_FILES if f.startswith("smoke_")])
def test_smoke_configs_point_at_committed_fixtures(name):
    """The smoke configs are the 'runs anywhere' path, so their data must be in
    the repo — not in the gitignored datasets/ directory."""
    cfg = load(name)
    resolved = os.path.normpath(
        os.path.join(CONFIG_DIR, cfg["dataset"]["path_to_data"])
    )
    assert os.path.isfile(resolved), f"{name}: {resolved} is not committed"
    assert "tests" in resolved and "fixtures" in resolved
