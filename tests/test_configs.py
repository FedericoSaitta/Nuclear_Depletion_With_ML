"""Every config in configs/ must actually be runnable.

These are cheap, import-light checks that catch the failure mode this repo has
hit repeatedly: a config that looks complete, is committed, and then dies on
first use because a key the code dereferences is absent — or worse, *runs* with
a silently substituted default.

`get_scaler` and `get_activation` now reject an unrecognised name, so a typo
fails at construction rather than training a model on unscaled data. These tests
catch the same typo earlier still — before a config is committed.
"""

import os

import pytest
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(REPO, "configs")

CONFIG_FILES = sorted(
    f for f in os.listdir(CONFIG_DIR) if f.endswith((".yaml", ".yml"))
)

# Keys dereferenced unconditionally by modes.py and both datamodules.
#
# Paths, device, workers, seed and output directories are deliberately absent:
# they are command-line flags with defaults, and a config that declared them
# would be describing a machine rather than a model.
REQUIRED = [
    ("dataset", "fraction_of_data"),
    ("dataset", "inputs"),
    ("dataset", "targets"),
    ("model", "kind"),  # main.py:MODELS[...], no default
    ("model", "name"),
    ("model", "layers"),
    ("model", "activation"),
    ("model", "dropout_probability"),
    ("model", "residual_connections"),
    ("train", "loss"),
    ("train", "learning_rate"),
    ("train", "weight_decay"),
    ("train", "lr_scheduler_patience"),
    ("train", "num_epochs"),
    ("train", "grad_clip"),  # modes.py:_build_trainer, no default
]

# Read by exactly one model's datamodule, so requiring them everywhere would
# force the other's configs to carry a key that cannot affect it.
DNN_ONLY = [
    ("dataset", "target_delta_conc"),  # dnn_datamodule.py:52
    ("train", "drop_last"),  # dnn_datamodule.py:53
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
VALID_KINDS = {"DNN", "NODE"}


def load(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def test_configs_directory_is_not_empty():
    assert CONFIG_FILES, "no configs found — has configs/ moved?"


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_required_keys_present(name):
    cfg = load(name)
    required = list(REQUIRED)
    if cfg.get("model", {}).get("kind") == "DNN":
        required += DNN_ONLY

    missing = [
        f"{section}.{key}"
        for section, key in required
        if not isinstance(cfg.get(section), dict) or key not in cfg[section]
    ]
    assert not missing, f"{name} is missing {missing} — it would crash on first run"


PAPER_CONFIGS = ["dnn.yaml", "dnn_no_state.yaml", "node.yaml"]


def test_the_paper_configs_agree_on_how_runs_are_split():
    """The head-to-head comparison is only paired while these three agree.

    With one strategy, one seed and one dataset the models hold out the same
    runs. Changing any of these back to `sequential_by_run` silently puts the
    DNN on a disjoint test set from the NODE's, which is exactly the defect this
    key was added to remove.
    """
    strategies = {
        name: load(name)["dataset"]["split"].get("strategy") for name in PAPER_CONFIGS
    }
    assert set(strategies.values()) == {"random_by_run"}, strategies

    fractions = {
        name: load(name)["dataset"]["split"]["train"] for name in PAPER_CONFIGS
    }
    assert len(set(fractions.values())) == 1, fractions


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_no_config_declares_a_path_or_a_runtime(name):
    """A config describes the model; the command line describes the run.

    These keys all used to live here, which meant a file could not be moved
    between machines without editing, and two of them (`mode`, `ckp_path`)
    encoded a decision the CLI now makes.
    """
    cfg = load(name)

    assert "runtime" not in cfg, f"{name}: runtime is built from CLI flags"
    strays = [k for k in cfg["dataset"] if "path" in k]
    assert not strays, f"{name}: {strays} — paths are --data and --out"


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_enumerated_values_are_recognised(name):
    """An unrecognised scaler, activation or loss must not reach a config.

    The registries reject one at construction; catching it here says which key
    of which file is wrong instead of failing partway into a run.
    """
    cfg = load(name)

    assert cfg["model"]["kind"] in VALID_KINDS
    assert str(cfg["train"]["loss"]).lower() in VALID_LOSSES
    assert str(cfg["model"]["activation"]).lower() in VALID_ACTIVATIONS
    if "output_activation" in cfg["model"]:
        assert str(cfg["model"]["output_activation"]).lower() in VALID_ACTIVATIONS

    for side in ("inputs", "targets"):
        for column, scaler in cfg["dataset"][side].items():
            assert str(scaler).lower() in VALID_SCALERS, (
                f"{name}: {side}.{column} uses scaler {scaler!r}, which "
                f"get_scaler does not recognise"
            )


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_fraction_of_data_is_a_usable_fraction(name):
    frac = load(name)["dataset"]["fraction_of_data"]
    assert 0 < frac <= 1.0, f"{name}: fraction_of_data={frac}"


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_a_matrix_ode_config_does_not_set_output_activation(name):
    """ODEFuncMatrix hardcodes "none" — it applies its own sign constraints.

    A config that set it would be stating something the code discards, which is
    worse than not stating it: it reads as a knob and is not one.
    """
    cfg = load(name)
    if not cfg["model"].get("matrix_ode"):
        pytest.skip("not a matrix ODE")
    assert "output_activation" not in cfg["model"]


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_node_configs_carry_solver_settings(name):
    cfg = load(name)
    if cfg["model"]["kind"] != "NODE":
        pytest.skip("DNN config")

    for key in ("solver", "rtol", "atol"):
        assert key in cfg["train"], f"{name}: NODE needs train.{key}"

    # Only fixed-step solvers read `step_size`; requiring it everywhere used to
    # enshrine a dead key in every adaptive-solver config.
    if cfg["train"]["solver"] == "rk4":
        assert "step_size" in cfg["train"], f"{name}: rk4 needs train.step_size"

    if cfg["model"].get("matrix_ode"):
        entries = cfg["model"].get("matrix_zero_entries")
        assert entries, f"{name}: matrix_ode is on but matrix_zero_entries is empty"
        n_targets = len(cfg["dataset"]["targets"])
        for i, j in entries:
            assert 0 <= i < n_targets and 0 <= j < n_targets, (
                f"{name}: matrix_zero_entries has ({i},{j}) but there are only "
                f"{n_targets} targets"
            )
