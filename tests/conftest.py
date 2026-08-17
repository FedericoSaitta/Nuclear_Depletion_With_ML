"""Shared test setup.

Two jobs: keep OpenMC-dependent tests out of the ML environment, and keep the
suite hermetic — a test run must not write into `results/` and must not need a
display.
"""

import importlib.util
import os

import pytest

from nuclear_surrogates.utils.quiet import silence_import_noise

# Runs before any test module imports lightning, which is when the warnings it
# suppresses would be emitted.
silence_import_noise()

# The two halves of this repo are never installed together (see the `conflicts`
# declaration in pyproject.toml), so in either environment some test modules
# cannot even be imported. Markers are not enough: pytest imports every module
# before it deselects anything, so a module whose *import* fails errors out
# during collection and takes the whole run with it.
#
# Hence collection-time ignores in both directions.
ML_ONLY = [
    "test_bundle.py",  # omegaconf
    "test_golden.py",  # lightning, via golden_setup
    "test_golden_eval.py",  # lightning, via golden_setup
    "test_pipeline_units.py",  # scikit-learn, via data_scalers
    "test_preprocessor.py",  # scikit-learn, via data_scalers
    "test_training.py",  # torch
]

# torch stands in for the whole `ml` extra: its packages are installed together
# or not at all, and torch is the one that unambiguously identifies it.
HAVE_ML = importlib.util.find_spec("torch") is not None

collect_ignore = []
if importlib.util.find_spec("openmc") is None:
    collect_ignore.append("OPENMC_tests/test_openmc_install.py")
if not HAVE_ML:
    collect_ignore.extend(ML_ONLY)


def pytest_configure(config):
    """Force a non-interactive matplotlib backend before anything imports pyplot.

    The datamodules and evaluation code draw figures as a side effect of
    running. On a headless CI runner an interactive backend either fails or
    hangs; Agg always works and never opens a window.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("openmc") is not None:
        return
    skip = pytest.mark.skip(reason="OpenMC not installed (ML environment)")
    for item in items:
        if "openmc" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def frozen_node_run(tmp_path_factory):
    """Evaluate the frozen paper NODE on the mini fixture, once per session.

    Both golden modules build their assertions on this. Integrating 10
    trajectories is the most expensive thing in the suite, so it is shared
    rather than repeated per module.
    """
    from golden_setup import build_inference_cfg, fixtures_present, run_inference

    if not fixtures_present():
        pytest.skip("golden fixtures are missing")

    out = tmp_path_factory.mktemp("frozen_node_run")
    model, dm, preds, trues = run_inference(build_inference_cfg(output_dir=out))
    return model, dm, preds, trues
