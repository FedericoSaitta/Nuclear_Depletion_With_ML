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

collect_ignore = []
if importlib.util.find_spec("openmc") is None:
    collect_ignore = ["OPENMC_tests/test_openmc_install.py"]


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
