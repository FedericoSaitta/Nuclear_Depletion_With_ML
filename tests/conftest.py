"""Skip OpenMC-dependent tests when running in the ML environment.

The repo has two environments (see README, "Generate data with OpenMC"): the
default `.venv` deliberately has no OpenMC, so collection must not fail there.
`collect_ignore` keeps the import out of collection entirely; the `openmc`
marker (declared in pyproject.toml) is what `pytest -m openmc` selects on.
"""

import importlib.util

import pytest

collect_ignore = []
if importlib.util.find_spec("openmc") is None:
    collect_ignore = ["OPENMC_tests/test_openmc_install.py"]


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("openmc") is not None:
        return
    skip = pytest.mark.skip(reason="OpenMC not installed (ML environment)")
    for item in items:
        if "openmc" in item.keywords:
            item.add_marker(skip)
