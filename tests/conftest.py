"""Skip OpenMC-dependent tests when running in the ML environment.

The repo has two environments (OUTPUT.md §4): .venv has no OpenMC by design.
Collection must not fail there.
"""
import importlib.util

import pytest

collect_ignore = []
if importlib.util.find_spec("openmc") is None:
    collect_ignore = ["test.py", "pinModel_Test.py",
                      "pinModelDepletion_Test.py", "variableDepletion.py"]


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("openmc") is not None:
        return
    skip = pytest.mark.skip(reason="OpenMC not installed (ML environment)")
    for item in items:
        if "openmc" in item.keywords:
            item.add_marker(skip)
