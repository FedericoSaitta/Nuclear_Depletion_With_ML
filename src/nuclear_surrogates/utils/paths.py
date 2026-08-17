"""Where a run's output goes.

Every path a run uses now arrives as a command-line argument and is resolved
against the working directory, like any other tool. There used to be a second
rule — paths inside a config file were anchored at that config file — which is
gone along with the path keys themselves; it cost two silently-wrong paths and
one key (`ckp_path`) that was documented as following the rule and did not.
"""

import os


def result_dir(cfg) -> str:
    """Return ``<runtime.output_dir>/<model.name>/``, creating it if absent."""
    path = os.path.join(cfg.runtime.output_dir, cfg.model.name) + os.sep
    os.makedirs(path, exist_ok=True)
    return path
