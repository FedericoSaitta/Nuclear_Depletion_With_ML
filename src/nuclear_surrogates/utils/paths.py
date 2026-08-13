"""Path resolution for a run.

Every path in a config file is written *relative to that config file*, never to
the working directory. `resolve_config_paths` rewrites them to absolute paths
once, at load time, so a run started from anywhere reads and writes the same
places. `result_dir` is then the single source of the per-run output directory.
"""

import os
from pathlib import Path

# Config keys holding a filesystem path, as (section, key).
_PATH_KEYS = (
    ("dataset", "path_to_data"),
    ("dataset", "path_to_inference_data"),
    ("runtime", "ckp_path"),
    ("runtime", "output_dir"),
    ("runtime", "model_database"),
)


def resolve_config_paths(cfg, config_path) -> None:
    """Rewrite every path key in *cfg* to an absolute path.

    Relative values are anchored at the directory holding *config_path*;
    absolute values and empty values are left alone. Mutates *cfg* in place.
    """
    anchor = Path(config_path).resolve().parent
    for section, key in _PATH_KEYS:
        if section not in cfg:
            continue
        value = cfg[section].get(key)
        if value:
            cfg[section][key] = str((anchor / value).resolve())


def result_dir(cfg) -> str:
    """Return ``<runtime.output_dir>/<model.name>/``, creating it if absent.

    A config predating ``runtime.output_dir`` falls back to a CWD-relative
    ``results/``, which is what those runs did before the key existed.
    """
    output_dir = cfg.runtime.get("output_dir") or "results"
    path = os.path.join(output_dir, cfg.model.name) + os.sep
    os.makedirs(path, exist_ok=True)
    return path
