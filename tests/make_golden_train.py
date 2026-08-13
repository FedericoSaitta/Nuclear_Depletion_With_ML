"""Regenerate the pinned training loss trajectories.

    uv run --extra ml python tests/make_golden_train.py

Needs no datasets — it trains on the committed 10-run fixture, so anyone can
regenerate it. Unlike the inference goldens, these numbers are only reproducible
on a comparable platform and library set, so the file records which one produced
them.

Regenerating is a deliberate act: it moves the reference these tests compare
against. Say why in the commit message.
"""

import json
import platform
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from test_training import (  # noqa: E402
    CONFIGS,
    GOLDEN_TRAIN,
    epoch_losses,
    load_cfg,
    train,
)


def main():
    # Reduction order depends on thread count, so pin it here and in CI.
    torch.set_num_threads(1)

    payload = {
        "_generated_on": {
            "platform": sys.platform,
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "_note": (
            "Per-epoch training loss on tests/fixtures/mini_casl_10runs.h5 at "
            "seed 0. Compared at rtol=1e-3, which absorbs BLAS-level "
            "differences but not a real change in the training path."
        ),
    }

    with tempfile.TemporaryDirectory() as tmp:
        for kind in CONFIGS:
            model, _, _ = train(load_cfg(kind, Path(tmp) / kind, **{"runtime.seed": 0}))
            losses = epoch_losses(model)
            payload[kind] = {"train_losses": losses}
            print(f"{kind}: {[f'{x:.6g}' for x in losses]}")

    with open(GOLDEN_TRAIN, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {GOLDEN_TRAIN}")


if __name__ == "__main__":
    main()
