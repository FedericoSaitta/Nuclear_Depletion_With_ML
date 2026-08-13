"""Silence third-party import-time noise this project cannot act on.

Both warnings below are emitted while `lightning` is being imported, so
`silence_import_noise()` has to run *before* that import — which is why the
entry points call it above their own imports.
"""

import logging
import warnings


def silence_import_noise() -> None:
    """Quieten warnings that are neither actionable nor informative here."""
    # `lightning`'s model summary imports `torch.utils.flop_counter`, which warns
    # at import time when triton is missing. Triton has no Windows wheel, and a
    # small MLP under an ODE solver has no triton kernels to count — the FLOP
    # total the summary prints is 0 either way.
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

    # `lightning.pytorch.utilities._pytree` builds a `LeafSpec()`, which torch
    # now deprecates. It is lightning's call to update, not ours.
    warnings.filterwarnings(
        "ignore",
        message=r".*isinstance\(treespec, LeafSpec\).*is deprecated.*",
    )
