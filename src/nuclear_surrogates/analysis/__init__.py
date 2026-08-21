"""Post-hoc analysis of a trained Neural ODE.

These are the passes that run once, after training, to produce the figures and
tables in the write-up: teacher-forced rollout, Jacobian sensitivity, the
learned depletion matrix, and per-step permutation importance.

They live outside the LightningModule on purpose. Each takes an explicit
`SolveContext` — the integrator, the right-hand side, the time grid and the
target unscaler — rather than reaching into a model, so every pass can be
called on plain arrays and tested without a Trainer. That is the same split
`evaluation.py` already applies to the metric maths.
"""

from nuclear_surrogates.analysis.depletion_matrix import (
    depletion_matrix_analysis,
    unscaling_matrix,
)
from nuclear_surrogates.analysis.importance import stepwise_importance
from nuclear_surrogates.analysis.jacobian import (
    forcing_jacobian,
    jacobian_analysis,
    state_jacobian,
)
from nuclear_surrogates.analysis.rollout import (
    SolveContext,
    single_step_batch,
    teacher_forced_predictions,
)

__all__ = [
    "SolveContext",
    "depletion_matrix_analysis",
    "forcing_jacobian",
    "jacobian_analysis",
    "single_step_batch",
    "state_jacobian",
    "stepwise_importance",
    "teacher_forced_predictions",
    "unscaling_matrix",
]
