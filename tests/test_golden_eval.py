"""Golden regression for the evaluation outputs that become paper figures.

`test_golden.py` pins the model's forward pass. This pins what is computed
*from* it: the teacher-forced rollout, the error-growth curves, and the MARE
comparison. Those are the numbers in the manuscript, and they are produced by
the analysis code that was extracted out of `neural_ode.py` — exactly the
refactor these tests exist to make safe.

The maths lives in `nuclear_surrogates.evaluation` so it can be called on plain
arrays; the model methods call the same functions, so pinning them here pins the
production path rather than a reimplementation of it.
"""

import json
import os

import numpy as np
import pytest
from golden_setup import (
    FIX,
    METRIC_RTOL,
    TRAJECTORY_ATOL,
    TRAJECTORY_RTOL,
    fixtures_present,
)

GOLDEN_TF = os.path.join(FIX, "golden_node_tf_preds.npy")
GOLDEN_EVAL = os.path.join(FIX, "golden_node_eval.json")

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(not fixtures_present(), reason="golden fixtures are missing"),
]


@pytest.fixture(scope="module")
def evaluated(frozen_node_run):
    """Frozen NODE -> unscaled AR + TF trajectories, the inputs to every figure."""
    import torch

    model, dm, ar_scaled, trues_scaled = frozen_node_run

    with torch.no_grad():
        tf_scaled = model._teacher_forced_predictions(
            dm.test_trajs[:, :, : dm.n_input_features].numpy(), trues_scaled
        )

    pre = dm.preprocessor
    n_targets = trues_scaled.shape[-1]

    def unscale(a):
        return pre.inverse_targets(a.reshape(-1, n_targets)).reshape(a.shape)

    return {
        "trues": unscale(trues_scaled),
        "ar": unscale(ar_scaled),
        "tf": unscale(tf_scaled),
        "tf_scaled": tf_scaled,
        "target_names": pre.target_names,
    }


def _require(path):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} not generated yet — run make_golden.py")


def test_teacher_forced_rollout_unchanged(evaluated):
    """The TF rollout is a separate model computation from the AR trajectories
    already pinned in test_golden.py — 99 single-step solves per run."""
    _require(GOLDEN_TF)
    np.testing.assert_allclose(
        evaluated["tf_scaled"],
        np.load(GOLDEN_TF),
        rtol=TRAJECTORY_RTOL,
        atol=TRAJECTORY_ATOL,
    )


def test_error_growth_curves_unchanged(evaluated):
    from nuclear_surrogates import evaluation

    _require(GOLDEN_EVAL)
    with open(GOLDEN_EVAL) as f:
        golden = json.load(f)

    curves = evaluation.error_growth_curves(
        evaluated["trues"], evaluated["ar"], evaluated["tf"]
    )
    for idx, name in enumerate(evaluated["target_names"]):
        for key in ("avg_ar_mae", "avg_tf_mae", "avg_ar_male", "avg_tf_male"):
            np.testing.assert_allclose(
                curves[idx][key],
                golden["error_growth"][name][key],
                rtol=TRAJECTORY_RTOL,
                atol=1e-12,
                err_msg=f"{name}/{key} moved",
            )


def test_mare_comparison_unchanged(evaluated):
    from nuclear_surrogates import evaluation

    _require(GOLDEN_EVAL)
    with open(GOLDEN_EVAL) as f:
        golden = json.load(f)

    comparison = evaluation.mare_comparison(
        evaluated["trues"], evaluated["ar"], evaluated["tf"]
    )
    for idx, name in enumerate(evaluated["target_names"]):
        np.testing.assert_allclose(
            [comparison[idx]["mare_tf"], comparison[idx]["mare_ar"]],
            [golden["mare"][name]["mare_tf"], golden["mare"][name]["mare_ar"]],
            rtol=METRIC_RTOL,
            err_msg=f"{name} MARE moved",
        )


# Channels whose timescale is much longer than the 10-day sampling interval, so
# autoregressive error genuinely compounds along the rollout.
COMPOUNDING = ("U238", "Pu240", "Pu241", "Pu242")

# U239 (t½ = 23 min) and Np239 (t½ = 2.36 d) are at secular equilibrium at every
# sampled point, so their concentration is slaved to the local capture rate
# rather than to the trajectory's history — feeding predictions back in cannot
# accumulate error for them. Pu239, the first long-lived member, is buffered by
# the same effect. This is the identifiability limit AUDIT.md P3 describes,
# stated here as a test rather than a caveat in prose.
EQUILIBRIUM_BUFFERED = ("U239", "Np239", "Pu239")


def test_error_compounds_under_rollout_only_where_physics_allows(evaluated):
    """A property, not a golden.

    Aggregate compounding is what a swapped AR/TF pair would break, and a pinned
    number alone would not reveal that. The per-channel split additionally
    encodes *which* channels the data can resolve.
    """
    from nuclear_surrogates import evaluation

    curves = evaluation.error_growth_curves(
        evaluated["trues"], evaluated["ar"], evaluated["tf"]
    )
    final = {
        name: (curves[i]["avg_ar_mae"][-1], curves[i]["avg_tf_mae"][-1])
        for i, name in enumerate(evaluated["target_names"])
    }

    total_ar = sum(ar for ar, _ in final.values())
    total_tf = sum(tf for _, tf in final.values())
    assert total_ar > total_tf, (
        f"autoregressive total error {total_ar:.3e} is below teacher-forced "
        f"{total_tf:.3e} — the two rollouts look swapped"
    )

    for name in COMPOUNDING:
        ar, tf = final[name]
        assert ar > tf, (
            f"{name} has a timescale far longer than the 10-day sampling "
            f"interval, so rollout error must compound: AR {ar:.3e} <= TF {tf:.3e}"
        )

    for name in EQUILIBRIUM_BUFFERED:
        ar, tf = final[name]
        assert 0.5 < ar / tf < 2.0, (
            f"{name} is at secular equilibrium on this grid, so AR and TF should "
            f"stay comparable; got a ratio of {ar / tf:.2f}"
        )


def test_error_growth_matches_a_hand_computation(evaluated):
    """Guards the extraction itself: the pure function must still compute what
    the inline code in `_plot_error_growth` used to."""
    from nuclear_surrogates import evaluation

    curves = evaluation.error_growth_curves(
        evaluated["trues"], evaluated["ar"], evaluated["tf"]
    )
    idx = 0
    gt = evaluated["trues"][:, :, idx]
    ar = evaluated["ar"][:, :, idx]
    expected = np.mean(np.abs(ar - gt), axis=0)
    np.testing.assert_allclose(curves[idx]["avg_ar_mae"], expected, rtol=0, atol=0)
