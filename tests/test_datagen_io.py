"""Tests for the data-generation half that do NOT need OpenMC.

This half of the repo had no coverage at all, for a structural reason: every
module imported OpenMC at import time, and CI has neither an OpenMC build nor
the 7 GB nuclear-data library. The pure layer — `config`, `dataset_io`,
`power_history`, `nuclides`, and the schema half of `tally_io` — exists so that
the parts which *can* be checked without a reactor are checked.

What matters most here is the format contract joining the two halves: the
generators write HDF5 and the ML pipeline reads it, and until now nothing
asserted the two agreed.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

# This module runs in BOTH environments — everything it imports is in the shared
# dependency set. The two tests that check the format contract are the exception:
# they need the ML side's reader, and therefore torch.
requires_ml = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="needs the `ml` extra — this is the simulation environment",
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATAGEN = os.path.join(REPO, "data_generation")

# data_generation/ is not a package; its own entry points import siblings by
# having their directory on sys.path. Do the same rather than restructuring it,
# because that structure is what makes `python data_generation/datagen.py` work.
if DATAGEN not in sys.path:
    sys.path.insert(0, DATAGEN)

import config as datagen_config  # noqa: E402
import dataset_io  # noqa: E402
import power_history  # noqa: E402
import tally_io  # noqa: E402
from nuclides import CAPTURE_NUCLIDES, FISSION_NUCLIDES  # noqa: E402

POWER_CSV = os.path.join(DATAGEN, "beavrs_cycle1_power.csv")
CONFIG_DIR = os.path.join(DATAGEN, "configs")


def _run(rows=4, label="run_test"):
    """A miniature version of what `depletion_results_frame` produces."""
    return {
        "run_label": [label] * rows,
        "time_days": np.arange(rows, dtype=float) * 10.0,
        "k_eff": np.linspace(1.2, 1.0, rows),
        "power_W_g": [30.0, 31.0, 32.0, float("nan")][:rows],
        "U238": np.linspace(2.2e-2, 2.1e-2, rows),
    }


# ── The format contract with the ML side ─────────────────────────────────────


@requires_ml
def test_written_runs_are_readable_by_the_ml_pipeline(tmp_path):
    """The generators write it; `read_h5_file` reads it. Nothing checked that.

    This is the seam between the two halves of the repo — two environments, two
    dependency sets, no shared import. A silent disagreement here means training
    on a file that is not what the simulation produced.
    """
    from nuclear_surrogates.datamodule.dataset_helper import read_h5_file

    data = _run()
    path = tmp_path / "worker_1.h5"
    dataset_io.write_run_h5(data, path)

    df = read_h5_file(str(path))

    assert df.columns == list(data), "column order must survive the round trip"
    assert df["run_label"].to_list() == data["run_label"]
    np.testing.assert_allclose(df["time_days"].to_numpy(), data["time_days"])
    np.testing.assert_allclose(df["U238"].to_numpy(), data["U238"])
    # NaN pads the extra results point; it must arrive as NaN, not as 0.
    assert np.isnan(df["power_W_g"].to_numpy()[-1])


@requires_ml
def test_the_two_readers_agree(tmp_path):
    """`dataset_io.read_run_h5` duplicates `read_h5_file` because the simulation
    environment has no torch. Pin them together so the copy cannot drift."""
    from nuclear_surrogates.datamodule.dataset_helper import read_h5_file

    path = tmp_path / "worker_1.h5"
    dataset_io.write_run_h5(_run(), path)

    mine = dataset_io.read_run_h5(path)
    theirs = read_h5_file(str(path))

    assert list(mine) == theirs.columns
    for col in mine:
        if col == "run_label":
            assert list(mine[col]) == theirs[col].to_list()
        else:
            np.testing.assert_allclose(
                np.asarray(mine[col], dtype=float), theirs[col].to_numpy()
            )


def test_columns_of_different_lengths_are_refused(tmp_path):
    """A per-step column that was not padded to the results grid is a bug that
    would otherwise reach the dataset as a silent truncation."""
    data = _run()
    data["power_W_g"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="differing lengths"):
        dataset_io.write_run_h5(data, tmp_path / "bad.h5")


# ── Merging ──────────────────────────────────────────────────────────────────


def test_merge_concatenates_workers_in_order(tmp_path):
    a = tmp_path / "worker_1.h5"
    b = tmp_path / "worker_2.h5"
    dataset_io.write_run_h5(_run(rows=4, label="a"), a)
    dataset_io.write_run_h5(_run(rows=4, label="b"), b)

    merged = dataset_io.read_run_h5(dataset_io.merge_h5([a, b], tmp_path / "all.h5"))

    assert merged["run_label"] == ["a"] * 4 + ["b"] * 4
    assert len(merged["time_days"]) == 8


def test_merge_keeps_a_column_only_one_worker_has(tmp_path):
    """A nuclide the chain produced for one worker and not another must not be
    dropped — the absent rows are NaN, which is visible downstream."""
    a_data, b_data = _run(label="a"), _run(label="b")
    b_data["Pu239"] = np.linspace(0.0, 1e-4, 4)

    a, b = tmp_path / "worker_1.h5", tmp_path / "worker_2.h5"
    dataset_io.write_run_h5(a_data, a)
    dataset_io.write_run_h5(b_data, b)

    merged = dataset_io.read_run_h5(dataset_io.merge_h5([a, b], tmp_path / "all.h5"))

    assert "Pu239" in merged
    assert np.all(np.isnan(merged["Pu239"][:4]))
    np.testing.assert_allclose(merged["Pu239"][4:], b_data["Pu239"])


def test_drop_empty_columns_keeps_all_nan_but_drops_all_zero():
    """All-zero means the chain tracks a nuclide this run never made. All-NaN
    means a measurement is missing, which is not the same thing."""
    data = {
        "run_label": ["a", "a"],
        "U238": [1.0, 2.0],
        "Xe135": [0.0, 0.0],
        "flux": [float("nan"), float("nan")],
    }
    kept = dataset_io.drop_empty_columns(data)

    assert "Xe135" not in kept
    assert "flux" in kept
    assert "U238" in kept and "run_label" in kept


# ── Provenance ───────────────────────────────────────────────────────────────


def test_manifest_records_the_seeds_and_the_chain(tmp_path):
    """A published dataset without this is a table nobody can regenerate."""
    chain = tmp_path / "chain.xml"
    chain.write_text("<depletion_chain/>")

    dataset_io.write_manifest(
        str(tmp_path),
        config={"enrichment": 3.1},
        worker_configs=[{"worker_id": "1_abc", "seed": 11}],
        chain_file=str(chain),
        extra={"pipeline": "test"},
    )

    manifest = json.loads((tmp_path / dataset_io.MANIFEST_NAME).read_text())

    assert manifest["config"] == {"enrichment": 3.1}
    assert manifest["workers"] == [{"worker_id": "1_abc", "seed": 11}]
    assert manifest["chain_file"]["name"] == "chain.xml"
    assert len(manifest["chain_file"]["sha256"]) == 64
    assert manifest["pipeline"] == "test"
    assert manifest["created_utc"]


# ── Power history ────────────────────────────────────────────────────────────


def test_the_committed_beavrs_history_parses():
    days, percents = power_history.parse_power_history(POWER_CSV)

    # 527 measured points over a 575-day cycle — the sampling is not daily
    # throughout, which is exactly why `resample_power` interpolates.
    assert len(days) == len(percents) == 527
    assert days[0] == 0.0
    assert days[-1] == 575.0
    # Strictly increasing: `np.interp` silently returns nonsense otherwise.
    assert np.all(np.diff(days) > 0)
    # Measured percentages of rated power, so within [0, 100].
    assert percents.min() >= 0.0
    assert percents.max() <= 100.0


def test_the_raw_beavrs_release_format_also_parses(tmp_path):
    """The committed file is bare rows, but the table as downloaded carries a
    banner and a header. Both must work, or a fresh download needs hand-editing.
    """
    path = tmp_path / "release.csv"
    path.write_text("Cycle 1\nDay, Percent\n0.0,50.0\n1.0,100.0\n\nCycle 2\n0.0,1.0\n")

    days, percents = power_history.parse_power_history(path)

    assert list(days) == [0.0, 1.0]
    assert list(percents) == [50.0, 100.0]


def test_a_file_with_no_rows_is_refused(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("Cycle 1\nDay, Percent\n")
    with pytest.raises(ValueError, match="no `day,percent` rows"):
        power_history.parse_power_history(path)


def test_resample_converts_percent_to_specific_power(tmp_path):
    path = tmp_path / "h.csv"
    path.write_text("0.0,0.0\n1.0,100.0\n2.0,50.0\n")

    powers, times = power_history.resample_power(
        path, rated_power_W_g=40.0, dt_days=1.0
    )

    assert list(times) == [0.0, 1.0]
    # 0 % -> 0 W/g, 100 % -> the rated power.
    np.testing.assert_allclose(powers, [0.0, 40.0])


def test_resample_windows_on_absolute_cycle_days(tmp_path):
    """The zoom pipeline's window must keep its true position in the cycle: a
    trajectory relabelled to start at day 0 is not comparable to the daily run.
    """
    path = tmp_path / "h.csv"
    path.write_text("0.0,0.0\n10.0,100.0\n")

    powers, times = power_history.resample_power(
        path, rated_power_W_g=10.0, dt_days=1.0, start_day=4.0, end_day=6.0
    )

    assert list(times) == [4.0, 5.0]
    np.testing.assert_allclose(powers, [4.0, 5.0])


def test_a_window_narrower_than_one_step_is_refused(tmp_path):
    path = tmp_path / "h.csv"
    path.write_text("0.0,0.0\n10.0,100.0\n")
    with pytest.raises(ValueError, match="steps"):
        power_history.resample_power(
            path, rated_power_W_g=10.0, dt_days=1.0, start_day=1.0, end_day=1.2
        )


# ── Tally schema ─────────────────────────────────────────────────────────────


def test_the_tally_placeholders_cover_exactly_the_declared_columns():
    """`step_keys` drives the dataset's columns and the placeholder rows fill
    them. If the two disagree, a decay-only step writes a short row."""
    keys = set(tally_io.step_keys())
    zero = tally_io.zero_tally()
    nan = tally_io.nan_tally()

    assert set(zero) == set(nan)
    # The placeholders carry no power fractions — those are computed, not read.
    fraction_keys = {f"{n}_fission_power_frac" for n in FISSION_NUCLIDES}
    assert keys - fraction_keys == set(zero)

    assert all(v == 0.0 for v in zero.values())
    assert all(np.isnan(v) for v in nan.values())
    assert f"{CAPTURE_NUCLIDES[0]}_capture" in zero


def test_fission_power_fractions_sum_to_one():
    tally = dict.fromkeys([f"{n}_fission" for n in FISSION_NUCLIDES], 1.0)
    fractions = tally_io.compute_fission_power_fractions(tally)

    assert sum(fractions.values()) == pytest.approx(1.0)
    # Weighted by Q-value, so the highest-Q nuclide takes the largest share.
    assert fractions["Pu241_fission_power_frac"] > fractions["U235_fission_power_frac"]


def test_fission_power_fractions_are_zero_at_zero_power():
    """A decay-only step divides by a zero total; it must not produce NaN."""
    fractions = tally_io.compute_fission_power_fractions(tally_io.zero_tally())
    assert all(v == 0.0 for v in fractions.values())

    fractions = tally_io.compute_fission_power_fractions(tally_io.nan_tally())
    assert all(v == 0.0 for v in fractions.values())


# ── Config loading ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "casl_pincell.yaml",
        "beavrs_quarterpin.yaml",
        "beavrs_zoom.yaml",
        "smoke_pincell.yaml",
    ],
)
def test_every_shipped_config_loads(name):
    """These are the files a Zenodo record points at. A broken one is only
    discovered when a multi-day job starts."""
    cfg = datagen_config.load(os.path.join(CONFIG_DIR, name))

    assert cfg["enrichment"] > 0
    assert cfg["particles"] > 0
    assert cfg["chain_file"].endswith(".xml")
    assert len(cfg["geometry_radii"]) >= 2


def test_an_unknown_key_is_refused(tmp_path):
    """The failure this catches: a typo that a job discovers after a day in the
    queue, having silently run with the default instead."""
    path = tmp_path / "c.yaml"
    path.write_text(
        "model: {enrichment: 3.1, fuel_density: 10.4, geometry_radii: [1, 2],\n"
        "        geometry_pitch: 1.0, particels: 500}\n"
        "transport: {particles: 100, batches: 10, inactive: 2}\n"
        "depletion: {chain_file: chain.xml}\n"
    )
    with pytest.raises(SystemExit, match="particels"):
        datagen_config.load(path)


def test_a_missing_required_key_is_refused(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "model: {enrichment: 3.1, fuel_density: 10.4, geometry_radii: [1, 2],\n"
        "        geometry_pitch: 1.0}\n"
        "transport: {particles: 100, batches: 10}\n"
        "depletion: {chain_file: chain.xml}\n"
    )
    with pytest.raises(SystemExit, match="inactive"):
        datagen_config.load(path)


def test_an_unknown_section_is_refused(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "model: {enrichment: 3.1, fuel_density: 10.4, geometry_radii: [1, 2],\n"
        "        geometry_pitch: 1.0}\n"
        "transport: {particles: 100, batches: 10, inactive: 2}\n"
        "depletion: {chain_file: chain.xml}\n"
        "runtime: {device: cpu}\n"
    )
    with pytest.raises(SystemExit, match="runtime"):
        datagen_config.load(path)


# ── Worker seeding ───────────────────────────────────────────────────────────


def test_the_same_master_seed_gives_the_same_worker_seeds():
    """The property that makes a dataset reproducible. It was documented in a
    docstring and checked by nothing."""
    base = {"enrichment": 3.1}

    first = datagen_config.create_worker_configs(base, 4, master_seed=7)
    second = datagen_config.create_worker_configs(base, 4, master_seed=7)

    assert [w["seed"] for w in first] == [w["seed"] for w in second]
    # Worker IDs come from uuid4, deliberately: two runs at the same seed must
    # not overwrite each other's output files.
    assert [w["worker_id"] for w in first] != [w["worker_id"] for w in second]


def test_different_master_seeds_give_different_worker_seeds():
    """`run_sweep` offsets the master seed per round precisely to avoid this
    collapsing — every round used to redraw the same histories."""
    base = {"enrichment": 3.1}

    first = datagen_config.create_worker_configs(base, 4, master_seed=7)
    second = datagen_config.create_worker_configs(base, 4, master_seed=8)

    assert [w["seed"] for w in first] != [w["seed"] for w in second]


def test_worker_configs_do_not_share_state():
    """Each worker mutates its own config in its own process; a shallow copy
    that leaked would cross-contaminate the seeds."""
    base = {"enrichment": 3.1}
    workers = datagen_config.create_worker_configs(base, 3, master_seed=1)

    assert len({w["seed"] for w in workers}) == 3
    assert "worker_id" not in base, "the base config must not be mutated"
