"""Tests for the Arduino lick-log importer.

Synthetic tests always run. The real-data test uses the first session
imported this way (Cera, 2852L) and skips if that folder is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iflip.align import build_peth
from iflip.io import (
    align_clocks,
    build_session_folder,
    discover_acquisitions,
    load_arduino_log,
    load_session,
    load_tidy,
)
from iflip.io.arduino_lick import first_lick_after, group_deliveries, mark_onsets

FS = 20.0
OFFSET_S = 67.9
SLOPE = 1 - 374e-6  # Arduino clock runs fast, as on the real rig

# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #


def _write_log(path: Path, *, sync_ms, solenoid_ms, lick_ms) -> Path:
    """Arduino log with CR line endings and blank lines, as the rig writes it."""
    lines = ["11 0", "116 15", "117 6", "127 0", "127 2"]
    events = [(1001, t) for t in sync_ms]
    for t in solenoid_ms:
        events += [(18, t), (14, t + 15)]
    events += [(32, t) for t in lick_ms]
    lines += [f"{c} {t}" for c, t in sorted(events, key=lambda e: e[1])]
    lines += ["0 999999"]
    path.write_bytes(("\r\n\r\n".join(lines) + "\r").encode())
    return path


def _write_tidy(folder: Path, stem: str, marks_at_s: np.ndarray, duration_s: float = 600.0):
    t = np.arange(0, duration_s, 1 / FS) + 0.5 / FS
    marks = np.zeros(t.size, dtype=int)
    marks[np.searchsorted(t, marks_at_s - 0.5 / FS)] = 4
    lifetime = 2.4 + np.zeros(t.size)
    df = pd.DataFrame(
        {
            "time": t,
            "intensity": 4e5,
            "lifetime": lifetime,
            "marks": marks,
            "tcspc_0": 10,
            "tcspc_0p1": 20,
            "tcspc_0p2": 5,
        }
    )
    df.to_csv(folder / f"{stem}_data.csv", index=False)
    pd.DataFrame(
        {
            "variable": [
                "filename",
                "header_state_samplingfreq",
                "header_state_t0",
                "header_starttime",
            ],
            "value": [stem, "20", "1.02", "14-Oct-2026 16:08:49.261"],
        }
    ).to_csv(folder / f"{stem}_param.csv", index=False)
    return folder / f"{stem}_data.csv"


@pytest.fixture
def synthetic(tmp_path: Path):
    # long enough (~500 s) that -374 ppm drift (~0.19 s) exceeds the 50 ms frame quantization
    sync_a = np.arange(0.3, 500, 5.0)  # Arduino clock, s
    marks_p = OFFSET_S - 60 + SLOPE * sync_a  # photometry clock
    deliveries_a = np.array([5.0, 20.0, 35.0])
    solenoid_a = np.concatenate([d + 0.18 * np.arange(6) for d in deliveries_a])
    licks_a = np.array([5.6, 5.8, 6.0, 36.2, 36.4])  # no licks after the 2nd delivery
    log = _write_log(
        tmp_path / "rig_log.csv",
        sync_ms=np.round(sync_a * 1000).astype(int),
        solenoid_ms=np.round(solenoid_a * 1000).astype(int),
        lick_ms=np.round(licks_a * 1000).astype(int),
    )
    data_csv = _write_tidy(tmp_path, "101426_m1_male_001", marks_p)
    return {
        "log": log,
        "data_csv": data_csv,
        "root": tmp_path,
        "deliveries_a": deliveries_a,
        "licks_a": licks_a,
        "sync_a": sync_a,
    }


# --------------------------------------------------------------------------- #
# Parsing and alignment
# --------------------------------------------------------------------------- #


def test_load_arduino_log_handles_cr_and_settings(synthetic):
    log = load_arduino_log(synthetic["log"])
    assert log.settings == {11: [0], 116: [15], 117: [6], 127: [0, 2]}
    assert log.sync_s.size == synthetic["sync_a"].size
    assert log.licks_s.size == 5
    assert log.solenoid_on_s.size == 18


def test_align_clocks_recovers_offset_and_drift():
    a = np.arange(0.3, 300, 5.0)
    p = 67.9 + SLOPE * a
    fit = align_clocks(a, p)
    assert fit.offset_s == pytest.approx(67.9, abs=1e-6)
    assert fit.drift_ppm == pytest.approx(-374, abs=0.5)
    assert fit.residual_max_s < 1e-6


def test_align_clocks_unequal_trains_need_explicit_shift():
    a = np.arange(0.3, 300, 5.0)
    p = 10 + a[3:-2]  # photometry started late and stopped early
    # periodic pulses: every shift fits perfectly, so refuse to guess
    with pytest.raises(ValueError, match="ambiguous"):
        align_clocks(a, p)
    fit = align_clocks(a, p, pulse_shift=3)
    assert fit.n_pulses == p.size
    assert fit.to_photometry(a[3]) == pytest.approx(p[0])
    with pytest.raises(ValueError, match="out of range"):
        align_clocks(a, p, pulse_shift=6)


def test_align_clocks_rejects_unrelated_trains():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="do not align"):
        align_clocks(np.sort(rng.uniform(0, 300, 40)), np.arange(0, 200, 5.0))


def test_mark_onsets_collapses_runs():
    t = np.arange(10) * 0.05
    marks = np.array([0, 4, 4, 0, 0, 4, 0, 0, 4, 4])
    np.testing.assert_allclose(mark_onsets(t, marks), [0.05, 0.25, 0.40])


def test_group_deliveries_and_first_lick():
    pulses = np.concatenate([d + 0.18 * np.arange(6) for d in (5.0, 20.0)])
    onsets = group_deliveries(pulses)
    np.testing.assert_allclose(onsets, [5.0, 20.0])
    first = first_lick_after(onsets, np.array([4.0, 5.5, 5.7, 30.0]))
    np.testing.assert_allclose(first, [5.5, 30.0])
    assert np.isnan(first_lick_after(onsets, np.array([5.5]))[1])


# --------------------------------------------------------------------------- #
# Session folder round trip
# --------------------------------------------------------------------------- #


def test_build_session_folder_round_trip(synthetic):
    out = build_session_folder(
        load_tidy(synthetic["data_csv"]),
        synthetic["log"],
        synthetic["root"],
        subject="m1",
        procedure="passive sucrose",
        extra_meta={"sensor": "FLIM-DA1.0"},
    )
    assert out == synthetic["root"] / "sessions" / "101426_m1_male"

    session = load_session(out)
    assert session.subject == "m1"
    assert session.fs == pytest.approx(FS)
    assert session.meta["sensor"] == "FLIM-DA1.0"
    assert session.meta["arduino_setting_117"] == "6"
    assert float(session.meta["sync_drift_ppm"]) == pytest.approx(-374, abs=50)

    ev = session.events
    onsets = ev.loc[ev.event_id_char == "solution_onset", "time"].to_numpy()
    expected = OFFSET_S - 60 + SLOPE * synthetic["deliveries_a"]
    np.testing.assert_allclose(onsets, expected, atol=0.03)  # within one frame
    assert (ev.event_id_char == "lick").sum() == 5
    assert (ev.event_id_char == "lick_post_solution").sum() == 2  # 2nd delivery had no licks

    # indicator columns mark one sample per event
    assert session.streams["solution_onset"].sum() == 3
    # PETH matches the R export layout and the PETH engine reads the session
    assert set(session.peth_event_types()) == {"solution_onset", "lick_post_solution"}
    assert session.peth.time_rel.min() == pytest.approx(-3.0)
    peth = build_peth(session, "solution_onset", signal="lifetime")
    assert peth.n_trials == 3


def test_discover_finds_flat_layout(synthetic):
    build_session_folder(load_tidy(synthetic["data_csv"]), synthetic["log"], synthetic["root"])
    [entry] = discover_acquisitions(synthetic["root"])
    assert entry.blockname == "101426_m1_male"
    assert entry.has_session
    assert entry.tidy_data_path == synthetic["data_csv"]


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #

CERA_DIR = Path(__file__).resolve().parents[3] / "Cera" / "iFLIP"
CERA_DATA = CERA_DIR / "101426_2852L_male_001_data.csv"
CERA_LOG = CERA_DIR / "2026_10_14_2852L_male.csv"


@pytest.mark.skipif(not CERA_DATA.is_file(), reason=f"Cera session not present at {CERA_DIR}")
def test_cera_2852L_import(tmp_path):
    out = build_session_folder(load_tidy(CERA_DATA), CERA_LOG, tmp_path, subject="2852L")
    session = load_session(out)
    counts = session.events["event_id_char"].value_counts()
    assert counts["solution_onset"] == 10
    assert counts["lick"] == 469
    assert counts["lick_post_solution"] == 10
    assert int(session.meta["sync_n_pulses"]) == 57
    assert float(session.meta["sync_residual_max_s"]) < 0.05
