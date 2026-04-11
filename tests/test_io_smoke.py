"""Smoke tests that exercise the IO loaders against the example session.

These are skipped automatically if the FLIPR data root is not present
(so they work on CI / checkouts without the real data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flipr.io import load_session, load_tidy
from flipr.io.session_csv import list_sessions
from flipr.io.tidy_csv import list_tidy_files, match_tidy_to_session

DATA_ROOT = Path(__file__).resolve().parents[2] / "FLIPR data"
EXAMPLE_BLOCK = "2026_04_09_acz02"

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.is_dir(),
    reason=f"FLIPR data root not present at {DATA_ROOT}",
)


def test_list_sessions_finds_example():
    sessions = list_sessions(DATA_ROOT)
    names = [p.name for p in sessions]
    assert EXAMPLE_BLOCK in names


def test_load_session_meta_and_streams():
    session = load_session(DATA_ROOT / "sessions" / EXAMPLE_BLOCK)
    assert session.blockname == EXAMPLE_BLOCK
    assert session.subject == "acz02"
    assert session.fs == pytest.approx(20.0)
    assert session.procedure == "passive sucrose"

    # streams have the expected columns and a plausible length (~1 hour at 20 Hz)
    for col in ("time", "intensity", "lifetime", "marks"):
        assert col in session.streams.columns
    assert len(session.streams) > 10_000

    # events contain solution_onset and lick timestamps
    ev_types = set(session.event_types)
    assert {"solution_onset", "lick", "lick_post_solution"}.issubset(ev_types)

    # lifetime column is in a sensible ns range for FLIM-DA0.5 (roughly 0.5–4 ns)
    lt = session.streams["lifetime"].to_numpy()
    assert np.nanmin(lt) > 0.1
    assert np.nanmax(lt) < 6.0


def test_peth_pivot_shape():
    session = load_session(DATA_ROOT / "sessions" / EXAMPLE_BLOCK)
    assert "solution_onset" in session.peth_event_types()
    mat = session.peth_for("solution_onset", signal="lifetime")
    assert not mat.empty
    assert mat.shape[0] == 50  # 50 unpredicted sucrose trials per meta note
    # pre- and post-event windows present
    assert mat.columns.min() < 0
    assert mat.columns.max() > 0


def test_load_tidy_and_tcspc_shape():
    tidy_files = list_tidy_files(DATA_ROOT)
    assert tidy_files, "no tidy files under data root"
    picked = match_tidy_to_session(tidy_files, EXAMPLE_BLOCK)
    assert picked is not None
    tidy = load_tidy(picked)

    # Stream table length matches the number of TCSPC rows
    assert tidy.n_time == len(tidy.streams)
    assert tidy.n_time > 10_000

    # 126 bins at 0.1 ns from 0.0 to 12.5 ns
    assert tidy.n_bins == 126
    assert tidy.tcspc_bins_ns[0] == pytest.approx(0.0)
    assert tidy.tcspc_bins_ns[-1] == pytest.approx(12.5)
    np.testing.assert_allclose(np.diff(tidy.tcspc_bins_ns), 0.1, atol=1e-9)

    # Intensity should roughly match the summed TCSPC row (to within a constant
    # scaling — the instrument reports photon counts per frame).
    row_sum = tidy.tcspc.sum(axis=1)
    intensity = tidy.streams["intensity"].to_numpy()
    assert np.corrcoef(row_sum, intensity)[0, 1] > 0.99

    # Header carries the fit parameters we'll need for the interval inspector
    assert 3.0 < tidy.tau1_ns < 5.0
    assert 0.3 < tidy.tau2_ns < 1.0
    assert tidy.fs == pytest.approx(20.0)


def test_integrate_histogram_window():
    tidy_files = list_tidy_files(DATA_ROOT)
    picked = match_tidy_to_session(tidy_files, EXAMPLE_BLOCK)
    tidy = load_tidy(picked)

    # First 10 seconds of data
    hist = tidy.integrate_histogram(0.0, 10.0)
    assert hist.shape == (tidy.n_bins,)
    assert hist.sum() > 0

    # Peak of the decay should sit near t0 (~1 ns per param.csv), not at bin 0
    peak_bin_ns = tidy.tcspc_bins_ns[int(np.argmax(hist))]
    assert 0.5 < peak_bin_ns < 2.5
