"""Round-trip parity tests for the ``.iFLiP2`` raw reader.

These tests load the example raw acquisition file *and* the lab MATLAB
pipeline's tidy CSV export of the same acquisition, and check that the
Python reader reproduces the CSV's stream columns, TCSPC histograms, and
key header values to within a tiny tolerance.

Tests are skipped automatically if the FLIPR data root (which lives
outside this repo) is not present, so the suite still passes on a CI
checkout that doesn't ship the data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iflip.io import load_iflip2, load_tidy
from iflip.io.iflip2 import FRAME_BYTES, N_BINS

# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #

DATA_ROOT = Path(__file__).resolve().parents[2] / "FLIPR data"
RAW_FILE = DATA_ROOT / "raw" / "2026_04_09_acz02_001.iFLiP2"
TIDY_DATA = DATA_ROOT / "tidy" / "2026_04_09_acz02_001_data.csv"
TIDY_PARAM = DATA_ROOT / "tidy" / "2026_04_09_acz02_001_param.csv"

EXPECTED_N_FRAMES = 31054

pytestmark = pytest.mark.skipif(
    not RAW_FILE.is_file() or not TIDY_DATA.is_file() or not TIDY_PARAM.is_file(),
    reason=f"FLIPR example acquisition not present at {DATA_ROOT}",
)


# --------------------------------------------------------------------------- #
# Fixtures: load both representations once per session
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def raw():
    """Parsed binary `.iFLiP2` file."""
    return load_iflip2(RAW_FILE)


@pytest.fixture(scope="module")
def tidy():
    """Parsed tidy CSV pair (ground truth)."""
    return load_tidy(TIDY_DATA, TIDY_PARAM)


# --------------------------------------------------------------------------- #
# Format / shape sanity
# --------------------------------------------------------------------------- #


def test_frame_count_matches_csv(raw, tidy):
    assert raw.n_frames == EXPECTED_N_FRAMES
    assert raw.n_frames == tidy.n_time
    assert raw.n_bins == N_BINS == tidy.n_bins
    assert raw.frame_bytes == FRAME_BYTES == 508


def test_file_size_matches_layout(raw):
    """Sanity check the inferred header + per-frame layout matches the file."""
    file_size = RAW_FILE.stat().st_size
    expected = raw.header_bytes + raw.n_frames * raw.frame_bytes
    assert expected == file_size, (
        f"layout mismatch: header({raw.header_bytes}) + "
        f"{raw.n_frames}*{raw.frame_bytes}={expected}, file is {file_size}"
    )


def test_tcspc_bin_axis(raw):
    bins = raw.tcspc_bins_ns
    assert bins.shape == (N_BINS,)
    assert bins[0] == pytest.approx(0.0)
    assert bins[-1] == pytest.approx(12.5)
    np.testing.assert_allclose(np.diff(bins), 0.1, atol=1e-12)


# --------------------------------------------------------------------------- #
# TCSPC parity (must be byte-exact: integer counts)
# --------------------------------------------------------------------------- #


def test_tcspc_matches_csv_exactly(raw, tidy):
    """Every TCSPC bin in every frame must equal the tidy CSV value exactly."""
    a = raw.tcspc.astype(np.int64)
    b = tidy.tcspc.astype(np.int64)
    assert a.shape == b.shape, f"shape: raw {a.shape} vs tidy {b.shape}"

    if not np.array_equal(a, b):
        diffs = np.argwhere(a != b)
        first = tuple(diffs[0])
        raise AssertionError(
            f"TCSPC mismatch: {len(diffs)} bins differ; first at frame={first[0]} "
            f"bin={first[1]}, raw={a[first]}, tidy={b[first]}"
        )


# --------------------------------------------------------------------------- #
# Stream column parity (intensity, lifetime, marks, time)
# --------------------------------------------------------------------------- #


def test_marks_match_csv_exactly(raw, tidy):
    """Marks are integers; must be byte-equal to the CSV."""
    a = raw.streams["marks"].to_numpy().astype(np.int64)
    b = tidy.streams["marks"].to_numpy().astype(np.int64)
    if not np.array_equal(a, b):
        diffs = np.flatnonzero(a != b)
        i = int(diffs[0])
        raise AssertionError(
            f"marks mismatch: {len(diffs)} frames differ; "
            f"first at frame={i}, raw={a[i]}, tidy={b[i]}"
        )


def test_intensity_matches_csv(raw, tidy):
    """Intensity is an integer photon count; must be byte-equal to the CSV."""
    a = raw.streams["intensity"].to_numpy().astype(np.int64)
    b = tidy.streams["intensity"].to_numpy().astype(np.int64)
    if not np.array_equal(a, b):
        diffs = np.flatnonzero(a != b)
        i = int(diffs[0])
        raise AssertionError(
            f"intensity mismatch: {len(diffs)} frames differ; "
            f"first at frame={i}, raw={a[i]}, tidy={b[i]} "
            f"(diff={int(a[i]) - int(b[i])})"
        )


def test_time_matches_csv(raw, tidy):
    """Frame center times: ``(i + 0.5) / fs``; CSV stores them rounded."""
    a = raw.streams["time"].to_numpy()
    b = tidy.streams["time"].to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-9, err_msg="time vector mismatch")


def test_lifetime_matches_csv(raw, tidy):
    """Lifetime is a derived float; allow small floating-point round-trip error."""
    a = raw.streams["lifetime"].to_numpy()
    b = tidy.streams["lifetime"].to_numpy()
    # The MATLAB pipeline writes ~14 significant digits; our derivation is
    # all float64 too. A 1e-9 absolute tolerance + 1e-9 relative tolerance
    # is a tighter bound than the CSV's text precision.
    np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9, err_msg="lifetime vector mismatch")


# --------------------------------------------------------------------------- #
# Header parity
# --------------------------------------------------------------------------- #


def test_header_numeric_fields(raw, tidy):
    """Key acquisition parameters from the binary header must match the
    param CSV.

    The binary stores full float64 precision while the CSV rounds to
    ~4-5 significant digits, so we use a relative tolerance of 1e-4
    for floats (well under the CSV's display precision) and exact
    equality for fields the MATLAB pipeline writes as integers.
    """

    def _flt(d, k):
        v = d.get(k)
        return float(v) if v is not None else float("nan")

    # (key, rtol, atol). Integer fields use rtol=0, atol=0.
    pairs = [
        ("header_state_t0", 1e-4, 5e-5),
        ("header_state_tau1", 1e-4, 5e-5),
        ("header_state_tau2", 1e-4, 5e-5),
        ("header_state_samplingfreq", 0.0, 0.0),
        ("header_state_acqtime", 0.0, 0.0),
        ("header_state_syncdiv", 0.0, 0.0),
        ("header_init_syncrate", 0.0, 0.0),
        ("header_init_deadtime_ns", 0.0, 0.0),
        ("header_state_afterpulseratio", 1e-4, 5e-5),
        ("header_state_spcrangelow", 1e-4, 5e-5),
        ("header_state_spcrangehigh", 1e-4, 5e-5),
    ]
    for key, rtol, atol in pairs:
        raw_v = _flt(raw.header, key)
        tidy_v = _flt(tidy.params, key)
        assert raw_v == pytest.approx(tidy_v, rel=rtol, abs=atol), (
            f"header field {key}: raw={raw_v}, tidy={tidy_v}"
        )


def test_header_string_fields(raw, tidy):
    """A few string fields the lab cares about: hardware, filename, start time."""
    # filename and start time should pass through verbatim
    raw_filename = str(raw.header.get("filename") or "")
    tidy_filename = str(tidy.params.get("filename") or "")
    assert raw_filename == tidy_filename

    # The hardware info string contains real newlines in the binary
    # (e.g. "TimeHarp 260 P\nPart# 930031;\nS/N: 1051615\nVer: ...") but
    # the MATLAB pipeline concatenates them with no separator in the
    # tidy CSV ("TimeHarp 260 PPart# 930031; S/N: 1051615Ver: ..."). The
    # substantive content is identical — we compare after stripping ALL
    # whitespace from both representations.
    def _strip_ws(s):
        return "".join(str(s).split())

    raw_hw = _strip_ws(raw.header.get("header_state_hardwareinfo", ""))
    tidy_hw = _strip_ws(tidy.params.get("header_state_hardwareinfo", ""))
    assert raw_hw == tidy_hw, f"hardwareinfo:\n  raw : {raw_hw!r}\n  tidy: {tidy_hw!r}"


def test_streams_dataframe_schema(raw):
    """Public DataFrame schema check."""
    cols = list(raw.streams.columns)
    assert cols == ["time", "intensity", "lifetime", "marks"]
    assert isinstance(raw.streams, pd.DataFrame)
    assert len(raw.streams) == raw.n_frames
