"""Loader for the ``tidy/`` per-file CSV exports.

Each acquisition produces two tidy files:

- ``<filename>_data.csv`` — one row per timepoint, columns:
  ``time, intensity, lifetime, marks, tcspc_0, tcspc_0p1, ..., tcspc_12p5``
  The ``tcspc_*`` columns are a TCSPC histogram with 126 bins at 0.1 ns resolution
  spanning 0–12.5 ns. Bin names use ``p`` in place of a decimal point
  (e.g. ``tcspc_0p1`` = 0.1 ns bin).
- ``<filename>_param.csv`` — two-column (variable, value) acquisition header,
  including ``header_state_t0``, ``header_state_tau1``, ``header_state_tau2``,
  ``header_state_samplingfreq``, ``header_init_syncrate``, etc.

:func:`load_tidy` returns a :class:`TidyData` container with the stream columns
as a pandas DataFrame, the TCSPC histograms as a numpy ``(n_time, n_bins)`` array,
the per-bin time axis in ns, and the parsed header dict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_TCSPC_COL_RE = re.compile(r"^tcspc_(\d+)(?:p(\d+))?$")


@dataclass
class TidyData:
    """Parsed contents of a ``tidy/`` file pair."""

    filename: str
    path: Path
    streams: pd.DataFrame  # columns: time, intensity, lifetime, marks
    tcspc: np.ndarray  # shape (n_time, n_bins), int
    tcspc_bins_ns: np.ndarray  # shape (n_bins,), bin centers in ns
    params: dict[str, str]

    @property
    def n_time(self) -> int:
        return self.tcspc.shape[0]

    @property
    def n_bins(self) -> int:
        return self.tcspc.shape[1]

    @property
    def fs(self) -> float:
        """Sampling rate in Hz from the header."""
        return float(self.params.get("header_state_samplingfreq", "nan"))

    @property
    def t0_ns(self) -> float:
        """Fitted IRF position / time-zero in ns."""
        return float(self.params.get("header_state_t0", "nan"))

    @property
    def tau1_ns(self) -> float:
        return float(self.params.get("header_state_tau1", "nan"))

    @property
    def tau2_ns(self) -> float:
        return float(self.params.get("header_state_tau2", "nan"))

    def time_index(self) -> np.ndarray:
        """Timepoint axis in seconds (from ``streams.time``)."""
        return self.streams["time"].to_numpy()

    def integrate_histogram(self, start_time: float, stop_time: float) -> np.ndarray:
        """Sum TCSPC histograms over a closed time interval [start_time, stop_time] in seconds.

        Returns a 1D array of length ``n_bins``.
        """
        t = self.time_index()
        mask = (t >= start_time) & (t <= stop_time)
        if not mask.any():
            return np.zeros(self.n_bins, dtype=np.int64)
        return self.tcspc[mask].sum(axis=0)


def _parse_tcspc_col(name: str) -> float | None:
    """Parse ``tcspc_0p1`` → 0.1 (ns). Returns None for non-matching columns."""
    m = _TCSPC_COL_RE.match(name)
    if m is None:
        return None
    whole = int(m.group(1))
    frac = m.group(2)
    if frac is None:
        return float(whole)
    return float(f"{whole}.{frac}")


def _load_params(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    # values are quoted strings; strip whitespace
    return {
        str(k).strip(): str(v).strip()
        for k, v in zip(df["variable"], df["value"], strict=False)
    }


def load_tidy(data_csv: str | Path, param_csv: str | Path | None = None) -> TidyData:
    """Load a tidy data/param CSV pair.

    Parameters
    ----------
    data_csv : str | Path
        Path to the ``<filename>_data.csv`` file.
    param_csv : str | Path | None
        Path to the matching ``<filename>_param.csv``. If None, inferred by
        replacing the ``_data.csv`` suffix with ``_param.csv``.
    """
    data_csv = Path(data_csv)
    if not data_csv.is_file():
        raise FileNotFoundError(f"Tidy data file not found: {data_csv}")

    if param_csv is None:
        if not data_csv.name.endswith("_data.csv"):
            raise ValueError(
                f"Cannot infer param file from {data_csv.name!r}; expected '*_data.csv'"
            )
        param_csv = data_csv.with_name(data_csv.name[: -len("_data.csv")] + "_param.csv")
    param_csv = Path(param_csv)
    if not param_csv.is_file():
        raise FileNotFoundError(f"Tidy param file not found: {param_csv}")

    df = pd.read_csv(data_csv)

    # Split stream columns from TCSPC columns, preserving bin order by ns value.
    tcspc_cols: list[tuple[float, str]] = []
    stream_cols: list[str] = []
    for col in df.columns:
        ns = _parse_tcspc_col(col)
        if ns is None:
            stream_cols.append(col)
        else:
            tcspc_cols.append((ns, col))
    tcspc_cols.sort(key=lambda t: t[0])
    tcspc_bins_ns = np.array([ns for ns, _ in tcspc_cols], dtype=np.float64)
    tcspc = df[[name for _, name in tcspc_cols]].to_numpy(dtype=np.int64)

    expected_streams = {"time", "intensity", "lifetime", "marks"}
    missing = expected_streams - set(stream_cols)
    if missing:
        raise ValueError(f"Tidy data file missing expected columns: {sorted(missing)}")
    streams = df[list(expected_streams)].copy()
    streams = streams[["time", "intensity", "lifetime", "marks"]]  # canonical order

    params = _load_params(param_csv)
    filename = params.get("filename", data_csv.stem.replace("_data", ""))

    return TidyData(
        filename=filename,
        path=data_csv,
        streams=streams,
        tcspc=tcspc,
        tcspc_bins_ns=tcspc_bins_ns,
        params=params,
    )


def list_tidy_files(data_root: str | Path) -> list[Path]:
    """List all ``*_data.csv`` files under ``<data_root>/tidy``."""
    tidy_dir = Path(data_root) / "tidy"
    if not tidy_dir.is_dir():
        return []
    return sorted(tidy_dir.glob("*_data.csv"))


def match_tidy_to_session(tidy_files: list[Path], blockname: str) -> Path | None:
    """Pick the tidy file whose stem (minus ``_NNN_data``) matches a session blockname.

    Example: blockname ``2026_04_09_acz02`` matches ``2026_04_09_acz02_001_data.csv``.
    Returns the first match (lexicographically earliest ``_NNN``), or None.
    """
    prefix = f"{blockname}_"
    matches = [p for p in tidy_files if p.name.startswith(prefix) and p.name.endswith("_data.csv")]
    if not matches:
        return None
    return sorted(matches)[0]
