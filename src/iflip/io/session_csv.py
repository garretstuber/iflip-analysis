"""Loader for the ``sessions/<blockname>/`` CSV exports.

A session folder contains four files produced by the lab's R pipeline:

- ``meta.csv`` — single-table (var, value, source) with subject, procedure, fs,
  baselines, and fit parameters.
- ``df_events.csv`` — long-format (event_id_char, time) behavioural events.
- ``df_streams_session.csv`` — wide-format (time, intensity, lifetime, marks,
  <event cols>) continuous streams sampled at ``fs`` Hz.
- ``df_streams_peth.csv`` — long-format event-aligned traces with ``time``,
  ``sample_rel`` (samples from event), ``trial_num``, ``event_type``,
  ``signal_id`` (intensity/lifetime), ``value``, and ``time_rel`` (s from event).

This module provides a single :func:`load_session` entry point, a
:class:`SessionData` container holding the parsed pieces, and
:func:`session_from_tcspc_source` to build a minimal ``SessionData``
from a tidy CSV or raw ``.iFLiP2`` file when no session folder exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from iflip.io.iflip2 import IFlip2File
    from iflip.io.tidy_csv import TidyData


@dataclass
class SessionData:
    """Parsed contents of a ``sessions/<blockname>/`` folder."""

    blockname: str
    path: Path
    meta: dict[str, str]
    events: pd.DataFrame  # columns: event_id_char, time
    streams: pd.DataFrame  # columns: time, intensity, lifetime, marks, <event indicator cols>
    peth: pd.DataFrame  # long-format PETH

    @property
    def fs(self) -> float:
        """Sampling rate in Hz, pulled from meta."""
        return float(self.meta.get("fs", "nan"))

    @property
    def subject(self) -> str:
        return self.meta.get("subject", "")

    @property
    def procedure(self) -> str:
        return self.meta.get("procedure", "")

    @property
    def event_types(self) -> list[str]:
        """Unique event type names found in events table."""
        if self.events.empty:
            return []
        return sorted(self.events["event_id_char"].unique().tolist())

    def peth_event_types(self) -> list[str]:
        """Event types for which PETH windows were computed."""
        if self.peth.empty:
            return []
        return sorted(self.peth["event_type"].unique().tolist())

    def peth_for(self, event_type: str, signal: str = "lifetime") -> pd.DataFrame:
        """Return PETH for a given event type and signal, as a (trial × sample_rel) matrix.

        Rows are trials, columns are ``time_rel`` (seconds from event), values are ``value``.
        """
        if self.peth.empty:
            return pd.DataFrame()
        sub = self.peth[
            (self.peth["event_type"] == event_type) & (self.peth["signal_id"] == signal)
        ]
        if sub.empty:
            return pd.DataFrame()
        return sub.pivot_table(index="trial_num", columns="time_rel", values="value")


def _load_meta(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["var"].astype(str), df["value"].astype(str), strict=False))


def load_session(session_dir: str | Path) -> SessionData:
    """Load a session folder.

    Parameters
    ----------
    session_dir : str | Path
        Path to a ``sessions/<blockname>/`` directory containing the four
        expected CSV files.

    Returns
    -------
    SessionData
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    required = {
        "meta": session_dir / "meta.csv",
        "events": session_dir / "df_events.csv",
        "streams": session_dir / "df_streams_session.csv",
        "peth": session_dir / "df_streams_peth.csv",
    }
    missing = [k for k, p in required.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Session {session_dir.name} is missing required file(s): {missing}"
        )

    meta = _load_meta(required["meta"])
    events = pd.read_csv(required["events"])
    streams = pd.read_csv(required["streams"])
    peth = pd.read_csv(required["peth"])

    blockname = meta.get("blockname", session_dir.name)

    return SessionData(
        blockname=blockname,
        path=session_dir,
        meta=meta,
        events=events,
        streams=streams,
        peth=peth,
    )


def list_sessions(data_root: str | Path) -> list[Path]:
    """List all session folders under ``<data_root>/sessions``."""
    sessions_dir = Path(data_root) / "sessions"
    if not sessions_dir.is_dir():
        return []
    return sorted(p for p in sessions_dir.iterdir() if p.is_dir())


# --------------------------------------------------------------------------- #
# Blockname extraction helpers
# --------------------------------------------------------------------------- #

#: Tidy data CSV filename → blockname. The filename ends with ``_data.csv``;
#: everything before that suffix is the "stem". When the stem ends with a
#: 3-digit acquisition-sequence number (e.g. ``_001``), strip it to get the
#: blockname. Otherwise the full stem *is* the blockname.
#:
#: Examples:
#:   ``2026_04_27_acz02_001_data.csv`` → ``2026_04_27_acz02``
#:   ``2026_04_27_bg001_data.csv``     → ``2026_04_27_bg001``
#:   ``2026_04_27_bg001_2_data.csv``   → ``2026_04_27_bg001_2``
_TIDY_DATA_SUFFIX = "_data.csv"
_IFLIP2_SUFFIX_RE = re.compile(r"\.iflip2$", re.IGNORECASE)
#: Trailing _NNN (3-digit acquisition number) at the end of a stem.
_ACQ_NUM_RE = re.compile(r"^(.+)_(\d{3})$")


def _strip_acq_number(stem: str) -> str:
    """Remove a trailing ``_NNN`` 3-digit acquisition number if present."""
    m = _ACQ_NUM_RE.match(stem)
    return m.group(1) if m else stem


def _blockname_from_tidy(name: str) -> str | None:
    """Extract blockname from a tidy data CSV filename."""
    if not name.endswith(_TIDY_DATA_SUFFIX):
        return None
    stem = name[: -len(_TIDY_DATA_SUFFIX)]
    return _strip_acq_number(stem)


def _blockname_from_iflip2(name: str) -> str | None:
    """Extract blockname from an .iFLiP2 filename."""
    m = _IFLIP2_SUFFIX_RE.search(name)
    if m is None:
        return None
    stem = name[: m.start()]
    return _strip_acq_number(stem)


# --------------------------------------------------------------------------- #
# Unified discovery
# --------------------------------------------------------------------------- #


@dataclass
class AcquisitionEntry:
    """Lightweight descriptor for a discovered acquisition."""

    blockname: str
    has_session: bool  # full session folder exists
    session_path: Path | None  # sessions/<blockname>/ if it exists
    tidy_data_path: Path | None  # tidy/<…>_data.csv if it exists
    iflip2_path: Path | None  # raw/<…>.iFLiP2 if it exists


def discover_acquisitions(data_root: str | Path) -> list[AcquisitionEntry]:
    """Find every unique acquisition under *data_root*.

    Scans ``sessions/``, ``tidy/``, and ``raw/`` and returns one
    :class:`AcquisitionEntry` per unique blockname, sorted alphabetically.
    An acquisition appears even if it only has a tidy CSV or a raw file
    (no full session folder required).
    """
    data_root = Path(data_root)
    entries: dict[str, AcquisitionEntry] = {}

    # 1. Session folders — the canonical source when available.
    for p in list_sessions(data_root):
        bn = p.name
        entries[bn] = AcquisitionEntry(
            blockname=bn,
            has_session=True,
            session_path=p,
            tidy_data_path=None,
            iflip2_path=None,
        )

    # 2. Tidy data CSVs
    tidy_dir = data_root / "tidy"
    if tidy_dir.is_dir():
        for f in sorted(tidy_dir.glob("*_data.csv")):
            bn = _blockname_from_tidy(f.name)
            if bn is None:
                continue
            if bn in entries:
                entries[bn].tidy_data_path = f
            else:
                entries[bn] = AcquisitionEntry(
                    blockname=bn,
                    has_session=False,
                    session_path=None,
                    tidy_data_path=f,
                    iflip2_path=None,
                )

    # 3. Raw .iFLiP2 files
    raw_dir = data_root / "raw"
    if raw_dir.is_dir():
        all_raw: set[Path] = set()
        all_raw.update(raw_dir.glob("*.iFLiP2"))
        all_raw.update(raw_dir.glob("*.iflip2"))
        for f in sorted(all_raw):
            bn = _blockname_from_iflip2(f.name)
            if bn is None:
                continue
            if bn in entries:
                entries[bn].iflip2_path = f
            else:
                entries[bn] = AcquisitionEntry(
                    blockname=bn,
                    has_session=False,
                    session_path=None,
                    tidy_data_path=None,
                    iflip2_path=f,
                )

    # Back-fill: for session-only entries whose blockname didn't match
    # a tidy/raw filename prefix, try the fuzzy matchers.
    needs_backfill = [
        e
        for e in entries.values()
        if e.has_session and (e.tidy_data_path is None or e.iflip2_path is None)
    ]
    if needs_backfill:
        from iflip.io.iflip2 import list_iflip2_files, match_iflip2_to_session
        from iflip.io.tidy_csv import list_tidy_files, match_tidy_to_session

        tidy_files = list_tidy_files(data_root) if tidy_dir.is_dir() else []
        iflip2_files = list_iflip2_files(data_root) if raw_dir.is_dir() else []
        for entry in needs_backfill:
            if entry.tidy_data_path is None and tidy_files:
                match = match_tidy_to_session(tidy_files, entry.blockname)
                if match is not None:
                    entry.tidy_data_path = match
            if entry.iflip2_path is None and iflip2_files:
                match = match_iflip2_to_session(iflip2_files, entry.blockname)
                if match is not None:
                    entry.iflip2_path = match

    return sorted(entries.values(), key=lambda e: e.blockname)


# --------------------------------------------------------------------------- #
# Minimal SessionData from a TCSPC source
# --------------------------------------------------------------------------- #


def session_from_tcspc_source(
    source: TidyData | IFlip2File,
    blockname: str | None = None,
) -> SessionData:
    """Build a minimal :class:`SessionData` from a tidy CSV or raw ``.iFLiP2``
    file, for acquisitions that lack a full ``sessions/`` folder.

    The resulting object has empty events and PETH DataFrames, so the
    Session Overview and Interval Inspector tabs work but Event PETH
    shows "no events".
    """
    from iflip.io.iflip2 import IFlip2File
    from iflip.io.tidy_csv import TidyData

    streams = source.streams.copy()
    # Ensure canonical column order
    for col in ("time", "intensity", "lifetime", "marks"):
        if col not in streams.columns:
            raise ValueError(f"TCSPC source is missing required stream column: {col}")

    if isinstance(source, TidyData):
        params = source.params
        path = source.path.parent
        bn = blockname or source.filename
    elif isinstance(source, IFlip2File):
        params = source.header
        path = source.path.parent
        bn = blockname or source.filename
    else:
        raise TypeError(f"Unsupported TCSPC source type: {type(source)}")

    # Build a meta dict from available header fields
    meta: dict[str, str] = {
        "blockname": bn,
        "subject": str(params.get("filename", bn)),
        "procedure": "unknown",
    }
    # Copy over relevant header fields
    for key in (
        "header_state_samplingfreq",
        "header_state_t0",
        "header_state_tau1",
        "header_state_tau2",
    ):
        val = params.get(key)
        if val is not None:
            meta[key] = str(val)
    # Map to session-level keys
    fs_val = params.get("header_state_samplingfreq")
    if fs_val is not None:
        meta["fs"] = str(fs_val)
    t0_val = params.get("header_state_t0")
    if t0_val is not None:
        meta["t0"] = str(t0_val)

    return SessionData(
        blockname=bn,
        path=path,
        meta=meta,
        events=pd.DataFrame(columns=["event_id_char", "time"]),
        streams=streams,
        peth=pd.DataFrame(),
    )
