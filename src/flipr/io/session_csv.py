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

This module provides a single :func:`load_session` entry point and a
:class:`SessionData` container holding the parsed pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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
