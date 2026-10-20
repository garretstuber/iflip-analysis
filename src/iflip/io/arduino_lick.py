"""Importer for Arduino lick-rig event logs.

Head-fixed sucrose sessions run on an Arduino lick rig write a plain-text
log of ``<code> <value>`` pairs, one per line (CR, CRLF, or LF line
endings, blank lines interleaved). The first block holds session
settings (``code value`` pairs written at start-up); after that every
line is ``code timestamp_ms`` on the Arduino clock. Codes used by the
rig:

- ``32`` — lick
- ``18`` / ``14`` — solenoid open / close (one sucrose delivery is a
  burst of short pulses, e.g. 6 × 15 ms every 180 ms)
- ``1001`` — sync pulse, also recorded by the iFLiP rig as a nonzero
  ``marks`` sample

The Arduino and TimeHarp clocks drift relative to each other (−374 ppm
on the first session imported, ~0.1 s over 5 min), so alignment fits an
offset *and* a slope to the matched sync pulses rather than a single
offset.

:func:`build_session_folder` turns a lick log plus a tidy CSV or raw
``.iFLiP2`` file into a ``sessions/<blockname>/`` folder with the same
four files and event names (``solution_onset``, ``lick``,
``lick_post_solution``) as the lab's R pipeline, so
:func:`iflip.io.load_session` and the dashboard read it unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from iflip.io.iflip2 import IFlip2File
    from iflip.io.tidy_csv import TidyData

LICK_CODE = 32
SOLENOID_ON_CODE = 18
SOLENOID_OFF_CODE = 14
SYNC_CODE = 1001

_LINE_SPLIT_RE = re.compile(r"\r\n|\r|\n")


@dataclass
class ArduinoLog:
    """Parsed Arduino lick log."""

    path: Path
    records: pd.DataFrame  # columns: code, value — every line, in file order
    settings: dict[int, list[int]]  # start-up block: code -> values

    def times_s(self, code: int) -> np.ndarray:
        """Timestamps (s, Arduino clock) of every record with ``code``."""
        vals = self.records.loc[self.records["code"] == code, "value"]
        return vals.to_numpy(dtype=np.float64) / 1000.0

    @property
    def licks_s(self) -> np.ndarray:
        return self.times_s(LICK_CODE)

    @property
    def sync_s(self) -> np.ndarray:
        return self.times_s(SYNC_CODE)

    @property
    def solenoid_on_s(self) -> np.ndarray:
        return self.times_s(SOLENOID_ON_CODE)


def load_arduino_log(path: str | Path) -> ArduinoLog:
    """Parse an Arduino lick log.

    Records before the first sync pulse whose code is not a timestamped
    event (lick, solenoid, sync) are treated as the start-up settings
    block and collected into :attr:`ArduinoLog.settings`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Arduino log not found: {path}")

    rows: list[tuple[int, int]] = []
    for line in _LINE_SPLIT_RE.split(path.read_text(errors="replace")):
        parts = line.replace(",", " ").split()
        if len(parts) != 2:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No '<code> <value>' records found in {path}")

    records = pd.DataFrame(rows, columns=["code", "value"])

    event_codes = {LICK_CODE, SOLENOID_ON_CODE, SOLENOID_OFF_CODE, SYNC_CODE}
    settings: dict[int, list[int]] = {}
    for code, value in rows:
        if code == SYNC_CODE:
            break
        if code not in event_codes:
            settings.setdefault(code, []).append(value)

    return ArduinoLog(path=path, records=records, settings=settings)


# --------------------------------------------------------------------------- #
# Clock alignment
# --------------------------------------------------------------------------- #


@dataclass
class ClockAlignment:
    """Linear map from Arduino time to photometry time: ``t_p = offset + slope · t_a``."""

    offset_s: float
    slope: float
    n_pulses: int
    pulse_shift: int  # index of the first matched pulse in the longer train
    residual_max_s: float
    residual_sd_s: float

    @property
    def drift_ppm(self) -> float:
        return (self.slope - 1.0) * 1e6

    def to_photometry(self, t_arduino_s: np.ndarray) -> np.ndarray:
        return self.offset_s + self.slope * np.asarray(t_arduino_s, dtype=np.float64)


def mark_onsets(time_s: np.ndarray, marks: np.ndarray) -> np.ndarray:
    """Times of the rising edges of the photometry ``marks`` stream."""
    on = np.asarray(marks) != 0
    rising = on & ~np.r_[False, on[:-1]]
    return np.asarray(time_s)[rising]


def align_clocks(
    sync_arduino_s: np.ndarray,
    mark_times_s: np.ndarray,
    *,
    pulse_shift: int | None = None,
    max_residual_s: float = 0.075,
) -> ClockAlignment:
    """Fit offset + drift between Arduino sync pulses and photometry marks.

    Pulse trains of equal length are paired one-to-one. If one train is
    longer (one recording started before or ended after the other),
    ``pulse_shift`` must say which pulse of the longer train pairs with
    the first pulse of the shorter one: the rig's sync is strictly
    periodic, so every shift fits equally well and the pairing cannot be
    inferred from the pulses alone.

    Raises
    ------
    ValueError
        If fewer than 2 pulses can be matched, the train lengths differ
        and no ``pulse_shift`` is given, or the fit leaves a residual
        above ``max_residual_s`` (default 1.5 frames at 20 Hz).
    """
    a = np.asarray(sync_arduino_s, dtype=np.float64)
    p = np.asarray(mark_times_s, dtype=np.float64)
    n = min(a.size, p.size)
    if n < 2:
        raise ValueError(
            f"need at least 2 sync pulses in each stream (arduino={a.size}, marks={p.size})"
        )
    if pulse_shift is None:
        if a.size != p.size:
            raise ValueError(
                f"sync pulse counts differ (arduino={a.size}, marks={p.size}) and the pulses are "
                "periodic, so the pairing is ambiguous; pass pulse_shift = index of the pulse in "
                "the longer train that matches the first pulse of the shorter one"
            )
        pulse_shift = 0
    if not 0 <= pulse_shift <= abs(a.size - p.size):
        raise ValueError(f"pulse_shift {pulse_shift} out of range 0..{abs(a.size - p.size)}")

    aa = a[pulse_shift : pulse_shift + n] if a.size > p.size else a[:n]
    pp = p[pulse_shift : pulse_shift + n] if p.size > a.size else p[:n]
    slope, offset = np.polyfit(aa, pp, 1)
    resid = pp - (offset + slope * aa)
    fit = ClockAlignment(
        offset_s=float(offset),
        slope=float(slope),
        n_pulses=n,
        pulse_shift=pulse_shift,
        residual_max_s=float(np.abs(resid).max()),
        residual_sd_s=float(resid.std()),
    )
    if fit.residual_max_s > max_residual_s:
        raise ValueError(
            f"sync pulses do not align: max residual {fit.residual_max_s * 1000:.0f} ms "
            f"> {max_residual_s * 1000:.0f} ms (arduino={a.size}, marks={p.size} pulses)"
        )
    return fit


# --------------------------------------------------------------------------- #
# Behavioural events
# --------------------------------------------------------------------------- #


def group_deliveries(solenoid_on_s: np.ndarray, *, gap_s: float = 1.0) -> np.ndarray:
    """Collapse solenoid pulse bursts into one onset time per delivery.

    A new delivery starts at any pulse more than ``gap_s`` after the previous one.
    """
    on = np.sort(np.asarray(solenoid_on_s, dtype=np.float64))
    if on.size == 0:
        return on
    return on[np.r_[True, np.diff(on) > gap_s]]


def first_lick_after(onsets_s: np.ndarray, licks_s: np.ndarray) -> np.ndarray:
    """First lick after each onset and before the next one (NaN if none)."""
    licks = np.sort(np.asarray(licks_s, dtype=np.float64))
    onsets = np.asarray(onsets_s, dtype=np.float64)
    bounds = np.r_[onsets[1:], np.inf]
    out = np.full(onsets.size, np.nan)
    for i, (start, stop) in enumerate(zip(onsets, bounds, strict=True)):
        j = np.searchsorted(licks, start, side="right")
        if j < licks.size and licks[j] < stop:
            out[i] = licks[j]
    return out


def build_events(
    log: ArduinoLog,
    alignment: ClockAlignment,
    *,
    delivery_gap_s: float = 1.0,
) -> pd.DataFrame:
    """Events table (``event_id_char``, ``time``) on the photometry clock."""
    onsets = alignment.to_photometry(group_deliveries(log.solenoid_on_s, gap_s=delivery_gap_s))
    licks = alignment.to_photometry(log.licks_s)
    post = first_lick_after(onsets, licks)
    post = post[~np.isnan(post)]

    frames = [
        pd.DataFrame({"event_id_char": name, "time": times})
        for name, times in (
            ("solution_onset", onsets),
            ("lick", licks),
            ("lick_post_solution", post),
        )
        if times.size
    ]
    events = pd.concat(frames, ignore_index=True)
    return events.sort_values(["time", "event_id_char"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Session folder writer
# --------------------------------------------------------------------------- #

#: Event types for which ``df_streams_peth.csv`` is pre-computed, and its
#: window, matching the R pipeline's export.
_PETH_EVENTS = ("solution_onset", "lick_post_solution")
_PETH_WINDOW_S = (-3.0, 6.0)


def _nearest_index(t: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Index of the sample in sorted ``t`` nearest to each of ``times``."""
    idx = np.clip(np.searchsorted(t, times), 1, t.size - 1)
    return idx - ((t[idx] - times) > (times - t[idx - 1]))


def _streams_with_indicators(streams: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = streams[["time", "intensity", "lifetime", "marks"]].copy()
    t = out["time"].to_numpy()
    for name in events["event_id_char"].unique():
        col = np.zeros(t.size, dtype=np.int64)
        col[_nearest_index(t, events.loc[events["event_id_char"] == name, "time"].to_numpy())] = 1
        out[name] = col
    return out


def _long_peth(streams: pd.DataFrame, events: pd.DataFrame, fs: float) -> pd.DataFrame:
    n_pre = int(round(-_PETH_WINDOW_S[0] * fs))
    n_post = int(round(_PETH_WINDOW_S[1] * fs))
    offsets = np.arange(-n_pre, n_post + 1)
    t = streams["time"].to_numpy()
    rows = []
    for event_type in _PETH_EVENTS:
        times = events.loc[events["event_id_char"] == event_type, "time"].to_numpy()
        for trial, center in enumerate(_nearest_index(t, times), start=1):
            idx = center + offsets
            keep = (idx >= 0) & (idx < t.size)
            for signal in ("intensity", "lifetime"):
                rows.append(
                    pd.DataFrame(
                        {
                            "time": t[idx[keep]],
                            "sample": idx[keep] + 1,
                            "sample_rel": offsets[keep],
                            "mark_id": 1,
                            "trial_num": trial,
                            "event_type": event_type,
                            "signal_id": signal,
                            "value": streams[signal].to_numpy()[idx[keep]],
                            "time_rel": offsets[keep] / fs,
                        }
                    )
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "time",
                "sample",
                "sample_rel",
                "mark_id",
                "trial_num",
                "event_type",
                "signal_id",
                "value",
                "time_rel",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _date_epoch(starttime: str | None) -> str:
    """``'14-Oct-2026 16:08:49.261'`` → midnight-UTC epoch seconds, as the R pipeline stores it."""
    if not starttime:
        return "NA"
    try:
        day = datetime.strptime(starttime.split()[0], "%d-%b-%Y").replace(tzinfo=UTC)
    except ValueError:
        return "NA"
    return str(int(day.timestamp()))


def build_session_folder(
    source: TidyData | IFlip2File,
    log_path: str | Path,
    data_root: str | Path,
    *,
    blockname: str | None = None,
    subject: str = "",
    procedure: str = "unknown",
    extra_meta: dict[str, str] | None = None,
    delivery_gap_s: float = 1.0,
    pulse_shift: int | None = None,
    max_residual_s: float = 0.075,
) -> Path:
    """Write ``<data_root>/sessions/<blockname>/`` from photometry + an Arduino log.

    ``blockname`` defaults to the photometry filename without its
    ``_NNN`` acquisition number, so discovery pairs the session with the
    matching tidy/raw file. ``pulse_shift`` is passed to
    :func:`align_clocks` and is only needed when the two recordings hold
    different numbers of sync pulses. Returns the session folder path.
    """
    from iflip.io.iflip2 import IFlip2File
    from iflip.io.session_csv import _strip_acq_number

    params = source.header if isinstance(source, IFlip2File) else source.params
    streams = source.streams
    fs = float(params.get("header_state_samplingfreq", "nan"))
    if not fs > 0:
        raise ValueError(f"photometry header has no valid sampling rate: {fs}")

    log = load_arduino_log(log_path)
    alignment = align_clocks(
        log.sync_s,
        mark_onsets(streams["time"].to_numpy(), streams["marks"].to_numpy()),
        pulse_shift=pulse_shift,
        max_residual_s=max_residual_s,
    )
    events = build_events(log, alignment, delivery_gap_s=delivery_gap_s)

    bn = blockname or _strip_acq_number(source.filename)
    out = Path(data_root) / "sessions" / bn
    out.mkdir(parents=True, exist_ok=True)

    meta: list[tuple[str, str, str]] = [
        ("date", _date_epoch(params.get("header_starttime")), "iflip_header"),
        ("subject", subject, "user"),
        ("blockname", bn, "importer"),
        ("filename", source.filename, "iflip_header"),
        ("procedure", procedure, "user"),
        ("fs", f"{fs:g}", "iflip_header"),
        ("t0", str(params.get("header_state_t0", "NA")), "iflip_header"),
        ("tau1", str(params.get("header_state_tau1", "NA")), "iflip_header"),
        ("tau2", str(params.get("header_state_tau2", "NA")), "iflip_header"),
        ("synch_offset", f"{alignment.offset_s:.4f}", "arduino_sync"),
        ("sync_slope", f"{alignment.slope:.8f}", "arduino_sync"),
        ("sync_drift_ppm", f"{alignment.drift_ppm:.1f}", "arduino_sync"),
        ("sync_n_pulses", str(alignment.n_pulses), "arduino_sync"),
        ("sync_residual_max_s", f"{alignment.residual_max_s:.4f}", "arduino_sync"),
        ("arduino_log", log.path.name, "importer"),
    ]
    meta += [
        (f"arduino_setting_{code}", ";".join(map(str, vals)), "arduino_log")
        for code, vals in sorted(log.settings.items())
    ]
    meta += [(k, v, "user") for k, v in (extra_meta or {}).items()]

    pd.DataFrame(meta, columns=["var", "value", "source"]).to_csv(out / "meta.csv", index=False)
    events.to_csv(out / "df_events.csv", index=False)
    session_streams = _streams_with_indicators(streams, events)
    session_streams.insert(0, "blockname", bn)
    session_streams.to_csv(out / "df_streams_session.csv", index=False)
    _long_peth(streams, events, fs).to_csv(out / "df_streams_peth.csv", index=False)
    return out
