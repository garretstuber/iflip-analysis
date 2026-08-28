"""Reader for the proprietary ``.iFLiP2`` raw acquisition format.

The ``.iFLiP2`` file is what the FLIPR (Fluorescence Lifetime Fiber Photometry)
rig writes during acquisition on a TimeHarp 260 P TCSPC board. It is
streamed directly from the MATLAB acquisition app and contains:

1.  An ASCII header block of MATLAB-style assignments terminated by the
    literal line ``header_end``::

        header.state.syncThresh.Value = -50;
        ...
        header.startTime = '09-Apr-2026 16:35:00.283';
        header_end

    Newlines are LF (0x0A); strings are single-quoted MATLAB strings;
    multi-line strings (e.g. ``hardwareInfo``) embed real newlines.

2.  Immediately after the ``header_end\\n`` line, a packed array of
    fixed-size acquisition frames. Each frame is::

        struct frame {
            uint32_le tcspc[126];   // 126 photon-count bins (0.0–12.5 ns,
                                    //   0.1 ns wide bins, edges left-aligned).
            uint32_le marks;        // event/TTL mark code (0 = no event).
        };

    so a frame is exactly ``126 * 4 + 4 = 508`` bytes. The number of
    frames is implicit (file size minus header size, divided by 508);
    in practice it equals ``samplingfreq * acqtime`` clipped to whatever
    point the operator stopped acquisition.

The ``time``, ``intensity`` and ``lifetime`` columns the MATLAB pipeline
exports to its tidy CSV are *derived* from the per-frame TCSPC histogram
and the header parameters — they are NOT independently stored in the
binary. We reproduce them here so that ``load_iflip2`` returns the same
``streams`` table the CSV does.

- ``time`` is the bin-center sample time, ``(i + 0.5) / fs`` seconds.
  The MATLAB rig samples at 20 Hz (one frame every 50 ms) and labels
  each frame with the *center* of its 50 ms integration window, so
  frame 0 has ``time = 0.025`` and frame 1 has ``time = 0.075``, etc.

- ``intensity`` is the total photon count for the frame, ``tcspc.sum()``
  across *all* 126 bins (the MATLAB pipeline sums the full histogram —
  it does NOT restrict this to the SPC range).

- ``lifetime`` is a first-moment ("mean arrival time") estimator over
  the full 126-bin histogram::

      lifetime = sum(c · t) / sum(c)

  where ``c`` and ``t`` run over **all** bins (0.0–12.5 ns). The SPC
  range values (``header_state_spcrangelow/high``) in the header are
  the fit range for tau1/tau2 extraction, NOT the integration range
  for this real-time pseudo-lifetime, which the MATLAB pipeline
  computes across the full histogram. The published value is the
  absolute mean arrival time (not ``− t0``); typical FLIM-DA0.5 values
  sit around 2.7 ns. Subtract ``header_state_t0`` yourself if you want
  the decay τ referenced to the IRF centre.

- ``marks`` is the integer mark code straight from the binary frame.

The function :func:`load_iflip2` returns an :class:`IFlip2File` dataclass
with the parsed header dict, the streams DataFrame, the raw TCSPC
``(n_frames, 126)`` int64 array, and the bin-centre axis in ns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Format constants
# --------------------------------------------------------------------------- #

#: Number of TCSPC histogram bins per frame (0.0–12.5 ns at 0.1 ns resolution).
N_BINS: int = 126

#: Bytes per TCSPC histogram value (uint32 little-endian).
TCSPC_DTYPE = np.dtype("<u4")

#: Bytes per per-frame "marks" value (uint32 little-endian).
MARKS_DTYPE = np.dtype("<u4")

#: Total bytes per frame: ``N_BINS * 4 + 4``.
FRAME_BYTES: int = N_BINS * TCSPC_DTYPE.itemsize + MARKS_DTYPE.itemsize  # 508

#: Marker line that terminates the ASCII header block.
HEADER_END_MARKER: bytes = b"header_end"


@dataclass
class IFlip2File:
    """Parsed contents of one ``.iFLiP2`` acquisition file."""

    filename: str
    path: Path
    header: dict[str, Any] = field(default_factory=dict)
    streams: pd.DataFrame = field(default_factory=pd.DataFrame)
    tcspc: np.ndarray = field(default_factory=lambda: np.zeros((0, N_BINS), dtype=np.int64))
    tcspc_bins_ns: np.ndarray = field(default_factory=lambda: np.zeros(N_BINS, dtype=np.float64))
    header_bytes: int = 0
    frame_bytes: int = FRAME_BYTES

    @property
    def n_frames(self) -> int:
        return int(self.tcspc.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.tcspc.shape[1])

    @property
    def fs(self) -> float:
        """Sampling rate in Hz, from the header."""
        v = self.header.get("header_state_samplingfreq")
        return float(v) if v is not None else float("nan")

    @property
    def t0_ns(self) -> float:
        v = self.header.get("header_state_t0")
        return float(v) if v is not None else float("nan")

    @property
    def tau1_ns(self) -> float:
        v = self.header.get("header_state_tau1")
        return float(v) if v is not None else float("nan")

    @property
    def tau2_ns(self) -> float:
        v = self.header.get("header_state_tau2")
        return float(v) if v is not None else float("nan")

    @property
    def params(self) -> dict[str, Any]:
        """Alias for ``header`` that matches :class:`iflip.io.tidy_csv.TidyData`'s
        attribute name, so either loader can be passed through the app
        interchangeably."""
        return self.header

    def time_index(self) -> np.ndarray:
        """Timepoint axis in seconds, mirroring
        :meth:`iflip.io.tidy_csv.TidyData.time_index`."""
        return self.streams["time"].to_numpy()

    def integrate_histogram(self, start_time: float, stop_time: float) -> np.ndarray:
        """Sum TCSPC histograms over a closed time interval
        ``[start_time, stop_time]`` in seconds. Returns a ``(n_bins,)``
        int64 array. Matches the signature of the tidy-CSV loader's
        method so the Streamlit app can use either source."""
        t = self.time_index()
        mask = (t >= start_time) & (t <= stop_time)
        if not mask.any():
            return np.zeros(self.n_bins, dtype=np.int64)
        return self.tcspc[mask].sum(axis=0).astype(np.int64)


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #

#: Mapping from MATLAB struct field names (in the binary header) to the
#: ``header_<group>_<key>`` keys used by the lab's tidy ``param.csv`` and that
#: we expose on :class:`IFlip2File.header`. The MATLAB names are camelCase;
#: the tidy/param keys are lower-case.
_HEADER_KEY_PREFIX = "header"


def _matlab_to_param_key(matlab_path: str) -> str:
    """Translate ``header.state.syncThresh.Value`` → ``header_state_syncthresh``.

    Strips the trailing ``.Value`` / ``.Text`` suffix that MATLAB UI elements
    use, joins the dotted path with underscores, and lower-cases everything
    so it matches the lab's ``param.csv`` schema.
    """
    parts = matlab_path.split(".")
    if parts and parts[-1] in {"Value", "Text"}:
        parts = parts[:-1]
    # join with underscores, lower-case all components
    return "_".join(parts).lower()


def _coerce_header_value(raw: str) -> Any:
    """Convert a MATLAB-syntax RHS literal to a Python value.

    Recognised forms:
        - Single-quoted string ``'foo bar'`` → ``"foo bar"``
            (MATLAB doubles single quotes to escape; we just unescape ``''`` → ``'``)
        - Bare numeric literal ``-50`` / ``1.25e-08`` → ``int`` or ``float``
        - Anything else → returned as a stripped string
    """
    raw = raw.strip()
    # Strip a trailing semicolon if present (MATLAB statement terminator).
    if raw.endswith(";"):
        raw = raw[:-1].strip()
    if not raw:
        return ""
    # MATLAB single-quoted string
    if raw.startswith("'") and raw.endswith("'") and len(raw) >= 2:
        inner = raw[1:-1]
        return inner.replace("''", "'")
    # Try int, then float
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_header(text: str) -> dict[str, Any]:
    """Parse the ASCII header block (without the trailing ``header_end`` line).

    Each statement looks like ``LHS = RHS;``, but a value can span multiple
    lines (multi-line single-quoted strings, e.g. ``hardwareInfo``). We
    re-join lines that don't start with the expected ``header.`` prefix back
    onto the previous logical statement.
    """
    # Re-stitch logical lines: a header assignment always begins with
    # "header." (or "header_" for any future variants). Lines that don't are
    # continuation of the previous string-valued assignment.
    raw_lines = text.split("\n")
    logical_lines: list[str] = []
    for line in raw_lines:
        stripped = line.lstrip()
        if stripped.startswith(_HEADER_KEY_PREFIX + ".") or stripped.startswith(
            _HEADER_KEY_PREFIX + "_"
        ):
            logical_lines.append(line.rstrip("\r"))
        else:
            if logical_lines:
                logical_lines[-1] = logical_lines[-1] + "\n" + line.rstrip("\r")
            # else: leading garbage before first assignment — ignore
    out: dict[str, Any] = {}
    for stmt in logical_lines:
        if "=" not in stmt:
            continue
        lhs, _, rhs = stmt.partition("=")
        key = _matlab_to_param_key(lhs.strip())
        if not key:
            continue
        out[key] = _coerce_header_value(rhs)
    return out


def _split_header_and_body(buf: bytes) -> tuple[bytes, int, int]:
    """Locate the ``header_end`` line and return ``(header_text_bytes,
    header_total_size, body_offset)``.

    ``header_total_size`` is the number of bytes from the start of the file
    up to and including the ``header_end\\n`` terminator. The first frame
    starts at ``body_offset == header_total_size``.
    """
    idx = buf.find(b"\n" + HEADER_END_MARKER)
    if idx < 0:
        # Maybe the file starts with header_end (degenerate / empty header)
        if buf.startswith(HEADER_END_MARKER):
            end = len(HEADER_END_MARKER)
            if end < len(buf) and buf[end : end + 1] == b"\n":
                end += 1
            return b"", end, end
        raise ValueError("iFLiP2 file is missing the 'header_end' marker")
    header_text = buf[:idx]
    # idx points at the leading '\n' of "\nheader_end". The header_end line
    # itself runs from idx+1 to idx+1+len(HEADER_END_MARKER); skip a
    # trailing newline if there is one.
    end = idx + 1 + len(HEADER_END_MARKER)
    if end < len(buf) and buf[end : end + 1] == b"\n":
        end += 1
    return header_text, end, end


# --------------------------------------------------------------------------- #
# Frame body decoding
# --------------------------------------------------------------------------- #


def _decode_frames(body: bytes, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode the binary frame block into ``(tcspc, marks)`` arrays.

    Returns
    -------
    tcspc : (n_frames, N_BINS) int64 array
    marks : (n_frames,) int64 array
    """
    expected = n_frames * FRAME_BYTES
    if len(body) < expected:
        raise ValueError(
            f"iFLiP2 body has {len(body)} bytes; expected at least "
            f"{expected} for {n_frames} frames of {FRAME_BYTES} bytes"
        )

    # View the body as a structured array of one frame per record. Using
    # numpy's structured dtype is the fastest vectorised path here.
    frame_dtype = np.dtype(
        [
            ("tcspc", TCSPC_DTYPE, (N_BINS,)),
            ("marks", MARKS_DTYPE),
        ]
    )
    frames = np.frombuffer(body, dtype=frame_dtype, count=n_frames)
    tcspc = frames["tcspc"].astype(np.int64, copy=True)
    marks = frames["marks"].astype(np.int64, copy=True)
    return tcspc, marks


# --------------------------------------------------------------------------- #
# Derived stream columns (intensity, lifetime, time)
# --------------------------------------------------------------------------- #


def _bin_centers_ns(n_bins: int = N_BINS) -> np.ndarray:
    """Bin centres in nanoseconds: ``[0.0, 0.1, 0.2, ..., (n_bins-1)*0.1]``.

    For ``n_bins = 126`` this runs from 0.0 to 12.5 ns inclusive, matching
    the column names ``tcspc_0`` through ``tcspc_12p5`` in the tidy CSV.
    """
    return np.arange(n_bins, dtype=np.float64) * 0.1


def _compute_intensity(tcspc: np.ndarray) -> np.ndarray:
    """Per-frame photon count: total counts across the full 126-bin histogram.

    The lab MATLAB pipeline writes intensity as ``sum(tcspc)`` over all
    bins, not restricted to the SPC integration window. Reproducing that
    here makes ``intensity`` byte-equal to the tidy CSV column.

    Returns an int64 array of shape ``(n_frames,)``.
    """
    return tcspc.sum(axis=1).astype(np.int64)


def _compute_lifetime(tcspc: np.ndarray, bin_ns: np.ndarray) -> np.ndarray:
    """Per-frame mean-arrival-time lifetime estimator (ns).

    ``lifetime[k] = sum_j (c_kj · t_j) / sum_j c_kj`` where the sum runs
    over **all** bins in the 126-bin histogram (0.0–12.5 ns). This is the
    real-time pseudo-lifetime that the FLIPR rig logs to its tidy CSV
    ``lifetime`` column — an *absolute* mean arrival time, NOT referenced
    to the IRF centre ``t0``. Matches the CSV bit-for-bit.
    """
    t = bin_ns.astype(np.float64)  # (N_BINS,)
    c = tcspc.astype(np.float64)  # (n_frames, N_BINS)
    counts = c.sum(axis=1)
    weighted = c @ t
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, weighted / counts, np.nan)


def _compute_time(n_frames: int, fs: float) -> np.ndarray:
    """Frame center times in seconds: ``(i + 0.5) / fs``.

    Frame 0 → 0.025 s at fs=20 Hz, frame 1 → 0.075 s, ... matching the
    tidy CSV's ``time`` column.
    """
    if not np.isfinite(fs) or fs <= 0:
        return np.full(n_frames, np.nan, dtype=np.float64)
    return (np.arange(n_frames, dtype=np.float64) + 0.5) / float(fs)


# --------------------------------------------------------------------------- #
# Public loader
# --------------------------------------------------------------------------- #


def load_iflip2(path: str | Path) -> IFlip2File:
    """Load a ``.iFLiP2`` raw acquisition file from disk.

    Parameters
    ----------
    path
        Filesystem path to the ``.iFLiP2`` file.

    Returns
    -------
    IFlip2File
        Parsed header dict, derived ``streams`` DataFrame
        (``time, intensity, lifetime, marks``), TCSPC histograms as a
        ``(n_frames, 126)`` int64 array, and the bin-centre axis in ns.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"iFLiP2 file not found: {path}")

    buf = path.read_bytes()
    header_text, header_size, body_offset = _split_header_and_body(buf)
    header = _parse_header(header_text.decode("latin-1", errors="replace"))

    body = buf[body_offset:]
    body_len = len(body)
    if body_len % FRAME_BYTES != 0:
        # Some acquisitions stop mid-frame; truncate to whole frames and
        # carry on rather than raising — but raise on a totally bogus size.
        n_frames = body_len // FRAME_BYTES
        if n_frames == 0:
            raise ValueError(
                f"iFLiP2 body of {body_len} bytes is shorter than one {FRAME_BYTES}-byte frame"
            )
    else:
        n_frames = body_len // FRAME_BYTES

    tcspc, marks = _decode_frames(body, n_frames)

    bin_ns = _bin_centers_ns(N_BINS)
    fs = float(header.get("header_state_samplingfreq", 20.0))

    intensity = _compute_intensity(tcspc)
    lifetime = _compute_lifetime(tcspc, bin_ns)
    time_s = _compute_time(n_frames, fs)

    streams = pd.DataFrame(
        {
            "time": time_s,
            "intensity": intensity,
            "lifetime": lifetime,
            "marks": marks,
        }
    )

    # Tidy ``param.csv`` exposes a top-level ``filename`` field that is just
    # the basename without the ``.iFLiP2`` extension. The binary header
    # stores the full Windows path under ``header_acq_filename`` (and the
    # directory under ``header_acq_pathname``), so reduce that here.
    acq_filename = str(header.get("header_acq_filename") or "")
    if acq_filename:
        # MATLAB writes Windows-style paths even on Unix; strip both
        # separators and the .iFLiP2 extension.
        base = acq_filename.replace("\\", "/").rsplit("/", 1)[-1]
        if base.lower().endswith(".iflip2"):
            base = base[: -len(".iflip2")]
        filename = base
    else:
        filename = path.stem
    header["filename"] = filename

    return IFlip2File(
        filename=filename,
        path=path,
        header=header,
        streams=streams,
        tcspc=tcspc,
        tcspc_bins_ns=bin_ns,
        header_bytes=header_size,
        frame_bytes=FRAME_BYTES,
    )


def list_iflip2_files(data_root: str | Path) -> list[Path]:
    """List all ``*.iFLiP2`` files under ``<data_root>/raw``.

    Returns an empty list if the ``raw/`` directory is missing.
    """
    raw_dir = Path(data_root) / "raw"
    if not raw_dir.is_dir():
        return []
    # Case-insensitive glob: .iFLiP2 / .iflip2
    out: set[Path] = set()
    out.update(raw_dir.glob("*.iFLiP2"))
    out.update(raw_dir.glob("*.iflip2"))
    return sorted(out)


def match_iflip2_to_session(files: list[Path], blockname: str) -> Path | None:
    """Pick the ``.iFLiP2`` file whose stem starts with ``<blockname>_``.

    Mirrors :func:`iflip.io.tidy_csv.match_tidy_to_session`: blockname
    ``2026_04_09_acz02`` matches ``2026_04_09_acz02_001.iFLiP2``. Returns
    the lexicographically earliest match or ``None``.
    """
    prefix = f"{blockname}_"
    matches = [p for p in files if p.name.startswith(prefix)]
    if not matches:
        return None
    return sorted(matches)[0]


__all__ = [
    "FRAME_BYTES",
    "N_BINS",
    "IFlip2File",
    "list_iflip2_files",
    "load_iflip2",
    "match_iflip2_to_session",
]
