"""Quick diagnostic for the iFLiP2 reader.

Loads the example raw acquisition + tidy CSV pair and prints a per-field
parity report. Useful when iterating on the binary layout / lifetime
formula in ``src/iflip/io/iflip2.py``.

Usage:
    .venv/bin/python scripts/iflip2_diagnose.py

Expects the FLIPR data root to live at ``../FLIPR data`` relative to this
repo (the same place :mod:`tests.test_iflip2` looks for it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iflip.io import load_iflip2, load_tidy

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO.parent / "FLIPR data"
RAW_FILE = DATA_ROOT / "raw" / "2026_04_09_acz02_001.iFLiP2"
TIDY_DATA = DATA_ROOT / "tidy" / "2026_04_09_acz02_001_data.csv"
TIDY_PARAM = DATA_ROOT / "tidy" / "2026_04_09_acz02_001_param.csv"


def hexdump(buf: bytes, offset: int, length: int = 64) -> str:
    """Tiny xxd-style dump."""
    out = []
    for i in range(0, length, 16):
        chunk = buf[offset + i : offset + i + 16]
        hexpart = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        out.append(f"  {offset + i:08x}  {hexpart:<48}  {ascii_part}")
    return "\n".join(out)


def main() -> None:
    print(f"raw file : {RAW_FILE}")
    print(f"size     : {RAW_FILE.stat().st_size} bytes\n")

    raw = load_iflip2(RAW_FILE)
    tidy = load_tidy(TIDY_DATA, TIDY_PARAM)

    print(f"n_frames : raw={raw.n_frames}  tidy={tidy.n_time}")
    print(f"n_bins   : raw={raw.n_bins}  tidy={tidy.n_bins}")
    print(f"frame_b  : {raw.frame_bytes}")
    print(f"hdr_b    : {raw.header_bytes}")
    print()

    # First frame TCSPC parity
    a0 = raw.tcspc[0]
    b0 = tidy.tcspc[0]
    print(f"frame 0 TCSPC equal? {np.array_equal(a0, b0)}")
    if not np.array_equal(a0, b0):
        d = np.argwhere(a0 != b0).ravel()[:5]
        print(f"  first mismatches at bins: {d.tolist()}")
        print(f"  raw[:8]={a0[:8].tolist()}")
        print(f"  tidy[:8]={b0[:8].tolist()}")

    # Streams
    for col in ("time", "intensity", "lifetime", "marks"):
        a = raw.streams[col].to_numpy()
        b = tidy.streams[col].to_numpy()
        if a.dtype.kind in "iu" and b.dtype.kind in "iu":
            eq = np.array_equal(a, b)
            print(f"{col:9s}: equal={eq}  raw[0]={a[0]}  tidy[0]={b[0]}")
        else:
            diff = np.abs(a - b)
            print(
                f"{col:9s}: max|diff|={np.nanmax(diff):.3e}  "
                f"mean|diff|={np.nanmean(diff):.3e}  "
                f"raw[0]={a[0]:.6g}  tidy[0]={b[0]:.6g}"
            )

    # Header sanity
    keys = [
        "header_state_t0",
        "header_state_tau1",
        "header_state_tau2",
        "header_state_samplingfreq",
        "header_state_acqtime",
        "header_init_syncrate",
        "header_state_spcrangelow",
        "header_state_spcrangehigh",
        "filename",
    ]
    print()
    print("header parity:")
    for k in keys:
        rv = raw.header.get(k)
        tv = tidy.params.get(k)
        print(f"  {k:35s}  raw={rv!r}  tidy={tv!r}")

    # First-frame raw bytes (helpful when re-checking the layout)
    body_offset = raw.header_bytes
    print()
    print(f"first 64 bytes after header (offset {body_offset}):")
    print(hexdump(RAW_FILE.read_bytes(), body_offset, 64))


if __name__ == "__main__":
    main()
