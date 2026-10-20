"""Build a ``sessions/<blockname>/`` folder from an iFLiP recording and an Arduino lick log.

The session folder is written under ``--data-root`` (default: the folder
holding the photometry file), so the dashboard picks it up next to the
tidy/raw files when pointed at that folder.

Usage:
    .venv/bin/python scripts/import_arduino_session.py \\
        --photometry "/path/to/101426_2852L_male_001_data.csv" \\
        --lick-log "/path/to/2026_10_14_2852L_male.csv" \\
        --subject 2852L --procedure "passive sucrose"

``--photometry`` takes either a tidy ``*_data.csv`` (its ``_param.csv``
must sit beside it) or a raw ``.iFLiP2`` file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iflip.io import build_session_folder, load_iflip2, load_session, load_tidy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--photometry", type=Path, required=True)
    ap.add_argument("--lick-log", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--blockname", default=None)
    ap.add_argument("--subject", default="")
    ap.add_argument("--procedure", default="unknown")
    ap.add_argument(
        "--note",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra meta.csv row, repeatable (e.g. --note sensor=FLIM-DA1.0)",
    )
    args = ap.parse_args()

    if args.photometry.suffix.lower() == ".iflip2":
        source = load_iflip2(args.photometry)
    else:
        source = load_tidy(args.photometry)
    extra = dict(n.split("=", 1) for n in args.note)

    out = build_session_folder(
        source,
        args.lick_log,
        args.data_root or args.photometry.parent,
        blockname=args.blockname,
        subject=args.subject,
        procedure=args.procedure,
        extra_meta=extra,
    )

    session = load_session(out)
    m = session.meta
    print(f"wrote {out}")
    print(
        f"  sync: {m['sync_n_pulses']} pulses, offset {float(m['synch_offset']):.3f} s, "
        f"drift {m['sync_drift_ppm']} ppm, max residual {float(m['sync_residual_max_s']) * 1000:.0f} ms"
    )
    for name, n in session.events["event_id_char"].value_counts().sort_index().items():
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
