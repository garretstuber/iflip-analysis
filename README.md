# flipr-analysis

Visualization and preprocessing tools for **Fluorescence Lifetime Fiber Photometry (FLIPR)** data acquired with the FLIM-DA0.5 sensor and TimeHarp 260P TCSPC hardware.

## What it does

- Loads FLIPR data from the lab's CSV export pipeline (`tidy/` + `sessions/`)
- Interactive Streamlit dashboard for session QC and visual inspection
- Per-interval TCSPC decay curve inspection with double-exponential re-fitting
- Event-aligned analyses (PETH, per-trial heatmaps, z-scored traces)
- Phasor analysis via [PhasorPy](https://www.phasorpy.org)

## Data layout expected

The app reads from a FLIPR data root containing:

```
<data_root>/
├── tidy/
│   ├── <blockname>_NNN_data.csv     # per-timepoint TCSPC histograms (126 bins @ 0.1 ns)
│   └── <blockname>_NNN_param.csv    # acquisition header (t0, IRF, fit params)
└── sessions/
    └── <blockname>/
        ├── meta.csv                 # subject, procedure, baselines
        ├── df_events.csv            # behavioral event timestamps
        ├── df_streams_session.csv   # long-format time/intensity/lifetime
        └── df_streams_peth.csv      # event-aligned PETH in long format
```

## Setup

```bash
uv venv
uv pip install -e .
```

## Run the dashboard

```bash
uv run streamlit run src/flipr/app/streamlit_app.py
```

Then point the sidebar at your FLIPR data root.

## Status

🚧 Early scaffold — IO loaders first, then Streamlit viewer.
