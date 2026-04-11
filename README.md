<p align="center">
  <img src="src/flipr/app/assets/FLIPRlogo.png" alt="FLIPR logo" width="320"/>
</p>

<h1 align="center">flipr-analysis</h1>

<p align="center">
  Visualization and preprocessing tools for <b>Fluorescence Lifetime Fiber Photometry (FLIPR)</b> —
  an interactive Streamlit dashboard, a library of reusable analysis modules,
  and a byte-exact reader for the lab's proprietary raw acquisition format.
</p>

---

## Overview

FLIPR records dopamine (and other) biosensor dynamics with a fiber-coupled TCSPC
rig, producing not just intensity traces but a full photon-arrival-time
histogram per frame at ~20 Hz. Because fluorescence lifetime is ratiometric
— independent of total photon count, excitation power, or fiber-coupling
efficiency — it provides a direct quantitative readout of sensor state
that intensity alone cannot. This package turns that data into a tool
researchers can actually drive: preprocess, inspect, event-align, and
phasor-analyse sessions from a single Streamlit app, with all of the
underlying analysis exposed as a clean Python library for scripted work.

This project targets data acquired on a **TimeHarp 260 P** board driven by
the lab's MATLAB acquisition software, with the **FLIM-DA0.5** dopamine
biosensor on the biology side. Nothing in the pipeline is hardcoded to
those specific devices or sensors — any 80 MHz rep-rate TCSPC photometry
setup with a 126-bin × 0.1 ns histogram at 20 Hz will work out of the
box. Different rig geometries just need a different period / bin axis.

---

## Features

### Data loading

- **Raw `.iFLiP2` reader** — byte-exact parser for the proprietary binary
  format produced by the acquisition rig (ASCII header block, followed by
  packed 508-byte frames of 126 uint32 TCSPC bins + 1 uint32 mark code).
  Loads directly from the raw file with no CSV intermediate. Verified
  against the lab's MATLAB tidy-CSV export on a real session:
  TCSPC counts are byte-equal, marks match exactly, intensity is exact,
  time and lifetime match to float64 round-off, and header numeric
  fields agree to 1e-4.
- **Tidy CSV loader** — reads the lab's R-pipeline tidy exports
  (`tidy/*_data.csv` + `tidy/*_param.csv`). Same data, same interface,
  used as a fallback when only the exported form is available.
- **Session CSV loader** — reads the lab's aligned session folders
  (`sessions/<block>/meta.csv`, `df_events.csv`, `df_streams_session.csv`,
  `df_streams_peth.csv`). Pulls subject, procedure, sampling rate, event
  list, and continuous streams into a single `SessionData` dataclass.

### Preprocessing

- **Double-exponential TCSPC fitting** with periodic wrap-around.
  Forward model is a numerically-stable sum-of-EMG with the rising edge
  computed via `scipy.special.erfcx` and the decay tail via direct `erfc`
  so the fit is finite across the full 0–12.5 ns window. The laser period
  (default 12.5 ns at 80 MHz) is baked into the model by summing the
  current pulse plus N previous-pulse wraps — essential on FLIM-DA0.5
  where τ₁ ≈ T/3 and ~4% of the slow component spills across periods.
- **Phasor analysis** via [PhasorPy](https://www.phasorpy.org). Per-frame
  `(G, S)` coordinates, phasor cloud over a session, reference-τ markers
  on the universal semicircle, apparent τ_phase / τ_modulation inversion.
- **Stream filtering** — Savitzky-Golay, moving-average boxcar, median,
  or raw passthrough. Global filter selection in the sidebar propagates
  to the session overview trace and to the PETH source, so event-aligned
  analyses automatically use the smoothed signal. The interval inspector
  and phasor explorer continue to read the raw TCSPC histograms so decay
  curves and phasor coordinates stay untouched by stream smoothing.
- **QC / artifact detection**
  - Robust photon-starvation flag (`median − 5·1.4826·MAD`, so a stable
    head-fixed recording flags ~0%).
  - Intensity-jump flag based on a robust z-score of `dI/dt`.
  - Rolling intensity/lifetime correlation shown as a **diagnostic trace
    only** — not part of the flag mask, because FLIM biosensor signals
    naturally produce correlated changes in brightness and lifetime.

### Event alignment

- **PETH engine** — rebuilds per-trial trial × time matrices from the
  raw streams and event list, so you can pick arbitrary pre/post windows
  without being tied to whatever the acquisition pipeline chose.
  Supports baseline correction, per-trial z-scoring, and AUC
  quantification over user-specified windows. Verified to match the
  lab's pre-computed `df_streams_peth.csv` on the example session.

### Streamlit dashboard

The app runs as a four-tab Streamlit application you point at any FLIPR
data root.

- **Session overview** — linked intensity and lifetime traces with event
  markers, QC metric strip (mean intensity, CV, mean τ, τ std), motion
  diagnostic panel with per-flag percentages and a collapsible rolling
  correlation trace, session metadata side panel. Selected interval
  from the Interval Inspector is shown as a translucent highlight band
  spanning both panels.
- **Interval inspector** — range slider on the session, sums TCSPC
  histograms inside the window, re-fits the double exponential, and
  displays:
  - log-y decay plot with fit overlay and shaded fit range
  - weighted residual subpanel
  - side-by-side fit parameter / instrument header tables
  - editable fit settings (IRF σ, t₀, fit range)
- **Phasor explorer** — full-session phasor trajectory at the 80 MHz
  fundamental, coloured by session time / trial / uniform, with the
  universal semicircle, reference-τ diamond markers from the header
  values, and a star highlighting the currently-selected interval
  window's phasor point. Clearly labelled as uncalibrated (a reference-
  dye calibration step is a planned v2 addition).
- **Event PETH** — event type + signal + normalisation selectors,
  configurable pre / post and baseline windows, raw / baseline-corrected
  / z-scored modes, per-trial heatmap + mean ± SEM trace, and a summary
  metric row (n trials, peak |mean|, peak time, post-event AUC ± SEM).

---

## Data layout

The app and library work with any FLIPR data root laid out like this.
At minimum, you need one of `tidy/` or `raw/` (for per-frame TCSPC
histograms) and a `sessions/<blockname>/` directory (for behavioural
events and continuous streams).

```text
<data_root>/
├── raw/                                 ← optional (byte-exact source)
│   └── <blockname>_NNN.iFLiP2           ← raw acquisition binary
│
├── tidy/                                ← alternative to raw/
│   ├── <blockname>_NNN_data.csv         ← per-frame TCSPC histograms
│   │                                      (126 bins @ 0.1 ns, tcspc_0 … tcspc_12p5)
│   │                                      plus time, intensity, lifetime, marks
│   └── <blockname>_NNN_param.csv        ← acquisition header
│                                          (t0, tau1, tau2, sync rate, ...)
│
└── sessions/
    └── <blockname>/
        ├── meta.csv                     ← subject, procedure, baselines, notes
        ├── df_events.csv                ← behavioural event timestamps
        ├── df_streams_session.csv       ← wide-format time/intensity/lifetime
        │                                  streams at fs Hz
        └── df_streams_peth.csv          ← pre-baked PETHs (long format) —
                                           optional; the PETH tab rebuilds
                                           matrices directly from streams.
```

### Expected columns

| File | Columns |
|---|---|
| `tidy/*_data.csv` | `time, intensity, lifetime, marks, tcspc_0, tcspc_0p1, …, tcspc_12p5` |
| `tidy/*_param.csv` | `variable, value` — TimeHarp header fields (`header_state_*`, `header_init_*`, `header_acq_*`, `header_starttime`) |
| `sessions/<block>/meta.csv` | `var, value, source` — subject, procedure, fs, `bl_fl`, `bl_int`, `IRF`, `t0`, `tau1`, `tau2` |
| `sessions/<block>/df_events.csv` | `event_id_char, time` — one row per event (e.g. `solution_onset`, `lick`) |
| `sessions/<block>/df_streams_session.csv` | `blockname, time, intensity, lifetime, marks, <event indicator cols>` |
| `sessions/<block>/df_streams_peth.csv` | `time, sample, sample_rel, mark_id, trial_num, event_type, signal_id, value, time_rel` |

If your rig produces a different TCSPC period or bin resolution, pass
the right `period_ns` to `phasor_from_histogram`, `fit_double_exp`, etc.
Nothing in the loaders is hardcoded beyond the column schema.

---

## Installation

### Requirements

- **Python ≥ 3.12** (PhasorPy requires this)
- A FLIPR data directory laid out as above
- macOS, Linux, or Windows (tested on macOS)

### Dependencies

Installed automatically via `pip install -e .`:

| Package | Used for |
|---|---|
| `numpy >= 1.26` | numerics |
| `pandas >= 2.2` | stream + event tables |
| `scipy >= 1.13` | `curve_fit`, `special.erfcx`, `ndimage` filters |
| `xarray >= 2024.6` | reserved for multi-dim container use |
| `plotly >= 5.22` | all dashboard figures |
| `streamlit >= 1.36` | dashboard UI |
| `phasorpy >= 0.9` | phasor transforms, semicircle, calibration |
| `pyarrow >= 16.0` | Streamlit dataframe serialisation backend |

Optional (dev) extras — `pip install -e ".[dev]"`:

| Package | Used for |
|---|---|
| `pytest >= 8.0` | test runner |
| `ruff >= 0.5` | linter + formatter |
| `ipykernel >= 6.29` | Jupyter / notebook interop |

### Setup with `uv` (recommended)

```bash
git clone https://github.com/garretstuber/flipr-analysis.git
cd flipr-analysis
uv venv --python 3.12
uv pip install -e ".[dev]"
```

### Setup with `pip` / `venv`

```bash
git clone https://github.com/garretstuber/flipr-analysis.git
cd flipr-analysis
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Quick start

### Run the dashboard

```bash
uv run streamlit run src/flipr/app/streamlit_app.py
# or, if .venv is activated:
streamlit run src/flipr/app/streamlit_app.py
```

Point the **Data root** field at your FLIPR data directory, pick a session
from the dropdown, and optionally switch between the raw `.iFLiP2` and
tidy CSV sources.

### Use the library from Python

```python
from pathlib import Path
from flipr.io import load_iflip2, load_session
from flipr.align import build_peth
from flipr.preprocess import filtered_session, compute_qc, fit_double_exp

data_root = Path("/path/to/FLIPR data")

# Load the session (streams + events) and the raw TCSPC file
session = load_session(data_root / "sessions" / "2026_04_09_acz02")
raw = load_iflip2(data_root / "raw" / "2026_04_09_acz02_001.iFLiP2")

# Smooth streams, run QC
smoothed = filtered_session(session, mode="savgol", window_s=0.3, polyorder=2)
qc = compute_qc(smoothed)
print(qc.summary())

# Fit a TCSPC interval
hist = raw.integrate_histogram(100.0, 120.0)
fit = fit_double_exp(hist, raw.tcspc_bins_ns)
print(f"τ_amp = {fit.tau_amp_weighted:.3f} ns, χ² = {fit.chi2_reduced:.2f}")

# Build a PETH around solution onsets
peth = build_peth(
    smoothed,
    event_type="solution_onset",
    signal="lifetime",
    pre_window=-3.0,
    post_window=5.0,
)
peth_bc = peth.baseline_corrected(window=(-2.5, -0.1))
print(f"mean response peak: {peth_bc.mean().max():+.4f} ns")
```

---

## Repository layout

```text
src/flipr/
├── io/
│   ├── iflip2.py        ← raw .iFLiP2 reader (byte-exact)
│   ├── session_csv.py   ← sessions/<block>/ loader
│   └── tidy_csv.py      ← tidy/ CSV loader
├── preprocess/
│   ├── lifetime.py      ← double-exp fit with periodic wrap
│   ├── phasor.py        ← PhasorPy wrapper
│   ├── filters.py       ← stream smoothing (boxcar / savgol / median)
│   └── motion.py        ← QC + artifact detection
├── align/
│   └── peth.py          ← event-aligned matrix builder
├── viz/
│   ├── traces.py        ← session overview figure
│   ├── tcspc.py         ← decay + fit overlay
│   ├── peth.py          ← per-trial heatmap + mean/SEM
│   └── phasor.py        ← phasor scatter + semicircle
└── app/
    ├── streamlit_app.py ← the dashboard
    ├── cli.py           ← console entry point
    └── assets/          ← logo image

tests/                   ← 73 tests (pytest)
scripts/
└── iflip2_diagnose.py   ← standalone parity report for the raw reader
```

---

## Development

### Tests

```bash
uv run pytest                     # full suite (73 tests)
uv run pytest -v tests/test_peth  # one module
uv run pytest -k lifetime         # filter by name
```

Real-data tests auto-skip if the `FLIPR data/` directory isn't next to
the repository; synthetic tests always run.

### Lint / format

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/
```

### Diagnostic scripts

```bash
# One-shot parity check: raw .iFLiP2 vs tidy CSV export
uv run python scripts/iflip2_diagnose.py
```

---

## Known limitations

- **Gaussian IRF only.** `fit_double_exp` models the IRF as a single
  Gaussian with configurable σ. This recovers the amplitude-weighted
  mean lifetime accurately (within a few percent of the instrument's
  reference fit) but can redistribute the individual τ₁ and τ₂
  components differently than a measured-IRF fit would. A numerical
  convolution against a measured reference IRF is a planned v2
  addition.
- **Uncalibrated phasor.** The phasor tab shows raw `(G, S)` coordinates.
  The instrument IRF rotates the whole cloud by a fixed angle, so
  points can sit slightly off the universal semicircle. Relative
  drift and cloud shape are still meaningful. Reference-dye
  calibration via `phasorpy.lifetime.phasor_calibrate` is planned for
  v2 once we have a matching calibration acquisition to validate
  against.
- **Head-fixed recordings only**, practically speaking. The QC
  thresholds and motion-flag defaults are tuned for stable fiber
  coupling. Freely-moving rigs with jittery fibers would want
  different sensitivities (and probably an accelerometer channel to
  regress against).

---

## Acknowledgements

- [PhasorPy](https://www.phasorpy.org) — the phasor-analysis library
  this project wraps for its phasor tab.
- TimeHarp 260 P (PicoQuant) — the TCSPC hardware the rig is built on.
- FLIM-DA0.5 — the fluorescence lifetime dopamine biosensor this
  project was built for.

---

## License

MIT — see `pyproject.toml` for the full declaration.
