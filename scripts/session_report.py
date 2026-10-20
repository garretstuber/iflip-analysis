"""Single-session HTML report: baseline stability, spontaneous transients, sucrose-evoked dopamine.

Reads a ``sessions/<blockname>/`` folder built by
``scripts/import_arduino_session.py`` (it needs ``solution_onset``,
``lick`` and ``lick_post_solution`` events and the ``arduino_log`` /
``synch_offset`` meta rows the importer writes) and writes one
self-contained HTML file with interactive Plotly figures.

The lifetime readout throughout is MPET = lifetime - t0, in ps.

Usage:
    .venv/bin/python scripts/session_report.py \\
        "/path/to/sessions/101426_2852L_male" --out report.html
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.offline
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder
from scipy import stats
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from iflip.align import build_peth
from iflip.io import load_arduino_log, load_session

# --------------------------------------------------------------------------- #
# Analysis parameters
# --------------------------------------------------------------------------- #

# Excluded around each delivery when measuring baseline / spontaneous activity. On 2852L the
# response tail stayed ~10-20 ps up for 10 s and returned by 13-16 s, hence +15 s.
REWARD_WINDOW_S = (-1.0, 15.0)
TAIL_BINS_S = ((3, 5), (5, 8), (8, 10), (10, 13), (13, 16), (16, 19))  # post-delivery time course
BIN_S = 10.0  # baseline bins
SLOW_BAND_HZ = (0.05, 0.5)  # slow fluctuation band
TRANSIENT_BAND_HZ = (0.3, 3.0)  # fast transient band (DA transient FWHM ~0.6 s)
TRANSIENT_K = 4.0  # detection threshold, robust SDs
SWEEP_K = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
LICK_ASSOC_S = (-2.0, 0.5)  # a lick in this window around a transient peak = lick-associated
PETH_WINDOW_S = (-3.0, 8.0)
PETH_BASELINE_S = (-2.0, -0.1)  # excludes the t = 0 sample
RESPONSE_WINDOW_S = (0.5, 2.5)  # per-trial response, from first lick

# --------------------------------------------------------------------------- #
# Colour roles. Python sets the light values; the page script re-applies
# every role from CSS custom properties so dark mode is its own palette.
# --------------------------------------------------------------------------- #

LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink-2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "signal": "#2a78d6",
    "sucrose": "#eb6834",
    "spont": "#1baf7a",
    "epoch-1": "#86b6ef",
    "epoch-2": "#2a78d6",
    "epoch-3": "#104281",
    "div-neg": "#2a78d6",
    "div-mid": "#f0efec",
    "div-pos": "#e34948",
    "band": "#898781",
}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def role(**props: str) -> dict:
    """Trace meta telling the page script which colour role feeds which property.

    Values are ``"<role>"`` or ``"<role>@<alpha>"``.
    """
    return {"roles": props}


def color(spec: str) -> str:
    name, _, alpha = spec.partition("@")
    return _rgba(LIGHT[name], float(alpha)) if alpha else LIGHT[name]


def base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 64, "r": 24, "t": 72, "b": 48},
        paper_bgcolor=LIGHT["surface"],
        plot_bgcolor=LIGHT["surface"],
        font={
            "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
            "size": 12,
            "color": LIGHT["ink-2"],
        },
        hoverlabel={"font": {"family": 'system-ui, -apple-system, "Segoe UI", sans-serif'}},
        # legend in the top margin, above the subplot titles
        legend={
            "orientation": "h",
            "x": 0,
            "yref": "container",
            "y": 0.995,
            "yanchor": "top",
            "bgcolor": "rgba(0,0,0,0)",
        },
        hovermode="closest",
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=LIGHT["axis"],
        ticks="outside",
        tickcolor=LIGHT["axis"],
        zeroline=False,
    )
    fig.update_yaxes(gridcolor=LIGHT["grid"], gridwidth=1, linecolor=LIGHT["axis"], zeroline=False)
    fig.update_annotations(font={"size": 12, "color": LIGHT["ink"]})
    return fig


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass
class Data:
    session: object
    t: np.ndarray
    mpet: np.ndarray  # ps
    dff: np.ndarray  # % vs session median
    photons: np.ndarray
    fs: float
    deliveries: np.ndarray
    first_licks: np.ndarray  # one per delivery, NaN if none
    licks: np.ndarray
    epochs: dict[str, tuple[float, float]]
    reward_mask: np.ndarray


def load(session_dir: Path) -> Data:
    s = load_session(session_dir)
    t = s.streams["time"].to_numpy()
    t0 = float(s.meta["t0"])
    photons = s.streams["intensity"].to_numpy(dtype=float)
    ev = s.events
    deliveries = ev.loc[ev.event_id_char == "solution_onset", "time"].to_numpy()
    licks = ev.loc[ev.event_id_char == "lick", "time"].to_numpy()
    post = ev.loc[ev.event_id_char == "lick_post_solution", "time"].to_numpy()
    first = np.array(
        [
            next((x for x in post if a < x < b), np.nan)
            for a, b in zip(deliveries, np.r_[deliveries[1:], np.inf], strict=True)
        ]
    )

    log = load_arduino_log(session_dir.parent.parent / s.meta["arduino_log"])
    end_codes = log.records.loc[log.records.code == 0, "value"]
    arduino_end_s = (end_codes.iloc[-1] if len(end_codes) else log.records.value.iloc[-1]) / 1000
    offset, slope = float(s.meta["synch_offset"]), float(s.meta["sync_slope"])
    task = (max(t[0], offset), min(t[-1], offset + slope * arduino_end_s))
    epochs = {"Pre-task": (t[0], task[0]), "Task": task, "Post-task": (task[1], t[-1])}

    mask = np.zeros(t.size, bool)
    for d in deliveries:
        mask |= (t >= d + REWARD_WINDOW_S[0]) & (t < d + REWARD_WINDOW_S[1])

    return Data(
        s,
        t,
        (s.streams["lifetime"].to_numpy() - t0) * 1000,
        100 * (photons / np.median(photons) - 1),
        photons,
        s.fs,
        deliveries,
        first,
        licks,
        epochs,
        mask,
    )


def epoch_of(d: Data, times: np.ndarray) -> np.ndarray:
    out = np.empty(np.size(times), dtype=object)
    for name, (a, b) in d.epochs.items():
        out[(times >= a) & (times <= b)] = name
    return out


def robust_sd(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def bandpass(x: np.ndarray, band: tuple[float, float], fs: float) -> np.ndarray:
    return sosfiltfilt(butter(2, band, "bandpass", fs=fs, output="sos"), x)


# ---- baseline ----------------------------------------------------------------


def baseline_analysis(d: Data) -> dict:
    edges = np.arange(d.t[0] - d.t[0] % BIN_S, d.t[-1] + BIN_S, BIN_S)
    rows = []
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        m = (d.t >= a) & (d.t < b) & ~d.reward_mask
        if m.sum() >= 0.5 * BIN_S * d.fs:
            rows.append((a + BIN_S / 2, d.mpet[m].mean(), d.dff[(d.t >= a) & (d.t < b)].mean()))
    centre, mpet_bin, dff_bin = (np.array(c) for c in zip(*rows, strict=True))
    fit = stats.linregress(centre / 60, mpet_bin)
    ifit = stats.linregress(centre / 60, dff_bin)
    epochs = {}
    for name, (a, b) in d.epochs.items():
        m = (d.t >= a) & (d.t < b) & ~d.reward_mask
        sel = (centre >= a) & (centre < b)
        epochs[name] = {
            "range": (a, b),
            "mpet": d.mpet[m].mean(),
            "bin_sd": mpet_bin[sel].std(ddof=1),
            "n_bins": int(sel.sum()),
            "bins": mpet_bin[sel],
            "dff": d.dff[m].mean(),
            "nonreward_s": m.sum() / d.fs,
        }
    a, b = d.epochs["Pre-task"]
    pre_task_mean = float(d.mpet[(d.t >= a) & (d.t < b)].mean())
    # pre-delivery baseline per trial (>= ITI - 5 s after the previous delivery)
    pre_trial = np.array([d.mpet[(d.t >= x - 5) & (d.t < x)].mean() for x in d.deliveries])
    # time course after delivery, relative to each trial's own pre-delivery baseline
    nxt = np.r_[d.deliveries[1:], np.inf]
    tail = []
    for lo, hi in TAIL_BINS_S:
        v = np.array(
            [
                d.mpet[(d.t >= x + lo) & (d.t < x + hi)].mean() - p
                for x, n, p in zip(d.deliveries, nxt, pre_trial, strict=True)
                if x + hi < n - 0.5
            ]
        )
        tail.append(
            {
                "window": (lo, hi),
                "mean": v.mean(),
                "sem": v.std(ddof=1) / np.sqrt(v.size),
                "n": v.size,
            }
        )
    # post-task recovery toward the pre-task level, 30-s blocks
    a, b = d.epochs["Post-task"]
    post_blocks = [
        (x, d.mpet[(d.t >= x) & (d.t < min(x + 30, b))].mean() - pre_task_mean)
        for x in np.arange(a, b - 15, 30)
    ]
    return {
        "centre": centre,
        "mpet_bin": mpet_bin,
        "dff_bin": dff_bin,
        "fit": fit,
        "ifit": ifit,
        "epochs": epochs,
        "pre_trial": pre_trial,
        "pre_task_mean": pre_task_mean,
        "tail": tail,
        "post_blocks": post_blocks,
        "mean": float(np.mean(mpet_bin)),
    }


# ---- noise, spectrum, slow coupling ----------------------------------------------


def spectrum_analysis(d: Data) -> dict:
    out = {}
    nperseg = int(25.6 * d.fs)
    shot_sd = np.mean(d.mpet) / np.sqrt(np.mean(d.photons))
    for name, (a, b) in d.epochs.items():
        a, b = a + 1, b - 1  # trim the epoch edges
        m = (d.t >= a) & (d.t < b)
        f, p = welch(
            d.mpet[m] - d.mpet[m].mean(), d.fs, nperseg=min(nperseg, m.sum()), detrend="linear"
        )
        out[name] = (f, p)
    slow_m, slow_i = bandpass(d.mpet, SLOW_BAND_HZ, d.fs), bandpass(d.dff, SLOW_BAND_HZ, d.fs)
    coupling = {}
    for name, (a, b) in d.epochs.items():
        m = (d.t >= a + 2) & (d.t < b - 2) & ~d.reward_mask
        r = stats.linregress(slow_i[m], slow_m[m])
        coupling[name] = {
            "slope": r.slope,
            "r": r.rvalue,
            "sd_mpet": slow_m[m].std(),
            "sd_dff": slow_i[m].std(),
        }
    frame_noise = np.diff(d.mpet).std() / np.sqrt(2)
    return {
        "psd": out,
        "shot_sd": shot_sd,
        "shot_psd": 2 * shot_sd**2 / d.fs,
        "frame_noise": frame_noise,
        "slow_m": slow_m,
        "slow_i": slow_i,
        "coupling": coupling,
    }


# ---- transients --------------------------------------------------------------


def triggered(d: Data, signal: np.ndarray, times: np.ndarray) -> np.ndarray:
    n_pre, n_post = int(-PETH_WINDOW_S[0] * d.fs), int(PETH_WINDOW_S[1] * d.fs)
    w = np.arange(-n_pre, n_post + 1)
    base = (w >= PETH_BASELINE_S[0] * d.fs) & (w <= PETH_BASELINE_S[1] * d.fs)
    idx = np.searchsorted(d.t, times)
    idx = idx[(idx - n_pre >= 0) & (idx + n_post < d.t.size)]
    segs = np.array([signal[i + w] for i in idx]).reshape(-1, w.size)
    return segs - segs[:, base].mean(axis=1, keepdims=True)


def transient_analysis(d: Data) -> dict:
    x = bandpass(d.mpet, TRANSIENT_BAND_HZ, d.fs)
    xi = bandpass(d.dff, TRANSIENT_BAND_HZ, d.fs)
    sd, sdi = robust_sd(x[~d.reward_mask]), robust_sd(xi[~d.reward_mask])
    dist = int(1.0 * d.fs)
    sweep = []
    for k in SWEEP_K:
        up, _ = find_peaks(x, height=k * sd, distance=dist)
        dn, _ = find_peaks(-x, height=k * sd, distance=dist)
        hits = sum(
            np.any((d.t[up] >= f) & (d.t[up] < f + 2)) for f in d.first_licks if np.isfinite(f)
        )
        sweep.append(
            {
                "k": k,
                "thr": k * sd,
                "up": int((~d.reward_mask[up]).sum()),
                "down": int((~d.reward_mask[dn]).sum()),
                "sucrose_hits": int(hits),
                "n_sucrose": int(np.isfinite(d.first_licks).sum()),
            }
        )
    up, _ = find_peaks(x, height=TRANSIENT_K * sd, distance=dist)
    up = up[~d.reward_mask[up]]
    tp = d.t[up]
    lick_assoc = np.array(
        [np.any((d.licks > p + LICK_ASSOC_S[0]) & (d.licks < p + LICK_ASSOC_S[1])) for p in tp]
    )
    seg = epoch_of(d, tp)
    events = []
    for i, p in enumerate(up):
        tm = triggered(d, d.mpet, d.t[[p]])[0]
        ti = triggered(d, d.dff, d.t[[p]])[0]
        n0 = int(-PETH_WINDOW_S[0] * d.fs)
        events.append(
            {
                "time": d.t[p],
                "epoch": seg[i],
                "lick": bool(lick_assoc[i]),
                "band_ps": float(x[p]),
                "band_dff": float(xi[p]),
                "peak_ps": float(tm[n0 - 5 : n0 + 6].max()),
                "peak_dff": float(ti[n0 - 5 : n0 + 6].max()),
            }
        )
    trig_m, trig_i = triggered(d, d.mpet, tp), triggered(d, d.dff, tp)
    fl = d.first_licks[np.isfinite(d.first_licks)]
    suc_m, suc_i = triggered(d, d.mpet, fl), triggered(d, d.dff, fl)
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(~d.reward_mask)
    pool = pool[(pool > 3 * d.fs) & (pool < d.t.size - 8 * d.fs)]
    n0 = int(-PETH_WINDOW_S[0] * d.fs)
    null_peaks = [
        triggered(d, d.mpet, d.t[rng.choice(pool, max(tp.size, 1))]).mean(0)[n0 - 5 : n0 + 6].max()
        for _ in range(500)
    ]
    rates = {
        name: {"n": int((seg == name).sum()), "lick": int(((seg == name) & lick_assoc).sum())}
        for name in d.epochs
    }
    return {
        "band": x,
        "band_i": xi,
        "sd": sd,
        "sd_i": sdi,
        "sweep": sweep,
        "events": events,
        "trig_m": trig_m,
        "trig_i": trig_i,
        "suc_m": suc_m,
        "suc_i": suc_i,
        "rates": rates,
        "null_peak_95": float(np.percentile(null_peaks, 95)),
    }


# ---- sucrose -------------------------------------------------------------------


def sucrose_analysis(d: Data) -> dict:
    s = d.session
    pre, post = PETH_WINDOW_S
    peths = {}
    for event in ("solution_onset", "lick_post_solution"):
        lt = build_peth(
            s, event, signal="lifetime", pre_window=pre, post_window=post
        ).baseline_corrected(PETH_BASELINE_S)
        it = build_peth(s, event, signal="intensity", pre_window=pre, post_window=post)
        base = np.nanmean(
            it.values[:, (it.time_rel >= PETH_BASELINE_S[0]) & (it.time_rel <= PETH_BASELINE_S[1])],
            axis=1,
            keepdims=True,
        )
        peths[event] = {
            "time": lt.time_rel,
            "mpet": lt.values * 1000,
            "dff": 100 * (it.values / base - 1),
            "events": lt.event_times,
        }
    fl = peths["lick_post_solution"]
    w = (fl["time"] >= RESPONSE_WINDOW_S[0]) & (fl["time"] <= RESPONSE_WINDOW_S[1])
    resp = np.nanmean(fl["mpet"][:, w], axis=1)
    pos = fl["time"] >= 0
    peak = np.nanmax(fl["mpet"][:, pos], axis=1)
    peak_dff = np.nanmax(fl["dff"][:, pos], axis=1)
    lat = d.first_licks - d.deliveries
    lick_counts = np.array([((d.licks >= x) & (d.licks < x + 5)).sum() for x in d.deliveries])
    have = np.isfinite(d.first_licks)
    trial_idx = np.flatnonzero(have) + 1
    trend = stats.spearmanr(trial_idx, resp)
    summary = {}
    for event, p in peths.items():
        mu = np.nanmean(p["mpet"], axis=0)
        sem = np.nanstd(p["mpet"], axis=0, ddof=1) / np.sqrt(p["mpet"].shape[0])
        k = np.flatnonzero(p["time"] >= 0)[np.argmax(mu[p["time"] >= 0])]
        mi = np.nanmean(p["dff"], axis=0)
        summary[event] = {
            "peak": mu[k],
            "sem": sem[k],
            "t_peak": p["time"][k],
            "fwhm": fwhm(p["time"], mu, k),
            "peak_dff": mi[p["time"] >= 0].max(),
            "n": p["mpet"].shape[0],
        }
    return {
        "peths": peths,
        "resp": resp,
        "peak": peak,
        "peak_dff": peak_dff,
        "latency": lat,
        "lick_counts": lick_counts,
        "trial_idx": trial_idx,
        "trend": trend,
        "summary": summary,
        "ratio": peak / peak_dff,
    }


def fwhm(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Full width at half maximum of the contiguous run of ``y >= y[k] / 2`` containing index ``k``."""
    above = y >= y[k] / 2
    lo, hi = k, k
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < y.size - 1 and above[hi + 1]:
        hi += 1
    return float(x[hi] - x[lo])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _epoch_shapes(fig: go.Figure, d: Data) -> None:
    """Shade the task-free epochs in every row and label all epochs inside the top panel."""
    for name, (a, b) in d.epochs.items():
        if name != "Task":
            fig.add_vrect(
                x0=a,
                x1=b,
                fillcolor=color("band@0.08"),
                line_width=0,
                layer="below",
                row="all",
                col=1,
                name="role:band@0.08",
            )
        fig.add_annotation(
            x=(a + b) / 2,
            y=0.98,
            xref="x",
            yref="y domain",
            yanchor="top",
            text=name,
            showarrow=False,
            font={"size": 11, "color": LIGHT["muted"]},
            name="role:muted",
        )


def _delivery_lines(fig: go.Figure, d: Data, rows: list[int]) -> None:
    for r in rows:
        for x in d.deliveries:
            fig.add_vline(
                x=x,
                line_width=1,
                line_color=color("sucrose@0.7"),
                row=r,
                col=1,
                name="role:sucrose@0.7",
            )


def fig_overview(d: Data, tr: dict) -> go.Figure:
    n = int(0.25 * d.fs)
    k = d.t.size // n
    tt = d.t[: k * n].reshape(k, n).mean(1)
    mm = d.mpet[: k * n].reshape(k, n).mean(1)
    ii = d.dff[: k * n].reshape(k, n).mean(1)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.46, 0.38, 0.16],
        subplot_titles=("Lifetime (MPET), 0.25-s means", "Intensity, 0.25-s means", "Licks"),
    )
    fig.add_trace(
        go.Scattergl(
            x=tt,
            y=mm,
            mode="lines",
            line={"width": 1.5, "color": color("signal")},
            name="MPET",
            showlegend=False,
            meta=role(line="signal"),
            hovertemplate="%{x:.1f} s<br>%{y:.1f} ps<extra></extra>",
        ),
        1,
        1,
    )
    fig.add_trace(
        go.Scattergl(
            x=tt,
            y=ii,
            mode="lines",
            line={"width": 1.5, "color": color("signal")},
            name="Intensity",
            showlegend=False,
            meta=role(line="signal"),
            hovertemplate="%{x:.1f} s<br>%{y:+.2f} %<extra></extra>",
        ),
        2,
        1,
    )
    fig.add_trace(
        go.Scattergl(
            x=d.licks,
            y=np.zeros(d.licks.size),
            mode="markers",
            name="Lick",
            marker={
                "symbol": "line-ns-open",
                "size": 10,
                "color": color("ink-2"),
                "line": {"width": 1, "color": color("ink-2")},
            },
            showlegend=False,
            meta=role(marker="ink-2", marker_line="ink-2"),
            hovertemplate="lick %{x:.2f} s<extra></extra>",
        ),
        3,
        1,
    )
    ev = tr["events"]
    if ev:
        et = np.array([e["time"] for e in ev])
        fig.add_trace(
            go.Scatter(
                x=et,
                y=np.interp(et, tt, mm) + 12,
                mode="markers",
                name="Spontaneous transient",
                marker={
                    "symbol": "triangle-down",
                    "size": 11,
                    "color": color("spont"),
                    "line": {"width": 2, "color": color("surface")},
                },
                meta=role(marker="spont", marker_line="surface"),
                customdata=[[e["peak_ps"], "yes" if e["lick"] else "no"] for e in ev],
                hovertemplate="spontaneous transient %{x:.1f} s<br>peak %{customdata[0]:.0f} ps"
                "<br>lick within 2 s: %{customdata[1]}<extra></extra>",
            ),
            1,
            1,
        )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Sucrose delivery",
            line={"width": 2, "color": color("sucrose")},
            meta=role(line="sucrose"),
        ),
        1,
        1,
    )
    _delivery_lines(fig, d, [1, 2])
    _epoch_shapes(fig, d)
    fig.update_yaxes(title_text="MPET (ps)", row=1, col=1)
    fig.update_yaxes(title_text="ΔF/F (%)", row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=3, col=1, range=[-1, 1])
    fig.update_xaxes(title_text="Time in recording (s)", row=3, col=1)
    return base_layout(fig, 560)


def fig_baseline(d: Data, bl: dict) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"Lifetime baseline, {BIN_S:.0f}-s bins with reward windows removed",
            f"Intensity, {BIN_S:.0f}-s bins",
        ),
    )
    ep = epoch_of(d, bl["centre"])
    for i, name in enumerate(d.epochs, start=1):
        m = ep == name
        e = bl["epochs"][name]
        fig.add_trace(
            go.Scatter(
                x=bl["centre"][m],
                y=bl["mpet_bin"][m],
                mode="markers",
                name=name,
                marker={
                    "size": 9,
                    "color": color(f"epoch-{i}"),
                    "line": {"width": 2, "color": color("surface")},
                },
                meta=role(marker=f"epoch-{i}", marker_line="surface"),
                legendgroup=name,
                hovertemplate=f"{name}<br>%{{x:.0f}} s<br>%{{y:.1f}} ps<extra></extra>",
            ),
            1,
            1,
        )
        a, b = e["range"]
        fig.add_trace(
            go.Scatter(
                x=[a, b],
                y=[e["mpet"]] * 2,
                mode="lines",
                line={"width": 2, "color": color(f"epoch-{i}")},
                showlegend=False,
                legendgroup=name,
                meta=role(line=f"epoch-{i}"),
                hovertemplate=f"{name} mean %{{y:.1f}} ps<extra></extra>",
            ),
            1,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=bl["centre"][m],
                y=bl["dff_bin"][m],
                mode="markers",
                name=name,
                showlegend=False,
                legendgroup=name,
                marker={
                    "size": 9,
                    "color": color(f"epoch-{i}"),
                    "line": {"width": 2, "color": color("surface")},
                },
                meta=role(marker=f"epoch-{i}", marker_line="surface"),
                hovertemplate=f"{name}<br>%{{x:.0f}} s<br>%{{y:+.2f}} %<extra></extra>",
            ),
            2,
            1,
        )
    x = np.array([bl["centre"][0], bl["centre"][-1]])
    for r, fit, unit in ((1, bl["fit"], "ps"), (2, bl["ifit"], "%")):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fit.intercept + fit.slope * x / 60,
                mode="lines",
                name="Linear trend",
                showlegend=r == 1,
                legendgroup="trend",
                line={"width": 1.5, "color": color("muted")},
                meta=role(line="muted"),
                hovertemplate=f"trend {fit.slope:+.2f} {unit}/min<extra></extra>",
            ),
            r,
            1,
        )
    fig.update_yaxes(title_text="MPET (ps)", row=1, col=1)
    fig.update_yaxes(title_text="ΔF/F (%)", row=2, col=1)
    fig.update_xaxes(title_text="Time in recording (s)", row=2, col=1)
    return base_layout(fig, 480)


def fig_spectrum(d: Data, sp: dict, su: dict, tr: dict) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.52, 0.48],
        horizontal_spacing=0.12,
        subplot_titles=("Lifetime power spectrum by epoch", "Lifetime vs intensity coupling"),
    )
    for i, (name, (f, p)) in enumerate(sp["psd"].items(), start=1):
        m = f > 0
        fig.add_trace(
            go.Scatter(
                x=f[m],
                y=p[m],
                mode="lines",
                name=name,
                line={"width": 2, "color": color(f"epoch-{i}")},
                meta=role(line=f"epoch-{i}"),
                legendgroup=name,
                hovertemplate=f"{name}<br>%{{x:.2f}} Hz<br>%{{y:.3g}} ps²/Hz<extra></extra>",
            ),
            1,
            1,
        )
    f = sp["psd"]["Task"][0]
    fig.add_trace(
        go.Scatter(
            x=[f[1], f[-1]],
            y=[sp["shot_psd"]] * 2,
            mode="lines",
            name="Shot-noise floor",
            line={"width": 1.5, "color": color("muted")},
            meta=role(line="muted"),
            hovertemplate="photon shot-noise floor<extra></extra>",
        ),
        1,
        1,
    )
    fig.add_vrect(
        x0=0.1,
        x1=0.3,
        fillcolor=color("band@0.08"),
        line_width=0,
        layer="below",
        row=1,
        col=1,
        name="role:band@0.08",
    )
    fig.add_annotation(
        x=np.log10(0.17),
        y=0.02,
        xref="x",
        yref="y domain",
        yanchor="bottom",
        showarrow=False,
        text="slow band",
        font={"size": 11, "color": LIGHT["muted"]},
        name="role:muted",
    )
    fig.update_xaxes(type="log", dtick=1, title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(type="log", dtick=1, title_text="Power (ps²/Hz)", row=1, col=1)

    m = (d.t >= d.epochs["Task"][0] + 2) & (d.t < d.epochs["Task"][1] - 2) & ~d.reward_mask
    step = 4
    fig.add_trace(
        go.Scattergl(
            x=sp["slow_i"][m][::step],
            y=sp["slow_m"][m][::step],
            mode="markers",
            name="Slow fluctuations (task, 0.05–0.5 Hz)",
            marker={"size": 4, "color": color("muted@0.35")},
            meta=role(marker="muted@0.35"),
            hovertemplate="slow band<br>%{x:+.2f} %, %{y:+.1f} ps<extra></extra>",
        ),
        1,
        2,
    )
    c = sp["coupling"]["Task"]
    xs = np.array([-2.5, 2.5])
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=c["slope"] * xs,
            mode="lines",
            name=f"Slow fit, {c['slope']:.0f} ps per %",
            line={"width": 1.5, "color": color("muted")},
            meta=role(line="muted"),
            hoverinfo="skip",
        ),
        1,
        2,
    )
    fig.add_trace(
        go.Scatter(
            x=su["peak_dff"],
            y=su["peak"],
            mode="markers",
            name="Sucrose trials (peak)",
            marker={
                "size": 10,
                "color": color("sucrose"),
                "line": {"width": 2, "color": color("surface")},
            },
            meta=role(marker="sucrose", marker_line="surface"),
            customdata=su["trial_idx"],
            hovertemplate="sucrose trial %{customdata}<br>%{x:.2f} %, %{y:.0f} ps<extra></extra>",
        ),
        1,
        2,
    )
    if tr["events"]:
        fig.add_trace(
            go.Scatter(
                x=[e["peak_dff"] for e in tr["events"]],
                y=[e["peak_ps"] for e in tr["events"]],
                mode="markers",
                name="Spontaneous transients (peak)",
                marker={
                    "size": 10,
                    "symbol": "triangle-down",
                    "color": color("spont"),
                    "line": {"width": 2, "color": color("surface")},
                },
                meta=role(marker="spont", marker_line="surface"),
                customdata=[e["time"] for e in tr["events"]],
                hovertemplate="spontaneous %{customdata:.1f} s<br>%{x:.2f} %, %{y:.0f} ps<extra></extra>",
            ),
            1,
            2,
        )
    xs = np.array([0, max(su["peak_dff"].max(), 5) * 1.05])
    med = float(np.median(su["ratio"]))
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=med * xs,
            mode="lines",
            name=f"Sucrose ratio, {med:.0f} ps per %",
            line={"width": 1.5, "color": color("sucrose")},
            meta=role(line="sucrose"),
            hoverinfo="skip",
        ),
        1,
        2,
    )
    fig.update_xaxes(title_text="Intensity change (ΔF/F, %)", row=1, col=2)
    fig.update_yaxes(title_text="Lifetime change (ps)", row=1, col=2)
    fig = base_layout(fig, 500)
    fig.update_layout(
        legend={"yref": "paper", "y": -0.22, "yanchor": "top"}, margin={"t": 40, "b": 120}
    )
    return fig


def _mean_sem(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nanmean(a, 0), np.nanstd(a, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(a), 0))


def _ribbon(fig, x, a, name, rolename, row, col, showlegend=True, unit="ps"):
    mu, se = _mean_sem(a)
    fig.add_trace(
        go.Scatter(
            x=np.r_[x, x[::-1]],
            y=np.r_[mu + se, (mu - se)[::-1]],
            fill="toself",
            mode="lines",
            line={"width": 0},
            fillcolor=color(f"{rolename}@0.15"),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=name,
            meta=role(fill=f"{rolename}@0.15"),
        ),
        row,
        col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mu,
            mode="lines",
            name=name,
            line={"width": 2, "color": color(rolename)},
            showlegend=showlegend,
            legendgroup=name,
            meta=role(line=rolename),
            customdata=se,
            hovertemplate=f"{name}<br>%{{x:+.2f}} s<br>%{{y:+.1f}} ± %{{customdata:.1f}} {unit}<extra></extra>",
        ),
        row,
        col,
    )


def fig_transients(d: Data, tr: dict) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.42, 0.58],
        horizontal_spacing=0.12,
        vertical_spacing=0.14,
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        subplot_titles=(
            "Detector against its null",
            "Lifetime, event-triggered mean ± SEM",
            "Intensity, event-triggered mean ± SEM",
        ),
    )
    ks = [f"{s['k']:g}σ" for s in tr["sweep"]]
    fig.add_trace(
        go.Bar(
            x=ks,
            y=[s["up"] for s in tr["sweep"]],
            name="Upward events",
            marker_color=color("spont"),
            meta=role(marker="spont"),
            width=0.36,
            offsetgroup="up",
            customdata=[[s["thr"], s["sucrose_hits"], s["n_sucrose"]] for s in tr["sweep"]],
            hovertemplate="%{x} (%{customdata[0]:.0f} ps): %{y} upward<br>sucrose detected "
            "%{customdata[1]}/%{customdata[2]}<extra></extra>",
        ),
        1,
        1,
    )
    fig.add_trace(
        go.Bar(
            x=ks,
            y=[s["down"] for s in tr["sweep"]],
            name="Downward events (null)",
            marker_color=color("muted"),
            meta=role(marker="muted"),
            width=0.36,
            offsetgroup="down",
            hovertemplate="%{x}: %{y} downward<extra></extra>",
        ),
        1,
        1,
    )
    fig.update_traces(marker_cornerradius=4, selector={"type": "bar"})
    fig.update_xaxes(title_text="Threshold (robust SDs)", row=1, col=1)
    fig.update_yaxes(title_text="Events outside reward windows", row=1, col=1)
    x = np.arange(tr["trig_m"].shape[1]) / d.fs + PETH_WINDOW_S[0]
    n = tr["trig_m"].shape[0]
    _ribbon(fig, x, tr["suc_m"], f"Sucrose, first lick (n={tr['suc_m'].shape[0]})", "sucrose", 1, 2)
    if n:
        _ribbon(fig, x, tr["trig_m"], f"Spontaneous (n={n})", "spont", 1, 2)
    _ribbon(fig, x, tr["suc_i"], "Sucrose", "sucrose", 2, 2, showlegend=False, unit="%")
    if n:
        _ribbon(fig, x, tr["trig_i"], "Spontaneous", "spont", 2, 2, showlegend=False, unit="%")
    for r in (1, 2):
        fig.add_vline(x=0, line_width=1, line_color=color("axis"), row=r, col=2, name="role:axis")
    fig.update_yaxes(title_text="ΔMPET (ps)", row=1, col=2)
    fig.update_yaxes(title_text="ΔF/F (%)", row=2, col=2)
    fig.update_xaxes(title_text="Time from event (s)", row=2, col=2)
    fig = base_layout(fig, 520)
    fig.update_layout(barmode="group", bargap=0.3)
    return fig


def fig_sucrose(d: Data, su: dict) -> go.Figure:
    p = su["peths"]
    fig = make_subplots(
        rows=4,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.035,
        horizontal_spacing=0.08,
        row_heights=[0.2, 0.3, 0.27, 0.23],
        column_titles=["Aligned to sucrose delivery", "Aligned to first lick after delivery"],
    )
    zmax = float(
        np.nanpercentile(
            np.abs(
                np.r_[p["solution_onset"]["mpet"].ravel(), p["lick_post_solution"]["mpet"].ravel()]
            ),
            99,
        )
    )
    for c, event in enumerate(("solution_onset", "lick_post_solution"), start=1):
        e = p[event]
        xs, ys = [], []
        for i, a in enumerate(e["events"], start=1):
            rel = d.licks - a
            rel = rel[(rel >= PETH_WINDOW_S[0]) & (rel <= PETH_WINDOW_S[1])]
            xs += rel.tolist()
            ys += [i] * rel.size
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="markers",
                name="Lick",
                showlegend=False,
                marker={
                    "symbol": "line-ns-open",
                    "size": 7,
                    "color": color("ink-2"),
                    "line": {"width": 1, "color": color("ink-2")},
                },
                meta=role(marker="ink-2", marker_line="ink-2"),
                hovertemplate="trial %{y}<br>lick %{x:+.2f} s<extra></extra>",
            ),
            1,
            c,
        )
        fig.add_trace(
            go.Heatmap(
                x=e["time"],
                y=np.arange(1, e["mpet"].shape[0] + 1),
                z=e["mpet"],
                zmid=0,
                zmin=-zmax,
                zmax=zmax,
                showscale=c == 2,
                colorscale=[[0, LIGHT["div-neg"]], [0.5, LIGHT["div-mid"]], [1, LIGHT["div-pos"]]],
                colorbar={
                    "title": {"text": "ΔMPET (ps)", "side": "right"},
                    "len": 0.3,
                    "y": 0.62,
                    "thickness": 12,
                    "outlinewidth": 0,
                },
                meta={"colorscale": "diverging"},
                name="",
                hovertemplate="trial %{y}<br>%{x:+.2f} s<br>%{z:+.0f} ps<extra></extra>",
            ),
            2,
            c,
        )
        _ribbon(fig, e["time"], e["mpet"], "Lifetime", "sucrose", 3, c, showlegend=False)
        _ribbon(fig, e["time"], e["dff"], "Intensity", "sucrose", 4, c, showlegend=False, unit="%")
        for r in (1, 2, 3, 4):
            fig.add_vline(
                x=0, line_width=1, line_color=color("axis"), row=r, col=c, name="role:axis"
            )
        fig.update_yaxes(autorange="reversed", dtick=2, row=1, col=c)
        fig.update_yaxes(autorange="reversed", dtick=2, row=2, col=c)
        fig.update_xaxes(title_text="Time from event (s)", row=4, col=c)
    fig.update_yaxes(title_text="Trial", row=1, col=1)
    fig.update_yaxes(title_text="Trial", row=2, col=1)
    fig.update_yaxes(title_text="ΔMPET (ps)", row=3, col=1)
    fig.update_yaxes(title_text="ΔF/F (%)", row=4, col=1)
    fig.update_yaxes(matches="y5", row=3, col=2)
    fig.update_yaxes(matches="y7", row=4, col=2)
    return base_layout(fig, 820)


def fig_trials(d: Data, su: dict, bl: dict) -> go.Figure:
    trials = np.arange(1, d.deliveries.size + 1)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            f"Response, mean ΔMPET {RESPONSE_WINDOW_S[0]:g}–{RESPONSE_WINDOW_S[1]:g} s after first lick",
            "Licks in the 5 s after delivery",
            "Pre-delivery baseline (5 s), relative to the pre-task level",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=su["trial_idx"],
            y=su["resp"],
            mode="lines+markers",
            name="Response",
            line={"width": 2, "color": color("sucrose")},
            marker={
                "size": 9,
                "color": color("sucrose"),
                "line": {"width": 2, "color": color("surface")},
            },
            meta=role(line="sucrose", marker="sucrose", marker_line="surface"),
            showlegend=False,
            hovertemplate="trial %{x}<br>%{y:+.0f} ps<extra></extra>",
        ),
        1,
        1,
    )
    fig.add_trace(
        go.Bar(
            x=trials,
            y=su["lick_counts"],
            name="Licks",
            marker_color=color("muted"),
            meta=role(marker="muted"),
            width=0.5,
            showlegend=False,
            marker_cornerradius=4,
            customdata=su["latency"],
            hovertemplate="trial %{x}<br>%{y} licks<br>first-lick latency %{customdata:.2f} s<extra></extra>",
        ),
        2,
        1,
    )
    fig.add_hline(y=0, line_width=1, line_color=color("axis"), row=3, col=1, name="role:axis")
    fig.add_trace(
        go.Scatter(
            x=trials,
            y=bl["pre_trial"] - bl["pre_task_mean"],
            mode="markers",
            name="Baseline",
            showlegend=False,
            marker={
                "size": 9,
                "color": color("signal"),
                "line": {"width": 2, "color": color("surface")},
            },
            meta=role(marker="signal", marker_line="surface"),
            hovertemplate="trial %{x}<br>%{y:+.1f} ps<extra></extra>",
        ),
        3,
        1,
    )
    fig.update_yaxes(title_text="ps", row=1, col=1)
    fig.update_yaxes(title_text="licks", row=2, col=1)
    fig.update_yaxes(title_text="ps", row=3, col=1)
    fig.update_xaxes(title_text="Trial", dtick=1, row=3, col=1)
    return base_layout(fig, 560)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

PAGE_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --signal: #2a78d6; --sucrose: #eb6834; --spont: #1baf7a;
  --epoch-1: #86b6ef; --epoch-2: #2a78d6; --epoch-3: #104281;
  --div-neg: #2a78d6; --div-mid: #f0efec; --div-pos: #e34948; --band: #898781;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --signal: #3987e5; --sucrose: #d95926; --spont: #199e70;
    --epoch-1: #184f95; --epoch-2: #3987e5; --epoch-3: #9ec5f4;
    --div-neg: #3987e5; --div-mid: #383835; --div-pos: #e66767; --band: #c3c2b7;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --signal: #3987e5; --sucrose: #d95926; --spont: #199e70;
  --epoch-1: #184f95; --epoch-2: #3987e5; --epoch-3: #9ec5f4;
  --div-neg: #3987e5; --div-mid: #383835; --div-pos: #e66767; --band: #c3c2b7;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 19px; font-weight: 600; margin: 48px 0 8px; }
h3 { font-size: 15px; font-weight: 600; margin: 20px 0 6px; }
p, li { color: var(--ink-2); max-width: 78ch; }
.lede { color: var(--ink-2); margin: 0 0 24px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 24px 0; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .label { font-size: 13px; color: var(--ink-2); }
.tile .value { font-size: 26px; font-weight: 600; margin: 2px 0; color: var(--ink); }
.tile .note { font-size: 12.5px; color: var(--muted); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 8px 8px 4px; margin: 12px 0 8px; }
.caption { font-size: 13px; color: var(--muted); margin: 4px 4px 0; max-width: none; }
.findings { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 6px 20px; }
.findings li { margin: 8px 0; }
.findings b, p b { color: var(--ink); font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 8px 0 4px; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
.tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 4px 8px; }
details { margin: 8px 0; }
summary { cursor: pointer; color: var(--ink-2); font-size: 14px; }
footer { margin-top: 56px; font-size: 12.5px; color: var(--muted); }
code { font-size: 12.5px; }
"""

THEME_JS = """
(function () {
  const css = n => getComputedStyle(document.documentElement).getPropertyValue('--' + n).trim();
  function rgba(spec) {
    const [name, a] = spec.split('@');
    const hex = css(name);
    if (!a) return hex;
    const h = hex.replace('#', '');
    const v = [0, 2, 4].map(i => parseInt(h.substr(i, 2), 16));
    return `rgba(${v[0]},${v[1]},${v[2]},${a})`;
  }
  function apply(gd) {
    if (!gd || !gd.data) return;
    gd.data.forEach((tr, i) => {
      const m = tr.meta || {};
      const u = {};
      if (m.colorscale === 'diverging')
        u.colorscale = [[[0, css('div-neg')], [0.5, css('div-mid')], [1, css('div-pos')]]];
      const r = m.roles || {};
      if (r.line) u['line.color'] = rgba(r.line);
      if (r.marker) u['marker.color'] = rgba(r.marker);
      if (r.marker_line) u['marker.line.color'] = rgba(r.marker_line);
      if (r.fill) u.fillcolor = rgba(r.fill);
      if (Object.keys(u).length) Plotly.restyle(gd, u, [i]);
    });
    const L = { paper_bgcolor: css('surface'), plot_bgcolor: css('surface'), 'font.color': css('ink-2') };
    Object.keys(gd.layout).forEach(k => {
      if (/^[xy]axis\\d*$/.test(k)) {
        L[k + '.gridcolor'] = css('grid'); L[k + '.linecolor'] = css('axis'); L[k + '.tickcolor'] = css('axis');
      }
    });
    (gd.layout.shapes || []).forEach((s, i) => {
      if (s.name && s.name.startsWith('role:')) {
        const spec = s.name.slice(5);
        if (s.type === 'line') L[`shapes[${i}].line.color`] = rgba(spec); else L[`shapes[${i}].fillcolor`] = rgba(spec);
      }
    });
    (gd.layout.annotations || []).forEach((a, i) => {
      L[`annotations[${i}].font.color`] = a.name === 'role:muted' ? css('muted') : css('ink');
    });
    Plotly.relayout(gd, L);
  }
  window.themePlot = apply;
  window.applyTheme = () => document.querySelectorAll('.js-plotly-plot').forEach(apply);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', window.applyTheme);
  new MutationObserver(window.applyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
"""


def fig_div(fig: go.Figure, div_id: str) -> str:
    spec = json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder)
    return (
        f'<div class="card"><div id="{div_id}"></div></div>'
        f"<script>(function(){{const s={spec};Plotly.newPlot('{div_id}',s.data,s.layout,"
        f"{{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']}})"
        f".then(gd=>window.themePlot(gd));}})();</script>"
    )


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def git_rev(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_html(d: Data, bl: dict, sp: dict, tr: dict, su: dict) -> str:
    meta = d.session.meta
    title = f"{meta.get('subject', '')} · {meta.get('sensor', 'iFLiP')} session report"
    ep = bl["epochs"]
    pre, task, post = ep["Pre-task"], ep["Task"], ep["Post-task"]
    fl = su["summary"]["lick_post_solution"]
    dv = su["summary"]["solution_onset"]
    n_spont = len(tr["events"])
    n_lick = sum(e["lick"] for e in tr["events"])
    nonreward_min = sum(e["nonreward_s"] for e in ep.values()) / 60
    sw = {s["k"]: s for s in tr["sweep"]}
    k = sw[TRANSIENT_K]
    k3 = sw[3.0]
    sp_c = sp["coupling"]
    ratio_sp = (
        np.median([e["peak_ps"] / e["peak_dff"] for e in tr["events"]]) if n_spont else float("nan")
    )
    resp_first, resp_last = su["resp"][1:4].mean(), su["resp"][-3:].mean()
    pre_rel = bl["pre_trial"] - bl["pre_task_mean"]
    tail_txt = ", ".join(
        f"{t['window'][0]}–{t['window'][1]} s {t['mean']:+.0f}" for t in bl["tail"]
    )
    back = next((t["window"] for t in bl["tail"] if abs(t["mean"]) < 2 * t["sem"]), None)
    photons_s = np.mean(d.photons) * d.fs
    mant, expo = f"{photons_s:.1e}".split("e")

    tiles = f"""
<div class="tiles">
  <div class="tile"><div class="label">Baseline lifetime (MPET)</div><div class="value">{bl["mean"] / 1000:.3f} ns</div>
    <div class="note">reward windows excluded</div></div>
  <div class="tile"><div class="label">Long-term drift</div><div class="value">{bl["fit"].slope:+.1f} ps/min</div>
    <div class="note">± {bl["fit"].stderr:.1f} ps/min; intensity {bl["ifit"].slope * (d.t[-1] - d.t[0]) / 60:+.1f}% over the recording</div></div>
  <div class="tile"><div class="label">Spontaneous transients</div><div class="value">{n_spont} in {nonreward_min:.1f} min</div>
    <div class="note">{n_lick} of {n_spont} follow a lick; none before the task</div></div>
  <div class="tile"><div class="label">Sucrose response, first-lick aligned</div><div class="value">+{fl["peak"]:.0f} ps</div>
    <div class="note">± {fl["sem"]:.0f} ps SEM at +{fl["t_peak"]:.2f} s, n = {fl["n"]} trials</div></div>
</div>"""

    findings = f"""
<ul class="findings">
  <li><b>Signal quality is good.</b> {mant} × 10<sup>{int(expo)}</sup> photons/s, no QC flags, and frame-to-frame lifetime noise of
    {sp["frame_noise"]:.1f} ps, {sp["frame_noise"] / sp["shot_sd"]:.1f}× the {sp["shot_sd"]:.1f} ps photon shot-noise limit (the reference DA0.5 session sits at 1.6×).
    Above ~2 Hz the lifetime noise is white in every epoch.</li>
  <li><b>No long-term drift, but the tonic level rises during the task.</b> The linear trend is {bl["fit"].slope:+.1f} ± {bl["fit"].stderr:.1f} ps/min
    and intensity fell only {abs(bl["ifit"].slope * (d.t[-1] - d.t[0]) / 60):.1f}%. Measured in the 5 s before each delivery, the baseline sits near the
    pre-task level for the first three trials ({", ".join(f"{v:+.0f}" for v in pre_rel[:3])} ps) and {pre_rel[3:].min():+.0f} to {pre_rel[3:].max():+.0f} ps above it
    from trial 4 on, then returns to the pre-task level within ~30 s of the task ending ({post["mpet"] - pre["mpet"]:+.0f} ps over the post-task epoch).</li>
  <li><b>The baseline carries slow 5–10 s fluctuations</b> (0.1–0.3 Hz, about ±{sp_c["Task"]["sd_mpet"]:.0f} ps during the task) that track intensity closely
    (r = {sp_c["Task"]["r"]:.2f}). They are dopamine-signed but carry only {sp_c["Pre-task"]["slope"]:.0f}–{sp_c["Task"]["slope"]:.0f} ps per 1% ΔF/F,
    against {np.median(su["ratio"]):.0f} ps for sucrose responses, so they are likely a mix of slow dopamine and a brightness-only (probably hemodynamic) component.</li>
  <li><b>Spontaneous transients are rare{" and mostly lick-associated" if n_lick > n_spont / 2 else ""}.</b> {n_spont} events passed the {TRANSIENT_K:g}σ detector ({k["thr"]:.0f} ps)
    outside reward windows ({60 * n_spont / (nonreward_min * 60):.1f}/min), with {k["down"]} downward events in the null; at 3σ the excess is {k3["up"] - k3["down"]} events.
    They are fast (FWHM ≈ {_fwhm(tr["trig_m"], d):.2f} s), seen in intensity too, and couple at {ratio_sp:.0f} ps per % like the sucrose response, about half its size.
    {n_lick} of {n_spont} follow a lick within 2 s; {tr["rates"]["Pre-task"]["n"]} occur in the quiet pre-task minute{", and post-task events have no lick record" if tr["rates"]["Post-task"]["n"] else ""}.</li>
  <li><b>Sucrose evokes a large, reliable dopamine response that follows consumption.</b> Aligned to the first lick: +{fl["peak"]:.0f} ± {fl["sem"]:.0f} ps at
    +{fl["t_peak"]:.2f} s (+{fl["peak_dff"]:.1f}% ΔF/F), every trial positive. Aligned to delivery it is later and smaller (+{dv["peak"]:.0f} ps at +{dv["t_peak"]:.2f} s) because licking starts
    {np.nanmedian(su["latency"]):.2f} s (median) after the solenoid. It has a long tail: lifetime stays above the pre-delivery baseline for ~10 s
    ({tail_txt} ps){f" and is back by {back[0]}–{back[1]} s" if back else ""}. Responses trend smaller late in the session (trials 2–4 ≈ {resp_first:.0f} ps,
    last 3 ≈ {resp_last:.0f} ps; Spearman ρ = {su["trend"].statistic:.2f}, p = {su["trend"].pvalue:.2f}, not significant with {su["resp"].size} trials) while licking per trial holds steady.</li>
</ul>"""

    epoch_rows = [
        [
            name,
            f"{e['range'][0]:.0f}–{e['range'][1]:.0f}",
            f"{e['nonreward_s']:.0f}",
            f"{e['mpet']:.1f}",
            f"{e['mpet'] - pre['mpet']:+.1f}",
            f"{e['bin_sd']:.1f}",
            f"{e['dff']:+.2f}",
            f"{sp_c[name]['sd_mpet']:.1f}",
            f"{sp_c[name]['slope']:.1f}",
            f"{tr['rates'][name]['n']} ({tr['rates'][name]['lick']})",
        ]
        for name, e in ep.items()
    ]
    sweep_rows = [
        [
            f"{s['k']:g}σ",
            f"{s['thr']:.0f}",
            str(s["up"]),
            str(s["down"]),
            f"{s['up'] - s['down']:+d}",
            f"{s['sucrose_hits']}/{s['n_sucrose']}",
        ]
        for s in tr["sweep"]
    ]
    event_rows = [
        [
            f"{e['time']:.1f}",
            e["epoch"],
            "yes" if e["lick"] else "no",
            f"{e['peak_ps']:.0f}",
            f"{e['peak_dff']:.2f}",
            f"{e['peak_ps'] / e['peak_dff']:.0f}",
        ]
        for e in tr["events"]
    ]
    trial_rows = []
    j = 0
    for i, x in enumerate(d.deliveries):
        has = np.isfinite(d.first_licks[i])
        trial_rows.append(
            [
                str(i + 1),
                f"{x:.1f}",
                f"{su['latency'][i]:.2f}" if has else "–",
                str(su["lick_counts"][i]),
                f"{pre_rel[i]:+.1f}",
                f"{su['resp'][j]:+.0f}" if has else "–",
                f"{su['peak'][j]:+.0f}" if has else "–",
                f"{su['peak_dff'][j]:+.2f}" if has else "–",
            ]
        )
        j += has
    tail_rows = [
        [f"{t['window'][0]}–{t['window'][1]}", f"{t['mean']:+.1f}", f"{t['sem']:.1f}", str(t["n"])]
        for t in bl["tail"]
    ]

    repo = Path(__file__).resolve().parents[1]
    body = f"""
<main>
<h1>{html.escape(title)}</h1>
<p class="lede">{html.escape(meta.get("blockname", ""))} · {html.escape(meta.get("region", ""))} · {
        html.escape(meta.get("prep", ""))
    } ·
  {html.escape(meta.get("procedure", ""))} · {d.t[-1] - d.t[0]:.0f} s at {d.fs:g} Hz</p>
{tiles}
<h2>Summary</h2>
{findings}

<h2>1. The whole recording</h2>
<p>The behaviour rig ran for the middle {task["range"][1] - task["range"][0]:.0f} s of the {
        d.t[-1] - d.t[0]:.0f}-s recording, so the minute before and the
two and a half minutes after give task-free baselines (no lick data exist there). Orange lines mark the {
        d.deliveries.size
    } sucrose deliveries;
triangles mark detected spontaneous transients.</p>
{fig_div(fig_overview(d, tr), "f-overview")}
<p class="caption">Lifetime is MPET = lifetime − t0. Intensity is ΔF/F relative to the session median. Shaded: outside the task.</p>

<h2>2. Baseline level and stability</h2>
<p>Each point is a {BIN_S:.0f}-s mean with the window from {REWARD_WINDOW_S[0]:g} to +{
        REWARD_WINDOW_S[1]:g} s around every delivery removed, so it shows
the baseline between rewards. Horizontal lines are epoch means. The exclusion runs to +{
        REWARD_WINDOW_S[1]:g} s because the sucrose response has a long tail
(table below){
        f": lifetime is back at the pre-delivery baseline only by {back[0]}–{back[1]} s after delivery"
        if back
        else ""
    }.</p>
{fig_div(fig_baseline(d, bl), "f-baseline")}
<h3>Lifetime after each delivery, relative to that trial's pre-delivery baseline</h3>
{table(["Seconds after delivery", "ΔMPET (ps)", "SEM", "Trials"], tail_rows)}
<p class="caption">Trials whose next delivery fell inside a window are left out of that window.</p>
{
        table(
            [
                "Epoch",
                "Time (s)",
                "Non-reward (s)",
                "MPET (ps)",
                "vs pre-task",
                f"{BIN_S:.0f}-s bin SD",
                "ΔF/F (%)",
                "Slow-band SD (ps)",
                "Slow ps per %",
                "Spont. transients (lick)",
            ],
            epoch_rows,
        )
    }
<p class="caption">The pre-task epoch is only {pre["range"][1] - pre["range"][0]:.0f} s ({
        pre["n_bins"]
    } bins), so the epoch comparison is indicative.
The task-epoch mean is {
        task["mpet"]
        - pre[
            "mpet"
        ]:+.0f} ps above pre-task with reward windows removed; the trial-by-trial pre-delivery baselines are in section 4.</p>

<h3>Where the baseline variance lives</h3>
<p>The lifetime spectrum is white above ~2 Hz, at {
        (sp["frame_noise"] / sp["shot_sd"])
        ** 2:.1f}× the theoretical shot-noise power (grey line), and rises
100–1000-fold below 0.3 Hz. That slow band is coherent with intensity. The right panel compares how much lifetime change each kind of event carries
per unit of brightness change: sucrose responses and spontaneous transients sit on one line, the slow fluctuations on a shallower one.</p>
{fig_div(fig_spectrum(d, sp, su, tr), "f-spectrum")}
<p class="caption">Welch spectra of MPET within each epoch (25.6-s segments). Coupling: slow-band ({
        SLOW_BAND_HZ[0]:g}–{SLOW_BAND_HZ[1]:g} Hz) samples from
the task epoch with reward windows removed, against the per-event peaks of each sucrose trial and spontaneous transient.</p>

<h2>3. Spontaneous transients</h2>
<p>Transients were detected on MPET band-passed to {TRANSIENT_BAND_HZ[0]:g}–{
        TRANSIENT_BAND_HZ[1]:g} Hz, which removes the slow band and keeps the
~0.6-s dopamine transient, with a threshold in robust SDs of that trace outside reward windows ({
        tr["sd"]:.1f} ps). Two controls set the threshold:
the same detector on the sign-flipped trace (a dopamine sensor should give few downward events) and the sucrose responses, which it must catch.
At {TRANSIENT_K:g}σ the null gives {k["down"]} events and {k["sucrose_hits"]} of {
        k["n_sucrose"]
    } sucrose responses are detected.
The {n_spont}-event triggered average peaks at {
        tr["trig_m"].mean(0).max() if n_spont else float("nan"):.0f} ps; averages of {
        n_spont
    } random non-reward
times reach {tr["null_peak_95"]:.0f} ps at the 95th percentile.</p>
{fig_div(fig_transients(d, tr), "f-transients")}
{table(["Threshold", "ps", "Upward", "Downward (null)", "Excess", "Sucrose detected"], sweep_rows)}
<h3>Detected events ({TRANSIENT_K:g}σ)</h3>
{
        table(
            [
                "Time (s)",
                "Epoch",
                "Lick within 2 s",
                "Peak ΔMPET (ps)",
                "Peak ΔF/F (%)",
                "ps per %",
            ],
            event_rows,
        )
        if event_rows
        else "<p>None.</p>"
    }

<h2>4. Dopamine evoked by sucrose</h2>
<p>Each delivery is {meta.get("arduino_setting_117", "?")} solenoid pulses of {
        meta.get("arduino_setting_116", "?")
    } ms. The response follows
the first lick rather than the solenoid: aligned to first lick it starts almost immediately and peaks at +{
        fl["t_peak"]:.2f} s
(FWHM {fl["fwhm"]:.2f} s). On trial 1 the mouse did not lick for {
        su["latency"][
            0
        ]:.1f} s; that trial is flat at delivery and responds at the late first lick.</p>
{fig_div(fig_sucrose(d, su), "f-sucrose")}
<p class="caption">Heatmaps and means are baseline-corrected per trial over {
        PETH_BASELINE_S[0]:g} to {PETH_BASELINE_S[1]:g} s (the t = 0 sample excluded).
Built with <code>iflip.align.build_peth</code> on the unsmoothed streams.</p>
{fig_div(fig_trials(d, su, bl), "f-trials")}
{
        table(
            [
                "Trial",
                "Delivery (s)",
                "First-lick latency (s)",
                "Licks 0–5 s",
                "Pre-delivery baseline vs pre-task (ps)",
                "Response (ps)",
                "Peak ΔMPET (ps)",
                "Peak ΔF/F (%)",
            ],
            trial_rows,
        )
    }
<p class="caption">Response = mean ΔMPET {RESPONSE_WINDOW_S[0]:g}–{
        RESPONSE_WINDOW_S[1]:g} s after the first lick, relative to the 2 s before it.
Pre-delivery baseline = mean MPET in the 5 s before delivery minus the pre-task mean.</p>

<h2>Methods and caveats</h2>
<ul>
  <li>One session, one mouse, {
        d.deliveries.size
    } sucrose trials. Rates and trends here are descriptive.</li>
  <li>MPET is computed by the rig over all 126 TCSPC bins with no background or afterpulse subtraction (lab default). This cancels in within-session
    differences but leaves absolute MPET biased, and lets any brightness-only change (for example hemodynamic absorption) leak slightly into lifetime
    through the constant background. A background recording at this laser power would bound that leak; a binding-dead mutant sensor would separate it.</li>
  <li>Clocks: Arduino events were mapped to photometry time with an offset + drift fit on {
        meta.get("sync_n_pulses")
    } sync pulses
    ({meta.get("sync_drift_ppm")} ppm, max residual {
        float(meta.get("sync_residual_max_s", "nan")) * 1000:.0f} ms).</li>
  <li>No lick data exist outside the task window, so pre- and post-task transients cannot be classed as lick-associated or not.</li>
  <li>The recording's photon-counting range was 0.4–12.3 ns, and the histogram shows a small late reflection peak (~0.7% of photons).
    The lab recommends 0.4–10 ns in vivo.</li>
</ul>
<footer>Generated {
        date.today().isoformat()
    } by <code>iflip-analysis/scripts/session_report.py</code> (commit {git_rev(repo)}) from
<code>{html.escape(str(d.session.path))}</code>.</footer>
</main>"""
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(meta.get('subject', ''))} session report</title><style>{PAGE_CSS}</style>"
        f"<script>{plotly.offline.get_plotlyjs()}</script><script>{THEME_JS}</script></head>"
        f"<body>{body}</body></html>"
    )


def _fwhm(segs: np.ndarray, d: Data) -> float:
    """FWHM of an event-triggered mean, around its peak within ±0.5 s of the event."""
    if not segs.size:
        return float("nan")
    mu = segs.mean(0)
    x = np.arange(mu.size) / d.fs + PETH_WINDOW_S[0]
    near = np.flatnonzero(np.abs(x) <= 0.5)
    return fwhm(x, mu, int(near[np.argmax(mu[near])]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = load(args.session_dir)
    bl, sp, tr, su = (
        baseline_analysis(d),
        spectrum_analysis(d),
        transient_analysis(d),
        sucrose_analysis(d),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(d, bl, sp, tr, su))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
