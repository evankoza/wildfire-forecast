"""Render the backtest as charts.

Four figures, each answering one question a reader would actually ask:

  calibration  -- when it says 15%, does 15% happen?
  pr-curve     -- how much better than "how big is it already"?
  regions      -- does it hold up in a region it has never seen?
  importance   -- what is it actually using?

Every figure is emitted twice, light and dark, because they are embedded in a
README that GitHub renders in either theme. The dark variants are stepped for
the dark surface rather than being an inverted copy of the light ones.

Palette is the validated two-hue categorical set (blue = the model, orange =
the comparison), checked with the dataviz validator in both modes: all-pairs
CVD ΔE 24.7 light / 26.8 dark, normal-vision 33.6 / 31.8, all slots >= 3:1
against their surface. Reference marks (the ideal diagonal, prevalence) are
muted ink, never a series hue -- they are not series.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from . import config

log = logging.getLogger(__name__)

DOCS_IMG = config.ROOT / "docs" / "img"


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    series_1: str  # the model
    series_2: str  # the comparison


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series_1="#2a78d6",
    series_2="#eb6834",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series_1="#3987e5",
    series_2="#d95926",
)

THEMES = (LIGHT, DARK)

SANS = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial", "sans-serif"]


DPI = 200


def _fig(theme: Theme, figsize):
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = SANS

    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.axis)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme.muted, labelsize=8, length=0, pad=6)
    ax.grid(True, color=theme.grid, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, theme: Theme, title: str, subtitle: str | None = None):
    ax.set_title(title, color=theme.ink, fontsize=11.5, fontweight="600",
                 loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, color=theme.ink_secondary,
                fontsize=8.5, va="bottom", ha="left")


def _save(fig, path: Path, theme: Theme):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=theme.surface, bbox_inches="tight", pad_inches=0.22)
    import matplotlib.pyplot as plt

    plt.close(fig)
    log.info("wrote %s", path.name)


# --- data loading ----------------------------------------------------------


def _preds() -> pl.DataFrame:
    path = config.MODELS / "escalation_test_predictions.parquet"
    if not path.exists():
        raise RuntimeError(
            "no test predictions on disk -- run `wildfire backtest` first "
            "(it writes escalation_test_predictions.parquet)"
        )
    return pl.read_parquet(path)


def _json(name: str) -> dict:
    path = config.MODELS / name
    if not path.exists():
        raise RuntimeError(f"missing {name} -- run the matching backtest first")
    return json.loads(path.read_text())


# --- figures ---------------------------------------------------------------


def _reliability_points(y: np.ndarray, p: np.ndarray, bins: int = 8):
    """Equal-count score buckets that survive heavy ties.

    Quantile edges are useless here. Isotonic calibration is a step function,
    so the shipped score takes only ~26 distinct values on the test set and
    42% of fires sit at exactly zero -- most quantile edges land on the same
    number and collapse into empty buckets, leaving three points and a line
    drawn across a gap where there is no data.

    So: walk the sorted scores, close a bucket once it holds `target` rows,
    then keep extending until the score actually changes. Ties can never be
    split across buckets, and a bucket may be much larger than the target
    (the zero bucket is). Returns bucket counts too, so the chart can size
    each marker by how many fires it represents rather than implying they
    all carry equal weight.
    """
    order = np.argsort(p, kind="stable")
    ps, ys_sorted = p[order], y[order]
    n = len(ps)
    target = max(30, n // bins)

    spans, i = [], 0
    while i < n:
        j = min(i + target, n)
        if j < n:  # extend across the tie group straddling the boundary
            edge = ps[j - 1]
            while j < n and ps[j] == edge:
                j += 1
        spans.append((i, j))
        i = j

    # A short trailing bucket is noise; fold it back into its predecessor.
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < target // 3:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()

    xs = np.array([ps[a:b].mean() for a, b in spans])
    ys = np.array([ys_sorted[a:b].mean() for a, b in spans])
    ns = np.array([b - a for a, b in spans])
    return xs, ys, ns


def _rounded_barh(ax, ypos, values, *, height, color, zorder=3):
    """Horizontal bars with a rounded data-end, anchored flat on the baseline.

    Drawn as thick round-capped lines rather than rectangles, with the cap
    radius subtracted from the length so the painted extent still ends exactly
    at the value -- a rounded cap that overshoots would inflate every bar by
    half its thickness, which on a magnitude chart is a lie.
    """
    from matplotlib.lines import Line2D

    ax.figure.canvas.draw()  # transforms are only valid once laid out
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    px_per_x = bbox.width / (x1 - x0)
    px_per_y = bbox.height / (y1 - y0)
    lw_px = height * px_per_y
    lw_pt = lw_px * 72.0 / ax.figure.dpi
    r_data = (lw_px / 2.0) / px_per_x  # cap radius, in x-data units

    for yv, v in zip(ypos, values):
        end = max(v - r_data, r_data * 0.02)
        ax.add_line(
            Line2D([0, end], [yv, yv], color=color, linewidth=lw_pt,
                   solid_capstyle="round", zorder=zorder)
        )


def chart_calibration(theme: Theme, out: Path):
    """Predicted vs observed, before and after isotonic calibration.

    The story is that calibration was necessary, so both curves are drawn --
    this is a two-series comparison, not one line with an ideal.
    """
    preds = _preds()
    y = preds["y"].to_numpy().astype(int)

    fig, ax = _fig(theme, (5.4, 4.2))

    hi = 0.0
    series = []
    for col, colour, label in (
        ("p_uncalibrated", theme.series_2, "Uncalibrated"),
        ("p", theme.series_1, "Calibrated (shipped)"),
    ):
        xs, ys, ns = _reliability_points(y, preds[col].to_numpy(), bins=16)
        hi = max(hi, float(xs.max()) if len(xs) else 0.0, float(ys.max()) if len(ys) else 0.0)
        series.append((xs, ys, ns, colour, label))

    lim = min(1.0, hi * 1.18 + 0.01)
    ax.plot([0, lim], [0, lim], color=theme.muted, linewidth=1.2,
            linestyle=(0, (4, 3)), zorder=1)
    # Along the 45 degree line rather than at its end, where it collided with
    # the top-right marker and got clipped by the axes.
    ax.text(lim * 0.62, lim * 0.635, "perfectly calibrated", color=theme.muted,
            fontsize=8, va="bottom", ha="center", rotation=45,
            rotation_mode="anchor")

    # Markers only, no connecting line: the buckets are far apart on the x
    # axis (most fires score ~0) and a line between them would draw a trend
    # through a region containing no fires at all.
    for xs, ys, ns, colour, label in series:
        ax.scatter(xs, ys, s=18 + 190 * np.sqrt(ns / ns.max()), color=colour,
                   edgecolor=theme.surface, linewidth=1.4, zorder=3,
                   alpha=0.9, label=label, clip_on=False)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted probability of escalation", color=theme.ink_secondary, fontsize=9)
    ax.set_ylabel("Observed escalation rate", color=theme.ink_secondary, fontsize=9)
    _title(ax, theme,
           "When it says 15%, does 15% happen?",
           "Held-out 2025 season · one dot per score bucket, sized by fires in it")

    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left", scatterpoints=1)
    for t in leg.get_texts():
        t.set_color(theme.ink_secondary)
    for h in leg.legend_handles:
        h.set_sizes([46])
    _save(fig, out, theme)


def chart_pr_curve(theme: Theme, out: Path):
    """Precision-recall, model vs the only baseline that matters."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    preds = _preds()
    y = preds["y"].to_numpy().astype(int)
    prevalence = float(y.mean())

    fig, ax = _fig(theme, (5.4, 4.2))

    for col, colour, label in (
        ("size_at_decision", theme.series_2, "Size at decision"),
        ("p", theme.series_1, "Model"),
    ):
        score = preds[col].to_numpy().astype(float)
        prec, rec, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        ax.plot(rec, prec, color=colour, linewidth=2.0, zorder=3,
                label=f"{label}  (PR-AUC {ap:.3f})")

    ax.axhline(prevalence, color=theme.muted, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.99, prevalence, f"prevalence {prevalence:.3f} ", color=theme.muted,
            fontsize=8, va="bottom", ha="right")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(1.0, max(0.6, prevalence * 12)))
    ax.set_xlabel("Recall", color=theme.ink_secondary, fontsize=9)
    ax.set_ylabel("Precision", color=theme.ink_secondary, fontsize=9)
    _title(ax, theme,
           "Precision-recall against the honest baseline",
           f"Held-out 2025 season · {prevalence:.2%} of fires escalate, "
           f"so PR-AUC leads, not ROC-AUC")

    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(theme.ink_secondary)
    _save(fig, out, theme)


def chart_regions(theme: Theme, out: Path):
    """Leave-one-agency-out: point estimate + bootstrap interval per fold.

    A dot-and-interval, not bars: the interval is the point of the chart, and
    a bar chart of a statistic with a wide CI invites reading the bar length
    as certainty.
    """
    data = _json("escalation_spatial_backtest_region.json")["result"]
    folds = sorted(data["folds"], key=lambda f: f["pr_auc_model"])
    pooled = data["pooled"]

    labels, mid, lo, hi, base = [], [], [], [], []
    for f in folds:
        ci = (f.get("ci") or {}).get("pr_auc_model") or [f["pr_auc_model"]] * 2
        labels.append(f"{f['holdout_agency']}  n={f['n_test']:,}  pos={f['n_positives_test']}")
        mid.append(f["pr_auc_model"])
        lo.append(ci[0])
        hi.append(ci[1])
        base.append(f["pr_auc_size_only"])

    # The pooled block spells this `n_positives`; the per-fold blocks spell it
    # `n_positives_test`. Read both rather than assuming either.
    p_pos = pooled.get("n_positives", pooled.get("n_positives_test"))
    p_ci = (pooled.get("ci") or {}).get("pr_auc_model") or [pooled["pr_auc_model"]] * 2
    labels.append(f"POOLED  n={pooled['n_test']:,}  pos={p_pos}")
    mid.append(pooled["pr_auc_model"])
    lo.append(p_ci[0])
    hi.append(p_ci[1])
    base.append(pooled["pr_auc_size_only"])

    n = len(labels)
    ypos = np.arange(n)
    fig, ax = _fig(theme, (6.6, 0.44 * n + 1.9))
    ax.grid(axis="y", visible=False)

    for i in range(n):
        is_pooled = i == n - 1
        ax.plot([lo[i], hi[i]], [ypos[i], ypos[i]], color=theme.series_1,
                linewidth=2.0, alpha=0.45, solid_capstyle="round", zorder=2)
        ax.plot(mid[i], ypos[i], marker="o", markersize=9 if is_pooled else 7.5,
                color=theme.series_1, markeredgecolor=theme.surface,
                markeredgewidth=1.5, zorder=4,
                label="Model PR-AUC (5-95%)" if i == 0 else None)
        ax.plot(base[i], ypos[i], marker="|", markersize=11, markeredgewidth=2.4,
                color=theme.series_2, zorder=3,
                label="Size-at-decision baseline" if i == 0 else None)

    # Separate the pooled row: it is an aggregate of the rows above, not a peer.
    ax.axhline(n - 1.5, color=theme.axis, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5, color=theme.ink_secondary)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, max(hi) * 1.08)
    ax.set_xlabel("PR-AUC on the held-out agency", color=theme.ink_secondary, fontsize=9)
    # Read the fold tally from the artefact rather than hardcoding it -- it has
    # already changed once (9 of 9 became 8 of 9 after the determinism fix) and
    # a caption that quietly goes stale is worse than no caption.
    macro = data.get("macro", {})
    beat = macro.get("folds_beating_baseline")
    n_folds = macro.get("n_folds", len(folds))
    tally = (f"{beat} of {n_folds} folds beat their own within-fold baseline"
             if beat is not None else "per-fold baselines shown")
    _title(ax, theme,
           "Every fire scored by a model that never saw its region",
           f"Leave-one-agency-out · {tally}")

    # Below the axes: inside the plot it lands on the shortest fold's interval.
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper center",
                    bbox_to_anchor=(0.5, -0.10 - 1.4 / n), ncol=2)
    for t in leg.get_texts():
        t.set_color(theme.ink_secondary)
    _save(fig, out, theme)


def chart_importance(theme: Theme, out: Path, top: int = 12):
    """What the model actually leans on. Single series, so no legend."""
    res = _json("escalation_backtest.json")["result"]
    feats = res["top_features"][:top][::-1]
    names = [f[0] for f in feats]
    vals = [f[1] for f in feats]

    fig, ax = _fig(theme, (6.0, 0.34 * len(names) + 1.7))
    ax.grid(axis="y", visible=False)

    ypos = np.arange(len(names))
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8.5, color=theme.ink_secondary)
    ax.set_xlabel("LightGBM split gain", color=theme.ink_secondary, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.02)
    ax.set_ylim(-0.7, len(names) - 0.3)
    _rounded_barh(ax, ypos, vals, height=0.58, color=theme.series_1)
    _title(ax, theme,
           "What the model leans on",
           "Satellite geometry and detection lead-time outrank the fire's reported size")
    _save(fig, out, theme)


# --- driver ----------------------------------------------------------------

FIGURES = {
    "calibration": chart_calibration,
    "pr-curve": chart_pr_curve,
    "regions": chart_regions,
    "importance": chart_importance,
}


def generate(outdir: Path | None = None, *, dpi: int = 200,
             suffix: str = "") -> list[Path]:
    """Render every figure in both themes.

    `dpi` is lowered for the dashboard, where eight figures are inlined as
    base64 and full-resolution PNGs would triple the page weight for detail
    nobody can see at the size they are displayed.
    """
    global DPI
    outdir = outdir or DOCS_IMG
    previous, DPI = DPI, dpi
    try:
        written = []
        for name, fn in FIGURES.items():
            for theme in THEMES:
                path = outdir / f"{name}-{theme.name}{suffix}.png"
                fn(theme, path)
                written.append(path)
        return written
    finally:
        DPI = previous
