"""Build the shareable dashboard: one self-contained HTML file.

Everything is inlined -- boundaries, scored fires, charts as base64 -- because
the page has to work with no network at all, and because a dashboard that
silently loses its basemap when a CDN moves is worse than no dashboard.

Regenerated from the pipeline rather than hand-assembled: `wildfire dashboard`
reads the same artefacts the CLI writes, so the page cannot drift from the
model it claims to describe.
"""

from __future__ import annotations

import base64
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from . import config, report

log = logging.getLogger(__name__)

ASSETS = Path(__file__).parent / "assets"
TEMPLATE = ASSETS / "dashboard.html.tmpl"
PROVINCES = ASSETS / "canada_provinces.json"
OUT = config.ROOT / "docs" / "dashboard.html"

# Where to place each province's label, in lon/lat. Centroids of the polygon
# would drop several of them in water (Nunavut, BC) or on top of the densest
# fire clusters, so these are placed by hand in clear space.
LABELS = [
    ("BC", -125.0, 54.5), ("AB", -114.5, 55.0), ("SK", -106.0, 55.0),
    ("MB", -98.5, 55.0), ("ON", -86.0, 51.0), ("QC", -72.0, 52.5),
    ("NL", -60.5, 53.5), ("NB", -66.4, 46.6), ("NS", -62.5, 44.8),
    ("YT", -135.0, 63.5), ("NT", -119.0, 64.0), ("NU", -95.0, 66.5),
]

CHART_META = [
    ("regions", "Leave-one-agency-out: model PR-AUC with a 5-95% bootstrap "
                "interval against each fold's own baseline.",
     "Every fire scored by a model that never saw its region. Eight of nine "
     "provinces beat the size-at-decision baseline; pooled, the model wins "
     "100% of bootstrap draws."),
    ("pr-curve", "Precision-recall curve for the held-out 2025 season.",
     "Against the only baseline that matters. With 2% of fires escalating, "
     "PR-AUC leads and ROC-AUC would flatter almost anything."),
    ("calibration", "Reliability diagram, predicted against observed rate.",
     "A score of 15% should mean 15% of those fires escalate. Isotonic "
     "calibration on a temporally held-back slice is what closed the gap."),
    ("importance", "Feature importance by LightGBM split gain.",
     "Distance to the nearest satellite detection and how long orbit saw the "
     "fire before the ground reported it both outrank its reported size."),
]

METHOD_HTML = """
<p>A fire enters Canada's national reporting feed. Twenty-four hours later this
model asks one question: <b>will it be a large fire &mdash; 100 hectares or
more &mdash; three days after it was first reported?</b></p>

<h3>Why the timing is the whole problem</h3>
<p>The obvious way to build this is also the wrong one. Fire records are
revised constantly: a fire logged at 0.1&nbsp;ha on Monday reads 4,000&nbsp;ha
by Thursday, in the same row. Train on the current state of the table and the
model learns from numbers that did not exist when the forecast would have been
made, then collapses in production.</p>
<p>What makes this tractable is that the national feed is <b>bitemporal</b>.
Every fire carries many revisions, each stamped with the window during which it
was the system's current belief. So &ldquo;what was known at 24&nbsp;hours&rdquo;
is a filter, not a reconstruction &mdash; and point-in-time correctness stops
being a matter of discipline and becomes a property of the query.</p>

<h3>What it reads</h3>
<dl class="kv">
  <dt>Fire state</dt><dd>Reported size, control status, cause and response type, as they stood at the 24&nbsp;hour mark &mdash; plus how many times the agency had already revised the record, which is a good proxy for how worried they were.</dd>
  <dt>Satellite</dt><dd>Hotspot detections within 10&nbsp;km, already carrying Canadian fire-behaviour outputs: head fire intensity, rate of spread, fuel type. The strongest single signal is how long orbit saw the fire <em>before</em> anyone reported it.</dd>
  <dt>Context</dt><dd>Location, day of year, and the agency's own preparedness level that day &mdash; the last of which, measured honestly, turned out not to help.</dd>
</dl>

<h3>How it is checked</h3>
<p>Two blocks, because one is not enough. Holding out a <b>season</b> shows it
works on a year it never saw. Holding out a <b>region</b> &mdash; dropping an
entire province from training and testing only there &mdash; shows it learned
fire behaviour rather than memorising Alberta. Doing both at once is the
strictest split the data supports, and it agrees.</p>
<p>Every figure is reported against a size-at-decision baseline, because a
model that cannot beat &ldquo;how big is it already&rdquo; has learned nothing
worth deploying, and with intervals, because 117 escalations is not many.</p>

<h3>What it does not do</h3>
<p>It does not know about terrain, roads, values at risk, or what crews are
actually doing. It cannot see a fire no one has reported yet. The scores are
coarse by construction &mdash; calibration produces about 26 distinct values, so
this is a shortlist, not a ranking, and small differences between adjacent rows
carry no information. And the model scoring the fires above was fitted on every
completed season, so its accuracy is <em>inferred</em> from the held-out tests,
never directly measured on these fires.</p>
"""


def _b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _fires() -> pl.DataFrame:
    path = config.MODELS / "current_risk.parquet"
    if not path.exists():
        raise RuntimeError("no scored fires -- run `wildfire predict` first")
    return pl.read_parquet(path)


def _readouts(df: pl.DataFrame, meta: dict) -> list[dict]:
    high = df.filter(pl.col("risk") >= 0.25).height
    elev = df.filter(pl.col("risk") >= 0.10).height
    oc = df.filter(pl.col("status_now") == "OC").height
    return [
        {"k": "Fires scored", "v": f"{df.height:,}",
         "s": "first reported in the last 21 days"},
        {"k": "Flagged elevated", "v": f"{elev:,}", "tone": "sev1" if elev else "",
         "s": "10% or higher"},
        {"k": "Flagged high", "v": f"{high:,}", "tone": "sev2" if high else "",
         "s": "25% or higher"},
        {"k": "Out of control now", "v": f"{oc:,}",
         "s": "agency-reported status"},
    ]


def build(*, out: Path | None = None, top_map: int | None = None) -> Path:
    out = out or OUT
    out.parent.mkdir(parents=True, exist_ok=True)

    df = _fires().sort("risk", descending=True)
    if top_map:
        df = df.head(top_map)

    fires = [
        {
            "id": (r["national_fire_id"] or "").split("_", 2)[-1],
            "ag": r["agency_code"] or "--",
            "lat": round(r["lat"], 3),
            "lon": round(r["lon"], 3),
            "risk": round(r["risk"], 4),
            "dec": round(r["size_at_decision"] or 0.0, 2),
            "now": round(r["size_now"] or 0.0, 1),
            "hs": int(r["hs_count"] or 0),
            "age": int((r["age_hours"] or 0) // 24),
            "st": {"OC": "out of control", "BH": "being held",
                   "UC": "under control", "EX": "out"}.get(r["status_now"], "unknown"),
        }
        for r in df.iter_rows(named=True)
    ]

    asof = df["t0"].max()
    meta = {
        "asof": datetime.now(timezone.utc).strftime("%d %b %Y"),
        "window": "fires first reported in the preceding 21 days",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    # Web-weight charts: eight full-resolution PNGs inlined as base64 would
    # roughly triple the page for detail invisible at the size they render.
    with tempfile.TemporaryDirectory() as tmp:
        report.generate(Path(tmp), dpi=110)
        charts = [
            {
                "light": _b64(Path(tmp) / f"{name}-light.png"),
                "dark": _b64(Path(tmp) / f"{name}-dark.png"),
                "alt": alt,
                "caption": caption,
            }
            for name, alt, caption in CHART_META
        ]

    payload = {
        "meta": meta,
        "fires": fires,
        "provinces": json.loads(PROVINCES.read_text(encoding="utf-8")),
        "labels": LABELS,
        "readouts": _readouts(df, meta),
        "charts": charts,
        "method": METHOD_HTML,
    }

    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__PAYLOAD__",
        # `</script>` inside a JSON string would close the host tag early.
        json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"),
    )
    out.write_text(html, encoding="utf-8")
    log.info("dashboard: %s fires, %.0f KB -> %s", len(fires),
             out.stat().st_size / 1024, out)
    return out
