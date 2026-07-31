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


SOURCES_HTML = """
<p>All public data, and the whole Canadian pipeline runs with <b>no API keys at
all</b>. The NRCan feeds below are published through the
<a href="https://cwfis.cfs.nrcan.gc.ca/datamart">CWFIS Datamart</a>.</p>

<div class="tablewrap srcwrap">
<table class="srctable">
  <thead>
    <tr><th>Source</th><th>What it gives</th><th>Role</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://geoserver.cwfif.nrcan.gc.ca/geoserver/wfs?service=WFS&amp;version=2.0.1&amp;request=GetCapabilities">CWFIF national GeoServer</a><span class="src-note">NRCan &middot; WFS</span></td>
      <td>Every reported fire with its full revision history &mdash; size, control status, cause, response type. The bitemporal stamps on these records are what make an honest forecast possible.</td>
      <td class="yes">Fire state and the label</td>
    </tr>
    <tr>
      <td><a href="https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/">CWFIS satellite hotspots</a><span class="src-note">NRCan &middot; daily + season archives</span></td>
      <td>Satellite detections already carrying Canadian fire-behaviour outputs: head fire intensity, rate of spread, fuel type and consumption. 9.1&nbsp;million detections across 2023&ndash;26.</td>
      <td class="yes">The strongest features</td>
    </tr>
    <tr>
      <td><a href="https://www.naturalearthdata.com/">Natural Earth</a></td>
      <td>Provincial and territorial boundaries, simplified from 705&nbsp;KB to 23&nbsp;KB so the page carries its own basemap.</td>
      <td class="yes">This map</td>
    </tr>
    <tr>
      <td><a href="https://ciffc.net/situation/">CIFFC situation report</a><span class="src-note">via <a href="https://api.ciffc.net/v1/sitrep">api.ciffc.net/v1/sitrep</a></span></td>
      <td>National and per-agency preparedness levels &mdash; the human judgement about how stretched crews and aircraft are. No machine feed produces this.</td>
      <td class="null">Ingested. Measured effect: none</td>
    </tr>
    <tr>
      <td><a href="https://open-meteo.com/en/docs/historical-weather-api">Open-Meteo ERA5 archive</a></td>
      <td>Hourly reanalysis weather &mdash; temperature, humidity, wind, gusts, precipitation &mdash; back to 1940. No key required.</td>
      <td class="spare">Already covered by hotspot FWI</td>
    </tr>
    <tr>
      <td><a href="https://firms.modaps.eosdis.nasa.gov/">NASA FIRMS</a><span class="src-note">MODIS &amp; VIIRS</span></td>
      <td>Near-real-time active fire detections worldwide, within about three hours of satellite overpass.</td>
      <td class="spare">Same satellites as CWFIS, less detail</td>
    </tr>
  </tbody>
</table>
</div>

<h3>Why three of these are not feeding the model</h3>

<p><b>CIFFC preparedness is a result, not a gap.</b> It is ingested, backfilled
to 2019, and joined onto 98.5% of fires with a point-in-time-safe as-of join on
publication time. Its effect on escalation was then measured at +0.002 PR-AUC,
with an interval straddling zero. The likely reason is structural: preparedness
is one value per agency per day, so every fire burning in one province that day
shares it &mdash; and day-of-year plus agency already encode most of that
seasonal-and-regional load pattern. It stays ingested because an
agency-and-day covariate is exactly the right shape for the ignition model,
where the unit of prediction is a grid cell and a day. Reporting it as helpful
would be easier and false.</p>

<p><b>Open-Meteo would mostly repeat what the hotspots already carry.</b> Every
satellite detection arrives with the Fire Weather Index computed at that pixel,
and FWI is precisely a compression of the drought, wind and humidity history
that ERA5 would supply. So the marginal gain is unproven, while the cost is one
request per grid cell per date window &mdash; thousands per season against a
free public service. The client is built and a flag turns it on; the experiment
to show it earns that cost has not been run, and until it has, leaving it off
is the honest default rather than the lazy one.</p>

<p><b>NASA FIRMS is a strict subset here.</b> It publishes raw detections from
the same satellites CWFIS uses &mdash; VIIRS and MODIS &mdash; but NRCan ships
them already enriched with Canadian Fire Behaviour Prediction outputs: head fire
intensity, rate of spread, fuel type, fuel consumption. Using FIRMS instead
would mean re-deriving fire science that has already been done properly. It
earns its place for two things this project does not currently need: latency
(about three hours after overpass, against next morning) and coverage outside
Canada. It is also the only source that requires a key.</p>
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
        "sources": SOURCES_HTML,
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
