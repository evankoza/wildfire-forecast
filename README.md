# wildfire-forecast

Predicting **wildfire escalation** in Canada from open government feeds: a fire
enters the national reporting system, and 24 hours later we ask whether it will
be a large fire (≥ 100 ha) three days after discovery.

Three legs, each load-bearing:

| Leg | Source | Why it is here |
|---|---|---|
| **API / OGC services** | CWFIF national GeoServer (WFS), CWFIS download services, Open-Meteo ERA5, NASA FIRMS | dense signal: fire state, satellite detections, weather |
| **Web scraping** | CIFFC national situation report | the only source of National Preparedness Level — a human judgement with no machine feed |
| **ML** | LightGBM escalation classifier, season-blocked backtest | the actual question |

---

## The idea the project is built on

The CWFIF reported-fires layer is **bitemporal**. Every fire carries many rows,
each with a `record_start` / `record_end` pair describing the window during
which that row was the system's current belief:

```
2026_PC_2026RM4   0.2 ha  UC   record_start 2026-05-14T15:45 → 21:45
2026_PC_2026RM4   0.05 ha EX   record_start 2026-05-14T21:45 → (open)
```

That means "what was known about this fire at time T" is a **filter**, not a
reconstruction:

```sql
WHERE record_start <= T
```

Point-in-time correctness — normally the hardest thing to get right in a
forecasting pipeline, and the usual source of silently inflated metrics — is
therefore guaranteed at the source rather than by discipline in our code. Every
feature goes through `features/asof.py`, and `tests/test_asof.py` asserts that a
revision recorded after the decision time can never leak into a feature.

Observed depth (2023–2026): **121,600 revision rows across 23,165 fires**, 4–6
revisions per fire. Deep enough to model on; `wildfire diagnose` re-checks this
and flags any year that was backfilled as a flat snapshot.

## No credentials required

The entire Canadian pipeline runs with **zero API keys**. NRCan publishes open
directory listings at `cwfis.cfs.nrcan.gc.ca/downloads/`, and Open-Meteo needs no
key. NASA FIRMS (which does need a free MAP_KEY) is wired up but **optional** —
it buys near-real-time latency (~3 h vs next-morning) and non-Canadian coverage,
not core capability.

A pleasant surprise: the CWFIS hotspot feed is materially richer than raw FIRMS.
Each detection already carries Canadian FBP System outputs computed at that
pixel — `hfi` (head fire intensity, kW/m), `ros` (rate of spread), fuel type,
fuel consumption, and `fwi`. That is a lot of fire science we do not have to
reimplement.

## Architecture

```
sources/          landing zone            features/            models/
┌──────────────┐  data/raw/**            ┌──────────────┐    ┌───────────────┐
│ cwfis_fires  │─ verbatim responses ───►│ asof.py      │───►│ escalation.py │
│  (WFS, page) │  + ETag/Last-Modified   │  ONLY way to │    │  LightGBM     │
│ cwfis_hotspot│  + fetched_at           │  read state  │    │  season-block │
│ openmeteo    │                         ├──────────────┤    │  PR-AUC +     │
│ firms  (opt) │  data/curated/**        │ build.py     │    │  calibration  │
│ ciffc (scrape│─ typed parquet ────────►│  DuckDB      │    └───────────────┘
└──────────────┘                         │  haversine   │
                                         └──────────────┘
```

**Storage is parquet + DuckDB, not Postgres/PostGIS.** Raw responses land
verbatim so a re-parse never costs another request; curated parquet is the
store; DuckDB is the compute engine for the spatial-temporal join (bounding-box
prefilter, then exact haversine). Postgres earns its place at the *serving*
layer, once there is an API to serve — it is not needed to answer the modelling
question, and requiring PostGIS to run a backtest would be friction for nothing.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/pip install -e .
cp .env.example .env          # optional; only needed for the FIRMS leg
```

```bash
python -m wildfire ingest-fires -y 2023 -y 2024 -y 2025 -y 2026
```

```bash
python -m wildfire ingest-hotspots -y 2023 -y 2024 -y 2025
```

```bash
python -m wildfire diagnose
```

```bash
python -m wildfire build
```

```bash
python -m wildfire backtest --test-years 2025
```

The scraping leg needs a browser, because CIFFC is a client-rendered React app
with no public JSON API:

```bash
.venv/Scripts/pip install -e ".[scrape]" && .venv/Scripts/playwright install chromium
```

```bash
python -m wildfire scrape-ciffc
```

## The modelling contract

Defined once, in `config.EscalationSpec`:

- **T0** — first appearance in the national feed.
- **Decision time** T0 + 24 h — everything the model sees.
- **Horizon** T0 + 72 h — where the label is read.
- **Label** — `fire_size ≥ 100 ha` at the horizon.
- **Exclusions** — fires already ≥ 100 ha at decision time (the question is
  whether a fire *becomes* large, not whether a large fire stays large), and
  fires too recent to have a 72 h outcome yet (otherwise the newest fires all
  look like non-escalations and the model learns misplaced optimism).

Evaluation reports **three** numbers, not one: the model, a size-at-decision
baseline, and prevalence. A booster that cannot beat "how big is it already"
has learned nothing worth deploying.

## First backtest (real numbers)

Train 2023–2024, test 2025. 12,097 fires train / 5,664 test, 2.07% escalation
rate in the test season.

| metric | model | size-at-decision baseline | prevalence |
|---|---|---|---|
| PR-AUC | **0.292** | 0.176 | 0.021 |
| Brier | **0.0166** | — | 0.0203 |
| ROC-AUC | 0.955 | — | 0.5 |

So: **1.66× the precision-recall of "how big is it already"**, and 14× the base
rate. ROC-AUC of 0.955 looks spectacular and mostly is not — with a 2%
positive rate it is dominated by easy negatives, which is exactly why PR-AUC
leads the table.

Calibration mattered. The first version used `class_weight="balanced"`, scored
PR-AUC 0.275, and had a Brier score *worse than predicting the base rate*
(0.0246 vs 0.0203) — its top bucket predicted 16.9% where 10.1% actually
escalated. Refitting unweighted and isotonic-calibrating on a temporally
held-back slice of the training window fixed both: top bucket now predicts
14.3% against 11.1% observed.

Top features are dominated by `hs_dist_min_km`, `hs_detection_lead_hours` and
`hs_hfi_max` — how close the nearest satellite detection is, how long the
satellite saw it before the agency reported it, and how intense it was burning.
That detection-lead signal is the satisfying one: fires that orbit sees well
before the ground reports them behave differently.

**Caveats on this result.** `lat`, `lon` and `doy` carry large gain, which is
partly real (fire regimes are geographic and seasonal) and partly
memorisation — this has not been tested on a held-out *region*, only a held-out
season. And 2025 is a single test season; the confidence interval on 0.292 with
117 positives is wide. Both are next steps, not conclusions.

## Honest notes

- **ERA5 is reanalysis.** It is used only for the window that had already
  elapsed at decision time. Using it across the *forecast* window would be
  leakage dressed as a feature; a production system would substitute the actual
  forecast available at T0 + 24 h.
- **Weather is off by default** (`build --weather` to enable). It costs one HTTP
  request per fire-cell/date window. The hotspot feed already carries FWI, so
  the model is not blind without it.
- **Non-vegetation detections are dropped** (`water`, `urban`, flares). Keeping
  them teaches the model that gas plants are fires.
- **A few `record_start` values are implausible** — e.g. a 2023 fire with a
  record dated 2011. Small in number, but the pipeline should quarantine rather
  than silently trust them. Not yet handled.
- **`percent_contained` and `severity_nearest_dsr` are mostly `-1`** (the feed's
  "not reported" sentinel, nulled on ingest). They are near-useless in practice.

## Status

Working: ingestion for all sources, the bitemporal as-of layer (7 passing
tests), the feature builder, and the backtest harness.

Next: **see [NEXT_STEPS.md](NEXT_STEPS.md)** — prioritised work, the invariants
not to break, and the gotchas already paid for. It is written to be picked up
cold.

The headline item is a spatially-blocked backtest: the current result holds out
a season but never a region, so "learned fire behaviour" and "memorised Alberta"
are not yet distinguishable.
