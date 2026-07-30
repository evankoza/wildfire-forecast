# wildfire-forecast

Predicting **wildfire escalation** in Canada from open government feeds: a fire
enters the national reporting system, and 24 hours later we ask whether it will
be a large fire (≥ 100 ha) three days after discovery.

Three legs, each load-bearing:

| Leg | Source | Why it is here |
|---|---|---|
| **API / OGC services** | CWFIF national GeoServer (WFS), CWFIS download services, Open-Meteo ERA5, NASA FIRMS | dense signal: fire state, satellite detections, weather |
| **Web scraping** | CIFFC national situation report | the only source of National Preparedness Level — a human judgement with no machine feed |
| **ML** | LightGBM escalation classifier, season- and region-blocked backtests | the actual question |

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

## Season-blocked backtest (real numbers)

Train 2023–2024, test 2025. 12,097 fires train / 5,664 test, 2.07% escalation
rate in the test season. Intervals are a 1,000-draw percentile bootstrap over
the test fires, 5th–95th.

| metric | model | size-at-decision baseline | prevalence |
|---|---|---|---|
| PR-AUC | **0.292** [0.237, 0.360] | 0.176 [0.141, 0.227] | 0.021 |
| Brier | **0.0166** | — | 0.0203 |
| ROC-AUC | 0.955 | — | 0.5 |

So: **1.66× the precision-recall of "how big is it already"**, and 14× the base
rate. ROC-AUC of 0.955 looks spectacular and mostly is not — with a 2%
positive rate it is dominated by easy negatives, which is exactly why PR-AUC
leads the table.

The interval matters here. 117 positives is not many, and the two marginal
intervals above nearly touch — which invites the wrong conclusion. The
comparison that settles it is the *paired* one, model and baseline scored on
the same resample: the difference is **+0.116 [0.053, 0.173]**, and the model
wins in **99.9%** of draws. Overlapping marginal intervals on two correlated
statistics are not evidence of a tie.

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

## Did it learn fire behaviour, or memorise Alberta?

`lat`, `lon` and `region_code` carry large gain, and a season-blocked split
cannot tell a geographic prior apart from a memorised one. So: hold out a
*region*. Each fold drops one reporting agency from training entirely and
tests on that agency's fires, with the size-at-decision baseline recomputed
inside the fold.

```bash
python -m wildfire backtest --holdout-agency ALL
```

| held out | n_test | pos | PR-AUC | 5–95% | size baseline | lift |
|---|---:|---:|---:|---|---:|---:|
| BC | 6,191 | 118 | 0.218 | [0.170, 0.273] | 0.160 | 1.36× |
| SK | 1,794 | 94 | 0.297 | [0.239, 0.365] | 0.247 | 1.21× |
| QC | 2,212 | 93 | 0.284 | [0.224, 0.360] | 0.197 | 1.44× |
| MB | 1,111 | 51 | 0.203 | [0.149, 0.289] | 0.115 | 1.77× |
| NT | 685 | 46 | 0.194 | [0.126, 0.272] | 0.118 | 1.64× |
| ON | 2,333 | 44 | 0.382 | [0.272, 0.495] | 0.194 | 1.97× |
| AB | 4,103 | 38 | 0.231 | [0.141, 0.349] | 0.169 | 1.36× |
| YT | 523 | 33 | 0.411 | [0.279, 0.547] | 0.133 | 3.10× |
| PC | 339 | 14 | 0.285 | [0.142, 0.458] | 0.294 | 0.97× |
| **pooled** | **19,291** | **531** | **0.267** | **[0.240, 0.300]** | 0.157 | **1.71×** |

**It generalises.** Every fire above was scored by a model that had never seen
its region, and out-of-region PR-AUC (0.267) lands within noise of the
season-blocked headline (0.292). 8 of 9 folds beat their own baseline; pooled,
the model wins 100% of bootstrap draws. Calibration survives the move too —
the top out-of-region bucket predicts 11.2% against 11.6% observed.

The one failure is instructive rather than worrying: **PC is Parks Canada**,
which is not a region at all but federal parkland scattered across the whole
country. Holding it out removes nothing the model can't find elsewhere, and it
is the fold where "unseen region" is the least meaningful frame.

Blocking *both* axes at once — train 2023–24 minus one agency, test that
agency in 2025 — is the strictest split this data supports, and thin enough
that only six agencies clear ten positives:

```bash
python -m wildfire backtest --holdout-agency ALL --test-years 2025
```

Pooled PR-AUC **0.276 [0.224, 0.341]** against a 0.185 baseline, 1.49× lift,
beating the baseline in 99.7% of draws; 5 of 6 folds win. BC is the loser
(0.61× lift) — in a quiet BC season with 13 escalations out of 1,353 fires,
raw size-at-decision was unusually predictive on its own.

### The follow-up that did not pan out

The obvious next move, if the model were leaning on geography, is to drop raw
`lat`/`lon` for transferable ecozone or fuel-regime features. That hypothesis
is testable directly, and it is wrong:

```bash
python -m wildfire backtest --holdout-agency ALL --drop-geography
```

Removing `lat`, `lon`, `agency_code` and `region_code` **drops** pooled
out-of-region PR-AUC from 0.267 to **0.209**, and takes two more folds below
their baseline. So coordinates are not a memorisation crutch — latitude is
carrying real, transferable fire-regime signal (a 60°N boreal fire behaves
differently from a 49°N one regardless of who reports it), and a model denied
it does worse in regions it has never seen, not better.

**Caveat, stated plainly.** The nine-fold table blocks region but shares
seasons: a Saskatchewan fire in train and a Manitoba fire in test can be
burning under the same synoptic ridge, which flatters it. That confound is why
the doubly-blocked run is reported alongside — it is the honest number, and it
is thin. Both point the same way, which is the reason to believe either.

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

Working: ingestion for all sources, the bitemporal as-of layer, the feature
builder, and the backtest harness — season-blocked, region-blocked, and both
at once, all with bootstrap intervals. 17 tests, no network, no data, under
ten seconds.

Not yet run: the CIFFC scraper (`sources/ciffc.py` is written but has never
touched the live page). Not yet built: the second model (ignition risk), and
any kind of scoring endpoint for the current season.

Next: **see [NEXT_STEPS.md](NEXT_STEPS.md)** — prioritised work, the invariants
not to break, and the gotchas already paid for. It is written to be picked up
cold.
