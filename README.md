# wildfire-forecast

Predicting **wildfire escalation** in Canada from open government feeds: a fire
enters the national reporting system, and 24 hours later we ask whether it will
be a large fire (≥ 100 ha) three days after discovery.

Three legs, each load-bearing:

| Leg | Source | Why it is here |
|---|---|---|
| **API / OGC services** | CWFIF national GeoServer (WFS), CWFIS download services, Open-Meteo ERA5, NASA FIRMS | dense signal: fire state, satellite detections, weather |
| **Browser-rendered source** | CIFFC national situation report | the only source of National Preparedness Level — a human judgement with no machine feed |
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

The CIFFC preparedness series backfills from a dated public API, no browser
required (see [below](#the-scraping-leg-that-turned-out-not-to-be-one)):

```bash
python -m wildfire ingest-ciffc -y 2023 -y 2024 -y 2025 -y 2026
```

The rendered-page fallback is still there, and still needs a browser, because
`ciffc.net/situation/` really is a 4 KB shell with an empty `#root`:

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
| PR-AUC | **0.270** [0.221, 0.329] | 0.176 [0.141, 0.228] | 0.021 |
| Brier | **0.0166** | — | 0.0203 |
| ROC-AUC | 0.950 | — | 0.5 |

So: **1.53× the precision-recall of "how big is it already"**, and 13× the base
rate. ROC-AUC of 0.950 looks spectacular and mostly is not — with a 2%
positive rate it is dominated by easy negatives, which is exactly why PR-AUC
leads the table.

The interval matters here. 117 positives is not many, and the two marginal
intervals above overlap — which invites the wrong conclusion. The comparison
that settles it is the *paired* one, model and baseline scored on the same
resample: the difference is **+0.094 [0.038, 0.141]**, and the model wins in
**99.8%** of draws. Overlapping marginal intervals on two correlated
statistics are not evidence of a tie.

> **This number used to read 0.292, and that was not reproducible.** Row order
> is not a modelling choice, but LightGBM's row subsampling and the
> calibration split's tie-break both read row positions, so the same data laid
> out differently scored anywhere from 0.258 to 0.296 — a spread wider than
> any feature effect this project has measured. Adding one upstream join was
> enough to move it. `_deterministic` now imposes a total order on
> `(t0, national_fire_id)` before anything reads a row position, and
> `tests/test_escalation.py` fails if row order ever changes a score again.
> 0.270 is the honest, reproducible figure; 0.292 was a lucky draw. Every
> number on this page is post-fix.

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
| BC | 6,191 | 118 | 0.189 | [0.148, 0.232] | 0.160 | 1.18× |
| SK | 1,794 | 94 | 0.328 | [0.263, 0.402] | 0.247 | 1.33× |
| QC | 2,212 | 93 | 0.275 | [0.214, 0.349] | 0.197 | 1.39× |
| MB | 1,111 | 51 | 0.183 | [0.136, 0.250] | 0.115 | 1.59× |
| NT | 685 | 46 | 0.209 | [0.134, 0.306] | 0.118 | 1.78× |
| ON | 2,333 | 44 | 0.290 | [0.206, 0.400] | 0.194 | 1.50× |
| AB | 4,103 | 38 | 0.235 | [0.146, 0.354] | 0.169 | 1.39× |
| YT | 523 | 33 | 0.366 | [0.244, 0.509] | 0.133 | 2.76× |
| PC | 339 | 14 | 0.302 | [0.162, 0.534] | 0.294 | 1.03× |
| **pooled** | **19,291** | **531** | **0.248** | **[0.221, 0.280]** | 0.157 | **1.59×** |

**It generalises.** Every fire above was scored by a model that had never seen
its region, and out-of-region PR-AUC (0.248) lands within noise of the
season-blocked headline (0.270). **All 9 folds** beat their own baseline;
pooled, the model wins 100% of bootstrap draws. Calibration survives the move
too — the top out-of-region bucket predicts 11.4% against 11.6% observed.

The weakest fold is instructive rather than worrying: **PC is Parks Canada**,
which is not a region at all but federal parkland scattered across the whole
country. Holding it out removes nothing the model can't find elsewhere, and it
is the fold where "unseen region" is the least meaningful frame — it scrapes
past its baseline at 1.03×.

Blocking *both* axes at once — train 2023–24 minus one agency, test that
agency in 2025 — is the strictest split this data supports, and thin enough
that only six agencies clear ten positives:

```bash
python -m wildfire backtest --holdout-agency ALL --test-years 2025
```

Pooled PR-AUC **0.243 [0.200, 0.299]** against a 0.185 baseline, 1.32× lift,
beating the baseline in 95.8% of draws; 5 of 6 folds win. BC is the loser
(0.41× lift) — in a quiet BC season with 13 escalations out of 1,353 fires,
raw size-at-decision was unusually predictive on its own.

### The follow-up that did not pan out

The obvious next move, if the model were leaning on geography, is to drop raw
`lat`/`lon` for transferable ecozone or fuel-regime features. That hypothesis
is testable directly, and it is wrong:

```bash
python -m wildfire backtest --holdout-agency ALL --drop-geography
```

Removing `lat`, `lon`, `agency_code` and `region_code` **drops** pooled
out-of-region PR-AUC from 0.248 to **0.223**. Fitting both variants on
identical folds and bootstrapping the difference paired gives **+0.025
[+0.004, +0.046]**, favouring geography in 98% of draws — a real effect, and a
modest one. So coordinates are not a memorisation crutch: latitude carries
transferable fire-regime signal (a 60°N boreal fire behaves differently from a
49°N one regardless of who reports it), and a model denied it does slightly
worse in regions it has never seen, not better. Ecozone features are still
worth adding — just *alongside* coordinates, not instead of them.

**Caveat, stated plainly.** The nine-fold table blocks region but shares
seasons: a Saskatchewan fire in train and a Manitoba fire in test can be
burning under the same synoptic ridge, which flatters it. That confound is why
the doubly-blocked run is reported alongside — it is the honest number, and it
is thin. Both point the same way, which is the reason to believe either.

## The scraping leg that turned out not to be one

CIFFC publishes the National Preparedness Level — the country's judgement
about how stretched suppression resources are. At PL 4–5 crews and aircraft
are rationed, which changes how a new fire is fought. There is no machine feed
for it, so this project was built to scrape it.

The premise was half right. `ciffc.net/situation/` really is a 4 KB shell with
an empty `#root`; plain HTTP gets you nothing. But the page loads no data
either — no XHR at all — which meant the numbers had to be reachable some
other way. They are: the JS bundle calls `api.ciffc.net/v1/sitrep`, and that
endpoint

- takes `?date=YYYY-MM-DD` and serves any past report,
- needs no credentials — the logged-out code path sends no `Authorization`
  header, and neither do we,
- carries far more than the page renders, including per-agency preparedness
  broken into its five components and each agency's own forecast of tomorrow's
  lightning- and human-caused ignitions.

`/v1/sitrep/archive` lists every published report — **1,025 of them, back to
2019** — and already contains the national PL for each, so the national series
costs one request.

This changed the leg from a liability into an asset. A scraped page only ever
shows *today*, so the series would have had to be accumulated forward and
could never have been a feature for a 2023–25 backtest. Because the API is
dated, the whole history comes down in one pass and preparedness becomes
**trainable**. Reports exist only for the fire season (April–October), which
is ~140 days a year rather than 365.

`render()` and `parse()` are kept as a documented fallback, and their
selectors were verified against the live DOM rather than left unproven — the
page does use real `<table>` elements, and the per-agency APL table it exposes
is now parsed too.

### Joining it without leaking

A sitrep is *dated* for a day but *published* late on that day — around 20:00
UTC — and it contains an explicit forecast of tomorrow's fire load. Joining on
`sitrep_date` would hand a fire first reported that morning a judgement made
hours after its decision instant. So ingestion carries `published_at` from
each record's `system_edit_timestamp`, and the feature join is an as-of
backward join on that column, per agency. `tests/test_preparedness.py` asserts
a report published after the decision instant cannot be seen.

### And it does nothing for escalation

Worth saying plainly, because the effort suggests otherwise. 6,565 agency-day
rows joined onto 98.5% of fires, and the effect on PR-AUC is nil:

| feature set | PR-AUC | Δ vs no CIFFC | 5–95% | P(helps) |
|---|---:|---:|---|---:|
| no CIFFC at all | 0.265 | — | — | — |
| full CIFFC block | 0.266 | +0.002 | [−0.020, +0.022] | 0.56 |
| CIFFC minus sitrep lag | **0.270** | +0.006 | [−0.015, +0.027] | 0.69 |
| national + agency PL only | 0.265 | −0.000 | [−0.020, +0.018] | 0.51 |
| sitrep lag only | 0.258 | −0.007 | [−0.030, +0.016] | 0.31 |

Every interval straddles zero. The likely reason is structural rather than a
data problem: preparedness varies by agency and by day, so every fire burning
in one agency on one day gets the same value — and `doy`, `agency_code` and
`region_code` already encode most of that seasonal-and-regional load pattern.
The judgement is real, but on this task it is largely redundant.

`ciffc_sitrep_lag_hours` — how stale the report was — is built and kept in the
modelling table but **excluded from the feature list**. It measures the
reporting calendar, not the fire, and it is the only variant that scored below
the no-CIFFC baseline. It briefly ranked as the #2 feature by gain, which is a
good reminder that gain measures how much a model *used* something, not
whether it should have.

The ingestion stays. Preparedness is a per-agency-day covariate, which is the
wrong shape for per-fire escalation but exactly the right shape for the
ignition-risk model (P5), where the unit of prediction is a grid cell and a
day. The negative result here is about the task, not the data.

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

Working: ingestion for all sources including the full CIFFC preparedness
backfill, the bitemporal as-of layer, the feature builder, and the backtest
harness — season-blocked, region-blocked, and both at once, all with bootstrap
intervals and all reproducible bit-for-bit. 24 tests, no network, no data,
under ten seconds.

Not yet built: the second model (ignition risk), and any kind of scoring
endpoint for the current season.

Next: **see [NEXT_STEPS.md](NEXT_STEPS.md)** — prioritised work, the invariants
not to break, and the gotchas already paid for. It is written to be picked up
cold.
