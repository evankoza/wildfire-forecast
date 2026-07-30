# Next steps / handoff

Written 2026-07-30, revised the same day after P1 and P2 landed. This file is
meant to be read cold — by a person or a fresh Claude Code instance with no
memory of how the project got here. Read `README.md` first for what the project
*is*; this file is what to *do next*.

---

## 1. Get running from a clean clone

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
```

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

24 tests should pass in under ten seconds. They need no network and no data —
they run on an in-memory bitemporal fixture and a synthetic modelling table.
**If these fail, fix that before anything else**: they pin the point-in-time
guarantee the whole project rests on (`test_asof.py`), the integrity of the
region block and the row-order determinism (`test_escalation.py`), and the
as-of preparedness join (`test_preparedness.py`).

`data/` is gitignored and holds ~640 MB. Nothing in it is precious; it all
rebuilds from public endpoints in about five minutes:

```bash
.venv/Scripts/python.exe -m wildfire ingest-fires -y 2023 -y 2024 -y 2025 -y 2026
```

```bash
.venv/Scripts/python.exe -m wildfire ingest-hotspots -y 2023 -y 2024 -y 2025
```

```bash
.venv/Scripts/python.exe -m wildfire ingest-ciffc -y 2023 -y 2024 -y 2025 -y 2026
```

```bash
.venv/Scripts/python.exe -m wildfire build
```

```bash
.venv/Scripts/python.exe -m wildfire backtest --test-years 2025
```

```bash
.venv/Scripts/python.exe -m wildfire backtest --holdout-agency ALL
```

### Baseline to reproduce

If the numbers below have moved, something changed — find out what before
building on top of it.

| metric | value |
|---|---|
| rows in modelling table | 21,503 (542 escalations, 2.52%) |
| n_train / n_test | 12,097 / 5,664 |
| PR-AUC model | 0.2703 (5–95%: 0.221–0.329) |
| PR-AUC size-at-decision baseline | 0.1762 (5–95%: 0.141–0.228) |
| PR-AUC prevalence | 0.0207 |
| paired delta, model − baseline | +0.0941 (5–95%: 0.038–0.141) |
| Brier model / prevalence | 0.0166 / 0.0203 |
| pooled out-of-region PR-AUC (9 folds) | 0.2482 (5–95%: 0.221–0.280) |
| pooled out-of-region, region+season blocked | 0.2434 (5–95%: 0.200–0.299) |
| pooled out-of-region, no geography | 0.2229 (5–95%: 0.196–0.253) |

These are now genuinely reproducible — bit-for-bit, not approximately. Both
the bootstrap (`seed=17`) and the fit are seeded, **and row order is pinned**.

> ⚠️ **The earlier version of this table said 0.2924, and it was not
> reproducible.** LightGBM's row subsampling and the calibration split's
> tie-break both read row positions, so the same data in a different physical
> order scored 0.258–0.296. Adding one upstream join was enough to move it,
> which is how this was found. `_deterministic` in `models/escalation.py` now
> sorts by `(t0, national_fire_id)` before anything reads a row position.
> **If you add a join to `build()` and the headline moves, that is a bug, not
> a result** — `tests/test_escalation.py::test_row_order_does_not_change_the_score`
> is there to catch it.

---

## 2. Invariants — do not break these

- **`features/asof.py` is the only sanctioned way to read fire state.** Any
  feature built by reaching into `reported_fires.parquet` directly will silently
  see the future. If you add a feature, route it through `state_asof` /
  `revisions_before` and add a test alongside the ones in `tests/test_asof.py`.
- **Splits are temporal, never random.** `train_and_backtest` enforces
  `train_years < min(test_years)` and drops later seasons entirely. Do not
  "just use `train_test_split`" — the numbers will look much better and mean
  nothing.
- **`FORBIDDEN` in `models/escalation.py`** lists the columns that encode the
  answer. `_feature_frame` raises if any reaches the feature matrix. Add to it
  when you add outcome columns.
- **ERA5 must not be read past the decision time.** It is reanalysis, not a
  forecast. `openmeteo.fetch_window` takes an explicit `until` for this reason.
- **The test frame's category levels come from the training frame.**
  `_feature_frame(test, categories=levels)` pins them. Build a test matrix
  without passing `categories` and pandas renumbers the codes per frame, so a
  level missing from one side shifts every level after it — nothing raises, the
  model just reads `BC` as `ON`. This is load-bearing for the region-blocked
  backtest, where the held-out agency is absent from training *by design*, and
  it is asserted in `tests/test_escalation.py`.
- **Row order must never affect a score.** `_deterministic` sorts on
  `(t0, national_fire_id)` before any code reads a row position, because
  LightGBM's subsampling and the calibration tie-break both do. Remove it and
  the headline becomes a function of how the parquet was laid out. See the
  warning in §1.

---

## 3. Settled — do not redo these

### ✅ P1 — Spatially-blocked backtest. **It generalises.**

`backtest --holdout-agency ALL` runs leave-one-agency-out; add `--test-years`
to block region *and* season at once. Full tables are in the README. The short
version: pooled out-of-region PR-AUC **0.248 [0.221, 0.280]** against a 0.157
within-fold baseline, **all 9 folds** beating their baseline, and the doubly
blocked run agreeing at 0.243 [0.200, 0.299]. Out-of-region calibration holds
(top bucket 11.4% predicted / 11.6% observed). The season-blocked headline was
not an artefact of memorised geography.

Two things worth knowing before you build on it:

- **The contingent next step is dead.** The plan was: if it fails, drop raw
  `lat`/`lon` for ecozone features. `--drop-geography` tests that directly and
  it goes the *wrong* way — pooled out-of-region PR-AUC falls 0.248 → 0.223,
  paired difference **+0.025 [+0.004, +0.046]**, geography favoured in 98% of
  draws. Real, if modest. Coordinates carry transferable fire-regime signal;
  do not spend time replacing them. If you want ecozone features, add them
  *alongside*.
- **PC (Parks Canada) is the weakest fold** at 1.03× lift, and it is not a
  region — it is federal parkland scattered nationwide. Treat it as a known
  non-result rather than a bug to chase.

### ✅ P2 — Confidence intervals

Every PR-AUC now ships a seeded 1,000-draw percentile bootstrap (5th/95th),
including the *paired* model-minus-baseline difference on the same resample.
That pairing is the point: the marginal intervals for model and baseline nearly
overlap, which reads as "indistinguishable", while the paired difference is
+0.094 [0.038, 0.141] and favours the model in 99.8% of draws. If you add a
metric, bootstrap it the same way — `_bootstrap` in `models/escalation.py`.

### ✅ P3 — CIFFC. **Backfillable via API; no browser; null effect.**

Three answers, in descending order of how much they matter.

1. **There is a public JSON API and the scraper is not needed.**
   `api.ciffc.net/v1/sitrep?date=YYYY-MM-DD` serves any past report with no
   credentials — the site's own logged-out path sends no `Authorization`
   header. `/v1/sitrep/archive` lists all 1,025 published reports back to 2019
   and already carries the national PL for each. Playwright was never
   installed. `ingest-ciffc` backfilled 2023–2026 in one pass: 6,565
   agency-day rows over 505 sitrep days (fire season only, April–October).
2. **It carries much more than the page renders** — per-agency preparedness
   split into its five components, plus each agency's own forecast of
   tomorrow's lightning- and human-caused ignitions.
3. **It does not help the escalation model.** Full block +0.002 PR-AUC
   [−0.020, +0.022]; national + agency PL alone −0.000. Every interval
   straddles zero. Probable reason: preparedness is a per-agency-day value, so
   every fire in one agency on one day shares it, and `doy` / `agency_code` /
   `region_code` already carry most of that signal.

The ingestion is kept anyway, on purpose: an agency-day covariate is the wrong
shape for per-fire escalation but the *right* shape for ignition risk (P5),
where the unit of prediction is a cell and a day. Re-test it there.

`ciffc_sitrep_lag_hours` is built but deliberately excluded from `NUMERIC` —
it measures the reporting calendar rather than the fire, and it was the only
variant to score below the no-CIFFC baseline. It reached #2 by feature gain
before being cut, which is worth remembering: gain says how much a model used
something, not whether it should have.

`render()`/`parse()` remain as a documented fallback, with their selectors now
verified against the live DOM (the page does use real `<table>` elements) and
the per-agency APL table newly parsed.

---

## 4. Priority work

### P4 — Current-season scoring

2026 hotspots have no archive zip yet, so `ingest-hotspots -y 2026` falls back
to ~210 daily file requests (`_load_days_for_year`). It works but is slow and
untested at that volume; consider restricting to fire-season months.

Then add a `predict` command: load `data/models/escalation_lgbm.joblib`, pull
currently-burning fires from `activefires.csv`, and score the ones within 24 h
of first report. That is the first thing that looks like a product.

### P5 — The second model (ignition risk)

The original design had two models; only escalation is built. Ignition risk is
per ~10 km grid cell per day: P(≥1 new fire detection tomorrow), features from
FWI + weather + fuel + lightning + distance to roads/settlement.

This is a much bigger lift than it sounds — it needs a grid, negative sampling
(~0.1% positive rate), and careful handling of the fact that **absence of a
detection is not absence of fire** (satellite overpass timing, cloud cover).
P1 has now cleared the gate it was waiting on: the simpler model does
generalise spatially, so this is unblocked. It is still the largest item here
by a wide margin — do P3 and P4 first unless there is a specific reason not to.

---

## 5. Smaller cleanups

- ~~**`.gitattributes`**~~ — done (`* text=auto eol=lf`, plus binary markers).
- ~~**No CI**~~ — done: `.github/workflows/ci.yml` runs `pytest` on push and PR
  against `ubuntu-latest` / Python 3.12. The suite needs no network or data.
  It has never actually run on GitHub, because nothing has been pushed yet.
- **Quarantine implausible `record_start`** — a handful of 2023 fires carry
  record timestamps from 2011. Currently trusted silently; they should be
  filtered with a logged count in `cwfis_fires._normalise`. Note this changes
  `data/curated/` and so may move the baseline table above: re-run both
  backtests afterwards and record the delta rather than quietly replacing the
  numbers.
- **Test whether weather earns its cost** — `build --weather` is off by default
  (one HTTP request per fire-cell/date window). Run it on a subsample and check
  whether PR-AUC moves at all; the hotspot feed already carries FWI, so it may
  add nothing.
- **`percent_contained` / `severity_nearest_dsr` are almost always `-1`** in the
  feed (nulled on ingest). Probably worth dropping from `NUMERIC` entirely.
- **`hs_estarea_sum` and `hs_bfc_mean` are declared in `NUMERIC` but never
  built** — the season archives lack `estarea` and `bfc` (see §6), so those two
  columns are absent from the modelling table and silently skipped by
  `_feature_frame`. Harmless, but misleading to read.

---

## 6. Gotchas already paid for

- **Row order silently moved the headline by 0.03 PR-AUC.** Covered in §1 and
  §2; repeated here because it is the single most expensive thing in this
  file. If a metric shifts after a pipeline change that should not have
  touched the data, suspect ordering before suspecting the feature.
- **CIFFC has a public JSON API; do not write a scraper.**
  `api.ciffc.net/v1/sitrep?date=YYYY-MM-DD`, no credentials. The tell was that
  the rendered page issues *no* data XHR at all, which meant the numbers had
  to be reachable another way — the endpoint is named in
  `static/js/main.*.chunk.js`. Reports exist April–October only.
- **`agencies_sitereps`** — yes, that is how the API spells it. Do not
  "correct" the key.

Things that cost time once. Do not rediscover them.

- **GeoServer caps the fires layer at 10,000 features per response**, whatever
  `count` asks for. `PAGE` in `cwfis_fires.py` must equal that cap — set it
  higher and a short page reads as "end of data" and the pull silently
  truncates. This is why the first ingest returned 2,821 fires instead of 23,165.
- **Hotspot season archives and rolling daily files have different columns.**
  The archives lack `estarea` and `bfc`. `hotspot_features` builds its SQL
  projection from whatever columns are present; keep it that way.
- **`CalibratedClassifierCV(cv="prefit")` was removed** from recent scikit-learn.
  The project uses `IsotonicRegression` directly instead — three lines, and
  immune to that API churn.
- **`class_weight="balanced"` wrecked calibration.** It improved tree splits on
  the rare class but inflated probabilities so badly the Brier score lost to
  predicting the base rate. Current approach: fit unweighted, then isotonic-
  calibrate on a temporally held-back slice of the training window. If you
  re-add class weighting, re-check Brier and the reliability table, not just
  PR-AUC.
- **Typer list options need a repeated flag**: `-y 2023 -y 2024`, not
  `-y 2023 2024`.
- **On Windows, set `PYTHONIOENCODING=utf-8`** or `rich`/`polars` box-drawing
  output crashes with a `cp1252` encode error.
- **All Canadian sources need zero credentials.** If you find yourself blocked
  on an API key, you are probably reaching for FIRMS — which is optional. The
  NRCan open directory listings at `cwfis.cfs.nrcan.gc.ca/downloads/` have
  everything the pipeline needs.
