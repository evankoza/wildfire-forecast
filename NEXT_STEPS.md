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

17 tests should pass in under ten seconds. They need no network and no data —
they run on an in-memory bitemporal fixture and a synthetic modelling table.
**If these fail, fix that before anything else**: they pin the point-in-time
guarantee the whole project rests on (`test_asof.py`) and the integrity of the
region block (`test_escalation.py`).

`data/` is gitignored and holds ~640 MB. Nothing in it is precious; it all
rebuilds from public endpoints in about five minutes:

```bash
.venv/Scripts/python.exe -m wildfire ingest-fires -y 2023 -y 2024 -y 2025 -y 2026
```

```bash
.venv/Scripts/python.exe -m wildfire ingest-hotspots -y 2023 -y 2024 -y 2025
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
| PR-AUC model | 0.2924 (5–95%: 0.237–0.360) |
| PR-AUC size-at-decision baseline | 0.1762 (5–95%: 0.141–0.227) |
| PR-AUC prevalence | 0.0207 |
| paired delta, model − baseline | +0.1162 (5–95%: 0.053–0.173) |
| Brier model / prevalence | 0.0166 / 0.0203 |
| pooled out-of-region PR-AUC (9 folds) | 0.2668 (5–95%: 0.240–0.300) |
| pooled out-of-region, region+season blocked | 0.2760 (5–95%: 0.224–0.341) |

The bootstrap is seeded (`seed=17` in `_bootstrap`), so intervals reproduce
exactly, not just approximately.

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

---

## 3. Settled — do not redo these

### ✅ P1 — Spatially-blocked backtest. **It generalises.**

`backtest --holdout-agency ALL` runs leave-one-agency-out; add `--test-years`
to block region *and* season at once. Full tables are in the README. The short
version: pooled out-of-region PR-AUC **0.267 [0.240, 0.300]** against a 0.157
within-fold baseline, 8 of 9 folds beating their baseline, and the doubly
blocked run agreeing at 0.276 [0.224, 0.341]. Out-of-region calibration holds
(top bucket 11.2% predicted / 11.6% observed). The season-blocked headline was
not an artefact of memorised geography.

Two things worth knowing before you build on it:

- **The contingent next step is dead.** The plan was: if it fails, drop raw
  `lat`/`lon` for ecozone features. `--drop-geography` tests that directly and
  it goes the *wrong* way — pooled out-of-region PR-AUC falls 0.267 → 0.209.
  Coordinates carry transferable fire-regime signal. Do not spend time
  replacing them; if you want ecozone features, add them *alongside*.
- **PC (Parks Canada) is the one losing fold**, and it is not a region — it is
  federal parkland scattered nationwide. Treat it as a known non-result rather
  than a bug to chase.

### ✅ P2 — Confidence intervals

Every PR-AUC now ships a seeded 1,000-draw percentile bootstrap (5th/95th),
including the *paired* model-minus-baseline difference on the same resample.
That pairing is the point: the marginal intervals for model and baseline nearly
touch, which reads as "indistinguishable", while the paired difference is
+0.116 [0.053, 0.173] and favours the model in 99.9% of draws. If you add a
metric, bootstrap it the same way — `_bootstrap` in `models/escalation.py`.

---

## 4. Priority work

### P3 — CIFFC scraper: run it, and settle the backfill question

The scraper is written (`sources/ciffc.py`) but **has never been run**. It needs:

```bash
.venv/Scripts/pip install -e ".[scrape]" && .venv/Scripts/playwright install chromium
```

```bash
.venv/Scripts/python.exe -m wildfire scrape-ciffc
```

Verify `parse()` actually finds the preparedness level and the per-agency table
against the live DOM — the selectors were written against the page as it looked
on 2026-07-30 and are not yet proven.

**The open question**: the sitrep page only ever shows *today*, so as written
this series has to be accumulated forward, which means it cannot be a feature
for the 2023–25 backtest. But the SPA's route table includes
`/en/sitrep-archive`, and `ciffc.net/situation/2026-07-14` renders a specific
past date. If archived sitreps are retrievable by date, the whole preparedness
series can be backfilled and it becomes usable as a training feature —
worth an hour to find out, because National Preparedness Level is the one
covariate in this project with no machine-feed equivalent.

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
