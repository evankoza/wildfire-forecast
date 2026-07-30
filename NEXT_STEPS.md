# Next steps / handoff

Written 2026-07-30. This file is meant to be read cold — by a person or a fresh
Claude Code instance with no memory of how the project got here. Read
`README.md` first for what the project *is*; this file is what to *do next*.

---

## 1. Get running from a clean clone

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
```

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

7 tests should pass in under a second. They need no network and no data — they
run on an in-memory bitemporal fixture. **If these fail, fix that before
anything else**: they pin the point-in-time guarantee the whole project rests
on.

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

### Baseline to reproduce

If the numbers below have moved, something changed — find out what before
building on top of it.

| metric | value |
|---|---|
| rows in modelling table | 21,503 (542 escalations, 2.52%) |
| n_train / n_test | 12,097 / 5,664 |
| PR-AUC model | 0.292 |
| PR-AUC size-at-decision baseline | 0.176 |
| PR-AUC prevalence | 0.021 |
| Brier model / prevalence | 0.0166 / 0.0203 |

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

---

## 3. Priority work

### P1 — Spatially-blocked backtest (highest value)

The current result holds out a *season*, never a *region*, and `lat`, `lon` and
`region_code` carry heavy feature gain. So we cannot currently distinguish
"learned fire behaviour" from "memorised Alberta". This is the test that decides
whether the model is worth anything.

Concretely: add `--holdout-agency` to `backtest`, run leave-one-agency-out over
the agencies with enough positives (BC, AB, ON, QC, SK, MB), and report PR-AUC
per fold against the size-at-decision baseline computed *within that fold*.

Expect degradation. The question is how much, and whether the model still beats
the baseline in a region it has never seen. If it does not, the honest next move
is to drop raw `lat`/`lon` in favour of ecozone / fuel-regime features that
transfer.

### P2 — Confidence intervals on the headline number

117 positives in the 2025 test set. PR-AUC 0.292 has a wide interval and it is
currently reported as a bare point estimate, which overstates what we know.
Bootstrap it (resample test fires with replacement, ~1000 draws, report the 5th
and 95th percentiles) and put the interval in the README table.

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
Do not start it until P1 has answered whether the simpler model generalises.

---

## 4. Smaller cleanups

- **`.gitattributes`** — git warns LF→CRLF on every file. Harmless on Windows,
  but a Linux clone or `ubuntu-latest` CI run will show spurious whole-file
  diffs. Three lines fixes it (`* text=auto eol=lf`).
- **Quarantine implausible `record_start`** — a handful of 2023 fires carry
  record timestamps from 2011. Currently trusted silently; they should be
  filtered with a logged count in `cwfis_fires._normalise`.
- **Test whether weather earns its cost** — `build --weather` is off by default
  (one HTTP request per fire-cell/date window). Run it on a subsample and check
  whether PR-AUC moves at all; the hotspot feed already carries FWI, so it may
  add nothing.
- **`percent_contained` / `severity_nearest_dsr` are almost always `-1`** in the
  feed (nulled on ingest). Probably worth dropping from `NUMERIC` entirely.
- **No CI.** A GitHub Actions job running `pytest` on push would be ~15 lines.

---

## 5. Gotchas already paid for

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
