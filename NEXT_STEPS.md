# Next steps / handoff

Written 2026-07-30, rewritten 2026-07-31 after the ignition model, the serving
layer and the paired-ablation harness landed. This file is meant to be read
cold — by a person or a fresh Claude Code instance with no memory of how the
project got here. Read `README.md` first for what the project *is*; this is
what to *do next*.

---

## 1. Get running from a clean clone

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev,serve]"
```

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

97 tests should pass in about ten seconds. They need no network and no data —
they run on in-memory fixtures. **If these fail, fix that before anything
else**: they pin the point-in-time guarantee the whole project rests on
(`test_asof.py`, `test_ignition.py`), the integrity of the region block and the
row-order determinism (`test_escalation.py`), the as-of preparedness join
(`test_preparedness.py`), the order-independence of the hotspot aggregation
(`test_determinism.py`), the grid's equal-area property and country filter
(`test_grid.py`), the sampling corrections (`test_ignition.py`), the timestamp
quarantine (`test_quarantine.py`) and the serving contract (`test_api.py`).

`data/` is gitignored and holds ~1.5 GB. Nothing in it is precious; it all
rebuilds from public endpoints:

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
.venv/Scripts/python.exe -m wildfire ingest-fwi -y 2023 -y 2024 -y 2025
```

```bash
.venv/Scripts/python.exe -m wildfire build && .venv/Scripts/python.exe -m wildfire build-ignition
```

The fire, hotspot and CIFFC pulls take about five minutes between them. The FWI
backfill is a single **670 MB** stream and will take longer on a slow link; it
is fetched once and re-parsed thereafter.

### Baseline to reproduce

If the numbers below have moved, something changed — find out what before
building on top of it.

**Escalation** (train 2023–24, test 2025)

| metric | value |
|---|---|
| rows in modelling table | 21,495 (542 escalations, 2.52%) |
| n_train / n_test | 12,089 / 5,664 |
| PR-AUC model | 0.2658 (5–95%: 0.2155–0.3231) |
| PR-AUC size-at-decision baseline | 0.1762 (5–95%: 0.1405–0.2282) |
| PR-AUC prevalence | 0.0207 |
| paired delta, model − baseline | +0.0896 (5–95%: 0.0343–0.1385) |
| Brier model / prevalence | 0.0169 / 0.0203 |
| pooled out-of-region PR-AUC (8 of 9 folds win) | 0.2664 (5–95%: 0.2393–0.2987) |
| pooled out-of-region, region+season blocked | 0.2428 (5–95%: 0.1993–0.2980) |
| pooled out-of-region, no geography | 0.2485 (5–95%: 0.2212–0.2790) |
| paired geography delta, out of region | +0.0189 (5–95%: 0.0026–0.0362) |

**Ignition** (train 2023–24, test 2025, `neg_rate=0.03`)

| metric | value |
|---|---|
| study area | 18,474 cells (2023–24 activity, inside Canada) |
| study-area coverage of test ignitions | 89.2% |
| panel rows | 335,811 (14,396 positive, 4.29% of the sample) |
| n_train / n_test | 247,977 / 87,834 |
| population prevalence (test) | 0.000917 |
| PR-AUC model | 0.01936 (5–95%: 0.01509–0.03150) |
| PR-AUC climatology baseline | 0.00629 |
| PR-AUC fire-weather baseline | 0.00168 |
| paired delta vs climatology | +0.0131 (5–95%: 0.0082–0.0247), P = 1.000 |
| Brier model / prevalence | 0.00091468 / 0.00091635 |
| pooled out-of-region (11 of 11 folds win) | 0.01874 (5–95%: 0.01690–0.02237) |
| pooled out-of-region, region+season blocked | 0.01229 (5–95%: 0.00975–0.02238) |

> ⚠️ **The escalation table has been wrong three times, for three independent
> reasons, and every time the number looked perfectly plausible.**
>
> 1. *In the model* (said 0.2924). LightGBM's row subsampling and the
>    calibration split's tie-break read row positions, so the same data in a
>    different physical order scored 0.258–0.296. Fixed by `_deterministic`.
> 2. *In the features* (said 0.2703). DuckDB's parallel `AVG` over float64 and
>    its `MODE` tie-break both vary with reduction/scan order, worth ~0.008
>    PR-AUC. Fixed by DECIMAL sums and an explicit `ROW_NUMBER()` tie-break in
>    `hotspot_features`.
> 3. *In the source data* (said 0.2641). 25 revision rows carried a
>    `record_start` that could not belong to their fire year — two dated 2011,
>    the rest dated 2025–26 against a stale `fire_year`. T0 is
>    `min(record_start)`, so each one moved a fire's whole decision instant.
>    Fixed by `quarantine_record_start`, worth +0.0017.
>
> **Verify a pipeline change by building the table twice and diffing it**,
> null-aware, NaN == NaN:
>
> ```bash
> python -m wildfire build && cp data/curated/modelling_table.parquet /tmp/a.parquet && python -m wildfire build
> ```
>
> **If you add a join to `build()` and the headline moves, that is a bug, not
> a result.** `tests/test_escalation.py::test_row_order_does_not_change_the_score`
> and `tests/test_determinism.py` catch the first two causes; a third showed up
> the same way.

---

## 2. Invariants — do not break these

- **`features/asof.py` is the only sanctioned way to read fire state.** Any
  feature built by reaching into `reported_fires.parquet` directly will
  silently see the future. Route it through `state_asof` / `revisions_before`
  and add a test alongside the ones in `tests/test_asof.py`.
- **The ignition panel has no such structural guarantee, so it has tests
  instead.** Its windows are arithmetic on dates, and a fencepost error widens
  one by a day without changing anything visible. `tests/test_ignition.py`
  plants an ignition and a hotspot *on* the decision day and demands neither
  appears in a feature. Add a window, add a test like those.
- **Splits are temporal, never random.** Both `train_and_backtest` functions
  enforce `train_years < min(test_years)` and drop later seasons entirely. Do
  not "just use `train_test_split`" — the numbers will look much better and
  mean nothing.
- **`FORBIDDEN` in both model modules** lists the columns that encode the
  answer. `_feature_frame` raises if any reaches the feature matrix. Add to it
  when you add outcome columns.
- **ERA5 must not be read past the decision time.** It is reanalysis, not a
  forecast. `openmeteo.fetch_window` takes an explicit `until` for this reason.
  Station FWI has the same rule enforced differently: the interpolation reads
  `day - 1`, because the observation is taken at noon local and at the 00:00
  decision instant today's does not exist.
- **The test frame's category levels come from the training frame.**
  `_feature_frame(test, categories=levels)` pins them. Build a test matrix
  without passing `categories` and pandas renumbers the codes per frame, so a
  level missing from one side shifts every level after it — nothing raises, the
  model just reads `BC` as `ON`. Load-bearing for every region-blocked fold,
  where the held-out agency is absent from training *by design*.
- **Row order must never affect a score.** `_deterministic` sorts before any
  code reads a row position, because LightGBM's subsampling and the calibration
  tie-break both do. See the warning in §1.
- **Aggregations must be order-independent, not just the model.** In
  `hotspot_features`, sums accumulate in `DECIMAL(18,6)` and the dominant fuel
  type is chosen by explicit `ROW_NUMBER()` ordering — never `AVG` over float64
  or `MODE`. `tests/test_determinism.py` re-runs the join on shuffled input and
  demands identical output.
- **The ignition study area comes from training seasons only.** `build-ignition`
  warns if you widen it to include a season you are about to evaluate on. The
  domain must not be drawn around the fires the model is about to forecast.
- **Never report an unweighted ignition metric.** Every PR-AUC, Brier and
  reliability figure takes `sample_weights`, which puts it back on the
  population's 0.09% prevalence. Unweighted the same model reads ~0.30 instead
  of 0.019.
- **Prior-correct with the rate stored in the model**, not the module default.
  `predict.score_ignition` reads `payload["neg_rate"]`. A model fitted under one
  rate and corrected with another is wrong by exactly the ratio, silently.
- **The API serves artefacts; it must never ingest or refit.** A rebuild is
  minutes of traffic against government endpoints. Putting it behind a request
  handler means the first curious visitor DDoSes NRCan on your behalf.

---

## 3. Settled — do not redo these

### ✅ P1 — Spatially-blocked backtest. **Both models generalise.**

`backtest --holdout-agency ALL` and `backtest-ignition --holdout-agency ALL` run
leave-one-agency-out; add `--test-years` to block region *and* season at once.
Full tables are in the README. Escalation: pooled out-of-region 0.266
[0.239, 0.299] against a 0.157 within-fold baseline, 8 of 9 folds winning.
Ignition: 11 of 11 folds beat both baselines, pooled 3.9× lift.

- **The contingent next step is dead.** The plan was: if it fails, drop raw
  `lat`/`lon` for ecozone features. `compare-geography` tests that directly,
  paired on the pooled out-of-region fires, and it goes the *wrong* way —
  +0.019 [+0.003, +0.036] in favour of keeping geography, helping in 97.6% of
  draws. Coordinates carry transferable fire-regime signal; do not spend time
  replacing them. If you want ecozone features, add them *alongside*.
- **PC (Parks Canada) is the one escalation fold that loses**, at 0.77×. It is
  also not a region: it is federal parkland scattered nationwide, and with 14
  positives its interval spans the baseline. A known non-result, not a bug.

### ✅ P2 — Confidence intervals, and then paired ones

Every PR-AUC ships a seeded 1,000-draw percentile bootstrap. More importantly,
`ablate` and `ablate-ignition` bootstrap *differences* between feature sets
**paired on the same resample**. That pairing is the point: with ~117
escalations every marginal interval swallows every other point estimate, so an
unpaired table says "no difference" whatever is true.

If you add a metric, bootstrap it the same way. If you add a feature block, add
it to `FEATURE_SETS` so its worth is measured rather than assumed.

### ✅ P3 — CIFFC. **Backfillable via API; no browser; null effect, twice.**

`api.ciffc.net/v1/sitrep?date=YYYY-MM-DD` serves any past report with no
credentials. `/v1/sitrep/archive` lists all 1,025 published reports back to
2019. Playwright was never installed. 6,565 agency-day rows over 505 sitrep
days.

It does not help escalation (+0.011 in favour of *dropping* it, interval
straddling zero) and it does not help ignition either (+0.0009 in favour of
dropping). The second result is the one that closes the question: the whole
argument for keeping the ingestion was that an agency-day covariate is the
right shape for a cell-day target, and it was measured there and still did
nothing. Do not re-litigate this without a new idea about *why* it would work.

`ciffc_sitrep_lag_hours` is built but deliberately excluded from `NUMERIC` — it
measures the reporting calendar rather than the fire. It reached #2 by feature
gain before being cut, which is worth remembering: gain says how much a model
used something, not whether it should have.

`render()`/`parse()` remain as a documented fallback.

### ✅ P4 — Current-season scoring, both models

`fit-final` / `predict` and `fit-final-ignition` / `predict-ignition`. Three
things are load-bearing:

- **Features come from `assemble_features`** — the same function that builds
  the training table, one per model. A separate serving path is the standard
  way a model that backtests well quietly starts seeing a different
  distribution than it was fitted on.
- **Category levels are restored from the fitted model**, not rebuilt from
  today's data.
- **`predict` uses a different model than the backtest reports on.** The
  backtest model is deliberately handicapped; `fit-final` refits on every
  labelled season. So the deployed model's accuracy is *inferred* from the
  backtest, never directly measured. That is the honest arrangement, but state
  it as such.

Scoring the current season needs `ingest-fwi -y <year> --daily`: the decadal
archive lags by most of a year, and the daily files have a completely different
schema (see §6).

### ✅ P5 — The ignition model

Built. `features/grid.py` (10 km Lambert azimuthal equal-area),
`features/ignition.py` (panel), `models/ignition.py` (fit, correction,
backtests, ablation). The design decisions and their justifications are in the
README and in the module docstrings; the ones that cost time are in §6.

### ✅ Serving layer

`wildfire serve` — FastAPI over the artefacts, optional extra `[serve]`. It
reads files the CLI wrote and never runs the pipeline. `tests/test_api.py`
covers the failure modes, which are the interesting part.

### ✅ Dashboard

`wildfire dashboard` builds `docs/dashboard.html` — a single self-contained
file, no external requests. Boundaries come from Natural Earth via
`src/wildfire/assets/canada_provinces.json`; charts are re-rendered at 110 dpi
rather than the README's 200.

Four things to know before editing it:

- **The map is Canvas, and it was SVG first, for three reasons.** 1,032 circles
  as DOM nodes made hover visibly laggy; SVG `text` sized in `px` inside a
  viewBox whose entire width is 0.87 *user units* renders each glyph about nine
  times the width of the map; and the y-flip is far easier to get right once in
  a transform than across a DOM tree.
- **Projected y grows north; screen y grows down.** The flip lives in `sy()`
  and nowhere else. Getting it wrong renders the whole country upside down, and
  no structural check catches it — the paths are all valid.
- **Hit-test synchronously, coalesce only the repaint.** An earlier version did
  both inside `requestAnimationFrame`; wherever rAF is throttled the "queued"
  flag latched on and hover stopped working permanently.
- **The canvas does not restyle itself on a theme change.** Add a token to the
  map and it must be read in `readPalette`.
- **The template must stay pure ASCII inside `<script>`.** The page declares no
  charset of its own, so a raw `·` or `—` in a JS string literal is at the mercy
  of the host's `Content-Type`. Check with
  `python -c "print(open(p,encoding='utf-8').read().isascii())"`.
- **`docs/dashboard.html` is committed but generated.** Regenerate rather than
  hand-edit, and avoid committing a rebuild unless the content actually changed
  — it is ~600 KB and will bloat history if it lands every run.

**How to verify it.** The published artifact is private, so the in-app browser
cannot open it; serve `docs/` over `http.server` and drive that. Screenshots
fail when the pane is not displayed, which is exactly how the upside-down map
shipped. Two techniques that work without a screenshot: sample the canvas into
a coarse ASCII grid (`getImageData`, classify by colour, print ~76x30), which
makes the country's shape and orientation obvious; and sample a patch of land
before and after flipping `data-theme` to prove the repaint fired.

---

## 4. Priority work

### P6 — A lightning feed for ignition

The single largest missing input. Roughly half of Canadian wildfire ignitions
are lightning-caused, and the model currently has no way to see a strike. The
Canadian Lightning Detection Network is not public; ECCC's alert feeds are
nowcasts rather than a strike archive. Worth an afternoon of looking before
committing: if a strike-density product exists at daily × ~10 km resolution,
it is the one covariate likely to move the ignition number materially, and it
would slot straight into `assemble_features` as another cell-day join.

### P7 — Human-caused ignition needs a human layer

`wx_dist_km` is currently doing double duty as "how good is this cell's
interpolated weather" and "how remote is it", and the ablation cannot separate
those. Distance to roads and settlement would separate them properly and is the
obvious feature for the human-caused half of the target. Natural Earth ships
roads; the accuracy is poor but it is free and already a dependency.

### P8 — A fourth season

Almost every interval in this project is wide because there are two training
seasons and one test season. 2026 will be complete after October and the
hotspot archive zip appears shortly after. Re-running everything with
2023–25 train / 2026 test is the cheapest available improvement to every
number on the page, and it is also the honest re-test of the fire-weather
null result in §3.

---

## 5. Smaller cleanups

- ~~**`.gitattributes`**~~ — done.
- ~~**No CI**~~ — done and green (`.github/workflows/ci.yml`).
- ~~**Quarantine implausible `record_start`**~~ — done, +0.0017 on the
  headline, 25 rows written to `curated/reported_fires_quarantine.parquet`.
- ~~**Re-run the CIFFC ablation**~~ — done, and now paired. See §3.
- ~~**Re-run the paired geography bootstrap**~~ — done: `compare-geography`.
- ~~**`percent_contained` / `severity_nearest_dsr`**~~ — dropped from the
  feature contract. Measured effect: exactly 0.0000 with a [0, 0] paired
  interval — the predictions were bit-identical.
- **Test whether ERA5 weather earns its cost on escalation** — `build --weather`
  is still off by default (one HTTP request per fire-cell/date window). The
  station-FWI result on ignition suggests it will do nothing, which is a reason
  to run it cheaply on a subsample rather than a reason not to run it.
- **The ignition reliability curve is a constant factor off, not a shape
  problem.** A single scalar recalibration per season would fix most of it, but
  fitting that needs the season's outcome — so it is only honest as a
  *reported* correction on historical data, never applied to today. Worth
  writing up rather than silently applying.
- **`docs/dashboard.html` shows only the escalation model.** The ignition
  layer has a map-shaped output (a value per cell) and no map on the page yet.
- **The API has no pagination beyond `limit`**, and `/v1/ignition` will happily
  return 5,000 cells. Fine for a research artefact, wrong for anything else.

---

## 6. Gotchas already paid for

Things that cost time once. Do not rediscover them.

### Reproducibility

- **Row order silently moved the headline by 0.03 PR-AUC.** Covered in §1 and
  §2; repeated here because it is the single most expensive thing in this file.
  If a metric shifts after a change that should not have touched the data,
  suspect ordering before suspecting the feature.
- **And then it happened again, one layer down — in DuckDB.** Fixing the row
  order in the *model* did not make the *features* reproducible. Float addition
  is not associative and DuckDB reduces partial sums in thread-completion
  order, so `AVG` drifted in its last bits between runs of the same query on
  the same data, and `MODE` broke ties by scan order. ~35 of 21,500 fires had a
  feature cross a split threshold: ~0.008 PR-AUC. Fixed with `DECIMAL(18,6)`
  sums (associative, and unlike `SET threads TO 1` it does not add minutes to a
  rebuild) and an explicit `ROW_NUMBER()` tie-break.

  The general lesson is the one worth keeping: **"reproducible" has to be
  measured, not asserted.** Build the artefact twice and diff it. Both bugs
  were invisible until something that could not possibly have changed a number
  appeared to change one — and the first version of that very check passed
  vacuously, because the build had failed and it compared a stale file against
  itself. **Check exit codes in verification scripts.**

### The ignition panel

- **A cross-joined offset table turns a hash join into a nested loop.** The
  natural way to write a neighbourhood feature is
  `JOIN activity ON activity.cell_x = panel.cell_x + offsets.ox`. DuckDB cannot
  hash a join key that is an expression over two relations, so it falls back to
  a nested loop over the whole activity table: the panel build went from four
  seconds to **not finishing in twenty minutes**. `_register_ring` materialises
  the eight neighbour keys into a real table first, and the planner hashes it.
- **The hotspot feed covers North America, not Canada.** An activity-derived
  study area reaches to 25°N and two fifths of the cell-days were in the United
  States — guaranteed negatives, all labelled `ON` because the nearest Canadian
  station to Florida is in Ontario. `grid.province_of` is the country filter.
- **`sum()` over an unsigned integer comes back from DuckDB as HUGEINT**, which
  arrives in pandas as an `object` column of `Decimal`s, which LightGBM refuses
  outright with "pandas dtypes must be int, float or bool". Cast counts to
  `Int64` on the way out.
- **An empty DuckDB result has no record batch**, and `pyarrow` will not build
  a table from none: `ValueError: Must pass schema, or at least one
  RecordBatch`. Use `con.execute(sql).pl()`, which handles it. This happens
  whenever a trailing window matches nothing, which is a normal day.
- **The bootstrap does not need to re-sort.** Resampling rows with replacement
  is a multinomial over the originals, and average precision reads only the
  score order plus each row's weight — so drawing a row *k* times is exactly
  giving it *k* times its weight. Sorting once instead of once per draw took
  the pooled region-blocked table from half an hour to about a minute.
  `tests/test_ignition.py` pins that identity against sklearn.

### The sources

- **CIFFC has a public JSON API; do not write a scraper.**
  `api.ciffc.net/v1/sitrep?date=YYYY-MM-DD`, no credentials. The tell was that
  the rendered page issues *no* data XHR at all. Reports exist April–October
  only.
- **`agencies_sitereps`** — yes, that is how the API spells it. Do not
  "correct" the key.
- **GeoServer caps the fires layer at 10,000 features per response**, whatever
  `count` asks for. `PAGE` in `cwfis_fires.py` must equal that cap — set it
  higher and a short page reads as "end of data" and the pull silently
  truncates. This is why the first ingest returned 2,821 fires instead of
  23,165.
- **Hotspot season archives and rolling daily files have different columns.**
  The archives lack `estarea` and `bfc`. `hotspot_features` builds its SQL
  projection from whatever columns are present; keep it that way.
- **There are two decadal FWI archives and they are not interchangeable.**
  `cwfis_fwi2020s.csv` (430 MB) stops on 2025-01-21 and carries no coordinates.
  `cwfis_fwi2020sv3.0_ll.csv` (670 MB) runs to 2025-08-31, carries lat/lon
  inline, and includes provincial stations the plain file is not licensed to
  publish. Use the `_ll` one. `PLAIN_URL` is kept named in the module so this
  is not rediscovered.
- **The daily station files are not the archive in miniature.** Upper-case
  headers, **the header repeated as the first data row**, `REPDATE` as a bare
  `YYYYMMDD`, every field space-padded, and no coordinates. Parsing them with
  the archive's reader yields an empty frame and no error.
- **The station list spells two provinces differently from the fire feed** —
  `SA` for Saskatchewan, `NF` for Newfoundland — and includes US states. This
  no longer matters (agencies come from the fire feed and the provincial
  outlines) but it will bite anything new that joins on `prov`.
- **Several columns in the station files are space-padded**, so a bare
  `.cast(Float64)` on `" 60.34"` raises rather than coercing. Strip first.
- **All Canadian sources need zero credentials.** If you find yourself blocked
  on an API key, you are probably reaching for FIRMS — which is optional.

### Environment

- **Typer list options need a repeated flag**: `-y 2023 -y 2024`, not
  `-y 2023 2024`.
- **On Windows, set `PYTHONIOENCODING=utf-8`** or `rich`/`polars` box-drawing
  output crashes with a `cp1252` encode error.
- **`CalibratedClassifierCV(cv="prefit")` was removed** from recent
  scikit-learn. The project uses `IsotonicRegression` directly instead — three
  lines, and immune to that API churn.
- **`class_weight="balanced"` wrecked calibration.** It improved tree splits on
  the rare class but inflated probabilities so badly the Brier score lost to
  predicting the base rate. Current approach: fit unweighted, then isotonic-
  calibrate on a temporally held-back slice. If you re-add class weighting,
  re-check Brier and the reliability table, not just PR-AUC.
