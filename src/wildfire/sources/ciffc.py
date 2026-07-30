"""CIFFC national situation report -- preparedness level and agency readiness.

What this gives us that no machine feed does:

  * **National Preparedness Level (1-5)** -- the country's aggregate judgement
    about how stretched suppression resources are. At PL 4-5 crews and
    aircraft are rationed, which changes how a new fire is fought and
    therefore how likely it is to escalate.
  * **Agency Preparedness Level**, plus its five components (fire danger,
    current load, anticipated 7-day load, resource levels, ability to respond
    to CIFFC requests). This is the same judgement at the scale the fire is
    actually fought at, and it joins straight onto `agency_code`.
  * **The agency's own occurrence prediction** for tomorrow, split into
    lightning and human causes -- a human-produced ignition forecast.

None of it exists in any satellite or weather product. All of it is a human
judgement recorded daily, which is exactly the sort of covariate a purely
machine-fed pipeline misses.

## This was written as a scraper. It is not one any more.

The original premise was that CIFFC is a client-rendered React app with no
public API, so the honest way in was to render the DOM. The first half is
true -- `https://ciffc.net/situation/` is a 4 KB shell with an empty `#root`.
The second half was wrong. The bundle calls `api.ciffc.net/v1/sitrep`, and
that endpoint:

  * takes `?date=YYYY-MM-DD` and serves any past report;
  * needs no credentials (the logged-out code path sends no `Authorization`
    header, and neither do we);
  * carries far more than the rendered page shows, including per-agency
    preparedness components that never appear in the HTML.

That matters beyond convenience. A scraped sitrep page only ever shows
*today*, so the series would have had to be accumulated forward and could
never have been a feature for a 2023-25 backtest. Because the API is dated,
the whole history back to 2019 is retrievable in one pass and preparedness
becomes a *trainable* feature rather than a going-forward one.

`render()` and `parse()` below are kept as a fallback for the day the API
changes shape, and because the rendered page remains the only human-readable
form of the report. They are no longer the primary path.

## Point-in-time

Every sitrep, national and per-agency, carries `system_edit_timestamp` -- the
instant it was published. That is stored as `published_at` and is the *only*
column features may filter on. Filtering on `sitrep_date` instead would leak:
the report for the 15th is published late on the 15th, so a fire that started
that morning must not see it.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone

import polars as pl

from .. import config
from ..http import get_json

log = logging.getLogger(__name__)

ARCHIVE_URL = f"{config.CIFFC_API}/sitrep/archive"
SITREP_URL = f"{config.CIFFC_API}/sitrep"

RAW_DIR = config.RAW / "ciffc"

AGENCIES = {
    "BC": "British Columbia", "YT": "Yukon", "AB": "Alberta",
    "NT": "Northwest Territories", "SK": "Saskatchewan", "MB": "Manitoba",
    "ON": "Ontario", "QC": "Quebec", "NL": "Newfoundland and Labrador",
    "NB": "New Brunswick", "NS": "Nova Scotia", "PE": "Prince Edward Island",
    "PC": "Parks Canada",
}

# Per-agency fields worth carrying. The sitrep has ~45; these are the ones
# that are a judgement about readiness rather than a restatement of fire
# counts we already get from the CWFIF feed.
AGENCY_FIELDS = {
    "field_preparedness": "agency_preparedness",
    "field_prep_hazard": "prep_hazard",
    "field_prep_current_load": "prep_current_load",
    "field_prep_expected_load": "prep_expected_load",
    "field_prep_resource_levels": "prep_resource_levels",
    "field_prep_resource_availability": "prep_resource_availability",
    "field_occurrence_prediction_lightning": "occurrence_pred_lightning",
    "field_occurrence_prediction_human": "occurrence_pred_human",
}


class PlaywrightMissing(RuntimeError):
    pass


# --- the API path ----------------------------------------------------------


def _ts(value) -> datetime | None:
    """Parse an ISO timestamp, normalising to naive UTC."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _num(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def archive_index(*, force: bool = False) -> pl.DataFrame:
    """Every published sitrep date and its national preparedness level.

    One request for the entire 2019-present series. The listing the site uses
    to build its archive calendar already carries the national PL, so the
    national series costs exactly one round trip -- only the per-agency detail
    needs a request per day.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "sitrep_archive.json"
    if force or not dest.exists():
        dest.write_text(json.dumps(get_json(ARCHIVE_URL)), encoding="utf-8")
    payload = json.loads(dest.read_text(encoding="utf-8"))

    rows = []
    for months in payload.values():
        for days in months.values():
            for rec in days.values():
                data = (rec or {}).get("data") or {}
                raw_date = str(data.get("field_date") or "")[:10]
                if not raw_date:
                    continue
                rows.append(
                    {
                        "sitrep_date": date.fromisoformat(raw_date),
                        "national_preparedness_level": data.get("field_preparedness_level"),
                    }
                )
    out = pl.DataFrame(rows).unique(subset=["sitrep_date"]).sort("sitrep_date")
    log.info("sitrep archive: %s published reports, %s -> %s", out.height,
             out["sitrep_date"].min(), out["sitrep_date"].max())
    return out


def fetch_day(day: date, *, force: bool = False) -> dict:
    """One day's full sitrep. Raw response lands verbatim, as everywhere else."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"sitrep_{day.isoformat()}.json"
    if force or not dest.exists():
        payload = get_json(f"{SITREP_URL}?date={day.isoformat()}")
        dest.write_text(json.dumps(payload), encoding="utf-8")
    return json.loads(dest.read_text(encoding="utf-8"))


def agency_rows(payload: dict) -> list[dict]:
    """Flatten one day's payload to one row per agency."""
    if not payload or not isinstance(payload, dict):
        return []

    raw_date = str(payload.get("field_date") or "")[:10]
    if not raw_date:
        return []
    sitrep_date = date.fromisoformat(raw_date)
    national_published = _ts(payload.get("system_edit_timestamp"))
    national_pl = payload.get("field_preparedness_level")

    rows = []
    for code, block in (payload.get("agencies_sitereps") or {}).items():
        sitrep = (block or {}).get("sitrep") or {}
        # Prefer the agency's own publication time; fall back to the national
        # one so a row is never admitted without a defensible as-of instant.
        published = _ts(sitrep.get("system_edit_timestamp")) or national_published
        if published is None:
            continue
        row = {
            "sitrep_date": sitrep_date,
            "agency_code": str(code).upper(),
            "published_at": published,
            "national_published_at": national_published,
            "national_preparedness_level": national_pl,
        }
        for src, dst in AGENCY_FIELDS.items():
            row[dst] = _num(sitrep.get(src))
        rows.append(row)
    return rows


def load_history(
    years: list[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Backfill the per-agency preparedness series and write it to curated.

    One request per published sitrep date. Reports only exist for the fire
    season (April-October), so a year is ~140 requests, not 365.
    """
    index = archive_index(force=force)
    if years:
        index = index.filter(pl.col("sitrep_date").dt.year().is_in(years))
    if index.is_empty():
        raise RuntimeError(f"no published sitreps for years {years}")

    days = index["sitrep_date"].to_list()
    log.info("fetching %s daily sitreps (cached responses are reused)", len(days))

    rows: list[dict] = []
    for i, day in enumerate(days, 1):
        try:
            rows.extend(agency_rows(fetch_day(day, force=force)))
        except Exception as exc:  # noqa: BLE001
            # A missing day is a gap in the series, not a reason to lose the
            # rest of it. Features join as-of, so gaps degrade gracefully.
            log.warning("sitrep %s failed: %s", day, exc)
        if i % 50 == 0:
            log.info("  %s/%s days", i, len(days))

    if not rows:
        raise RuntimeError("no agency sitrep rows parsed")

    df = (
        pl.DataFrame(rows)
        .unique(subset=["sitrep_date", "agency_code"], keep="last")
        .sort(["published_at", "agency_code"])
    )

    dest = config.CURATED / "ciffc_sitreps.parquet"
    df.write_parquet(dest)
    log.info("ciffc sitreps: %s rows, %s agencies, %s -> %s",
             df.height, df["agency_code"].n_unique(),
             df["sitrep_date"].min(), df["sitrep_date"].max())
    return df


def preparedness_coverage(df: pl.DataFrame) -> pl.DataFrame:
    """Per-season summary, for `diagnose`-style sanity checking."""
    return (
        df.group_by(pl.col("sitrep_date").dt.year().alias("year"))
        .agg(
            n_days=pl.col("sitrep_date").n_unique(),
            n_rows=pl.len(),
            mean_national_pl=pl.col("national_preparedness_level").mean().round(2),
            max_national_pl=pl.col("national_preparedness_level").max(),
            mean_agency_pl=pl.col("agency_preparedness").mean().round(2),
        )
        .sort("year")
    )


# --- the rendered-page fallback --------------------------------------------
#
# Retained deliberately. If `api.ciffc.net` changes shape, this still works,
# and the rendered report is the only human-readable form. It cannot backfill
# -- the page only ever shows today -- which is precisely why the API is the
# primary path.


def render(url: str = config.CIFFC_SITREP, *, timeout_ms: int = 45_000) -> str:
    """Return the fully rendered HTML of the sitrep page."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise PlaywrightMissing(
            "The rendered-page fallback needs a browser. Install with:\n"
            "  pip install playwright && playwright install chromium\n"
            "The API path (`wildfire ingest-ciffc`) needs neither."
        ) from exc

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(user_agent=config.USER_AGENT)
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            page.wait_for_function(
                "() => document.body.innerText.includes('National Preparedness Level')",
                timeout=timeout_ms,
            )
            return page.content()
        finally:
            browser.close()


def parse(html: str) -> dict:
    """Pull the structured numbers out of the rendered sitrep.

    Verified against the live DOM on 2026-07-30: the page really does use
    `<table>` elements (20 of them), the preparedness line really does read
    "National Preparedness Level: N", and the first table with both `natural`
    and `human` headers really is New Wildland Fires Yesterday.
    """
    import re

    from selectolax.parser import HTMLParser

    pl_re = re.compile(r"National Preparedness Level:\s*([1-5])", re.I)
    date_re = re.compile(r"For:\s*([A-Z][a-z]+ \d{1,2}, \d{4})")

    tree = HTMLParser(html)
    text = tree.body.text(separator="\n") if tree.body else ""

    out: dict = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "preparedness_level": None,
        "sitrep_date": None,
    }
    if m := pl_re.search(text):
        out["preparedness_level"] = int(m.group(1))
    if m := date_re.search(text):
        try:
            out["sitrep_date"] = datetime.strptime(m.group(1), "%B %d, %Y").date().isoformat()
        except ValueError:
            pass

    tables = []
    for node in tree.css("table"):
        rows = []
        for tr in node.css("tr"):
            cells = [c.text(strip=True) for c in tr.css("th, td")]
            if any(cells):
                rows.append(cells)
        if len(rows) > 1:
            tables.append(rows)
    out["tables"] = tables
    out["new_fires_by_agency"] = _new_fires(tables)
    out["agency_preparedness"] = _agency_apl(tables)
    return out


def _cell_num(s: str) -> float | None:
    s = s.replace(",", "").strip()
    if not s or s in {"-", "--"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _new_fires(tables: list[list[list[str]]]) -> list[dict]:
    """Find the 'New Wildland Fires Yesterday' table by its header shape."""
    name_to_code = {v: k for k, v in AGENCIES.items()}
    for rows in tables:
        header = [h.lower() for h in rows[0]]
        if "natural" in header and "human" in header:
            idx = {name: i for i, name in enumerate(header)}
            recs = []
            for r in rows[1:]:
                if not r or r[0].strip().lower() == "total":
                    continue
                agency = r[0].strip()
                recs.append(
                    {
                        "agency": agency,
                        # The rendered table spells agencies out in full while
                        # everything else in the project keys on the code.
                        "agency_code": name_to_code.get(agency, agency),
                        "natural": _cell_num(r[idx["natural"]]) if "natural" in idx else None,
                        "human": _cell_num(r[idx["human"]]) if "human" in idx else None,
                        "total": _cell_num(r[idx["total"]]) if "total" in idx else None,
                    }
                )
            return recs
    return []


def _agency_apl(tables: list[list[list[str]]]) -> list[dict]:
    """The per-agency Preparedness Levels table (`Agency | APL`)."""
    name_to_code = {v: k for k, v in AGENCIES.items()}
    for rows in tables:
        header = [h.lower() for h in rows[0]]
        if header[:2] == ["agency", "apl"]:
            out = []
            for r in rows[1:]:
                if len(r) < 2 or r[0].strip().lower() == "total":
                    continue
                agency = r[0].strip()
                out.append(
                    {
                        "agency_code": name_to_code.get(agency, agency),
                        "agency_preparedness": _cell_num(r[1]),
                    }
                )
            return out
    return []


def snapshot(*, force: bool = False) -> dict:
    """Render and parse today's report. Kept for the fallback path."""
    today = date.today().isoformat()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    html_path = RAW_DIR / f"sitrep_{today}.html"

    if not html_path.exists() or force:
        html_path.write_text(render(), encoding="utf-8")

    parsed = parse(html_path.read_text(encoding="utf-8"))
    (RAW_DIR / f"sitrep_rendered_{today}.json").write_text(json.dumps(parsed, indent=2))
    return parsed
