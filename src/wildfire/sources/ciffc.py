"""CIFFC national situation report -- the scraping leg.

This is the only genuinely *scraped* source in the project, and deliberately
so. Everything else here is a file or an OGC service; CIFFC is a client-
rendered React app with no public JSON API (its Strapi instance serves only
CMS assets, and the app's data routes are not reachable unauthenticated).
So the honest way in is to render the page and parse the DOM.

What it gives us that no machine feed does:

  * National Preparedness Level (1-5) -- the country's aggregate judgement
    about how stretched suppression resources are. When PL is 4-5, crews and
    aircraft are rationed, which changes how a new fire is fought and
    therefore how likely it is to escalate. That is an operational covariate
    with no equivalent in any satellite or weather product.
  * Per-agency new-fire counts split by cause, and OC/BH/UC counts with
    hectares under full vs modified response.

Rendering needs Playwright:
    pip install playwright && playwright install chromium

The rest of the pipeline does not depend on this module -- if Playwright is
absent, the preparedness features are simply absent.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone

import polars as pl

from .. import config

log = logging.getLogger(__name__)

PL_RE = re.compile(r"National Preparedness Level:\s*([1-5])", re.I)
DATE_RE = re.compile(r"For:\s*([A-Z][a-z]+ \d{1,2}, \d{4})")

AGENCIES = {
    "BC": "British Columbia", "YT": "Yukon", "AB": "Alberta",
    "NT": "Northwest Territories", "SK": "Saskatchewan", "MB": "Manitoba",
    "ON": "Ontario", "QC": "Quebec", "NL": "Newfoundland and Labrador",
    "NB": "New Brunswick", "NS": "Nova Scotia", "PE": "Prince Edward Island",
    "PC": "Parks Canada",
}


class PlaywrightMissing(RuntimeError):
    pass


def render(url: str = config.CIFFC_SITREP, *, timeout_ms: int = 45_000) -> str:
    """Return the fully rendered HTML of the sitrep page."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise PlaywrightMissing(
            "CIFFC is client-rendered and needs a browser. Install with:\n"
            "  pip install playwright && playwright install chromium"
        ) from exc

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(user_agent=config.USER_AGENT)
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            # The summary text is the last thing to populate.
            page.wait_for_function(
                "() => document.body.innerText.includes('National Preparedness Level')",
                timeout=timeout_ms,
            )
            return page.content()
        finally:
            browser.close()


def parse(html: str) -> dict:
    """Pull the structured numbers out of the rendered sitrep."""
    from selectolax.parser import HTMLParser

    tree = HTMLParser(html)
    text = tree.body.text(separator="\n") if tree.body else ""

    out: dict = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "preparedness_level": None,
        "sitrep_date": None,
    }

    if m := PL_RE.search(text):
        out["preparedness_level"] = int(m.group(1))
    if m := DATE_RE.search(text):
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
    return out


def _num(s: str) -> float | None:
    s = s.replace(",", "").strip()
    if not s or s in {"-", "--"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _new_fires(tables: list[list[list[str]]]) -> list[dict]:
    """Find the 'New Wildland Fires Yesterday' table by its header shape."""
    for rows in tables:
        header = [h.lower() for h in rows[0]]
        if "natural" in header and "human" in header:
            idx = {name: i for i, name in enumerate(header)}
            recs = []
            for r in rows[1:]:
                if not r or r[0].strip().lower() == "total":
                    continue
                recs.append(
                    {
                        "agency": r[0].strip(),
                        "natural": _num(r[idx["natural"]]) if "natural" in idx else None,
                        "human": _num(r[idx["human"]]) if "human" in idx else None,
                        "total": _num(r[idx["total"]]) if "total" in idx else None,
                    }
                )
            return recs
    return []


def snapshot(*, force: bool = False) -> dict:
    """Render, parse, and append to the daily preparedness-level history.

    The sitrep page only ever shows *today*, so the history is something we
    accumulate by running this daily -- exactly the kind of series that has to
    be built rather than downloaded.
    """
    today = date.today().isoformat()
    raw_dir = config.RAW / "ciffc"
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_path = raw_dir / f"sitrep_{today}.html"

    if not html_path.exists() or force:
        html_path.write_text(render(), encoding="utf-8")

    parsed = parse(html_path.read_text(encoding="utf-8"))
    (raw_dir / f"sitrep_{today}.json").write_text(json.dumps(parsed, indent=2))

    _append_history(parsed)
    return parsed


def _append_history(parsed: dict) -> None:
    hist = config.CURATED / "ciffc_preparedness.parquet"
    row = pl.DataFrame(
        {
            "sitrep_date": [parsed.get("sitrep_date")],
            "preparedness_level": [parsed.get("preparedness_level")],
            "scraped_at": [parsed.get("scraped_at")],
        }
    )
    if hist.exists():
        prior = pl.read_parquet(hist)
        row = pl.concat([prior, row], how="vertical_relaxed").unique(
            subset=["sitrep_date"], keep="last"
        )
    row.sort("sitrep_date").write_parquet(hist)
