"""One HTTP client for every source, with the manners a scraper owes a host.

Retries with exponential backoff, a real User-Agent, a per-host rate limit,
and conditional requests via ETag / Last-Modified so that re-running the
pipeline does not re-download files that have not changed.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import config

log = logging.getLogger(__name__)

_CACHE_META = config.RAW / "_http_cache.json"
_lock = threading.Lock()
_last_hit: dict[str, float] = {}


def _throttle(url: str) -> None:
    """Keep at most ~3 requests/second against any single host."""
    host = urlparse(url).netloc
    with _lock:
        gap = time.monotonic() - _last_hit.get(host, 0.0)
        if gap < config.RATE_LIMIT_SLEEP:
            time.sleep(config.RATE_LIMIT_SLEEP - gap)
        _last_hit[host] = time.monotonic()


def _load_meta() -> dict:
    if _CACHE_META.exists():
        try:
            return json.loads(_CACHE_META.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_meta(meta: dict) -> None:
    _CACHE_META.write_text(json.dumps(meta, indent=2, sort_keys=True))


def client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": config.USER_AGENT},
        timeout=config.REQUEST_TIMEOUT,
        follow_redirects=True,
    )


@retry(
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=2, max=45),
    retry=retry_if_exception_type(
        (httpx.TransportError, httpx.HTTPStatusError, httpx.ReadTimeout)
    ),
    reraise=True,
)
def _get(c: httpx.Client, url: str, headers: dict | None = None) -> httpx.Response:
    _throttle(url)
    r = c.get(url, headers=headers or {})
    # 4xx other than 429 are not worth retrying -- they will not fix themselves.
    if r.status_code == 429 or 500 <= r.status_code < 600:
        r.raise_for_status()
    return r


def fetch_to_file(url: str, dest: Path, *, conditional: bool = True) -> tuple[Path, bool]:
    """Download `url` to `dest`. Returns (path, downloaded).

    `downloaded` is False when the server answered 304 Not Modified, or when
    the file is already on disk and we have no validator to check against.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = _load_meta()
    entry = meta.get(url, {})

    headers = {}
    if conditional and dest.exists():
        if etag := entry.get("etag"):
            headers["If-None-Match"] = etag
        if lastmod := entry.get("last_modified"):
            headers["If-Modified-Since"] = lastmod
        if not headers:
            # No validator available but the bytes are already here.
            return dest, False

    with client() as c:
        r = _get(c, url, headers)

    if r.status_code == 304:
        log.debug("304 not modified: %s", url)
        return dest, False

    if r.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"{r.status_code} for {url}", request=r.request, response=r
        )

    dest.write_bytes(r.content)
    meta[url] = {
        "etag": r.headers.get("ETag"),
        "last_modified": r.headers.get("Last-Modified"),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bytes": len(r.content),
        "status": r.status_code,
    }
    _save_meta(meta)
    log.info("fetched %s (%s bytes) -> %s", url, len(r.content), dest.name)
    return dest, True


def get_text(url: str) -> str:
    with client() as c:
        r = _get(c, url)
        r.raise_for_status()
        return r.text


def get_json(url: str) -> dict:
    with client() as c:
        r = _get(c, url)
        r.raise_for_status()
        return r.json()
