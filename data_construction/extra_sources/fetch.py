"""Polite, cached downloader for screenplay files.

- one request per host every `delay` seconds, identifying user agent, robots.txt respected;
- hosts whose published policy forbids AI-training use of their scripts are refused (POLICY_BLOCKED);
- every response is cached under cache_dir/<host>/<sha1(url)[:16]>.<ext>, indexed in cache_dir/index.jsonl
  (url -> file, sha1, bytes, status), so builds are reproducible offline and provenance is auditable.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import urllib.robotparser
from urllib.parse import quote, urlsplit, urlunsplit

USER_AGENT = "CML-Bench-dataset-research/0.1 (+https://github.com/DuNGEOnmassster/CML-Bench)"
# robots.txt + ai.txt: "AI model training is NOT permitted" (checked 2026-10-09).
POLICY_BLOCKED = {"www.scriptslug.com": "ai.txt forbids AI training use", "scriptslug.com": "ai.txt forbids AI training use",
                  "assets.scriptslug.com": "ai.txt forbids AI training use"}
LOGIN_WALLED = {"www.scribd.com", "scribd.com", "drive.google.com"}


class PolicyError(RuntimeError):
    pass


def _safe_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=quote(parts.path, safe="/%:@+,;=~!$&'()*-._"), query=parts.query))


class Fetcher:
    def __init__(self, cache_dir: str, delay: float = 2.0, timeout: int = 60, offline: bool = False):
        self.cache_dir = cache_dir
        self.delay = delay
        self.timeout = timeout
        self.offline = offline
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        os.makedirs(cache_dir, exist_ok=True)
        self.index_path = os.path.join(cache_dir, "index.jsonl")
        self.index: dict[str, dict] = {}
        if os.path.exists(self.index_path):
            with open(self.index_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.index[rec["url"]] = rec

    def _allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        host = parts.netloc.lower()
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                req = urllib.request.Request(f"{parts.scheme}://{host}/robots.txt", headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=20) as r:
                    body = r.read().decode("utf-8", "replace")
                rp.parse(body.splitlines() if "user-agent" in body.lower() else [])
            except (urllib.error.URLError, TimeoutError, ValueError):
                rp.parse([])
            self._robots[host] = rp
        rp = self._robots[host]
        return rp is None or rp.can_fetch(USER_AGENT, url)

    def path_for(self, url: str) -> str | None:
        rec = self.index.get(url)
        return os.path.join(self.cache_dir, rec["file"]) if rec and rec.get("file") else None

    def get(self, url: str) -> tuple[bytes, dict]:
        """Return (body, index record). Raises PolicyError for blocked hosts, RuntimeError on HTTP failure."""
        rec = self.index.get(url)
        if rec and rec.get("file"):
            with open(os.path.join(self.cache_dir, rec["file"]), "rb") as f:
                return f.read(), rec
        if rec and rec.get("status") not in (None, 200):
            raise RuntimeError(f"cached failure {rec['status']}: {url}")
        host = urlsplit(url).netloc.lower()
        if host in POLICY_BLOCKED:
            raise PolicyError(f"{host}: {POLICY_BLOCKED[host]}")
        if host in LOGIN_WALLED:
            raise PolicyError(f"{host}: login-walled")
        if self.offline:
            raise RuntimeError(f"offline and not cached: {url}")
        if not self._allowed(url):
            raise PolicyError(f"robots.txt disallows {url}")
        wait = self._last.get(host, 0) + self.delay - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.time()
        req = urllib.request.Request(_safe_url(url), headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body, status, ctype = r.read(), r.status, r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            self._record({"url": url, "status": exc.code})
            raise RuntimeError(f"HTTP {exc.code}: {url}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise RuntimeError(f"network error {exc}: {url}") from exc
        sha1 = hashlib.sha1(body).hexdigest()
        ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
        ext = ext if ext in ("pdf", "txt", "html", "htm", "doc", "rtf", "docx") else ("pdf" if "pdf" in ctype else "html")
        rel = os.path.join(host, f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.{ext}")
        os.makedirs(os.path.join(self.cache_dir, host), exist_ok=True)
        with open(os.path.join(self.cache_dir, rel), "wb") as f:
            f.write(body)
        rec = {"url": url, "status": status, "file": rel, "sha1": sha1, "bytes": len(body), "content_type": ctype,
               "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self._record(rec)
        return body, rec

    def _record(self, rec: dict) -> None:
        self.index[rec["url"]] = rec
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
