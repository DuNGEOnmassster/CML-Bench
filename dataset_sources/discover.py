"""Discover METADATA, never full texts or permission to reuse them.

Official bulk/catalog endpoints only. HTTP responses are size-limited, cached,
hashed, and recorded. A failed source makes the run visibly partial (exit 2).
"""

from __future__ import annotations

import argparse
import csv
import http.client
import io
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .common import digest, registry, write_json, write_jsonl

PG_CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
DRACOR = "https://dracor.org/api/v1"
ALLOWED_HOSTS = {
    "www.gutenberg.org",
    "dracor.org",
    "raw.githubusercontent.com",
    "query.wikidata.org",
}
FILM_QUERY = """SELECT ?film ?filmLabel ?basedOn ?basedOnLabel WHERE {
  { SELECT ?film ?basedOn WHERE {
      ?film wdt:P31 wd:Q11424; wdt:P144 ?basedOn .
    } ORDER BY ?film ?basedOn LIMIT 500 }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,zh". }
}"""
WIKIDATA_CATALOG = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
    {"query": FILM_QUERY, "format": "json"}
)


def check_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username
        or parsed.password
    ):
        raise ValueError("discovery URL outside the official metadata endpoints")
    allowed = (
        (
            parsed.hostname == "www.gutenberg.org"
            and parsed.path == "/cache/epub/feeds/pg_catalog.csv"
        )
        or (
            parsed.hostname == "dracor.org"
            and re.fullmatch(r"/api/v1/corpora(?:/[a-z]+)?", parsed.path)
        )
        or (parsed.hostname == "query.wikidata.org" and url == WIKIDATA_CATALOG)
        or (
            parsed.hostname == "raw.githubusercontent.com"
            and re.fullmatch(
                r"/COST-ELTeC/ELTeC-[a-z]{3}/[^/]+/ELTeC-[a-z]{3}_metadata.tsv", parsed.path
            )
        )
    )
    if not allowed:
        raise ValueError("full-text downloads are not part of discovery")


class CatalogRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Client:
    def __init__(self, cache, *, refresh=False, timeout=30):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.refresh, self.timeout = refresh, timeout
        self.receipts = []
        self.opener = urllib.request.build_opener(CatalogRedirect())
        self.last_request = 0.0

    def get(self, url):
        check_url(url)
        key = digest(url.encode())
        body_path, meta_path = self.cache / (key + ".body"), self.cache / (key + ".json")
        if not self.refresh and body_path.exists() and meta_path.exists():
            body = body_path.read_bytes()
            receipt = json.loads(meta_path.read_text())
            if receipt.get("url") != url or digest(body) != receipt.get("sha256"):
                raise ValueError("catalog cache hash mismatch; use --refresh")
        else:
            time.sleep(max(0, 0.5 - (time.monotonic() - self.last_request)))
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "CML-Bench-SourceSurvey/1.0 (+https://github.com/DuNGEOnmassster/CML-Bench)",
                    "Accept": "application/json,text/csv,text/tab-separated-values,*/*;q=0.1",
                },
            )
            with self.opener.open(req, timeout=self.timeout) as response:
                body = response.read(100_000_001)
                if len(body) > 100_000_000:
                    raise ValueError("catalog exceeds 100 MB limit")
                receipt = {
                    "url": url,
                    "sha256": digest(body),
                    "bytes": len(body),
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "etag": response.headers.get("ETag"),
                }
            self.last_request = time.monotonic()
            body_path.write_bytes(body)
            write_json(meta_path, receipt)
        self.receipts.append(receipt)
        return body.decode("utf-8-sig")


def candidate(source, key, title, kind, language, creators, url, **extra):
    return {
        "candidate_id": f"{source}:{key}",
        "source_id": source,
        "source_record_id": str(key),
        "title": title,
        "kind": kind,
        "language": language or "und",
        "creators": creators,
        "source_url": url,
        "rights_status": "unreviewed",
        "release_eligible": False,
        "work_id": None,
        "story_family_id": None,
        **extra,
    }


def gutenberg(csv_text):
    reader = csv.DictReader(io.StringIO(csv_text))
    if not {"Text#", "Title", "Subjects"} <= set(reader.fieldnames or []):
        raise ValueError("Gutenberg catalog schema changed or response is not CSV")
    for row in reader:
        if row.get("Type", "Text") != "Text":
            continue
        subjects = row.get("Subjects", "")
        # Subject-based candidate classification, NOT verified literary form.
        if re.search(r"\b(screenplays?|motion picture plays)\b", subjects, re.I):
            kind = "screenplay"
        elif re.search(r"\b(drama|plays)\b", subjects, re.I):
            kind = "stage_play"
        elif re.search(r"\b(fiction|novels|stories|romances)\b", subjects, re.I):
            kind = "prose_fiction"
        else:
            continue
        key = row["Text#"]
        yield candidate(
            "gutenberg",
            key,
            row["Title"],
            kind,
            row.get("Language", "und"),
            [x.strip() for x in row.get("Authors", "").split(";") if x.strip()],
            f"https://www.gutenberg.org/ebooks/{key}",
            subjects=subjects,
            catalog_release_date=row.get("Issued"),
            original_publication_year=None,
            reported_rights="Not determined by CSV; inspect the specific edition and license",
            rights_scope="unreviewed",
            metadata_license="CC0-1.0",
            classification_method="catalog_subject_heuristic",
        )


def dracor(corpus, data):
    for play in data["plays"]:
        key = f"{corpus['name']}/{play['name']}"
        yield candidate(
            "dracor",
            key,
            play.get("title", play["name"]),
            "stage_play",
            play.get("language", "und"),
            [a.get("name", "") for a in play.get("authors", [])],
            f"https://dracor.org/{key}",
            original_publication_year=play.get("yearPrinted"),
            normalized_work_year=play.get("yearNormalized"),
            wikidata_id=play.get("wikidataId"),
            corpus=corpus["name"],
            source_revision=data.get("commit", corpus.get("commit")),
            reported_rights=data.get("licence", corpus.get("licence", "unknown")),
            license_url=data.get("licenceUrl", corpus.get("licenceUrl")),
            rights_scope="corpus declaration; original/translation/edition require review",
            classification_method="drama_corpus",
        )


def eltec(tsv_text, language, url):
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    if not {"xmlid", "title", "author-name"} <= set(reader.fieldnames or []):
        raise ValueError("ELTeC catalog schema changed or response is not TSV")
    for row in reader:
        key = row["xmlid"]
        yield candidate(
            "eltec",
            f"{language}/{key}",
            row["title"],
            "novel",
            row.get("language", language),
            [row["author-name"]],
            f"https://github.com/COST-ELTeC/ELTeC-{language}",
            original_publication_year=row.get("first-edition"),
            author_death=row.get("author-death"),
            word_count=row.get("numwords"),
            catalog_url=url,
            reported_rights="Inspect per-text TEI and corpus license",
            rights_scope="unreviewed",
            classification_method="curated_novel_corpus",
        )


def wikidata(data):
    """Bounded film/adaptation metadata sample; never a screenplay license."""
    films = {}
    for binding in data["results"]["bindings"]:
        film = binding["film"]["value"].rsplit("/", 1)[-1]
        original = binding["basedOn"]["value"].rsplit("/", 1)[-1]
        if not re.fullmatch(r"Q\d+", film) or not re.fullmatch(r"Q\d+", original):
            raise ValueError("invalid Wikidata entity ID")
        row = films.setdefault(
            film,
            candidate(
                "wikidata",
                film,
                binding.get("filmLabel", {}).get("value", film),
                "film_metadata",
                "und",
                [],
                "https://www.wikidata.org/wiki/" + film,
                metadata_license="CC0-1.0",
                rights_status="metadata_only",
                reported_rights="No screenplay/text rights supplied",
                classification_method="P31=film, P144=based-on; bounded 500-relation sample",
                adaptations=[],
            ),
        )
        row["adaptations"].append(
            {
                "work_wikidata_id": original,
                "title": binding.get("basedOnLabel", {}).get("value", original),
            }
        )
    return sorted(films.values(), key=lambda r: r["candidate_id"])


def survey(args):
    sources = registry()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    client = Client(args.cache, refresh=args.refresh, timeout=args.timeout)
    rows, errors, corpora = [], [], []

    def attempt(label, action):
        try:
            rows.extend(action())
        except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as exc:
            errors.append({"source": label, "error": str(exc)[:500]})

    if "gutenberg" in args.sources:
        attempt("gutenberg", lambda: list(gutenberg(client.get(PG_CATALOG))))
    if "dracor" in args.sources:
        try:
            corpora = json.loads(client.get(DRACOR + "/corpora"))
            if not isinstance(corpora, list) or not all(
                isinstance(c, dict) and "name" in c for c in corpora
            ):
                corpora = []
                raise ValueError("DraCor corpus inventory schema changed")
            unknown = set(args.dracor_corpora or []) - {c["name"] for c in corpora}
            if unknown:
                errors.append(
                    {
                        "source": "dracor",
                        "error": "unknown requested corpora: " + ",".join(sorted(unknown)),
                    }
                )
        except (OSError, ValueError, http.client.HTTPException) as exc:
            errors.append({"source": "dracor", "error": str(exc)[:500]})
        for corpus in corpora:
            if args.dracor_corpora and corpus["name"] not in args.dracor_corpora:
                continue
            attempt(
                "dracor:" + corpus["name"],
                lambda c=corpus: list(
                    dracor(c, json.loads(client.get(DRACOR + "/corpora/" + c["name"])))
                ),
            )
    if "eltec" in args.sources:
        for lang in args.eltec_languages:
            if not re.fullmatch("[a-z]{3}", lang):
                raise ValueError("ELTeC language must be a three-letter code")
            url = f"https://raw.githubusercontent.com/COST-ELTeC/ELTeC-{lang}/master/ELTeC-{lang}_metadata.tsv"
            attempt(
                "eltec:" + lang,
                lambda u=url, language=lang: list(eltec(client.get(u), language, u)),
            )
    if "wikidata" in args.sources:
        attempt("wikidata", lambda: wikidata(json.loads(client.get(WIKIDATA_CATALOG))))
    original_count = len(rows)
    unique = {r["candidate_id"]: r for r in rows}
    rows = sorted(unique.values(), key=lambda r: r["candidate_id"])
    assert all(r["source_id"] in sources and r["release_eligible"] is False for r in rows)
    write_jsonl(out / "candidates.jsonl", rows)
    write_jsonl(
        out / "sample.jsonl",
        [r for source in args.sources for r in [x for x in rows if x["source_id"] == source][:5]],
    )
    report = {
        "schema_version": "1.0",
        "status": "partial" if errors else "complete",
        "scope": "metadata candidates, not cleared works or benchmark examples",
        "candidate_records": len(rows),
        "distinct_work_count": None,
        "globally_cleared_works": 0,
        "released_segments": 0,
        "by_source": dict(Counter(r["source_id"] for r in rows)),
        "by_kind": dict(Counter(r["kind"] for r in rows)),
        "by_language": dict(Counter(r["language"] for r in rows)),
        "by_dracor_corpus": dict(Counter(r["corpus"] for r in rows if "corpus" in r)),
        "duplicate_candidate_ids": original_count - len(unique),
        "requested_sources": args.sources,
        "receipts": client.receipts,
        "errors": errors,
        "candidate_manifest_sha256": digest((out / "candidates.jsonl").read_bytes()),
    }
    write_json(out / "report.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "candidate_records", "by_source", "errors")},
            ensure_ascii=False,
        )
    )
    return 2 if errors else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sources",
        nargs="+",
        choices=["gutenberg", "dracor", "eltec", "wikidata"],
        default=["gutenberg", "dracor", "eltec"],
    )
    ap.add_argument("--eltec-languages", nargs="+", default=["eng", "fra", "deu"])
    ap.add_argument("--dracor-corpora", nargs="+")
    ap.add_argument("--cache", default="dataset_sources/work/cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()
    if not 1 <= args.timeout <= 60:
        ap.error("timeout must be 1–60 seconds")
    raise SystemExit(survey(args))


if __name__ == "__main__":
    main()
