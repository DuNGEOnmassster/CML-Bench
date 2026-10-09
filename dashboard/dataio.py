"""Data access for the CML dataset dashboard.

Reads a private Hugging Face dataset repo (DATASET_REPO + HF_TOKEN) or, for development, a local folder
with the same layout (DATA_DIR). Layout: item files (*.jsonl, *.jsonl.gz, *.parquet) anywhere in the repo,
plus optional build_status.json / info.json / stats.json / contract_report.json next to them. Files under
`pilot/` form the "pilot" release, everything else the "main" release.

A background thread polls the repo head; only files whose blob changed are re-downloaded and re-parsed,
and per-item contract checks are cached by (item_id, content sha1, abstract hash), so a new batch only
costs the checks of its own items.
"""
from __future__ import annotations

import concurrent.futures as cf
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import checks

log = logging.getLogger("cml-dashboard")

ITEM_SUFFIXES = (".jsonl", ".jsonl.gz", ".parquet")
AUX_FILES = ("build_status.json", "info.json", "stats.json", "contract_report.json", "abstract_checks_summary.json")
AUX_MAX_BYTES = 50_000_000
ITEM_COLUMNS = ("c01", "c02", "c03", "c04", "c07", "c08", "c09", "c10", "c11", "c12", "c13", "c14", "c15", "c16",
                "c18", "c19", "c20", "c21", "c22", "c23", "c24")
CHECK_FIELDS = ("c07", "c08", "c11", "c12", "c13", "c14", "c16", "c19", "c20", "c21", "c22", "c23", "c12b", "c12c", "c13b",
                "sha_ok", "junk", "dup_scene", "artefacts_1k", "garble_rate", "bad_char_ratio", "max_element_chars",
                "heading_ratio", "num_speakers", "dialogue_ratio", "ascii_content", "gt_overlap", "backslashes", "contd",
                "quote_inner_1k", "orphan_lines", "speaker_splits", "abs_words", "paragraphs", "target_lo", "target_hi",
                "target_center", "hard", "soft", "top1", "top3", "grounding", "thirds", "ungrounded", "check_error")
INLINE_CHECK_LIMIT = 150
PUBLISH_EVERY_S = 20


@dataclass(frozen=True)
class RepoEntry:
    path: str
    blob_id: str
    size: int


def release_of(path: str) -> str:
    return "pilot" if path.split("/")[0] == "pilot" else "main"


def is_subset_file(path: str) -> bool:
    """An id list for the eval_safe subset (e.g. eval_safe_ids.txt, splits/eval_safe.jsonl), not extra items."""
    name = path.rsplit("/", 1)[-1]
    return "eval_safe" in name and name.endswith((".jsonl", ".txt", ".json"))


def is_item_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.endswith(ITEM_SUFFIXES) and not name.startswith((".", "abstract_checks")) and not is_subset_file(path)


def read_subset_ids(local_path: str) -> set[str]:
    with open(local_path, encoding="utf-8") as f:
        text = f.read()
    if local_path.endswith(".txt"):
        return {line.strip() for line in text.splitlines() if line.strip()}
    if local_path.endswith(".json"):
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("item_ids") or data.get("items") or data.get("eval_safe") or []
        return {x if isinstance(x, str) else str(x.get("item_id")) for x in data}
    ids = set()
    for line in text.splitlines():
        if line.strip():
            rec = json.loads(line)
            ids.add(rec if isinstance(rec, str) else str(rec.get("item_id")))
    return ids


# --- sources -----------------------------------------------------------------------------------


class LocalSource:
    kind = "local"

    def __init__(self, root: str):
        self.root = Path(root)
        self.label = str(self.root)

    def _walk(self) -> list[tuple[str, os.stat_result]]:
        if not self.root.is_dir():
            return []
        out = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(self.root).parts):
                out.append((p.relative_to(self.root).as_posix(), p.stat()))
        return out

    def head(self) -> tuple[str | None, datetime | None]:
        files = self._walk()
        if not self.root.is_dir():
            return None, None
        sig = hashlib.sha1(json.dumps([(p, s.st_size, s.st_mtime_ns) for p, s in files]).encode()).hexdigest()
        latest = max((s.st_mtime for _, s in files), default=None)
        return sig, (datetime.fromtimestamp(latest, tz=timezone.utc) if latest else None)

    def list_files(self, rev: str) -> list[RepoEntry]:
        return [RepoEntry(p, f"{s.st_size}-{s.st_mtime_ns}", s.st_size) for p, s in self._walk()]

    def fetch(self, entry: RepoEntry, rev: str) -> str:
        return str(self.root / entry.path)

    def commits(self, n: int = 12) -> list[dict]:
        return []

    def prune(self, keep_rev: str) -> None:
        pass


class HFSource:
    kind = "hf"

    def __init__(self, repo_id: str, token: str | None):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)
        self.label = repo_id

    def head(self) -> tuple[str | None, datetime | None]:
        from huggingface_hub.utils import RepositoryNotFoundError

        try:
            info = self.api.dataset_info(self.repo_id)
        except RepositoryNotFoundError:
            return None, None
        return info.sha, info.last_modified

    def list_files(self, rev: str) -> list[RepoEntry]:
        from huggingface_hub.hf_api import RepoFile

        out = []
        for e in self.api.list_repo_tree(self.repo_id, repo_type="dataset", revision=rev, recursive=True):
            if isinstance(e, RepoFile):
                out.append(RepoEntry(e.path, (e.lfs.sha256 if e.lfs else e.blob_id), e.size))
        return out

    def fetch(self, entry: RepoEntry, rev: str) -> str:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(self.repo_id, entry.path, repo_type="dataset", revision=rev, token=self.token)
        return os.path.realpath(p)

    def commits(self, n: int = 12) -> list[dict]:
        try:
            cs = self.api.list_repo_commits(self.repo_id, repo_type="dataset")
        except Exception:
            return []
        return [{"time": c.created_at, "title": c.title, "sha": c.commit_id} for c in cs[:n]]

    def prune(self, keep_rev: str) -> None:
        """Drop cached revisions other than the live one; blobs shared with it are kept."""
        from huggingface_hub import scan_cache_dir

        try:
            cache = scan_cache_dir()
            old = [r.commit_hash for repo in cache.repos if repo.repo_id == self.repo_id and repo.repo_type == "dataset"
                   for r in repo.revisions if r.commit_hash != keep_rev]
            if old:
                cache.delete_revisions(*old).execute()
        except Exception as exc:
            log.warning("cache prune failed: %s", exc)


# --- records -----------------------------------------------------------------------------------

_TITLE_YEAR = re.compile(r"^(.*)_(\d{4})$")


def title_year(movie_name: str) -> tuple[str, int | None]:
    m = _TITLE_YEAR.match(movie_name or "")
    if m:
        return m.group(1).replace("_", " ").strip(), int(m.group(2))
    return (movie_name or "").replace("_", " ").strip(), None


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def normalize(rec: dict, idx: int, release: str, gt_ids: frozenset, gt_titles: frozenset) -> tuple[dict, str]:
    content = checks.record_content(rec)
    summary = checks.record_abstract(rec).strip()
    movie_name = str(rec.get("movie_name") or rec.get("title") or "")
    imdb_id = str(rec.get("imdb_id") or "")
    item_id = str(rec.get("item_id") or f"{imdb_id or 'item'}-{idx:05d}")
    title, name_year = title_year(movie_name)
    year = _num(rec.get("year")) or name_year
    tokens = _num(rec.get("script_tokens")) or _num(rec.get("content_tokens"))
    if tokens is None:
        from cml_format import count_tokens

        tokens = count_tokens(content)
    tag_counts = rec.get("tag_counts") if isinstance(rec.get("tag_counts"), dict) else {}
    scenes = _num(rec.get("num_scenes")) or tag_counts.get("<scene>") or content.count("<scene>")
    dialogue = tag_counts.get("<dialogue>") if tag_counts else content.count("<dialogue>")
    genres = rec.get("genres") if isinstance(rec.get("genres"), list) else []
    sha = rec.get("content_sha1") or None
    related, rel_type, rel_movie = checks.gt_relation(rec)
    eval_safe = rec.get("eval_safe")
    if not isinstance(eval_safe, bool):
        subsets = rec.get("subsets") or rec.get("splits")
        eval_safe = ("eval_safe" in subsets) if isinstance(subsets, list) else None
    row = {
        "item_id": item_id,
        "release": release,
        "movie_name": movie_name,
        "imdb_id": imdb_id,
        "film_key": imdb_id or movie_name,
        "title": title,
        "year": year,
        "film": f"{title} ({year})" if year else title,
        "source": str(rec.get("source_dataset") or rec.get("source") or "unknown"),
        "split": rec.get("source_split"),
        "script_tokens": int(tokens),
        "num_scenes": int(scenes),
        "dialogue_turns": int(dialogue or 0),
        "has_abstract": bool(summary),
        "summary": summary,
        "summary_words": len(summary.split()) if summary else 0,
        "summary_tokens": _num(rec.get("summary_tokens")) if summary else None,
        "imdb_rating": _num(rec.get("imdb_rating")),
        "imdb_votes": _num(rec.get("imdb_votes")),
        "genres": tuple(str(g) for g in genres),
        "prompt_version": rec.get("abstract_prompt_version"),
        "author": rec.get("abstract_author"),
        "normalization": rec.get("content_normalization"),
        "scene_start": _num(rec.get("scene_start")),
        "scene_end": _num(rec.get("scene_end")),
        "segment_index": _num(rec.get("segment_index")),
        "relative_position": _num(rec.get("relative_position")),
        "content_sha1": sha,
        "source_url": rec.get("source_url"),
        "imdb_url": rec.get("imdb_url"),
        "content_chars": len(content),
        "gt_related": related,
        "gt_rel_type": rel_type,
        "gt_rel_movie": rel_movie,
        "eval_safe": eval_safe,
        **checks.metadata_checks(rec, gt_ids, gt_titles),
    }
    abs_hash = hashlib.sha1(summary.encode()).hexdigest()[:12]
    row["check_key"] = f"{item_id}|{sha or hashlib.sha1(content.encode()).hexdigest()}|{abs_hash}"
    return row, content


@dataclass
class ParsedFile:
    rows: list[dict]
    path: str | None  # set when rows carry byte offsets into this jsonl file
    inline: dict[str, str] = field(default_factory=dict)  # check_key -> content for gz/parquet files


def parse_item_file(local_path: str, rel_path: str, gt_ids: frozenset, gt_titles: frozenset) -> ParsedFile:
    release = release_of(rel_path)
    rows: list[dict] = []
    if local_path.endswith(".jsonl"):
        with open(local_path, "rb") as fh:
            idx = 0
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                rec = json.loads(line)
                if not isinstance(rec, dict) or not checks.record_content(rec):
                    if idx == 0:
                        return ParsedFile([], None)  # not an item file (e.g. a checks log)
                    continue
                row, _ = normalize(rec, idx, release, gt_ids, gt_titles)
                row["file"], row["ref"], row["path"] = local_path, offset, rel_path
                rows.append(row)
                idx += 1
        return ParsedFile(rows, local_path)

    if local_path.endswith(".gz"):
        with gzip.open(local_path, "rt", encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
    else:
        recs = pd.read_parquet(local_path).to_dict("records")
    parsed = ParsedFile([], None)
    for idx, rec in enumerate(r for r in recs if isinstance(r, dict) and checks.record_content(r)):
        row, content = normalize(rec, idx, release, gt_ids, gt_titles)
        row["file"], row["ref"], row["path"] = None, None, rel_path
        parsed.rows.append(row)
        parsed.inline[row["check_key"]] = content
    return parsed


def read_item(row: dict, inline: dict[str, str]) -> tuple[str, str]:
    if row.get("file"):
        with open(row["file"], "rb") as fh:
            fh.seek(int(row["ref"]))
            rec = json.loads(fh.readline())
        return checks.record_content(rec), checks.record_abstract(rec)
    return inline.get(row["check_key"], ""), row.get("summary", "")


# --- frames ------------------------------------------------------------------------------------


def build_frame(rows: list[dict], check_cache: dict[str, dict], subset_ids: set[str] | None = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["item_id", "film_key", "source", "has_abstract", "summary", "script_tokens", "num_scenes"])
    df = pd.DataFrame(rows)
    chk = pd.DataFrame([check_cache.get(k, {}) for k in df["check_key"]], index=df.index)
    for c in CHECK_FIELDS:
        df[c] = chk[c] if c in chk else None
    df = df.astype({c: object for c in CHECK_FIELDS})

    # eval_safe: explicit field > subset id file > derived from gt_related (when the schema has it)
    df["eval_safe_basis"] = None
    if df["eval_safe"].notna().any():
        df["eval_safe_basis"] = "field"
    elif subset_ids:
        df["eval_safe"] = df["item_id"].isin(subset_ids)
        df["eval_safe_basis"] = "id list"
    elif df["gt_related"].notna().any():
        df["eval_safe"] = df["gt_related"].map(lambda r: None if r is None else not r)
        df["eval_safe_basis"] = "derived from gt_related"

    dup_id = df["item_id"].duplicated(keep=False)
    dup_sha = df["content_sha1"].notna() & df["content_sha1"].duplicated(keep=False)
    sha_ok = df["sha_ok"]
    df["c04"] = [None if s is None else (bool(s) and not a and not b) for s, a, b in zip(sha_ok, dup_id, dup_sha)]
    df["c09"] = df["script_tokens"].between(2000, 10000)
    df["c10"] = df["num_scenes"].between(12, 24)

    c18 = pd.Series(True, index=df.index)
    ranged = df[df["scene_start"].notna() & df["scene_end"].notna()]
    for _, g in ranged.groupby("film_key"):
        if len(g) < 2:
            continue
        g = g.sort_values("scene_start")
        prev_end = g["scene_end"].shift()
        bad = g.index[(g["scene_start"] <= prev_end).fillna(False).to_numpy()]
        for i in bad:
            c18.loc[i] = False
            pos = g.index.get_loc(i)
            c18.loc[g.index[pos - 1]] = False
    df["c18"] = c18.where(df["scene_start"].notna(), None)

    first8 = df["summary"].map(lambda s: " ".join(s.split()[:8]).lower() if s else None)
    dup8 = first8.notna() & first8.duplicated(keep=False)
    df["c24"] = [None if f is None else not d for f, d in zip(first8, dup8)]

    vals = df[list(ITEM_COLUMNS)]
    df["checked"] = df["c07"].notna()
    df["n_fail"] = (vals == False).sum(axis=1)  # noqa: E712
    df["failed"] = vals.apply(lambda r: ", ".join(c.upper() for c, v in r.items() if v is False), axis=1)
    proposed = df[list(checks.PROPOSED_COLUMNS)]
    df["failed_v2"] = proposed.apply(lambda r: ", ".join(checks.COLUMN_CID[c] for c, v in r.items() if v is False), axis=1)
    return df


# --- GT reference ------------------------------------------------------------------------------


@dataclass
class GTData:
    df: pd.DataFrame
    ids: frozenset
    titles: frozenset
    shingles: set


def load_gt(repo_id: str = "songdj/CML-Bench", token: str | None = None, local_dir: str | None = None) -> GTData | None:
    try:
        if local_dir:
            gt_path, info_path = os.path.join(local_dir, "gt_100.json"), os.path.join(local_dir, "gt_100_info.json")
        else:
            from huggingface_hub import hf_hub_download

            gt_path = hf_hub_download(repo_id, "ground_truth/gt_100.json", repo_type="dataset", token=token)
            info_path = hf_hub_download(repo_id, "gt_100_info.json", repo_type="dataset", token=token)
        with open(gt_path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        with open(info_path, encoding="utf-8") as f:
            info = {x["imdb_id"]: x for x in json.load(f)["individual_results"]}
    except Exception as exc:
        log.warning("GT reference unavailable: %s", exc)
        return None

    rows, contents = [], []
    for i, rec in enumerate(recs):
        meta = info.get(rec["imdb_id"], {})
        merged = dict(rec)
        merged["script_segment"] = rec["script_segment"].strip()  # GT wraps every segment in newlines
        merged.update({k: meta[k] for k in ("script_tokens", "summary_tokens", "tag_counts", "imdb_rating", "genres") if k in meta})
        merged["item_id"] = f"gt-{rec['imdb_id']}"
        merged["source_dataset"] = "CML-Bench GT"
        row, content = normalize(merged, i, "gt", frozenset(), frozenset())
        row.update({"c03": None, "c15": None, "file": None, "ref": None, "path": "ground_truth/gt_100.json"})
        res = checks.item_checks(content, row["summary"], row["script_tokens"], None, None)
        res["sha_ok"] = None
        rows.append(row)
        contents.append(content)
        row["_checks"] = res
    cache = {r["check_key"]: r.pop("_checks") for r in rows}
    df = build_frame(rows, cache)
    df["c04"] = None
    df["c18"] = None
    from build_segments import norm_title

    return GTData(df=df, ids=frozenset(r["imdb_id"] for r in recs), titles=frozenset(norm_title(r["movie_name"]) for r in recs),
                  shingles=checks.gt_shingle_set(contents))


# --- store -------------------------------------------------------------------------------------


@dataclass
class ReleaseView:
    name: str
    df: pd.DataFrame
    aux: dict
    verdicts: dict


@dataclass
class Snapshot:
    version: int = 0
    status: str = "loading"  # loading | ok | empty | missing | error
    message: str = "Loading dataset..."
    rev: str | None = None
    updated_at: datetime | None = None
    loaded_at: datetime | None = None
    releases: dict[str, ReleaseView] = field(default_factory=dict)
    commits: list[dict] = field(default_factory=list)
    checks_total: int = 0
    checks_done: int = 0


class DataStore:
    def __init__(self, source, gt: GTData | None, refresh_seconds: int = 300, target_items: int | None = None, workers: int = 2):
        self.source = source
        self.gt = gt
        self.refresh_seconds = refresh_seconds
        self.target_items = target_items
        self.workers = max(1, workers)
        self.snapshot = Snapshot()
        self._rev: str | None = None
        self._updated: datetime | None = None
        self._commits: list[dict] = []
        self._files: dict[tuple[str, str], ParsedFile] = {}
        self._checks: dict[str, dict] = {}
        self._aux: dict[str, dict] = {}
        self._subsets: dict[str, set[str]] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._version = 0

    @classmethod
    def from_env(cls) -> "DataStore":
        token = os.environ.get("HF_TOKEN") or None
        data_dir = os.environ.get("DATA_DIR")
        if data_dir:
            source = LocalSource(data_dir)
        else:
            repo = os.environ.get("DATASET_REPO") or f"{os.environ.get('HF_ACCOUNT', '')}/CML-Dataset-Expanded"
            source = HFSource(repo, token)
        gt = load_gt(os.environ.get("GT_REPO", "songdj/CML-Bench"), token, os.environ.get("GT_DIR"))
        target = int(os.environ["TARGET_ITEMS"]) if os.environ.get("TARGET_ITEMS") else None
        return cls(source, gt, int(os.environ.get("REFRESH_SECONDS", "300")), target, int(os.environ.get("CHECK_WORKERS", "2")))

    # public API

    def start(self) -> None:
        threading.Thread(target=self._loop, name="dataset-refresh", daemon=True).start()

    def request_refresh(self) -> None:
        self._wake.set()

    def view(self, release: str) -> ReleaseView | None:
        return self.snapshot.releases.get(release)

    def item(self, release: str, item_id: str) -> tuple[dict | None, str, str]:
        rv = self.view(release)
        if rv is None or rv.df.empty:
            return None, "", ""
        hit = rv.df[rv.df["item_id"] == item_id]
        if hit.empty:
            return None, "", ""
        row = hit.iloc[0].to_dict()
        inline = {}
        for pf in list(self._files.values()):
            if row["check_key"] in pf.inline:
                inline = pf.inline
                break
        try:
            content, summary = read_item(row, inline)
        except OSError:
            content, summary = "", row.get("summary", "")
        return row, content, summary

    # refresh loop

    def _loop(self) -> None:
        while True:
            try:
                self.refresh()
            except Exception as exc:
                log.error("refresh failed: %s\n%s", exc, traceback.format_exc())
                self._publish(status="error", message=f"Refresh failed: {type(exc).__name__}: {exc}")
            self._wake.wait(self.refresh_seconds)
            self._wake.clear()

    def refresh(self, force: bool = False) -> bool:
        with self._lock:
            rev, updated = self.source.head()
            if rev is None:
                self._rev = None
                self._publish(status="missing", message=f"Dataset {self.source.label} does not exist yet (or the token cannot read it).")
                return False
            if rev == self._rev and not force:
                return False
            entries = self.source.list_files(rev)
            live = set()
            for e in entries:
                if not is_item_file(e.path):
                    continue
                key = (e.path, e.blob_id)
                live.add(key)
                if key not in self._files:
                    local = self.source.fetch(e, rev)
                    self._files[key] = parse_item_file(local, e.path, self.gt.ids if self.gt else frozenset(),
                                                       self.gt.titles if self.gt else frozenset())
            for key in list(self._files):
                if key not in live:
                    del self._files[key]
            subsets: dict[str, set[str]] = {}
            for e in entries:
                if is_subset_file(e.path):
                    try:
                        subsets.setdefault(release_of(e.path), set()).update(read_subset_ids(self.source.fetch(e, rev)))
                    except (OSError, ValueError, AttributeError) as exc:
                        log.warning("could not read subset file %s: %s", e.path, exc)
            self._subsets = subsets
            aux: dict[str, dict] = {}
            for e in sorted(entries, key=lambda x: x.path.count("/")):
                name = e.path.rsplit("/", 1)[-1]
                if name in AUX_FILES and e.size <= AUX_MAX_BYTES:
                    slot = aux.setdefault(release_of(e.path), {})
                    if name not in slot:
                        try:
                            with open(self.source.fetch(e, rev), encoding="utf-8") as f:
                                slot[name] = json.load(f)
                        except (OSError, json.JSONDecodeError) as exc:
                            log.warning("could not read %s: %s", e.path, exc)
            self._aux = aux
            self._rev, self._updated = rev, updated
            self._commits = self.source.commits()
            self._publish()
            self._run_checks()
            self._publish()
            self.source.prune(rev)
            return True

    def _rows(self) -> list[dict]:
        return [r for pf in self._files.values() for r in pf.rows]

    def _publish(self, status: str | None = None, message: str | None = None) -> None:
        rows = self._rows()
        releases = {}
        for name in ("main", "pilot"):
            rel_rows = [r for r in rows if r["release"] == name]
            if not rel_rows:
                continue
            df = build_frame(rel_rows, self._checks, self._subsets.get(name))
            aux = self._aux.get(name, {})
            releases[name] = ReleaseView(name, df, aux, checks.release_verdicts(df, aux.get("info.json")))
        total = len(rows)
        done = sum(1 for r in rows if r["check_key"] in self._checks)
        if status is None:
            status = "ok" if releases else ("empty" if self._rev else "missing")
            message = "" if releases else "The dataset repo exists but has no item files yet."
        self._version += 1
        prev = self.snapshot
        self.snapshot = Snapshot(
            version=self._version,
            status=status,
            message=message or "",
            rev=self._rev if status != "error" else prev.rev,
            updated_at=self._updated,
            loaded_at=datetime.now(timezone.utc),
            releases=releases if (releases or status != "error") else prev.releases,
            commits=self._commits,
            checks_total=total,
            checks_done=done,
        )

    def _run_checks(self) -> None:
        todo: dict[str | None, list] = {}
        seen = set()
        for pf in self._files.values():
            for r in pf.rows:
                k = r["check_key"]
                if k in self._checks or k in seen:
                    continue
                seen.add(k)
                ref = (pf.inline[k], r["summary"]) if pf.path is None else r["ref"]
                todo.setdefault(pf.path, []).append((k, ref, r["script_tokens"], r["content_sha1"]))
        n = sum(len(v) for v in todo.values())
        if not n:
            return
        gt_sh = self.gt.shingles if self.gt else None
        tasks = [(path, entries[i:i + 100]) for path, entries in todo.items() for i in range(0, len(entries), 100)]
        t0, last = time.time(), time.time()
        if n <= INLINE_CHECK_LIMIT or self.workers == 1:
            checks.init_worker(gt_sh)
            for t in tasks:
                self._checks.update(checks.check_chunk(t))
                if time.time() - last > PUBLISH_EVERY_S:
                    self._publish()
                    last = time.time()
        else:
            ctx = mp.get_context("spawn")
            with cf.ProcessPoolExecutor(self.workers, mp_context=ctx, initializer=checks.init_worker, initargs=(gt_sh,)) as ex:
                for fut in cf.as_completed([ex.submit(checks.check_chunk, t) for t in tasks]):
                    self._checks.update(fut.result())
                    if time.time() - last > PUBLISH_EVERY_S:
                        self._publish()
                        last = time.time()
        log.info("checked %d items in %.1fs", n, time.time() - t0)
