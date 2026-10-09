"""Frozen item schema for the expanded CML dataset (every source, content set and release alike).

One record per item. The first four fields are CML-Bench `gt_100.json`'s, in the same order. Items without
an abstract yet (the `content` set) carry `summary == ""` and null summary/abstract fields.

Versioning: fields are never renamed, retyped or removed within a major version; new optional fields may be
appended (minor bump). `schema.md` in the project store is the human-readable copy of FIELDS.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

SCHEMA_VERSION = "1.0"

TAG_KEYS = ("<scene>", "<stage_direction>", "<scene_description>", "<parenthetical>", "<character>", "<dialogue>")
QUALITY_KEYS = ("dialogue_turns", "num_speakers", "top_speakers", "dialogue_char_ratio", "max_element_chars",
                "heading_ratio", "bad_character_tag_ratio", "garble_rate", "rare_word_rate",
                "tokenization_artefacts_per_1k_words", "gt_ngram_overlap")
GT_RELATED_TYPES = ("series", "universe", "same_source")

# (name, json type(s), nullable, description)
FIELDS = [
    ("movie_name", "string", False, "`Title_Year` exactly as the source/IMDb gives it (gt_100 field)"),
    ("imdb_id", "string", False, "`tt` + 7-8 digits; one source per imdb_id across the dataset (gt_100 field)"),
    ("script_segment", "string", False, "the excerpt in CML: `<script><scene>...</scene>...</script>` (gt_100 field)"),
    ("summary", "string", False, "the abstract; empty string until an abstract is merged (gt_100 field)"),
    ("item_id", "string", False, "`{imdb_id}-s{scene_start:04d}-{scene_end:04d}`; unique across the dataset"),
    ("source_dataset", "string", False, "source corpus name, e.g. `MovieSum`"),
    ("source_split", "string", True, "split inside the source corpus (MovieSum: train/val/test)"),
    ("source_url", "string", False, "landing page of the source corpus"),
    ("source_file", "string", False, "file inside the source corpus that holds the screenplay"),
    ("imdb_url", "string", False, "`https://www.imdb.com/title/{imdb_id}/`"),
    ("segment_index", "integer", False, "0-based index of this window among the movie's accepted windows"),
    ("scene_start", "integer", False, "index of the first scene in the source screenplay's raw scene list"),
    ("scene_end", "integer", False, "index of the last scene (inclusive) in the raw scene list"),
    ("num_scenes", "integer", False, "number of `<scene>` elements in script_segment"),
    ("dropped_scenes", "array", False, "raw scenes inside [scene_start, scene_end] that are not in the segment: "
                                       "[{scene, reason}], reason in no_body (heading-only/empty) | duplicate"),
    ("relative_position", "number", False, "window start / number of scenes in the cleaned screenplay (0-1)"),
    ("script_tokens", "integer", False, "tiktoken cl100k_base tokens of script_segment (gt_100_info.json convention)"),
    ("tag_counts", "object", False, "count of each of the 6 CML tags in script_segment (gt_100_info.json keys)"),
    ("summary_tokens", "integer", True, "cl100k_base tokens of summary; null until an abstract is merged"),
    ("summary_words", "integer", True, "whitespace words of summary; null until an abstract is merged"),
    ("imdb_rating", "number", True, "IMDb average rating (non-commercial dumps)"),
    ("imdb_votes", "integer", True, "IMDb vote count"),
    ("genres", "array", False, "IMDb genres (strings)"),
    ("year", "integer", True, "IMDb start year"),
    ("gt_related", "object", True, "null, or {type, gt_movie, gt_imdb_id} when the movie shares a franchise/story with a "
                                   "CML-Bench GT movie (type in series | universe | same_source; remakes are excluded)"),
    ("eval_safe", "boolean", False, "true iff gt_related is null; the `eval_safe` config holds only these items"),
    ("content_normalization", "string", False, "cleaning version applied to the source text, e.g. `moviesum_clean_detok_v3`"),
    ("content_sha1", "string", False, "sha1 of script_segment (UTF-8)"),
    ("build_id", "string", False, "content build: `{source_slug}-{normalization}-{sha1 of sorted content_sha1s, 8 hex}`"),
    ("batch_id", "string", False, "abstract work unit inside the build, e.g. `moviesum-b0001`"),
    ("abstract_prompt_version", "string", True, "prompt that produced summary; null until merged"),
    ("abstract_author", "string", True, "model/agent that wrote summary; null until merged"),
    ("quality", "object", False, "content QC metrics: " + ", ".join(QUALITY_KEYS)),
]
FIELD_NAMES = [f[0] for f in FIELDS]
_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def source_slug(source_dataset: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", source_dataset.lower()).strip("-")


def build_id(slug: str, normalization: str, shas) -> str:
    digest = hashlib.sha1("\n".join(sorted(shas)).encode()).hexdigest()[:8]
    short = normalization.replace(f"{slug}_", "").replace("clean_", "")
    return f"{slug}-{short}-{digest}"


def load_gt_related(path: str | None = None) -> dict:
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_related.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gt_relation(imdb_id: str, table: dict) -> dict | None:
    """The table entry for imdb_id ({type, gt_movie, gt_imdb_id}), or None."""
    for rel in table["relations"]:
        if rel["imdb_id"] == imdb_id:
            return {"type": rel["type"], "gt_movie": rel["gt_movie"], "gt_imdb_id": rel["gt_imdb_id"]}
    return None


def make_record(item: dict, *, batch_id: str, build: str, abstract: dict | None = None, count_tokens=None) -> dict:
    """Schema-1.0 record from a built segment (build_segments.py output) and an optional abstract file."""
    summary = abstract["abstract"].strip() if abstract else ""
    if abstract and count_tokens is None:
        from cml_format import count_tokens  # noqa: PLC0415
    rec = {
        "movie_name": item["movie_name"],
        "imdb_id": item["imdb_id"],
        "script_segment": item["script_segment"],
        "summary": summary,
        "item_id": item["item_id"],
        "source_dataset": item["source_dataset"],
        "source_split": item.get("source_split"),
        "source_url": item["source_url"],
        "source_file": item["source_file"],
        "imdb_url": item["imdb_url"],
        "segment_index": item["segment_index"],
        "scene_start": item["scene_start"],
        "scene_end": item["scene_end"],
        "num_scenes": item["num_scenes"],
        "dropped_scenes": item["dropped_scenes"],
        "relative_position": item["relative_position"],
        "script_tokens": item["content_tokens"],
        "tag_counts": {k: item["tag_counts"].get(k, 0) for k in TAG_KEYS},
        "summary_tokens": count_tokens(summary) if abstract else None,
        "summary_words": len(summary.split()) if abstract else None,
        "imdb_rating": item.get("imdb_rating"),
        "imdb_votes": item.get("imdb_votes"),
        "genres": item.get("genres") or [],
        "year": item.get("year"),
        "gt_related": item["gt_related"],
        "eval_safe": item["gt_related"] is None,
        "content_normalization": item["content_normalization"],
        "content_sha1": item["content_sha1"],
        "build_id": build,
        "batch_id": batch_id,
        "abstract_prompt_version": abstract.get("prompt_version") if abstract else None,
        "abstract_author": abstract.get("author") if abstract else None,
        "quality": {k: item.get(k) for k in QUALITY_KEYS},
    }
    return rec


def validate_record(rec: dict) -> list[str]:
    """Structural schema check (types, nullability, field order, id formats). Content rules live in the contract."""
    problems = []
    if list(rec)[: len(FIELD_NAMES)] != FIELD_NAMES:
        missing = [f for f in FIELD_NAMES if f not in rec]
        problems.append(f"field_order_or_missing:{missing[:3]}")
    for name, typ, nullable, _ in FIELDS:
        if name not in rec:
            continue
        v = rec[name]
        if v is None:
            if not nullable:
                problems.append(f"null:{name}")
            continue
        if isinstance(v, bool) and typ in ("integer", "number"):
            problems.append(f"type:{name}")
        elif not isinstance(v, _JSON_TYPES[typ]):
            problems.append(f"type:{name}")
    if problems:
        return problems
    if not re.fullmatch(r".+_\d{4}", rec["movie_name"]):
        problems.append("movie_name_format")
    if not re.fullmatch(r"tt\d{7,8}", rec["imdb_id"]):
        problems.append("imdb_id_format")
    if rec["item_id"] != f"{rec['imdb_id']}-s{rec['scene_start']:04d}-{rec['scene_end']:04d}":
        problems.append("item_id_format")
    if hashlib.sha1(rec["script_segment"].encode()).hexdigest() != rec["content_sha1"]:
        problems.append("content_sha1_mismatch")
    if set(rec["tag_counts"]) != set(TAG_KEYS):
        problems.append("tag_counts_keys")
    if set(rec["quality"]) != set(QUALITY_KEYS):
        problems.append("quality_keys")
    if rec["eval_safe"] != (rec["gt_related"] is None):
        problems.append("eval_safe_inconsistent")
    if rec["gt_related"] is not None and (set(rec["gt_related"]) != {"type", "gt_movie", "gt_imdb_id"}
                                          or rec["gt_related"]["type"] not in GT_RELATED_TYPES):
        problems.append("gt_related_shape")
    for d in rec["dropped_scenes"]:
        if set(d) != {"scene", "reason"} or not rec["scene_start"] < d["scene"] < rec["scene_end"]:
            problems.append("dropped_scenes_shape")
            break
    has_summary = bool(rec["summary"])
    if has_summary != (rec["abstract_prompt_version"] is not None) or has_summary != (rec["summary_words"] is not None):
        problems.append("summary_fields_inconsistent")
    return problems


def schema_json() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "fields": [{"name": n, "type": t, "nullable": nl, "description": d} for n, t, nl, d in FIELDS],
        "tag_keys": list(TAG_KEYS),
        "quality_keys": list(QUALITY_KEYS),
        "gt_related_types": list(GT_RELATED_TYPES),
    }
