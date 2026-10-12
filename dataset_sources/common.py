"""Small, dependency-free I/O helpers. No credentials or screenplay text in reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KINDS = {"screenplay", "stage_play", "novel", "prose_fiction"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{n}: invalid JSON") from exc


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def registry():
    data = load_json(ROOT / "registry.json")
    entries = data["sources"]
    ids = [s["id"] for s in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source ID")
    for s in entries:
        if s["route"] not in {"review", "permission", "metadata", "excluded"}:
            raise ValueError(f"invalid route: {s['id']}")
        if not s["evidence_urls"] or not s["checked_on"]:
            raise ValueError(f"missing evidence: {s['id']}")
    return {s["id"]: s for s in entries}
