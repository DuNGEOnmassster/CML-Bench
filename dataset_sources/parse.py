"""Conservative local parsers. Prose remains prose, speaker attribution is explicit.

No LLM rewriting, OCR, screenplay inference, or CML fabrication for novels.
All XML must be uncompressed UTF-8 without DTDs/entities.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET


def tag(element):
    return element.tag.rsplit("}", 1)[-1]


def text(element):
    return " ".join("".join(element.itertext()).split())


def xml_root(raw):
    if re.search(rb"<!\s*(DOCTYPE|ENTITY)\b", raw, re.I):
        raise ValueError("DTD/entity declarations not supported")
    try:
        return ET.fromstring(raw.decode("utf-8-sig"))
    except ET.ParseError as exc:
        raise ValueError("malformed XML") from exc


def check_inline(node):
    if any(tag(e) in {"stage", "note", "sp", "speaker"} for e in list(node.iter())[1:]):
        raise ValueError("mixed inline speech/stage/note requires a reviewed structural conversion")


def tei(raw, kind):
    root = xml_root(raw)
    if tag(root) != "TEI":
        raise ValueError("expected one TEI work (not a corpus wrapper)")
    body = next((e for e in root.iter() if tag(e) == "body"), None)
    if body is None:
        raise ValueError("TEI body missing")
    blocks = []

    def visit(node, path):
        name = tag(node)
        locator = node.get("{http://www.w3.org/XML/1998/namespace}id") or path
        if name == "sp":
            if kind not in {"stage_play", "screenplay"}:
                raise ValueError("speech in prose manifest: review work classification")
            speakers = [text(e) for e in node if tag(e) == "speaker"]
            speaker = " / ".join(speakers) or node.get("who")
            if not speaker:
                raise ValueError("speech has no speaker; do not guess")
            for i, child in enumerate(node):
                t = tag(child)
                if t == "speaker":
                    continue
                if t not in {"p", "l", "lg", "ab", "stage", "note"}:
                    raise ValueError(f"unsupported speech child {t}; curator review needed")
                check_inline(child)
                value = text(child)
                if value:
                    blocks.append(
                        {
                            "locator": f"{locator}/{t}[{i}]",
                            "type": "stage_direction"
                            if t == "stage"
                            else "note"
                            if t == "note"
                            else "dialogue",
                            "speaker": speaker if t not in {"stage", "note"} else None,
                            "text": value,
                        }
                    )
            return
        if name in {"p", "l", "lg", "ab", "stage", "head", "note"}:
            check_inline(node)
            value = text(node)
            if value:
                block_type = {"head": "heading", "stage": "stage_direction", "note": "note"}.get(
                    name, "narration"
                )
                blocks.append(
                    {"locator": locator, "type": block_type, "speaker": None, "text": value}
                )
            return
        if name in {"pb", "lb", "milestone", "castList", "front", "back"}:
            return  # apparatus/page layout, explicitly recorded by parser version
        if name not in {"body", "div", "div1", "div2", "div3", "div4"}:
            raise ValueError(f"unsupported TEI block {name}; extend parser with fidelity tests")
        if node.text and node.text.strip():
            raise ValueError("unwrapped TEI body text; review instead of dropping")
        for i, child in enumerate(node):
            visit(child, f"{path}/{tag(child)}[{i}]")
            if child.tail and child.tail.strip():
                raise ValueError("unwrapped TEI tail text; review instead of dropping")

    visit(body, "body")
    if not blocks:
        raise ValueError("no usable TEI blocks")
    return blocks


def cml(raw, kind):
    if kind != "screenplay":
        raise ValueError("CML import is restricted to actual screenplay assets")
    root = xml_root(raw)
    if tag(root) != "script" or any(tag(s) != "scene" for s in root):
        raise ValueError("expected script with scene children")
    blocks = []
    mapping = {
        "stage_direction": "heading",
        "scene_description": "narration",
        "parenthetical": "stage_direction",
        "dialogue": "dialogue",
    }
    for i, scene in enumerate(root):
        speaker = None
        for j, e in enumerate(scene):
            name, value = tag(e), text(e)
            if name == "character":
                speaker = value
                continue
            if name not in mapping:
                raise ValueError(f"unknown CML element {name}")
            if name == "dialogue" and not speaker:
                raise ValueError("unattributed CML dialogue")
            if value:
                blocks.append(
                    {
                        "locator": f"scene[{i}]/{name}[{j}]",
                        "type": mapping[name],
                        "speaker": speaker if name == "dialogue" else None,
                        "text": value,
                    }
                )
    if not blocks:
        raise ValueError("empty CML")
    return blocks


def plain(raw, kind):
    if kind not in {"novel", "prose_fiction"}:
        raise ValueError(
            "plain text cannot establish screenplay/play speakers; use reviewed TEI/CML"
        )
    value = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    if "START OF THE PROJECT GUTENBERG" in value or "END OF THE PROJECT GUTENBERG" in value:
        raise ValueError(
            "raw Gutenberg wrapper: provide a reviewed body-only derivative and retain original provenance"
        )
    return [
        {
            "locator": f"paragraph[{i}]",
            "type": "narration",
            "speaker": None,
            "text": " ".join(p.split()),
        }
        for i, p in enumerate(re.split(r"\n\s*\n", value))
        if p.strip()
    ]


def parse(raw, fmt, kind):
    if fmt not in {"tei", "cml", "text"}:
        raise ValueError("supported formats: tei, cml, text (no automatic PDF/OCR)")
    return {"tei": tei, "cml": cml, "text": plain}[fmt](raw, kind)


def windows(blocks, max_chars=24000, min_chars=2000):
    """Whole-block non-overlapping windows; never truncate a long speech silently."""
    if not 0 < min_chars <= max_chars:
        raise ValueError("require 0 < min_chars <= max_chars")
    output, rejected, current, size = [], [], [], 0

    def flush():
        nonlocal current, size
        if current:
            if size >= min_chars:
                output.append(current)
            else:
                rejected.append(
                    {"locator": current[0]["locator"], "reason": "short_window", "chars": size}
                )
        current, size = [], 0

    for block in blocks:
        length = len(block["text"])
        if length > max_chars:
            flush()
            rejected.append(
                {"locator": block["locator"], "reason": "oversize_block", "chars": length}
            )
            continue
        if size + length + bool(current) > max_chars:
            flush()
        size += length + bool(current)
        current.append(block)
    flush()
    return output, rejected
