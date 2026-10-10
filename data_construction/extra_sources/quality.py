"""Script-level quality gates for extra-source screenplays (applied before windowing; the per-window filters
of build_segments.py still apply afterwards).

Rejects: image-only PDFs that would need OCR, OCR garbage (letter-spaced text, fused digits, rare words),
transcripts and other non-screenplays (no scene headings / no action / "NAME: line" format),
incomplete or partial scripts, treatments without dialogue, non-English text and parser failures.
"""
from __future__ import annotations

import re

from cml_format import garble_rate, is_bad_character_tag, rare_word_rate, speaker_name

_ABSORBED_RE = re.compile(r"(?:^|[.!?]\s+)([A-Z][A-Za-z'\-]+)\s+([a-z]+(?:s|ed))\b")
_SPEECH_VERBS = {"was", "has", "is", "does", "says", "needs", "wants", "likes", "loves", "knows", "thinks", "gets", "goes",
                 "seems", "looks", "called", "said", "used", "asked", "told"}


def absorbed_action_rate(scenes) -> float:
    """Share of dialogue elements containing a sentence that opens with a speaker's name and a third-person
    verb ("... Like an announcement. Annie confers with Karin"): action lines swallowed by the preceding
    dialogue when a file has dialogue at the action indent and no blank line after it.
    Calibration: CML-Bench GT p95 0.040, MovieSum segments p95 0.019 / p99 0.046."""
    speakers = {speaker_name(t) for s in scenes for tag, t in s.elements if tag == "character"}
    first = {n.split()[0] for n in speakers if n}
    dialogue = [t for s in scenes for tag, t in s.elements if tag == "dialogue"]
    bad = sum(1 for t in dialogue if any(m.group(1).upper() in first and m.group(2) not in _SPEECH_VERBS
                                         for m in _ABSORBED_RE.finditer(t)))
    return bad / max(1, len(dialogue))

QUALITY_CONFIG = {
    "min_chars_per_pdf_page": 600,
    "max_letter_spaced_rate": 0.01,
    "min_words": 10000,
    "min_headings": 30,
    "max_speaker_colon_share": 0.05,
    "min_action_share": 0.08,
    "min_dialogue_share": 0.10,
    "max_dialogue_share": 0.85,
    "min_character_cues": 150,
    "max_bad_character_tag_ratio": 0.08,
    "max_garble_rate": 0.004,
    "max_rare_word_rate": 0.012,
    "min_ascii_letter_share": 0.97,
    "min_stopword_share": 0.25,
    "max_median_scene_words": 600,
    "max_absorbed_action_rate": 0.019,  # MovieSum segments p95 (the GT p95 0.04 let Margaret's swallowed action through)
    "max_oneoff_short_speakers": 4,
}
OCR_PRODUCER_RE = re.compile(r"paper capture|image conversion|clearscan|abbyy|finereader|omnipage|readiris|tesseract|ocr", re.I)
_STOP = set("the a an and of to in is it that he she they you i we his her their was for on with as at but not this be are".split())


def script_quality(text: str, scenes, diag: dict, pdf_meta: dict, vocab) -> tuple[list[str], dict]:
    cfg = QUALITY_CONFIG
    reasons = []
    nonspace = len(re.sub(r"\s", "", text))
    pages = pdf_meta.get("pages") or 0
    metrics = {"text_chars": nonspace, "pdf_pages": pages, "pdf_producer": pdf_meta.get("producer", ""),
               "ocr_pdf": bool(OCR_PRODUCER_RE.search(pdf_meta.get("producer", "") + " " + pdf_meta.get("creator", "")))}
    if pages and nonspace / pages < cfg["min_chars_per_pdf_page"]:
        return ["needs_ocr"], metrics | {"chars_per_page": round(nonspace / pages)}

    # OCR letter spacing ("t a k e s"): letters inside runs of >= 3 single-letter tokens
    tokens = text.split()
    spaced, run = 0, 0
    for t in tokens + [""]:
        if len(t) == 1 and t.isalpha():
            run += 1
        else:
            spaced += run if run >= 3 else 0
            run = 0
    metrics["letter_spaced_rate"] = round(spaced / max(1, len(tokens)), 4)
    if metrics["letter_spaced_rate"] > cfg["max_letter_spaced_rate"]:
        reasons.append("ocr_letter_spaced")

    elements = [(t, x) for s in scenes for t, x in s.elements]
    body = "\n".join(x for _, x in elements)
    words = body.split()
    metrics["words"] = len(words)
    metrics["headings"] = diag.get("headings", 0)
    metrics["scenes"] = len(scenes)
    chars = {tag: sum(len(x) for t, x in elements if t == tag) for tag in ("scene_description", "dialogue", "character")}
    total = sum(len(x) for _, x in elements) or 1
    metrics["action_share"] = round(chars["scene_description"] / total, 3)
    metrics["dialogue_share"] = round(chars["dialogue"] / total, 3)
    cues = [x for t, x in elements if t == "character"]
    metrics["character_cues"] = len(cues)
    metrics["speakers"] = len({speaker_name(c) for c in cues})
    metrics["bad_character_tag_ratio"] = round(sum(map(is_bad_character_tag, cues)) / max(1, len(cues)), 4)
    short = {}
    for c in cues:
        name = speaker_name(c)
        if len(re.sub(r"[^A-Za-z]", "", name)) <= 2:
            short[name] = short.get(name, 0) + 1
    metrics["oneoff_short_speakers"] = sum(1 for v in short.values() if v < 3)
    metrics["speaker_colon_share"] = round(diag.get("speaker_colon_lines", 0) / max(1, diag.get("nonblank_lines", 1)), 4)
    scene_words = sorted(sum(len(x.split()) for _, x in s.elements) for s in scenes) or [0]
    metrics["median_scene_words"] = scene_words[len(scene_words) // 2]
    metrics["absorbed_action_rate"] = round(absorbed_action_rate(scenes), 4)
    metrics["garble_rate"] = round(garble_rate(body), 5)
    metrics["rare_word_rate"] = round(rare_word_rate(body, vocab), 5)
    letters = [c for c in body if c.isalpha()]
    metrics["ascii_letter_share"] = round(sum(c.isascii() for c in letters) / max(1, len(letters)), 4)
    lw = [w.lower() for w in re.findall(r"[A-Za-z']+", body)]
    metrics["stopword_share"] = round(sum(w in _STOP for w in lw) / max(1, len(lw)), 3)

    if metrics["speaker_colon_share"] > cfg["max_speaker_colon_share"] or (
        metrics["headings"] < 10 and metrics["action_share"] < cfg["min_action_share"]
    ):
        reasons.append("transcript_like")
    if metrics["headings"] < cfg["min_headings"]:
        reasons.append("few_scene_headings")
    if metrics["words"] < cfg["min_words"]:
        reasons.append("too_short_or_incomplete")
    if metrics["action_share"] < cfg["min_action_share"]:
        reasons.append("no_action_text")
    if not cfg["min_dialogue_share"] <= metrics["dialogue_share"] <= cfg["max_dialogue_share"] or metrics["character_cues"] < cfg["min_character_cues"]:
        reasons.append("dialogue_structure")
    if metrics["bad_character_tag_ratio"] > cfg["max_bad_character_tag_ratio"]:
        reasons.append("parse_bad_cues")
    if metrics["oneoff_short_speakers"] >= cfg["max_oneoff_short_speakers"]:
        reasons.append("watermark_fragments")  # a vertical watermark split into "IM", "AL", "ON" speaker tags
    if metrics["median_scene_words"] > cfg["max_median_scene_words"]:
        reasons.append("merged_scenes")
    if metrics["absorbed_action_rate"] > cfg["max_absorbed_action_rate"]:
        reasons.append("dialogue_action_merged")
    if metrics["garble_rate"] > cfg["max_garble_rate"] or metrics["rare_word_rate"] > cfg["max_rare_word_rate"]:
        reasons.append("ocr_garbage")
    if metrics["ascii_letter_share"] < cfg["min_ascii_letter_share"] or metrics["stopword_share"] < cfg["min_stopword_share"]:
        reasons.append("not_english")
    return reasons, metrics
