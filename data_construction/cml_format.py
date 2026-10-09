"""Parse, clean, render and measure MovieSum-style Cinematic Markup Language (CML) screenplays.

MovieSum scripts look like::

    <script>
      <scene>
        <stage_direction>INT. ROOM - NIGHT</stage_direction>
        <scene_description>...</scene_description>
        <character>WELLES</character>
        <parenthetical>(quietly)</parenthetical>
        <dialogue>...</dialogue>
      </scene>
      ...
    </script>

CML-Bench's ground-truth segments (`ground_truth/gt_100.json`) use the same tag set and layout.
"""
from __future__ import annotations

import html
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache

ELEMENT_TAGS = ("stage_direction", "scene_description", "character", "dialogue", "parenthetical", "action")
ALLOWED_TAGS = frozenset(("script", "scene") + ELEMENT_TAGS)

SCENE_RE = re.compile(r"<scene>(.*?)</scene>", re.S)
ELEMENT_RE = re.compile(r"<(" + "|".join(ELEMENT_TAGS) + r")>(.*?)</\1>", re.S)

HEADING_RE = re.compile(r"^\s*(INT|EXT|I/E|INT\.?/EXT|EXT\.?/INT)\b", re.I)
CONTINUOUS_RE = re.compile(r"\b(CONTINUOUS|SAME|MOMENTS? LATER|SECONDS? LATER|CONT'?D)\b", re.I)
TRANSITION_END_RE = re.compile(r"\b(CUT TO|FADE OUT|FADE TO BLACK|DISSOLVE TO|SMASH CUT|MATCH CUT)\b\W*$", re.I)
TRANSITION_RE = re.compile(
    r"^(FADE (IN|OUT|TO BLACK|TO WHITE)|(QUICK |SLOW |RIPPLE )?DISSOLVE( TO)?|(SMASH |MATCH |HARD )?CUT( TO)?|"
    r"WIPE( TO)?|IRIS (IN|OUT)|INTERCUT|BACK TO( SCENE)?|THE END|END CREDITS|TITLE CARD|SUPER(IMPOSE)?)\W*$",
    re.I,
)
CAMERA_RE = re.compile(
    r"^(NEW ANGLE|REVERSE ANGLE|WIDE ANGLE|HIGH ANGLE|LOW ANGLE|ANGLE|CLOSE ON|CLOSE UP|CLOSEUP|CLOSE SHOT|WIDE SHOT|"
    r"EXTREME CLOSE|INSERT|POV|SERIES OF SHOTS|MONTAGE|BACK TO|BACK ON|MOMENTS LATER|SECONDS LATER|MINUTES LATER|"
    r"LATER|CONTINUOUS|SUPER|TITLE)\b",
    re.I,
)
CREDITS_RE = re.compile(
    r"\b(written by|screenplay by|story by|based on (the|a) (novel|book|play)|shooting script|revised draft|"
    r"first draft|final draft|copyright|all rights reserved)\b|\u00a9",
    re.I,
)

_CTRL_RE = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f\ufffd]")
_NOISE_CHARS_RE = re.compile(r"[\u2022\u25a0\u25aa\u00b7\u25cf\u25a1\u2023\u2043]+")
_INLINE_JUNK_RE = re.compile(r"\(?\b(CONTINUED|CONT'D|OMITTED|OMIT)\b\)?:?")
_ASTERISK_RE = re.compile(r"\*+")
# Shooting-script scene numbers printed on both margins ("128A 128A", "132pt 132pt", "64 ... 64").
_DUP_SCENE_NO_RE = re.compile(r"\b(\d{1,3}[A-Z]{1,3}|\d{1,3}pt)\s+\1\b")
_MARGIN_SCENE_NO_RE = re.compile(r"^(\d{1,3}[A-Z]{0,3})\s+(.+?)\s+\1$")
_LEADING_DOT_RE = re.compile(r"^\.\s+(?=\w)")
_JUNK_ELEMENT_RE = re.compile(r"^(\(?(MORE|CONTINUED|CONT'?D|OMITTED)\)?[:.]?|[\W\d_]*)$", re.I)
_WS_RE = re.compile(r"\s+")

# MovieSum text is partly Penn-Treebank tokenized ("Welles 's", "do n't", "( V.O . )").
# These rules undo only the unambiguous cases; " - " is left alone because it is
# indistinguishable from a dash.
_DETOK_RULES = (
    (re.compile(r"-LRB-"), "("),
    (re.compile(r"-RRB-"), ")"),
    (re.compile(r"-LSB-"), "["),
    (re.compile(r"-RSB-"), "]"),
    (re.compile(r"-LCB-"), "{"),
    (re.compile(r"-RCB-"), "}"),
    (re.compile(r"\b(gon|wan|got) (na|ta)\b", re.I), r"\1\2"),
    (re.compile(r"\b(lem|gim) (me)\b", re.I), r"\1\2"),
    (re.compile(r"``|''"), '"'),
    (re.compile(r"`"), "'"),
    (re.compile(r"(\w) (n't)\b", re.I), r"\1\2"),
    (re.compile(r"(\w) ('(?:s|re|ve|ll|d|m))\b", re.I), r"\1\2"),
    (re.compile(r"(\w[a-z]s) ' (?=\w)"), r"\1' "),
    (re.compile(r"\( +"), "("),
    (re.compile(r" +\)"), ")"),
    (re.compile(r" ([,.;:!?%])(?=\s|$|[\"')\]])"), r"\1"),
    (re.compile(r"\$ (\d)"), r"$\1"),
)
_TOKENIZATION_ARTEFACT_RE = re.compile(
    r"\w (n't|'s|'re|'ll|'ve)\b| [,.;:!?](?=\s|$)|\( | \)|-[LR][RSC]B-|\b(gon|wan|got) (na|ta)\b", re.I
)

_GARBLE_RE = re.compile(r"^(?=[^\d]*\d)(?=.*[A-Za-z])[A-Za-z\d']+$")
_GARBLE_OK_RE = re.compile(
    r"^(\d+(st|nd|rd|th|s|'s|am|pm|mm|cm|km|ft|lb|lbs|k|m|x|d|g|hz|mph|kg|ml|p)|[A-Z]{1,3}-?\d+[A-Z]?|\d{2,4}s|\d+[A-Z])$",
    re.I,
)

_SPEAKER_SUFFIX_RE = re.compile(r"\s*\((?:[^)]*)\)\s*|\s+(V\.?O\.?|O\.?S\.?|O\.?C\.?|CONT'?D\.?)\s*$", re.I)


@dataclass
class Scene:
    index: int
    elements: list = field(default_factory=list)  # [(tag, text), ...]

    @property
    def heading(self) -> str:
        for tag, text in self.elements:
            if tag == "stage_direction":
                return text
        return ""

    def has_body(self) -> bool:
        return any(tag != "stage_direction" for tag, _ in self.elements)


def detokenize(text: str) -> str:
    for pattern, repl in _DETOK_RULES:
        text = pattern.sub(repl, text)
    return text


def clean_text(text: str, detok: bool = True) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL_RE.sub("", text)
    text = _NOISE_CHARS_RE.sub(" ", text)
    text = _ASTERISK_RE.sub(" ", text)
    text = _INLINE_JUNK_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    text = _DUP_SCENE_NO_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    text = _MARGIN_SCENE_NO_RE.sub(r"\2", text)
    if detok:
        text = detokenize(text)
        # MovieSum turned a leading ellipsis into ". " ("<dialogue>. just some girl")
        text = _LEADING_DOT_RE.sub("...", text)
    return text


def is_junk_element(text: str) -> bool:
    return not text or bool(_JUNK_ELEMENT_RE.match(text))


def retag_orphan_characters(elements: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """MovieSum tags many all-caps lines (transitions, camera directions) as <character>.
    A <character> that is a transition or is not followed by dialogue/parenthetical becomes a description."""
    out, i = [], 0
    while i < len(elements):
        tag, text = elements[i]
        if tag == "character":
            nxt_tag, nxt_text = elements[i + 1] if i + 1 < len(elements) else (None, "")
            if TRANSITION_RE.match(text) or CAMERA_RE.match(text):
                if nxt_tag == "dialogue":
                    out.append(("scene_description", f"{text.rstrip(': ')}: {nxt_text}"))
                    i += 2
                    continue
                tag = "scene_description"
            elif nxt_tag not in ("dialogue", "parenthetical"):
                tag = "scene_description"
        out.append((tag, text))
        i += 1
    return out


def parse_script(script: str, detok: bool = True) -> list[Scene]:
    """Parse a MovieSum script into cleaned scenes. Scene.index is the 0-based index in the raw script."""
    scenes = []
    for idx, raw in enumerate(SCENE_RE.findall(script)):
        elements = []
        for tag, text in ELEMENT_RE.findall(raw):
            text = clean_text(text, detok=detok)
            if is_junk_element(text):
                continue
            elements.append((tag, text))
        scene = Scene(idx, retag_orphan_characters(elements))
        if scene.has_body():
            scenes.append(scene)
    return scenes


def drop_duplicate_scenes(scenes: list[Scene], min_chars: int = 200) -> list[Scene]:
    """Drop a scene whose body repeats an earlier scene verbatim (draft paste-overs)."""
    seen, out = set(), []
    for scene in scenes:
        body = "\n".join(t for tag, t in scene.elements if tag != "stage_direction")
        if len(body) >= min_chars and body in seen:
            continue
        seen.add(body)
        out.append(scene)
    return out


def drop_front_matter(scenes: list[Scene], max_scan: int = 5) -> list[Scene]:
    """Drop title-page/credit scenes at the very start of a script."""
    start = 0
    for i, scene in enumerate(scenes[:max_scan]):
        text = " ".join(t for _, t in scene.elements)
        if CREDITS_RE.search(text):
            start = i + 1
    return scenes[start:]


def xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(scenes: list[Scene]) -> str:
    lines = ["<script>"]
    for scene in scenes:
        lines.append("  <scene>")
        for tag, text in scene.elements:
            lines.append(f"    <{tag}>{xml_escape(text)}</{tag}>")
        lines.append("  </scene>")
    lines.append("</script>")
    return "\n".join(lines)


def validate_cml(content: str) -> list[str]:
    """Return a list of structural problems (empty list == valid)."""
    problems = []
    if not content.startswith("<script>") or not content.endswith("</script>"):
        problems.append("not_wrapped_in_script")
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        return problems + [f"xml_parse_error:{exc}"]
    if root.tag != "script":
        problems.append("root_not_script")
    for scene in root:
        if scene.tag != "scene":
            problems.append(f"non_scene_child:{scene.tag}")
            continue
        if len(scene) == 0:
            problems.append("empty_scene")
        for el in scene:
            if el.tag not in ELEMENT_TAGS:
                problems.append(f"bad_tag:{el.tag}")
            if len(el):
                problems.append(f"nested_tag_in:{el.tag}")
            if not (el.text or "").strip():
                problems.append(f"empty_element:{el.tag}")
    return sorted(set(problems))


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """cl100k_base token count; this reproduces `script_tokens` in CML-Bench's gt_100_info.json exactly."""
    return len(_encoder().encode(text, disallowed_special=()))


def speaker_name(character_tag_text: str) -> str:
    return _SPEAKER_SUFFIX_RE.sub(" ", character_tag_text).strip().upper()


_BAD_SPEAKER_RE = re.compile(r"[!?:~|\\^{}<>=+_]|^[(\-]|-$|^\d+[A-Z]{0,2}(\s+\d+[A-Z]{0,2})*$")


def is_bad_character_tag(text: str) -> bool:
    """Speaker tags that are really headings, camera directions, exclamations, scene numbers or OCR debris."""
    return (
        bool(HEADING_RE.match(text) or CAMERA_RE.match(text) or _BAD_SPEAKER_RE.search(text))
        or "CUT TO" in text.upper()
        or len(text) > 40
        or len(text.split()) > 6
        or len(re.sub(r"[^A-Za-z]", "", text)) < 2
    )


def rare_word_rate(text: str, vocab, max_count: int = 3) -> float:
    """OCR-noise proxy: share of words (>= 3 letters) seen <= max_count times in the whole corpus,
    ignoring words repeated >= 3 times in this text (invented names, places)."""
    words = re.findall(r"[A-Za-z]{3,}", text)
    if not words or not vocab:
        return 0.0
    local = {}
    for w in words:
        lw = w.lower()
        local[lw] = local.get(lw, 0) + 1
    rare = sum(1 for w in words if vocab.get(w.lower(), 0) <= max_count and local[w.lower()] < 3)
    return rare / len(words)


def garble_rate(text: str) -> float:
    """Share of words that look like OCR garbage (letters and digits fused, e.g. CREDI1'S)."""
    words = re.findall(r"[A-Za-z\d']+", text)
    if not words:
        return 0.0
    bad = sum(1 for w in words if _GARBLE_RE.match(w) and not _GARBLE_OK_RE.match(w))
    return bad / len(words)


def tokenization_artefact_rate(text: str) -> float:
    """PTB-tokenization artefacts per 1k words."""
    words = max(1, len(text.split()))
    return 1000 * len(_TOKENIZATION_ARTEFACT_RE.findall(text)) / words


def segment_stats(scenes: list[Scene], content: str | None = None, vocab=None) -> dict:
    content = content if content is not None else render(scenes)
    elements = [(tag, text) for s in scenes for tag, text in s.elements]
    tag_counts = {f"<{t}>": 0 for t in ("scene",) + ELEMENT_TAGS}
    tag_counts["<scene>"] = len(scenes)
    for tag, _ in elements:
        tag_counts[f"<{tag}>"] += 1
    dialogue_chars = sum(len(t) for tag, t in elements if tag == "dialogue")
    total_chars = sum(len(t) for _, t in elements) or 1
    speakers = {}
    for tag, text in elements:
        if tag == "character":
            name = speaker_name(text)
            speakers[name] = speakers.get(name, 0) + 1
    char_tags = [t for tag, t in elements if tag == "character"]
    body_text = "\n".join(t for _, t in elements)
    return {
        "num_scenes": len(scenes),
        "content_tokens": count_tokens(content),
        "content_chars": len(content),
        "tag_counts": tag_counts,
        "dialogue_turns": tag_counts["<dialogue>"],
        "num_speakers": len(speakers),
        "top_speakers": [n for n, _ in sorted(speakers.items(), key=lambda kv: -kv[1])[:5]],
        "dialogue_char_ratio": round(dialogue_chars / total_chars, 4),
        "max_element_chars": max((len(t) for _, t in elements), default=0),
        "heading_ratio": round(sum(1 for s in scenes if s.heading) / max(1, len(scenes)), 4),
        "bad_character_tag_ratio": round(sum(map(is_bad_character_tag, char_tags)) / max(1, len(char_tags)), 4),
        "garble_rate": round(garble_rate(body_text), 5),
        "rare_word_rate": round(rare_word_rate(body_text, vocab), 5),
        "tokenization_artefacts_per_1k_words": round(tokenization_artefact_rate(body_text), 3),
    }
