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
# Also matches self-closing elements (`<dialogue />`), which MovieSum uses for empty lines.
ELEMENT_ANY_RE = re.compile(r"<(" + "|".join(ELEMENT_TAGS) + r")>(.*?)</\1>|<(" + "|".join(ELEMENT_TAGS) + r")\s*/>", re.S)

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
_NOISE_IN_WORD_RE = re.compile(r"\b([A-Za-z]+)[\u2022\u25a0\u25aa\u00b7\u25cf\u25a1\u2023\u2043]+([A-Za-z]+)\b")
# Markdown escapes left by the PDF->text step ("\*", "\[", "\_"); a backslash is never screenplay text.
_MD_ESCAPE_RE = re.compile(r"\\([^\w\s\\]|_)")
_BACKSLASH_RE = re.compile(r"\\+")
# Runs after detokenization so the PTB form "( CONT 'D . )" is caught too; uppercase only.
_INLINE_JUNK_RE = re.compile(r"\(?\b(CONTINUED|OMITTED|OMIT)\b\)?:?")
_CONTD_RE = re.compile(r"\s*[;,]?\s*\bCONT\s*['\u2019]?\s*D\b\.?")
_EMPTY_PARENS_RE = re.compile(r"\(\s*[;,.]?\s*\)")
_ASTERISK_RE = re.compile(r"\*+")
# Shooting-script scene numbers printed on both margins ("128A 128A", "132pt 132pt", "64 ... 64").
_DUP_SCENE_NO_RE = re.compile(r"\b(\d{1,3}-?[A-Z]{1,3}|\d{1,3}pt)\s+\1\b")
_MARGIN_SCENE_NO_RE = re.compile(r"^(\d{1,3}-?[A-Z]{0,3})\s+(.+?)\s+\1$")
_LEADING_DOT_RE = re.compile(r"^\.\s+(?=[\w\"'])")
_JUNK_ELEMENT_RE = re.compile(r"^(\(?(MORE|CONTINUED|CONT'?D|OMITTED)\)?[:.]?|[\W\d_]*)$", re.I)
# Dialogue is real speech unless it is empty or an explicit page-break/revision marker ("More.", "...", "?!",
# "926 - 3143." are lines).
_JUNK_DIALOGUE_RE = re.compile(r"^(\((MORE|CONTINUED|CONT'?D)\)|(CONTINUED|OMITTED)[:.]?)$")
# A bare number in a dialogue tag is a page number when nobody real says it: either no speaker precedes it, or the
# "speaker" is a revision stamp or heading fragment ("6/15/15 - YELLOW", "INT. OFFICE -- CONTINUOUS") that never
# says anything with letters. "478." from a real speaker stays.
_NUMBER_LINE_RE = re.compile(r"^\d{1,4}[A-Z]?\.?$")
_HEADING_LEAD_NO_RE = re.compile(r"^\d{1,3}-?[A-Z]{0,2}\.?\s+(?=(INT|EXT|I/E)\b)")
_HEADING_TAIL_NO_RE = re.compile(
    r"\b(DAY|NIGHT|MORNING|EVENING|AFTERNOON|DAWN|DUSK|LATER|SUNSET|SUNRISE|CONTINUOUS)\s+\d{1,3}-?[A-Z]{0,2}\.?$"
)
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
    (re.compile(r"``\s*"), '"'),
    (re.compile(r"\s*''"), '"'),
    (re.compile(r"`"), "'"),
    (re.compile(r"(\w) (n't)\b", re.I), r"\1\2"),
    (re.compile(r"(\w) ('(?:s|re|ve|ll|d|m))\b", re.I), r"\1\2"),
    (re.compile(r"(\w) ('(?:s|re|ve|ll|d|m))\b", re.I), r"\1\2"),  # again for doubled clitics ("Jo 's 's")
    (re.compile(r"(\w[a-z]s) ' (?=\w)(?!(?:nt|t|s|d|ll|re|ve|m)\b)"), r"\1' "),
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


_OPEN_QUOTE_SPACE_RE = re.compile(r'(^|[\s(\[:,])" (?=\w)')
_CLOSE_QUOTE_SPACE_RE = re.compile(r'(?<=[\w.!?,;:\'])\s"(?=\s|$|[.,;:!?)\]])')


def fix_quote_spaces(text: str) -> str:
    """`" Son, sit down . "` -> `"Son, sit down."`: pair quotes inside an element when their count is even,
    otherwise only fix the unambiguous opening/closing positions (a quote spanning two elements)."""
    if '"' not in text:
        return text
    parts = text.split('"')
    if len(parts) % 2 == 1:
        for i in range(1, len(parts), 2):
            parts[i] = parts[i].strip()
        return '"'.join(parts)
    text = _OPEN_QUOTE_SPACE_RE.sub(r'\1"', text)
    if text.startswith('" '):
        text = '"' + text[2:]
    return _CLOSE_QUOTE_SPACE_RE.sub('"', text)


def _noise_in_word(m: re.Match) -> str:
    """A bullet between letters stood for an apostrophe ("You•re"), a space ("in·position") or nothing ("J·im")."""
    left, right = m.group(1), m.group(2)
    if right.lower() in ("s", "re", "m", "ve", "ll", "d", "t"):
        return f"{left}'{right}"
    return f"{left} {right}" if len(left) >= 2 and len(right) >= 2 else left + right


def clean_text(text: str, detok: bool = True) -> str:
    text = html.unescape(html.unescape(text))  # some sources are escaped twice ("Bed Bath &amp;amp; Beyond")
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL_RE.sub("", text)
    text = _MD_ESCAPE_RE.sub(r"\1", text)
    text = _BACKSLASH_RE.sub(" ", text)
    text = _NOISE_IN_WORD_RE.sub(_noise_in_word, text)
    text = _NOISE_CHARS_RE.sub(" ", text)
    text = _ASTERISK_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    text = _DUP_SCENE_NO_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    text = _MARGIN_SCENE_NO_RE.sub(r"\2", text)
    if detok:
        text = detokenize(text)
    text = _CONTD_RE.sub("", text)
    text = _INLINE_JUNK_RE.sub(" ", text)
    text = _EMPTY_PARENS_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if detok:
        # MovieSum turned a leading ellipsis into ". " ("<dialogue>. just some girl")
        text = _LEADING_DOT_RE.sub("...", text)
        text = fix_quote_spaces(text)
    return text


def clean_heading(text: str) -> str:
    """Strip shooting-script scene numbers from a scene heading ("64 INT. BAR - NIGHT", "EXT. ROAD - DAY 64").
    A number is only removed before INT/EXT or after a time of day, so "EXT. ROUTE 66" survives."""
    return _HEADING_TAIL_NO_RE.sub(r"\1", _HEADING_LEAD_NO_RE.sub("", text)).strip()


def is_junk_element(text: str) -> bool:
    return not text or bool(_JUNK_ELEMENT_RE.match(text))


def is_junk_dialogue(text: str) -> bool:
    return not text or bool(_JUNK_DIALOGUE_RE.match(text))


def drop_junk_elements(elements: list[tuple[str, str]], known_speakers: set[str]) -> list[tuple[str, str]]:
    """Remove page-break/revision junk without ever deleting a real line of dialogue.

    - dialogue is dropped only when empty or an explicit marker ((MORE), CONTINUED); its speaker line (and any
      parentheticals in between) goes with it instead of being left behind as a stray name;
    - other elements are dropped when they are markers, page numbers or punctuation only;
    - a bare speaker name that never gets a line (a <character> followed by no dialogue, or a <scene_description>
      that is exactly a speaker's name) is dropped when it is a known speaker of the script: it carries no text."""
    elements = [(t, x) for t, x in elements if t == "dialogue" or not is_junk_element(x)]
    out, i, n = [], 0, len(elements)
    while i < n:
        tag, text = elements[i]
        if tag == "character":
            j = i + 1
            while j < n and elements[j][0] == "parenthetical":
                j += 1
            if j < n and elements[j][0] == "dialogue" and is_junk_dialogue(elements[j][1]):
                if TRANSITION_RE.match(text) or CAMERA_RE.match(text):
                    out.append(("scene_description", text))
                i = j + 1
                continue
            if j < n and elements[j][0] == "dialogue" and _NUMBER_LINE_RE.match(elements[j][1]) \
                    and speaker_name(text) not in known_speakers:
                i = j + 1
                continue
            nxt = elements[i + 1][0] if i + 1 < n else None
            if nxt not in ("dialogue", "parenthetical") and not (TRANSITION_RE.match(text) or CAMERA_RE.match(text)) \
                    and speaker_name(text) in known_speakers:
                i += 1
                continue
        elif tag == "scene_description" and text.isupper() and text.strip() == speaker_name(text) and text.strip() in known_speakers:
            i += 1
            continue
        elif tag == "dialogue" and (is_junk_dialogue(text) or (_NUMBER_LINE_RE.match(text) and
                                                               (not out or out[-1][0] not in ("character", "parenthetical")))):
            i += 1
            continue
        out.append((tag, text))
        i += 1
    return out


def known_speakers_of(raw_scenes: list[list[tuple[str, str]]]) -> set[str]:
    """Speaker names that have at least one real line somewhere in the script."""
    known = set()
    for elements in raw_scenes:
        for k, (tag, text) in enumerate(elements):
            if tag != "character":
                continue
            j = k + 1
            while j < len(elements) and elements[j][0] == "parenthetical":
                j += 1
            if j < len(elements) and elements[j][0] == "dialogue" and re.search(r"[A-Za-z]", elements[j][1]) \
                    and not is_junk_dialogue(elements[j][1]):
                name = speaker_name(text)
                # "CUT TO:" or a heading tagged as a speaker is not a name, even when a line follows it.
                if len(re.sub(r"[^A-Z]", "", name)) >= 2 and not (TRANSITION_RE.match(name) or is_bad_character_tag(name)):
                    known.add(name)
    return known


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


def parse_script(script: str, detok: bool = True, dropped: dict | None = None) -> list[Scene]:
    """Parse a MovieSum script into cleaned scenes. Scene.index is the 0-based index in the raw script.
    Scenes left without a body (heading only, or empty) are skipped and recorded in `dropped` as "no_body"."""
    raw_scenes = []
    for raw in SCENE_RE.findall(script):
        elements = []
        for m in ELEMENT_ANY_RE.finditer(raw):
            tag = m.group(1) or m.group(3)
            text = clean_text(m.group(2) or "", detok=detok)
            if tag == "stage_direction":
                text = clean_heading(text)
            elements.append((tag, text))
        raw_scenes.append(elements)
    known = known_speakers_of(raw_scenes)
    scenes = []
    for idx, elements in enumerate(raw_scenes):
        scene = Scene(idx, retag_orphan_characters(drop_junk_elements(elements, known)))
        if scene.has_body():
            scenes.append(scene)
        elif dropped is not None:
            dropped[idx] = "no_body"
    return scenes


def drop_duplicate_scenes(scenes: list[Scene], min_chars: int = 200, dropped: dict | None = None) -> list[Scene]:
    """Drop a scene whose body repeats an earlier scene verbatim (draft paste-overs)."""
    seen, out = set(), []
    for scene in scenes:
        body = "\n".join(t for tag, t in scene.elements if tag != "stage_direction")
        if len(body) >= min_chars and body in seen:
            if dropped is not None:
                dropped[scene.index] = "duplicate"
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
