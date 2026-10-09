"""Plain-text screenplays (IMSDb/Daily Script HTML <pre>, TXT, `pdftotext -layout`) -> CML scenes.

The parser infers the page layout from indentation (action ~15 cols, dialogue ~25, cue ~37 in a typical
file; everything at column 0 in flush-left files) and classifies lines as scene headings (INT./EXT.),
transitions, character cues, parentheticals, dialogue and action paragraphs. Output scenes use the same
`Scene` objects and cleaning as the MovieSum pipeline, so windowing, filters and rendering are shared.
"""
from __future__ import annotations

import html
import os
import re
import subprocess
import tempfile
from collections import Counter

from cml_format import CAMERA_RE, Scene, clean_text, is_junk_element, speaker_name

PARSER_VERSION = "text_screenplay_v1"

_HEADING_CORE = r"(?:INT|EXT|INT\.?\s*/\s*EXT|EXT\.?\s*/\s*INT|I\s*/\s*E|E\s*/\s*I|INTERIOR|EXTERIOR|EST)"
HEADING_LINE_RE = re.compile(rf"^(?:(\d{{1,4}}[A-Z]{{0,3}})[.)]?\s+)?({_HEADING_CORE}(?:[.:\-\s].*)?)$")
_TRAILING_SCENE_NO_RE = re.compile(r"\s+\d{1,4}[A-Z]{0,3}\.?\s*$")
TRANSITION_LINE_RE = re.compile(
    r"^((FADE|IRIS) (IN|OUT|UP|TO BLACK|TO WHITE)|((QUICK|SLOW|MATCH|SMASH|JUMP|HARD|STRAIGHT|FLASH) )?CUT( BACK)?( TO)?|"
    r"(QUICK |SLOW |RIPPLE |LAP )?DISSOLVE( TO)?|WIPE( TO)?|CUT TO BLACK|BACK TO( SCENE)?|INTERCUT( WITH)?|THE END|END CREDITS)\b[\s\w.:]*$"
)
# Page furniture. Only the typographic forms are removed ("(MORE)", "CONTINUED:", a page number on its own
# line at a page break or the right margin), so dialogue such as "More." or "478." is never dropped.
_PAGE_NO_RE = re.compile(r"^\s*(page\s*)?[-\u2013]?\s*\d{1,3}\s*[-\u2013]?[.)]?\s*$", re.I)
_CONTINUED_RE = re.compile(
    r"^\s*\d{0,4}[A-Z]{0,2}\s*(\(\s*(CONTINUED|CONT'?D|MORE|continued|more)\s*\)|CONTINUED\s*:?|-+\s*MORE\s*-+)"
    r"\s*(\(\d+\))?\s*\d{0,4}[A-Z]{0,2}\s*$"
)
_OMITTED_RE = re.compile(r"^\s*\d{0,4}[A-Z]{0,3}\s*(OMITTED|OMIT)\s*\d{0,4}[A-Z]{0,3}\s*$")
_REVISION_MARK_RE = re.compile(r"\\*\*+\\*")
_CUE_CONTD_RE = re.compile(r"\(\s*CONT(INUED|'?D|\.)?[.\s]*\)", re.I)
_STRAY_BACKSLASH_RE = re.compile(r"\\+")
_REVISION_RE = re.compile(
    r"\b(rev(ision|ised|s)?\.?|draft|pink|blue|yellow|green|goldenrod|buff|salmon|cherry)\b.{0,40}\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}", re.I
)
_SPEAKER_COLON_RE = re.compile(r"^\s*[A-Z][A-Za-z .'\-]{0,24}:\s+\S")
_CUE_EXT_RE = re.compile(r"\s*\([^()]*\)\s*$")
_CUE_CHARS_RE = re.compile(r"^[A-Z0-9#'\"][A-Za-z0-9 .,'\-&#/\"]*$")
_HONORIFIC_END_RE = re.compile(r"\b(MR|MRS|MS|DR|JR|SR|ST|LT|SGT|CAPT|COL|GEN|PROF|REV|NO)\.$")
_NOT_A_NAME = {"THE END", "CONTINUED", "MORE", "OMITTED", "BLACK", "SILENCE", "CREDITS", "TITLE", "SUPER", "INSERT",
               "FADE IN", "FADE OUT", "LATER", "CONTINUOUS", "MONTAGE", "FLASHBACK", "END FLASHBACK", "BACK TO SCENE",
               "END OF MONTAGE", "INTERCUT", "DAY", "NIGHT"}


def decode(body: bytes) -> str:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        text = body.decode("cp1252", errors="replace")
    if "\u00e2\u20ac" in text:  # UTF-8 read as cp1252 ("â€™")
        try:
            text = text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return text


def html_to_text(page: str) -> str:
    """Script text from an HTML page: the longest <pre> block if it holds the script, else <br>/<p> text."""
    pres = re.findall(r"<pre[^>]*>(.*?)</pre>", page, re.S | re.I)
    body = max(pres, key=len) if pres else ""
    if len(body) < 5000:
        m = re.search(r"<td class=\"?scrtext\"?[^>]*>(.*?)</td>", page, re.S | re.I)
        body = m.group(1) if m else re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", "", page)
        body = re.sub(r"(?i)<br\s*/?>", "\n", body)
        body = re.sub(r"(?i)</p>|<p[^>]*>", "\n\n", body)
    body = re.sub(r"<[^>]+>", "", body)
    return html.unescape(body).replace("\u00a0", " ")


def pdf_to_text(body: bytes) -> tuple[str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "in.pdf")
        with open(path, "wb") as f:
            f.write(body)
        try:
            text = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", path, "-"], capture_output=True, timeout=120).stdout
            info = subprocess.run(["pdfinfo", path], capture_output=True, timeout=60).stdout.decode("utf-8", "replace")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "", {"pdf_error": True}
    meta = dict(re.findall(r"^(\w[\w ]*?):\s+(.*)$", info, re.M))
    pages = int(meta.get("Pages", "0") or 0)
    return text.decode("utf-8", "replace"), {"pages": pages, "producer": meta.get("Producer", ""), "creator": meta.get("Creator", "")}


def extract_text(body: bytes, fmt: str) -> tuple[str, dict]:
    if fmt == "pdf":
        return pdf_to_text(body)
    text = decode(body)
    if fmt == "html" or re.search(r"<(html|pre|body)\b", text[:5000], re.I):
        text = html_to_text(text)
    return text, {}


def _is_noise_line(s: str, prev_blank: bool = True) -> bool:
    """Page furniture. A bare number is a page number only at a page break or far right; inside a dialogue
    block (directly under a cue or another dialogue line) it is dialogue."""
    if _PAGE_NO_RE.match(s):
        return prev_blank or len(s) - len(s.lstrip()) > 55
    return bool(
        _CONTINUED_RE.match(s) or _OMITTED_RE.match(s)
        or (len(s.strip()) < 80 and _REVISION_RE.search(s) and not re.search(r"[a-z]{3,} [a-z]{3,} [a-z]{3,}", s))
        or (prev_blank and re.fullmatch(r"\s*([_=~]+|[-.]{5,})\s*", s))
    )


def _strip_marks(line: str) -> str:
    """Revision asterisks (with any escaping backslash) and stray backslashes, keeping column positions."""
    line = _REVISION_MARK_RE.sub(lambda m: " " * len(m.group()), line)
    return _STRAY_BACKSLASH_RE.sub(lambda m: " " * len(m.group()), line)


def drop_orphan_cues(elements: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int]:
    """A cue whose dialogue is missing is removed rather than turned into an action line."""
    out, dropped = [], 0
    for i, (tag, text) in enumerate(elements):
        nxt = elements[i + 1][0] if i + 1 < len(elements) else None
        if tag == "character" and nxt not in ("dialogue", "parenthetical"):
            dropped += 1
            continue
        out.append((tag, text))
    return out, dropped


def heading_text(line: str) -> str | None:
    s = line.strip().rstrip("*").strip()
    m = HEADING_LINE_RE.match(s)
    if not m:
        return None
    text = m.group(2)
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return None
    if m.group(1) or _TRAILING_SCENE_NO_RE.search(text):
        text = _TRAILING_SCENE_NO_RE.sub("", text)
    return text.strip()


def is_transition(s: str) -> bool:
    s = s.strip()
    return bool(s) and s.upper() == s and bool(TRANSITION_LINE_RE.match(s))


def cue_name(s: str) -> str:
    name = s.strip()
    for _ in range(2):
        name = _CUE_EXT_RE.sub("", name)
    return name.strip()


def looks_like_cue(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 50 or heading_text(s) or is_transition(s) or CAMERA_RE.match(s):
        return False
    name = cue_name(s)
    if not name or len(name) > 35 or len(name.split()) > 5 or name in _NOT_A_NAME:
        return False
    if name[-1] in ":!?,;" or (name.endswith(".") and not _HONORIFIC_END_RE.search(name)):
        return False
    if not _CUE_CHARS_RE.match(name):
        return False
    letters = [c for c in name if c.isalpha()]
    return len(letters) >= 2 and sum(c.isupper() for c in letters) / len(letters) >= 0.8


def _mode(xs, default=0):
    return Counter(xs).most_common(1)[0][0] if xs else default


def infer_layout(lines: list[str]) -> dict:
    indents = [(len(l) - len(l.lstrip()), l.strip()) for l in lines]
    cue_ind, dlg_ind = [], []
    for i, (ind, s) in enumerate(indents[:-1]):
        nxt_ind, nxt = indents[i + 1]
        if s and nxt and looks_like_cue(s):
            cue_ind.append(ind)
            if not nxt.startswith("("):
                dlg_ind.append(nxt_ind)
    action_ind = _mode([ind for ind, s in indents if len(s) > 50], 0)
    cue, dlg = _mode(cue_ind, 0), _mode(dlg_ind, 0)
    indented = len(cue_ind) >= 20 and cue >= action_ind + 8 and dlg >= action_ind + 4
    return {"layout": "indented" if indented else "flush", "action_indent": action_ind, "cue_indent": cue,
            "dialogue_indent": dlg, "cue_candidates": len(cue_ind)}


def text_to_scenes(text: str, detok: bool = True) -> tuple[list[Scene], dict]:
    """Parse screenplay text into cleaned scenes (content before the first scene heading is dropped)."""
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n").expandtabs(8).split("\n")
    lines, prev_blank = [], True
    for raw in raw_lines:
        l = _strip_marks(raw).rstrip()
        lines.append("" if l.strip() and _is_noise_line(l, prev_blank) else l)
        prev_blank = not l.strip()
    lay = infer_layout(lines)
    indented = lay["layout"] == "indented"
    a_ind, c_ind, d_ind = lay["action_indent"], lay["cue_indent"], lay["dialogue_indent"]
    cue_min = (d_ind + c_ind) / 2 - 2 if indented else 0

    def ind(l):
        return len(l) - len(l.lstrip())

    scenes_raw: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] | None = None
    para: list[str] = []
    pre_heading_lines = 0

    def flush_para():
        nonlocal para
        if para and cur is not None:
            cur.append(("scene_description", " ".join(para)))
        para = []

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()
        if not s:
            flush_para()
            i += 1
            continue
        h = heading_text(s)
        if h:
            flush_para()
            cur = [("stage_direction", h)]
            scenes_raw.append(cur)
            i += 1
            continue
        if cur is None:
            pre_heading_lines += 1
            i += 1
            continue
        if is_transition(s):
            flush_para()
            cur.append(("scene_description", s))
            i += 1
            continue
        nxt_i = i + 1
        if indented and nxt_i < n and not lines[nxt_i].strip() and nxt_i + 1 < n:
            # tolerate one blank line between a cue and its dialogue in indented files
            cand = lines[nxt_i + 1]
            if cand.strip() and a_ind + 3 < ind(cand) < c_ind - 2:
                nxt_i += 1
        nxt = lines[nxt_i] if nxt_i < n else ""
        is_cue = (
            looks_like_cue(s) and nxt.strip() != "" and not heading_text(nxt.strip())
            and (ind(line) >= cue_min if indented else not para)
            and (not indented or ind(nxt) > a_ind + 2 or nxt.strip().startswith("("))
        )
        if not is_cue:
            para.append(s)
            i += 1
            continue
        flush_para()
        cur.append(("character", s))
        i = nxt_i
        dlg: list[str] = []
        paren: list[str] = []
        while i < n:
            l2 = lines[i]
            t = l2.strip()
            if not t or heading_text(t) or is_transition(t) or (indented and ind(l2) <= a_ind + 2 and len(t) > 0):
                break
            if indented and ind(l2) >= cue_min and looks_like_cue(t) and dlg:
                break
            if paren or t.startswith("("):
                if dlg and not paren:
                    cur.append(("dialogue", " ".join(dlg)))
                    dlg = []
                paren.append(t)
                if ")" in t:
                    cur.append(("parenthetical", " ".join(paren)))
                    paren = []
            else:
                dlg.append(t)
            i += 1
        if paren:
            cur.append(("parenthetical", " ".join(paren)))
        if dlg:
            cur.append(("dialogue", " ".join(dlg)))
    flush_para()

    scenes, orphan_cues = [], 0
    for idx, elements in enumerate(scenes_raw):
        cleaned = []
        for tag, t in elements:
            if tag == "character":
                t = _CUE_CONTD_RE.sub(" ", t)
            t = clean_text(t, detok=detok)
            # dialogue is real text however short ("More.", "...", "478."); page furniture was removed per line
            if (tag in ("dialogue", "parenthetical") and t.strip()) or not is_junk_element(t):
                cleaned.append((tag, t))
        cleaned, dropped = drop_orphan_cues(cleaned)
        orphan_cues += dropped
        scenes.append(Scene(idx, cleaned))
    speakers = {speaker_name(t) for s in scenes for tag, t in s.elements if tag == "character"}
    for s in scenes:
        keep = []
        for tag, t in s.elements:
            if tag == "scene_description" and len(t) < 30 and t.upper() == t and speaker_name(t.rstrip(".:")) in speakers:
                orphan_cues += 1
                continue
            keep.append((tag, t))
        s.elements = keep
    heading_only = [s.index for s in scenes if not s.has_body()]
    scenes = [s for s in scenes if s.has_body()]
    diag = {**lay, "headings": len(scenes_raw), "scenes_with_body": len(scenes), "pre_heading_lines": pre_heading_lines,
            "orphan_cues_dropped": orphan_cues, "heading_only_scenes": heading_only,
            "nonblank_lines": sum(1 for l in lines if l.strip()),
            "speaker_colon_lines": sum(1 for l in lines if _SPEAKER_COLON_RE.match(l))}
    return scenes, diag
