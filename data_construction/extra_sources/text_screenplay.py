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

from cml_format import CAMERA_RE, Scene, clean_text, is_junk_element, speaker_name, strip_page_furniture

PARSER_VERSION = "text_screenplay_v1"

_HEADING_CORE = r"(?:INT|EXT|INT\.?\s*/\s*EXT|EXT\.?\s*/\s*INT|I\s*/\s*E|E\s*/\s*I|INTERIOR|EXTERIOR|EST)"
HEADING_LINE_RE = re.compile(rf"^(?:(\d{{1,4}}[A-Z]{{0,3}})[.)]?\s+)?({_HEADING_CORE}(?:[.:\-\s].*)?)$")
# Location sluglines without INT./EXT. ("THE HIGHWAY - DAY"); MovieSum keeps these as stage directions too.
_SLUG_TOD_RE = re.compile(
    r"^(?:\d{1,4}[A-Z]{0,3}[.)]?\s+)?([A-Z0-9][A-Z0-9 .,'&/()\-]{2,60}?\s*(?:-|--|\u2013|\u2014)\s*"
    r"(?:DAY|NIGHT|DAWN|DUSK|MORNING|AFTERNOON|EVENING|SUNSET|SUNRISE|LATER|CONTINUOUS|SAME|MOMENTS LATER|SAME TIME)\.?)$"
)
_TOD_END_RE = re.compile(r"(DAY|NIGHT|DAWN|DUSK|MORNING|AFTERNOON|EVENING|SUNSET|SUNRISE|LATER|CONTINUOUS|SAME|FLASHBACK)\W*$")
_TRAILING_SCENE_NO_RE = re.compile(r"\s+\d{1,4}[A-Z]{0,3}\.?\s*$")
TRANSITION_LINE_RE = re.compile(
    r"^((FADE|IRIS) (IN|OUT|UP|TO BLACK|TO WHITE)|((QUICK|SLOW|MATCH|SMASH|JUMP|HARD|STRAIGHT|FLASH) )?CUT( BACK)?( TO)?|"
    r"(QUICK |SLOW |RIPPLE |LAP )?DISSOLVE( TO)?|WIPE( TO)?|CUT TO BLACK|BACK TO( SCENE)?|INTERCUT( WITH)?|THE END|END CREDITS)\b[\s\w.:]*$"
)
# Page furniture. Only the typographic forms are removed ("(MORE)", "CONTINUED:", a page number on its own
# line at a page break or the right margin), so dialogue such as "More." or "478." is never dropped.
_PAGE_NO_RE = re.compile(r"^\s*(page\s*)?[-\u2013\[]?\s*\d{1,3}\s*[-\u2013\]]?[.)]?\s*(of\s+\d{1,3}\s*)?$", re.I)
# Scene numbers around it ("12A", "A12") always hold a digit, so "JO (CONT'D)" stays a cue.
_SCENE_NO = r"([A-Z]?\d{1,4}[A-Z]{0,2})?"
_CONTINUED_RE = re.compile(
    rf"^\s*{_SCENE_NO}\s*(\(\s*(CONTINUED|CONT'?D|MORE|continued|more)\s*\)|CONTINUED\s*:?|-+\s*MORE\s*-+)"
    rf"\s*(\(\d+\))?\s*{_SCENE_NO}\s*$"
)
_SCENE_NOTE_RE = re.compile(r"^\s*SCENES?\s+\d+[A-Z]?(\s*(-|TO|AND|THRU|THROUGH)\s*\d+[A-Z]?)?(\s+(INCORPORATED INTO|MOVED TO|"
                            r"COMBINED WITH|DELETED|OMITTED)(\s+SCENES?\s+\d+[A-Z]?)?)?\s*$")
_DUAL_CUE_RE = re.compile(r"^(\s*)(\S(?:.*?\S)?)\s{4,}(\S(?:.*?\S)?)\s*$")
_DOUBLED_SCENE_NO_RE = re.compile(r"^(\d{1,3}[A-Z]?)\s+\1\s+(?=\S)")
_OMITTED_RE = re.compile(rf"^\s*{_SCENE_NO}\s*(OMITTED|OMIT)\s*{_SCENE_NO}\s*$")
_REVISION_MARK_RE = re.compile(r"\\*\*+\\*")
_CUE_CONTD_RE = re.compile(r"\(\s*CONT(INUED|['\u2019]?D|\.)?[.\s]*\)", re.I)
_STRAY_BACKSLASH_RE = re.compile(r"\\+")
_REVISION_RE = re.compile(
    r"\b(rev(ision|ised|s)?\.?|draft|pink|blue|yellow|green|goldenrod|buff|salmon|cherry)\b.{0,40}"
    r"(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\(mm/dd/yy\))", re.I
)
_SPEAKER_COLON_RE = re.compile(r"^\s*[A-Z][A-Za-z .'\-]{0,24}:\s+\S")
_CUE_EXT_RE = re.compile(r"\s*\([^()]*\)\s*$")
_CUE_DASH_RE = re.compile(r"^[-\u2013]\s+")
_CUE_VOICE_RE = re.compile(r"\s+[VO]\.\s?[SOC0]\.?$")  # unbracketed "V.O.", "O.S.", "O.C." (and the "V.0." typo)
_INITIALISM_END_RE = re.compile(r"(^|\s)([A-Z]\.){1,3}$")  # "M.E.", "STORE P.A."
_CUE_CHARS_RE = re.compile(r"^[A-Z0-9#'\"][A-Za-z0-9 .,'\u2019\-&#/\"]*$")
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
        _CONTINUED_RE.match(s) or _OMITTED_RE.match(s) or _SCENE_NOTE_RE.match(s)
        or (len(s.strip()) < 80 and _REVISION_RE.search(s) and not re.search(r"[a-z]{3,} [a-z]{3,} [a-z]{3,}", s))
        or (prev_blank and re.fullmatch(r"\s*([_=~]+|[-.]{5,})\s*", s))
    )


def _header_key(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().lower()))


def _strip_glued_header(line: str, headers: set[str]) -> str:
    """A running header printed on the same line as text, after a wide gap ("So I-     Blue Rev. (mm/dd/yy)  3.")."""
    if not headers:
        return line
    m = re.match(r"^(.*?\S)\s{3,}(\S.*?)\s*$", line)
    if m and _header_key(m.group(2)) in headers:
        return m.group(1)
    m = re.match(r"^(\s*)(\S.*?)\s{3,}(\S.*)$", line)
    if m and _header_key(m.group(2)) in headers:
        return m.group(1) + " " * len(m.group(2)) + "   " + m.group(3)
    return line


def repeated_page_headers(lines: list[str], min_repeats: int = 5) -> set[str]:
    """Page headers/footers printed on every page ('"Ricky Stanicky" 11.13.09 Bushell 93.', 'Deceptions by Richard
    Taylor 32.', 'WONDERSTRUCK 2.', 'WRECK-IT RALPH 62'): lines carrying a number whose text, with numbers masked,
    repeats >= min_repeats times. Scene headings and transitions never qualify. Short all-caps lines (shot slugs
    "151 SARK 151", cues) only qualify as a title with a trailing page number that takes >= 3 values."""
    counts: dict[str, int] = {}
    pages: dict[str, set] = {}
    for l in lines:
        s = l.strip()
        if len(s) < 6 or not re.search(r"\d", s) or heading_text(s) or is_transition(s):
            continue
        letters = [c for c in s if c.isalpha()]
        shot_or_cue = len(s.split()) <= 6 and letters and sum(c.isupper() for c in letters) / len(letters) >= 0.9
        k = _header_key(s)
        title_page = re.fullmatch(r"[^\d#]*[a-z][^#]*\s#\.?", k)  # "wonderstruck #." : no number but the trailing page
        if (looks_like_cue(s) or shot_or_cue) and not title_page:
            continue  # numbered shot headings ("151 SARK 151") repeat too, but they are content
        if len(re.sub(r"[^a-z]", "", k)) >= 4:
            counts[k] = counts.get(k, 0) + 1
            pages.setdefault(k, set()).add(re.findall(r"\d+", s)[-1])
    return {k for k, c in counts.items() if c >= min_repeats and len(pages[k]) >= 3}


def _strip_marks(line: str) -> str:
    """Revision asterisks (with any escaping backslash) and stray backslashes, keeping column positions."""
    line = _REVISION_MARK_RE.sub(lambda m: " " * len(m.group()), line)
    return _STRAY_BACKSLASH_RE.sub(lambda m: " " * len(m.group()), line)


def drop_orphan_cues(elements: list[tuple[str, str]], talkers: set[str] | None = None) -> tuple[list[tuple[str, str]], int]:
    """A cue whose dialogue is missing is removed (contract C12c) when that name speaks elsewhere in the script;
    otherwise it was an uppercase action line taken for a cue ("EXPLOSION OF COLORS") and stays as action."""
    out, dropped = [], 0
    for i, (tag, text) in enumerate(elements):
        nxt = elements[i + 1][0] if i + 1 < len(elements) else None
        if tag == "character" and nxt not in ("dialogue", "parenthetical"):
            if talkers is not None and speaker_name(text) not in talkers:
                out.append(("scene_description", text))
                continue
            dropped += 1
            continue
        out.append((tag, text))
    return out, dropped


def heading_text(line: str) -> str | None:
    s = line.strip().rstrip("*").strip()
    m = HEADING_LINE_RE.match(s)
    if not m:
        slug = _SLUG_TOD_RE.match(_TRAILING_SCENE_NO_RE.sub("", s))
        return slug.group(1).strip() if slug and not CAMERA_RE.match(slug.group(1)) else None
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
    """'JAMIE (CONT'D).' / '- LEWIS' / 'SARK V.0.' -> 'JAMIE' / 'LEWIS' / 'SARK'"""
    name = re.sub(r"\)\s*\.$", ")", _CUE_DASH_RE.sub("", s.strip()))
    for _ in range(2):
        name = _CUE_EXT_RE.sub("", name)
    return _CUE_VOICE_RE.sub("", name).strip()


def looks_like_cue(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 50 or heading_text(s) or is_transition(s) or CAMERA_RE.match(s):
        return False
    name = cue_name(s)
    if not name or len(name) > 35 or len(name.split()) > 5 or name in _NOT_A_NAME:
        return False
    if name[-1] in ":!?,;" or (name.endswith(".") and not _HONORIFIC_END_RE.search(name)
                               and not (_INITIALISM_END_RE.search(name) and not re.search(r"\d", name))):
        return False
    if not _CUE_CHARS_RE.match(name):
        return False
    letters = [c for c in name if c.isalpha()]
    return len(letters) >= 2 and sum(c.isupper() for c in letters) / len(letters) >= 0.8


def join_lines(parts: list[str]) -> str:
    """Join wrapped lines; a single hyphen at a line end before a lowercase word is a wrap ("black-and-" + "white")."""
    out = ""
    for p in parts:
        if out.endswith("-") and not out.endswith("--") and out[-2:-1].isalpha() and p[:1].islower():
            out += p
        else:
            out = f"{out} {p}" if out else p
    return out


def _mode(xs, default=0):
    return Counter(xs).most_common(1)[0][0] if xs else default


def is_dual_cue(line: str) -> bool:
    m = _DUAL_CUE_RE.match(line)
    return bool(m and looks_like_cue(m.group(2)) and looks_like_cue(m.group(3)))


def infer_layout(lines: list[str]) -> dict:
    indents = [(len(l) - len(l.lstrip()), l.strip()) for l in lines]
    cue_ind, dlg_ind = [], []
    for i, (ind, s) in enumerate(indents[:-1]):
        nxt_ind, nxt = indents[i + 1]
        if s and nxt and looks_like_cue(s) and not is_dual_cue(s):
            cue_ind.append(ind)
            if not nxt.startswith("("):
                dlg_ind.append(nxt_ind)
    action_ind = _mode([ind for ind, s in indents if len(s) > 50], 0)
    cue, dlg = _mode(cue_ind, 0), _mode(dlg_ind, 0)
    indented = len(cue_ind) >= 20 and cue >= action_ind + 8 and dlg >= action_ind + 4
    known = {cue_name(s) for i, (ind, s) in enumerate(indents[:-1])
             if s and indents[i + 1][1] and looks_like_cue(s) and not is_dual_cue(s) and indents[i + 1][0] >= action_ind + 4}
    return {"layout": "indented" if indented else "flush", "action_indent": action_ind, "cue_indent": cue,
            "dialogue_indent": dlg, "cue_candidates": len(cue_ind), "known_cues": known}


def text_to_scenes(text: str, detok: bool = True) -> tuple[list[Scene], dict]:
    """Parse screenplay text into cleaned scenes (content before the first scene heading is dropped)."""
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n").expandtabs(8).split("\n")
    headers = repeated_page_headers(raw_lines)
    lines, prev_blank = [], True
    for raw in raw_lines:
        l = _strip_marks(raw).rstrip()
        l = _strip_glued_header(l, headers)
        lines.append("" if l.strip() and (_is_noise_line(l, prev_blank) or _header_key(l) in headers) else l)
        prev_blank = not l.strip()
    lay = infer_layout(lines)
    indented = lay["layout"] == "indented"
    a_ind, c_ind, d_ind = lay["action_indent"], lay["cue_indent"], lay["dialogue_indent"]
    cue_min = (d_ind + c_ind) / 2 - 2 if indented else 0
    # cues indented but dialogue at the action indent (common in IMSDb HTML): cues by indent, dialogue by blank lines
    cue_by_indent = indented or (lay["cue_candidates"] >= 20 and c_ind >= a_ind + 8)

    def ind(l):
        return len(l) - len(l.lstrip())

    scenes_raw: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] | None = None
    para: list[str] = []
    pre_heading_lines = 0

    def flush_para():
        nonlocal para
        if para and cur is not None:
            cur.append(("scene_description", join_lines(para)))
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
            # a heading broken over two lines: "INT. DRUG STORE, GARY, INDIANA - PHONEBOOTH -" / "DILLINGER - NIGHT"
            nxt2 = lines[i + 1].strip() if i + 1 < n else ""
            if (nxt2 and not _TOD_END_RE.search(h) and nxt2.upper() == nxt2 and len(nxt2) <= 60
                    and _TOD_END_RE.search(_TRAILING_SCENE_NO_RE.sub("", nxt2)) and not HEADING_LINE_RE.match(nxt2)):
                h = f"{h.rstrip(' -,')} - {heading_text(nxt2) or _TRAILING_SCENE_NO_RE.sub('', nxt2).strip()}"
                i += 1
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
        dual = _DUAL_CUE_RE.match(line)
        if is_dual_cue(line) and i + 1 < n and lines[i + 1].strip():
            # two-column (dual) dialogue: "BARNES          TAYLOR" / "Co --           Copy that."
            flush_para()
            col2 = line.index(dual.group(3), len(dual.group(1)) + len(dual.group(2)))
            sides = ([], [])
            i += 1
            while i < n and lines[i].strip() and not heading_text(lines[i].strip()) and not is_transition(lines[i].strip()):
                l2 = lines[i].rstrip()
                chunks = [c for c in re.split(r"\s{3,}", l2.strip()) if c]
                if len(chunks) >= 2:
                    sides[0].append(chunks[0])
                    sides[1].append(" ".join(chunks[1:]))
                else:
                    sides[1 if ind(l2) >= col2 - 6 else 0].append(chunks[0])
                i += 1
            for name, side in zip((dual.group(2), dual.group(3)), sides):
                if not side:
                    continue
                cur.append(("character", name))
                speech = []
                for t in side:
                    if t.startswith("(") and t.endswith(")"):
                        if speech:
                            cur.append(("dialogue", join_lines(speech)))
                            speech = []
                        cur.append(("parenthetical", t))
                    else:
                        speech.append(t)
                if speech:
                    cur.append(("dialogue", join_lines(speech)))
            continue
        nxt_i = i + 1
        if nxt_i < n and not lines[nxt_i].strip():
            # blank lines between a cue and its parenthetical (any layout) or indented dialogue: one blank line, or a
            # page break (several blank lines where the page furniture was removed)
            j = nxt_i
            while j < n and j - nxt_i < 12 and not lines[j].strip():
                j += 1
            cand = lines[j] if j < n else ""
            if (j - nxt_i == 1 or not cand.strip()[:1].isupper() or indented) and (
                    cand.strip().startswith("(") or (indented and cand.strip() and a_ind + 3 < ind(cand) < c_ind - 2)):
                nxt_i = j
        nxt = lines[nxt_i] if nxt_i < n else ""
        # a known speaker at the cue indent whose line (or parenthetical) sits at the action indent: the dialogue then
        # runs to a blank line ("LOIS" / "(pre-occupied)" / "Uh-huh ..." all at column 0 under a tab-indented cue)
        flat_dialogue = (indented and ind(line) >= cue_min and ind(nxt) <= a_ind + 2
                         and (nxt.strip().startswith("(") or cue_name(s) in lay["known_cues"]))
        is_cue = (
            looks_like_cue(s) and nxt.strip() != "" and not heading_text(nxt.strip()) and not is_transition(nxt.strip())
            and (ind(line) >= cue_min if indented else (not para and (not cue_by_indent or ind(line) >= c_ind - 4)))
            and (not indented or ind(nxt) > a_ind + 2 or nxt.strip().startswith("(") or flat_dialogue)
        )
        if not is_cue:
            if not _PAGE_NO_RE.match(s):  # a bare number inside action is a page number (in dialogue it is speech)
                para.append(s)
            i += 1
            continue
        flush_para()
        cur.append(("character", re.sub(r"\)\s*\.$", ")", _CUE_DASH_RE.sub("", s))))
        i = nxt_i
        dlg: list[str] = []
        paren: list[str] = []
        while i < n:
            l2 = lines[i]
            t = l2.strip()
            if not t and not dlg and not paren and cur[-1][0] == "parenthetical" and i + 1 < n:
                # "CUE / (paren) / blank / line": the line after the blank is still this speaker's dialogue
                t2 = lines[i + 1].strip()
                if t2 and not heading_text(t2) and not is_transition(t2) and not looks_like_cue(t2):
                    i += 1
                    continue
            # "(a beat)" at the left margin between two lines of indented dialogue is still a parenthetical
            margin_paren = (t.startswith("(") and t.endswith(")") and len(t) <= 40 and i + 1 < n and lines[i + 1].strip()
                            and ind(lines[i + 1]) > a_ind + 2 and not looks_like_cue(lines[i + 1].strip()))
            if not t or heading_text(t) or is_transition(t) or (
                    indented and not flat_dialogue and ind(l2) <= a_ind + 2 and len(t) > 0 and not margin_paren):
                break
            if indented and ind(l2) >= cue_min and looks_like_cue(t) and dlg:
                break
            if paren or t.startswith("("):
                if dlg and not paren:
                    cur.append(("dialogue", join_lines(dlg)))
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
            cur.append(("dialogue", join_lines(dlg)))
    flush_para()

    cleaned_scenes = []
    for elements in scenes_raw:
        cleaned = []
        for tag, t in elements:
            if tag == "character":
                t = _CUE_CONTD_RE.sub(" ", t)
            t = clean_text(t, detok=detok)
            # dialogue is real text however short ("More.", "...", "478."); page furniture was removed per line
            if (tag in ("dialogue", "parenthetical") and t.strip()) or not is_junk_element(t):
                cleaned.append((tag, t))
        cleaned_scenes.append(cleaned)
    cleaned_scenes = [[(tag, _DOUBLED_SCENE_NO_RE.sub("", t) if tag == "scene_description" else t) for tag, t in els if t.strip()]
                      for els in strip_page_furniture(cleaned_scenes)]
    talkers = {speaker_name(t) for els in cleaned_scenes for i, (tag, t) in enumerate(els)
               if tag == "character" and i + 1 < len(els) and els[i + 1][0] in ("dialogue", "parenthetical")}
    scenes, orphan_cues = [], 0
    for idx, cleaned in enumerate(cleaned_scenes):
        cleaned, dropped = drop_orphan_cues(cleaned, talkers)
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
