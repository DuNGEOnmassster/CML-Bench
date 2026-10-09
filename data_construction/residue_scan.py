"""Count cleaning residue in a segments.jsonl or release jsonl (the evaluator's patterns plus C12b/C12c/C13b).

  python data_construction/residue_scan.py data_construction/work/build/segments.jsonl
"""
from __future__ import annotations

import json
import re
import sys

PATTERNS = {
    "ptb_brackets": r"-[LR][RSC]B-",
    "asterisk": r"\*",
    "backslash": r"\\",
    "cont_d_marker": r"\(?\bCONT\s*['\u2019]?\s*D\b\.?\)?|\(?\bCONTINUED\b\)?|\bOMITTED\b",
    "quote_inner_space": r'(?:^|[\s>])" [A-Za-z]|[a-z.!?,] "(?=[\s<])',
    "leading_dot": r">\. [A-Za-z\"']",
    "ptb_nt_s": r"\w (n't|'s)\b",
    "space_before_punct": r"\w [,.;:!?](?=\s|<)",
    "noise_char": r"[\u2022\u25a0\u25aa\u00b7\u25cf\u25a1\u2023\u2043]",
    "html_entity": r"&amp;(amp|quot|lt|gt|#\d+);",
}
HEADING_NO_RE = re.compile(
    r"<stage_direction>(\d{1,3}[A-Z]{0,2}\s+(INT|EXT|I/E)\b[^<]*|[^<]*\b(DAY|NIGHT|MORNING|EVENING|AFTERNOON|DAWN|DUSK|LATER|"
    r"CONTINUOUS|SUNSET|SUNRISE)\s+\d{1,3}[A-Z]{0,2})</stage_direction>"
)
SCENE_DESC_RE = re.compile(r"<scene_description>([^<]{2,29})</scene_description>")
CHAR_RE = re.compile(r"<character>([^<]+)</character>")
SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*|\s+(V\.?O\.?|O\.?S\.?|O\.?C\.?|CONT'?D\.?)\s*$", re.I)


def speaker(t: str) -> str:
    return SUFFIX_RE.sub(" ", t).strip().upper()


def orphan_speaker_lines(content: str) -> int:
    speakers = {speaker(c) for c in CHAR_RE.findall(content)}
    return sum(1 for d in SCENE_DESC_RE.findall(content) if d.strip().upper() in speakers and d.strip().isupper())


def scan(rows) -> dict:
    out = {k: [0, 0] for k in list(PATTERNS) + ["orphan_speaker_line", "heading_scene_number"]}
    headings = 0
    for r in rows:
        c = r["script_segment"]
        for k, p in PATTERNS.items():
            n = len(re.findall(p, c))
            out[k][0] += n
            out[k][1] += n > 0
        n = orphan_speaker_lines(c)
        out["orphan_speaker_line"][0] += n
        out["orphan_speaker_line"][1] += n > 0
        n = len(HEADING_NO_RE.findall(c))
        out["heading_scene_number"][0] += n
        out["heading_scene_number"][1] += n > 0
        headings += c.count("<stage_direction>")
    return {"items": len(rows), "headings": headings, "residue": {k: {"total": v[0], "items": v[1]} for k, v in out.items()}}


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print(json.dumps(scan(rows), indent=1))
