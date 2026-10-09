"""Stage 4: automated checks for agent-written abstracts (mechanical + lexical proxies, not a quality judge).

  python check_abstracts.py --run_dir RUN            # all items of a run -> RUN/abstract_checks.jsonl + summary
  python check_abstracts.py --run_dir RUN --batch B  # only the items of one batch manifest
  python check_abstracts.py --gt_baseline GT.json    # same metrics on CML-Bench GT summaries (calibration)

Hard failures make the item ineligible for assembly; soft flags are reported for the evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cml_format import parse_script, speaker_name  # noqa: E402

HARD_MIN_WORDS, HARD_MAX_WORDS = 90, 300
MAX_PARAGRAPHS = 3
META_RE = re.compile(
    r"^\s*(here is|here's|summary\s*:|abstract\s*:)|\b(this|the) (excerpt|segment|screenplay)\b|\bin this (scene|script|excerpt|segment)\b|"
    r"\bthe script\b|\bas an ai\b|\bthe scene (then )?shifts\b",
    re.I,
)
LIST_RE = re.compile(r"^\s*([-*\u2022#>]|\d+[.)])\s", re.M)
MARKDOWN_RE = re.compile(r"\*\*|__|`|^#", re.M)
STOPWORDS = set(
    """a about above after again against all also am an and any are as at be because been before being below between both but by
    can could did do does doing down during each few for from further had has have having he her here hers herself him himself his
    how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours out over own
    same she should so some such than that the their theirs them themselves then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you your yours yourself""".split()
)
SENTENCE_STARTERS = set(
    """The He She They It His Her Their When As After Before Later Meanwhile Back In At On Inside Outside Upon Then Once While With
    Without Despite Although Though Elsewhere Soon Finally Eventually Next Afterward Afterwards Suddenly That This These Those
    There Here Now Still Yet But And Or So Both Each Every Some One Two Three Several Many Most Another Other During Over Under
    Across Along Through From Into Onto Near Far Alone Together Unable Hoping Determined Frustrated Realizing Left Following
    Moments Hours Days Years Night Morning Evening Afternoon Dawn Dusk Day By For To Of A An If Not No Even Only Just Almost Its
    Mr Mrs Ms Dr Miss Sir Lady Lord Captain Officer Detective Agent Sergeant Doctor Professor Father Mother Uncle Aunt""".split()
)
NAME_TITLES = {"MR", "MRS", "MS", "DR", "MISS", "SIR", "THE", "OLD", "YOUNG", "LITTLE", "BIG", "OFFICER", "DETECTIVE", "CAPTAIN",
               "AGENT", "SERGEANT", "SGT", "LT", "COLONEL", "GENERAL", "DOCTOR", "PROFESSOR", "FATHER", "MOTHER", "UNCLE", "AUNT",
               "JR", "SR", "MAN", "WOMAN", "GIRL", "BOY", "GUY", "VOICE", "LADY", "KID"}
ACRONYMS_OK = re.compile(r"^(FBI|CIA|NSA|USA|US|UK|TV|DNA|NYPD|LAPD|CEO|DJ|OK|AM|PM|IRS|SWAT|NASA|KGB|MI6|ER|ICU|II|III|IV|VIP|AI)$")


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", text)


def stem(w: str) -> str:
    w = w.lower().replace("\u2019", "'")
    w = re.sub(r"'s$|'$", "", w)
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def content_stems(text: str) -> set[str]:
    return {stem(w) for w in words(text) if len(w) >= 4 and w.lower() not in STOPWORDS}


def name_tokens(speaker: str) -> list[str]:
    return [t for t in re.findall(r"[A-Z][A-Z'\-]+", speaker.upper()) if len(t) >= 3 and t not in NAME_TITLES]


def check_one(abstract: str, content: str, target: list[int] | None = None) -> dict:
    hard, soft = [], []
    text = abstract.strip()
    wc = len(text.split())
    if not text:
        return {"hard": ["empty"], "soft": [], "words": 0}
    if not HARD_MIN_WORDS <= wc <= HARD_MAX_WORDS:
        hard.append("word_count_out_of_bounds")
    elif target and not target[0] <= wc <= target[1]:
        soft.append("word_count_outside_target")
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > MAX_PARAGRAPHS:
        hard.append("too_many_paragraphs")
    if LIST_RE.search(text) or MARKDOWN_RE.search(text):
        hard.append("markdown_or_list")
    if text[0] in "\"'\u201c" and text[-1] in "\"'\u201d":
        hard.append("wrapped_in_quotes")
    if META_RE.search(text):
        hard.append("meta_language")
    letters = [c for c in text if c.isalpha()]
    ascii_share = sum(c.isascii() for c in letters) / max(1, len(letters))
    ws = [w.lower() for w in words(text)]
    stop_share = sum(w in STOPWORDS for w in ws) / max(1, len(ws))
    if ascii_share < 0.97 or stop_share < 0.2:
        hard.append("not_english")

    scenes = parse_script(content, detok=False)
    elements = [(t, x) for s in scenes for t, x in s.elements]
    content_lower = content.lower()
    speakers = Counter(speaker_name(x) for t, x in elements if t == "character")
    top = [n for n, _ in speakers.most_common(3)]
    abstract_upper = text.upper()

    def mentioned(speaker):
        toks = name_tokens(speaker)
        return any(re.search(rf"\b{re.escape(t)}\b", abstract_upper) for t in toks) if toks else True

    top1_mentioned = mentioned(top[0]) if top else True
    top3_mentioned = sum(mentioned(s) for s in top)
    if not top1_mentioned:
        soft.append("top_speaker_missing")
    if len(top) >= 3 and top3_mentioned < 2:
        soft.append("few_top_speakers_mentioned")

    speaker_tokens = {t for s in speakers for t in name_tokens(s)}
    caps = [w for w in re.findall(r"\b[A-Z][A-Z'\-]{2,}\b", text) if not ACRONYMS_OK.match(w)]
    if any(w in speaker_tokens for w in caps):
        hard.append("all_caps_character_name")

    initial, inner = set(), set()
    for m in re.finditer(r"\b[A-Z][a-z][A-Za-z'\u2019\-]*", text):
        w = re.sub(r"['\u2019]s?$", "", m.group())
        before = text[: m.start()].rstrip(" \"'\u201c(")
        (initial if not before or before[-1] in ".!?:;\n" else inner).add(w)
    candidates = [w for w in inner | (initial & inner) if w not in SENTENCE_STARTERS and w.lower() not in STOPWORDS]
    ungrounded = sorted({w for w in candidates if w.lower() not in content_lower})
    if len(ungrounded) >= 3 or (candidates and len(ungrounded) / len(set(candidates)) > 0.25):
        hard.append("ungrounded_proper_nouns")
    elif ungrounded:
        soft.append("some_ungrounded_proper_nouns")

    a_stems = content_stems(text)
    c_stems = content_stems(" ".join(x for _, x in elements))
    grounding = len(a_stems & c_stems) / max(1, len(a_stems))
    if grounding < 0.4:
        soft.append("low_lexical_grounding")

    thirds = [scenes[i * len(scenes) // 3 : (i + 1) * len(scenes) // 3] for i in range(3)]
    third_stems = [content_stems(" ".join(x for s in part for _, x in s.elements)) for part in thirds]
    covered = []
    for i in range(3):
        distinctive = third_stems[i] - third_stems[(i + 1) % 3] - third_stems[(i + 2) % 3]
        covered.append(bool(a_stems & distinctive))
    if not covered[2]:
        soft.append("last_third_not_covered")
    if not covered[0]:
        soft.append("first_third_not_covered")

    return {
        "hard": hard,
        "soft": soft,
        "words": wc,
        "paragraphs": len(paragraphs),
        "top1_speaker_mentioned": top1_mentioned,
        "top3_speakers_mentioned": top3_mentioned,
        "ungrounded_proper_nouns": ungrounded,
        "lexical_grounding": round(grounding, 3),
        "thirds_covered": covered,
    }


def load_abstract(path: str, item_id: str) -> tuple[str | None, dict, list[str]]:
    if not os.path.exists(path):
        return None, {}, ["missing_abstract_file"]
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, {}, ["invalid_json"]
    problems = []
    if rec.get("item_id") != item_id:
        problems.append("item_id_mismatch")
    if not isinstance(rec.get("abstract"), str):
        problems.append("abstract_not_string")
        return None, rec, problems
    for key in ("prompt_version", "author"):
        if not rec.get(key):
            problems.append(f"missing_{key}")
    return rec["abstract"], rec, problems


def summarize(results: list[dict]) -> dict:
    n = len(results) or 1
    return {
        "items": len(results),
        "hard_pass": sum(not r["hard"] for r in results),
        "hard_failures": dict(Counter(h for r in results for h in r["hard"])),
        "soft_flags": dict(Counter(s for r in results for s in r["soft"])),
        "mean_words": round(sum(r.get("words", 0) for r in results) / n, 1),
        "top1_speaker_mentioned_rate": round(sum(r.get("top1_speaker_mentioned", False) for r in results) / n, 3),
        "all_thirds_covered_rate": round(sum(all(r.get("thirds_covered", [False])) for r in results) / n, 3),
        "mean_lexical_grounding": round(sum(r.get("lexical_grounding", 0) for r in results) / n, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir")
    ap.add_argument("--batch", help="path to a batch dir (checks only its items)")
    ap.add_argument("--gt_baseline", help="gt_100.json: compute metrics for the original GT summaries")
    args = ap.parse_args()

    if args.gt_baseline:
        with open(args.gt_baseline, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        results = [{"item_id": r["imdb_id"], **check_one(r["summary"], r["script_segment"])} for r in rows]
        print(json.dumps(summarize(results), indent=2))
        return

    if args.batch:
        with open(os.path.join(args.batch, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        entries = manifest["items"]
    else:
        with open(os.path.join(args.run_dir, "items.jsonl"), encoding="utf-8") as f:
            items = [json.loads(line) for line in f]
        from make_abstract_batches import target_words

        entries = [
            {
                "item_id": it["item_id"],
                "abstract_path": os.path.join(args.run_dir, "abstracts", f"{it['item_id']}.json"),
                "content": it["script_segment"],
                "target_words": target_words(it["content_tokens"]),
            }
            for it in items
        ]

    results = []
    for e in entries:
        content = e.get("content")
        if content is None:
            with open(e["content_path"], encoding="utf-8") as f:
                content = f.read()
        abstract, rec, problems = load_abstract(e["abstract_path"], e["item_id"])
        if abstract is None:
            res = {"hard": problems, "soft": [], "words": 0}
        else:
            res = check_one(abstract, content, e.get("target_words"))
            res["hard"] = problems + res["hard"]
        results.append({"item_id": e["item_id"], "author": rec.get("author"), **res})

    summary = summarize(results)
    if args.run_dir and not args.batch:
        with open(os.path.join(args.run_dir, "abstract_checks.jsonl"), "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(args.run_dir, "abstract_checks_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    for r in results:
        if r["hard"]:
            print(f"HARD FAIL {r['item_id']}: {r['hard']}")
    print(json.dumps(summary, indent=2))
    sys.exit(1 if summary["hard_pass"] < summary["items"] else 0)


if __name__ == "__main__":
    main()
