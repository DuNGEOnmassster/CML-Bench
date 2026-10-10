"""Render one item: the CML segment as a screenplay page, the raw markup, and the abstract with its checks."""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET

ELEMENT_CLASS = {
    "stage_direction": "sp-slug",
    "scene_description": "sp-action",
    "action": "sp-action",
    "character": "sp-char",
    "parenthetical": "sp-paren",
    "dialogue": "sp-dlg",
}
HEADING_RE = re.compile(r"^\s*(\d+[A-Z]?\s+)?(INT|EXT|INT\./EXT|EXT\./INT|I/E)[\s.:/-]", re.I)


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def screenplay_html(content: str, scene_start: int | None = None) -> str:
    if not content:
        return '<div class="sp-empty">No content.</div>'
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        return (f'<div class="banner banner-warn">This segment is not well-formed XML ({esc(exc)}); showing the raw markup.</div>'
                f'<pre class="cml-raw">{esc(content)}</pre>')
    out = ['<div class="sp-page">']
    for i, scene in enumerate(root.findall("scene") or [root]):
        num = (scene_start + i) if isinstance(scene_start, (int, float)) else i + 1
        out.append(f'<section class="sp-scene"><div class="sp-scene-no">SCENE {int(num)}</div>')
        for el in scene:
            text = (el.text or "").strip()
            cls = ELEMENT_CLASS.get(el.tag, "sp-action")
            if el.tag == "stage_direction" and not HEADING_RE.match(text):
                cls = "sp-direction"
            if el.tag == "parenthetical" and text and not text.startswith("("):
                text = f"({text})"
            out.append(f'<p class="{cls}" title="&lt;{esc(el.tag)}&gt;">{esc(text)}</p>')
        out.append("</section>")
    out.append("</div>")
    return "".join(out)


def raw_html(content: str) -> str:
    return f'<pre class="cml-raw">{esc(content)}</pre>'


def _badge(ok, label: str, tip: str) -> str:
    if ok is None:
        return f'<span class="badge badge-na" title="{esc(tip)}">{esc(label)}</span>'
    return f'<span class="badge {"badge-ok" if ok else "badge-bad"}" title="{esc(tip)}">{"✓" if ok else "✗"} {esc(label)}</span>'


def _val(row: dict, k: str):
    v = row.get(k)
    try:
        if v != v:  # NaN
            return None
    except (TypeError, ValueError):
        pass
    return v


def abstract_html(row: dict | None, summary: str) -> str:
    if row is None:
        return '<div class="card abs-card"><div class="muted">Select an item in the table above.</div></div>'
    genres = ", ".join(row.get("genres") or ()) or "—"
    rating = _val(row, "imdb_rating")
    rel_chip = ""
    if row.get("gt_related") is True:
        movie = row.get("gt_rel_movie")
        rel_chip = (f'<span class="chip chip-warn">GT-related: {esc(row.get("gt_rel_type") or "related")}'
                    f'{" → " + esc(movie) if movie else ""}</span>')
    safe = row.get("eval_safe")
    chips = [
        rel_chip,
        '<span class="chip chip-ok">eval-safe</span>' if safe is True else
        ('<span class="chip chip-warn">not eval-safe</span>' if safe is False else ""),
        f'<span class="chip">{esc(row.get("source"))}</span>',
        f'<span class="chip">{esc(row.get("split") or "—")} split</span>' if row.get("split") else "",
        f'<span class="chip">IMDb {rating:.1f}</span>' if rating is not None else "",
        f'<span class="chip">{esc(genres)}</span>',
    ]
    s0, s1 = _val(row, "scene_start"), _val(row, "scene_end")
    facts = [
        ("Scenes", f"{int(row['num_scenes'])}" + (f" ({int(s0)}–{int(s1)})" if s0 is not None and s1 is not None else "")),
        ("Content tokens", f"{int(row['script_tokens']):,}"),
        ("Dialogue turns", f"{int(row['dialogue_turns']):,}"),
        ("Position in film", f"{float(_val(row, 'relative_position')):.0%}" if _val(row, "relative_position") is not None else "—"),
        ("Abstract prompt", esc(row.get("prompt_version") or "—")),
        ("Content cleaning", esc(row.get("normalization") or "—")),
        ("Author", esc(row.get("author") or "—")),
    ]
    paras = [p.strip() for p in re.split(r"\n\s*\n", summary or "") if p.strip()]
    if paras:
        words = len(summary.split())
        lo, hi = _val(row, "target_lo"), _val(row, "target_hi")
        target = f" · target {int(lo)}–{int(hi)}" if lo is not None and hi is not None else ""
        body = "".join(f"<p>{esc(p)}</p>" for p in paras)
        meta = f'<div class="abs-meta">{words} words · {len(paras)} paragraph{"s" if len(paras) != 1 else ""}{target}</div>'
    else:
        body = '<p class="muted">Abstract not written yet.</p>'
        meta = ""

    checked = row.get("checked")
    g = _val(row, "grounding")
    hard = row.get("hard") if isinstance(row.get("hard"), list) else []
    ung = row.get("ungrounded") if isinstance(row.get("ungrounded"), list) else []
    badges = [
        _badge(_val(row, "c07"), "valid CML", "C07: well-formed CML"),
        _badge(_val(row, "c08"), "no residue", "C08: no LLM wrapper text"),
        _badge(_val(row, "c09"), "2k–10k tokens", "C09: content length"),
        _badge(_val(row, "c10"), "12–24 scenes", "C10: scene count"),
        _badge(_val(row, "c11"), "dialogue", f"C11: {_val(row, 'num_speakers') or '?'} speakers, dialogue share {_val(row, 'dialogue_ratio') or '?'}"),
        _badge(_val(row, "c12"), "clean text", f"C12: {_val(row, 'artefacts_1k') if _val(row, 'artefacts_1k') is not None else '?'} artefacts / 1k words"),
        _badge(_val(row, "c13"), "low noise", f"C13: garble {_val(row, 'garble_rate')}, mis-tagged speakers {_val(row, 'bad_char_ratio')}"),
        _badge(_val(row, "c15"), "not GT movie", "C15"),
        _badge(_val(row, "c16"), "no GT overlap", f"C16: 13-gram overlap {_val(row, 'gt_overlap')}"),
        _badge(_val(row, "c19"), "abstract hard-pass", "C19: " + (", ".join(hard) if hard else "no hard failures")),
        _badge(_val(row, "c20"), "in target length", "C20"),
        _badge(_val(row, "c21"), "top speaker named", "C21"),
        _badge(_val(row, "c22"), "all thirds covered", f"C22: {row.get('thirds')}"),
        _badge(_val(row, "c23"), f"grounding {g:.2f}" if g is not None else "grounding", "C23: lexical grounding ≥ 0.35"),
        _badge(_val(row, "c24"), "unique opening", "C24"),
    ]
    splits = row.get("speaker_splits") if isinstance(row.get("speaker_splits"), list) else []
    v2_badges = [
        _badge(_val(row, "c12b"), "no \\ / CONT'D.", f"C12b: {_val(row, 'backslashes')} backslashes, {_val(row, 'contd')} CONT'D., "
               f"{_val(row, 'quote_inner_1k')} quote-inner spaces / 1k words"),
        _badge(_val(row, "c12c"), "no orphan speakers", f"C12c: {_val(row, 'orphan_lines')} orphan speaker lines"),
        _badge(_val(row, "c13b"), "no speaker splits", "C13b: " + (", ".join(splits) if splits else "none")),
        _badge(_val(row, "c15b"), "not a GT remake", "C15b (needs the gt_related field)"),
    ]
    pending = "" if checked else '<div class="muted small">Content checks are still running for this item.</div>'
    ung_html = f'<div class="muted small">Ungrounded capitalized words: {esc(", ".join(ung))}</div>' if ung else ""
    links = []
    if row.get("imdb_url"):
        links.append(f'<a href="{esc(row["imdb_url"])}" target="_blank" rel="noopener">IMDb</a>')
    if row.get("source_url"):
        links.append(f'<a href="{esc(row["source_url"])}" target="_blank" rel="noopener">source</a>')
    facts_html = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in facts)
    return f"""
<div class="card abs-card">
  <div class="abs-title">{esc(row.get('film'))}</div>
  <div class="abs-id mono">{esc(row.get('item_id'))}{' · ' + ' · '.join(links) if links else ''}</div>
  <div class="chips">{''.join(chips)}</div>
  <div class="abs-label">Abstract</div>
  <div class="abs-body">{body}</div>
  {meta}
  <dl class="facts">{facts_html}</dl>
  <div class="abs-label">Checks · contract v1.2</div>
  <div class="badges">{''.join(badges)}</div>
  <div class="abs-label abs-label-v2">Proposed v2 checks (evaluator)</div>
  <div class="badges">{''.join(v2_badges)}</div>
  {pending}{ung_html}
</div>"""
