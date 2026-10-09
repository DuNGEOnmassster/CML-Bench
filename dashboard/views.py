"""HTML fragments and Plotly figures for the dashboard (pure functions of a ReleaseView and the GT frame)."""
from __future__ import annotations

import html
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from checks import CONTRACT, GROUPS
from dataio import release_label

C_EXP = "#4f46e5"
C_EXP_LIGHT = "#c7d2fe"
C_GT = "#f59e0b"
C_OK = "#059669"
C_BAD = "#dc2626"
C_MUTED = "#94a3b8"
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def fmt_int(n) -> str:
    return "—" if n is None or (isinstance(n, float) and math.isnan(n)) else f"{int(n):,}"


def fmt_big(n: float) -> str:
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= div:
            return f"{n / div:.1f}{suf}"
    return f"{n:,.0f}"


def ago(ts: datetime | None) -> str:
    if ts is None:
        return "—"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    s = (datetime.now(timezone.utc) - ts).total_seconds()
    if s < 90:
        return "just now"
    if s < 5400:
        return f"{s / 60:.0f} min ago"
    if s < 172800:
        return f"{s / 3600:.0f} h ago"
    return ts.strftime("%Y-%m-%d")


def rate_of(df: pd.DataFrame, col: str) -> tuple[float | None, int, int]:
    if df is None or col not in df:
        return None, 0, 0
    s = df[col].dropna()
    if not len(s):
        return None, 0, 0
    k = int(s.astype(bool).sum())
    return k / len(s), k, len(s)


# --- header / banners ---------------------------------------------------------------------------


def header_html(snap, source_label: str, refresh_s: int) -> str:
    rev = (snap.rev or "")[:7]
    rev_html = f'<span class="chip mono">rev {esc(rev)}</span>' if rev and len(snap.rev or "") >= 7 else ""
    return f"""
<div class="hdr">
  <div>
    <div class="hdr-kicker">CML-Bench · dataset expansion</div>
    <h1>Expanded CML dataset</h1>
    <div class="hdr-sub">Screenplay segments in Cinematic Markup Language, each paired with an AI-written abstract.</div>
  </div>
  <div class="hdr-meta">
    <span class="chip"><span class="dot dot-live"></span>{esc(source_label)}</span>
    {rev_html}
    <span class="chip">updated {esc(ago(snap.updated_at))}</span>
    <span class="chip">auto-refresh {refresh_s // 60 if refresh_s >= 60 else refresh_s}{" min" if refresh_s >= 60 else " s"}</span>
  </div>
</div>"""


def banner_html(snap, release: str | None) -> str:
    parts = []
    if snap.status == "loading":
        parts.append(('info', "Loading the dataset from Hugging Face…"))
    elif snap.status == "missing":
        parts.append(('warn', esc(snap.message) + " The dashboard will pick up data automatically once the first batch lands."))
    elif snap.status == "empty":
        parts.append(('warn', esc(snap.message)))
    elif snap.status == "error":
        parts.append(('bad', esc(snap.message) + " Showing the last good snapshot."))
    is_pilot = bool(release) and release.startswith("pilot")
    if is_pilot and "main" not in snap.releases and snap.releases:
        parts.append(('info', "<b>Showing the pilot release.</b> The full build has not landed in the dataset repo yet; "
                              "the dashboard switches to it automatically when it does."))
    rv = snap.releases.get(release) if release else None
    pilot_meta = (rv.aux.get("pilot.json") or {}) if (is_pilot and rv is not None) else {}
    if pilot_meta:
        bits = [f"<b>{esc(release_label(release))}</b>: {esc(pilot_meta.get('prompt_version'))}, {esc(pilot_meta.get('content_normalization'))}"]
        if pilot_meta.get("status"):
            bits.append(esc(pilot_meta["status"]))
        if pilot_meta.get("note"):
            bits.append(f"<span class='muted'>{esc(pilot_meta['note'])}</span>")
        parts.append(('info', " · ".join(bits)))
    if snap.checks_total and snap.checks_done < snap.checks_total:
        pct = snap.checks_done / snap.checks_total
        parts.append(('info', f"Running contract checks on new items: {snap.checks_done:,} / {snap.checks_total:,} ({pct:.0%}). "
                              "Check-based numbers fill in as they finish."))
    return "".join(f'<div class="banner banner-{k}">{t}</div>' for k, t in parts)


# --- overview -----------------------------------------------------------------------------------


def _kpi(label: str, value: str, sub: str = "", accent: str = "") -> str:
    return (f'<div class="kpi {accent}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def kpis_html(rv, gt: pd.DataFrame | None) -> str:
    df = rv.df
    n = len(df)
    n_abs = int(df["has_abstract"].sum()) if n else 0
    films = df["film_key"].nunique() if n else 0
    sources = sorted(df["source"].unique()) if n else []
    toks = int(df["script_tokens"].sum()) if n else 0
    med = int(df["script_tokens"].median()) if n else 0
    gt_med = int(gt["script_tokens"].median()) if gt is not None else None
    words = df.loc[df["has_abstract"], "summary_words"]
    gt_words = gt["summary_words"].mean() if gt is not None else None
    auto = [c for c in CONTRACT if c.group in "ABCD"]
    passed = sum(1 for c in auto if rv.verdicts.get(c.cid, (None,))[0] is True)
    judged = sum(1 for c in auto if rv.verdicts.get(c.cid, (None,))[0] is not None)
    failing_items = int((df["n_fail"] > 0).sum()) if n else 0
    safe = df["eval_safe"] if n and "eval_safe" in df else pd.Series(dtype=object)
    if safe.notna().any():
        n_rel = int((df["gt_related"] == True).sum())  # noqa: E712
        safe_card = _kpi("Eval-safe", fmt_int(int((safe == True).sum())),  # noqa: E712
                         f"of {n:,} · {n_rel:,} GT-related", "kpi-ok")
    else:
        safe_card = _kpi("Eval-safe", "—", "no eval_safe / gt_related field yet")
    cards = [
        _kpi("Items with abstract", fmt_int(n_abs), f"of {n:,} items uploaded"),
        safe_card,
        _kpi("Films", fmt_int(films), f"{n / films:.1f} segments per film" if films else ""),
        _kpi("Sources", str(len(sources)), esc(", ".join(sources[:3])) + ("…" if len(sources) > 3 else "")),
        _kpi("Content tokens", fmt_big(toks), f"median {med:,}" + (f" · GT {gt_med:,}" if gt_med else "")),
        _kpi("Abstract words", f"{words.mean():.0f}" if len(words) else "—",
             "mean" + (f" · GT {gt_words:.0f}" if gt_words else "")),
        _kpi("Contract (automated)", f"{passed}/{judged}" if judged else "—",
             f"{failing_items:,} items fail ≥ 1 check", "kpi-ok" if judged and passed == judged else ("kpi-warn" if judged else "")),
    ]
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


def _first(d: dict, keys: tuple, kind=int):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return kind(v)
    return None


TOTAL_KEYS = ("items_total", "total_items", "target_items", "total", "target")


def progress_numbers(rv, env_target: int | None) -> dict:
    """Progress from build_status.json (status_format 1: nested `totals`) cross-checked with the loaded rows."""
    df = rv.df
    bs = rv.aux.get("build_status.json") or {}
    tot = bs.get("totals") if isinstance(bs.get("totals"), dict) else bs
    n = len(df)
    merged_seen = int(df["has_abstract"].sum()) if n else 0
    status_total = _first(tot, TOTAL_KEYS)
    if rv.name.startswith("pilot"):
        target, source = n, "items in the pilot"
    elif status_total:
        target, source = status_total, "build_status.json"
    elif env_target:
        target, source = env_target, "configured TARGET_ITEMS"
    else:
        target, source = n, "items uploaded"
    written = _first(tot, ("items_with_abstract",))
    return {
        "uploaded": n,
        "merged": max(merged_seen, _first(tot, ("items_merged",)) or 0),
        "with_abstract": max(written or 0, merged_seen),
        "checks_passed": _first(tot, ("items_checks_passed",)),
        "target": max(target, 1),
        "target_source": source,
        "batches_done": _first(tot, ("batches_merged", "batches_done", "batches_complete")),
        "batches_rejected": _first(tot, ("batches_rejected",)),
        "batches_total": _first(tot, ("batches_total", "total_batches")),
        "status_updated": bs.get("updated_at") or bs.get("timestamp"),
        "phase": bs.get("phase"),
        "gates": bs.get("gates") if isinstance(bs.get("gates"), dict) else {},
        "per_source": bs.get("per_source") if isinstance(bs.get("per_source"), dict) else {},
    }


def progress_html(rv, env_target: int | None) -> str:
    p = progress_numbers(rv, env_target)
    t = p["target"]
    m = min(p["merged"] / t, 1.0)
    w = min(max(p["with_abstract"] - p["merged"], 0) / t, 1.0 - m)
    a = min(p["with_abstract"] / t, 1.0)
    chips = []
    if p["batches_total"]:
        rej = f" · {p['batches_rejected']:,} rejected" if p["batches_rejected"] else ""
        chips.append(f'<span class="chip">batches merged {fmt_int(p["batches_done"])} / {fmt_int(p["batches_total"])}{rej}</span>')
    if p["status_updated"]:
        chips.append(f'<span class="chip">status {esc(str(p["status_updated"])[:16].replace("T", " "))} UTC</span>')
    stages = (f'<div class="stages"><span>uploaded <b>{p["uploaded"]:,}</b></span><span>abstract written <b>{p["with_abstract"]:,}</b></span>'
              + (f'<span>checks passed <b>{p["checks_passed"]:,}</b></span>' if p["checks_passed"] is not None else "")
              + f'<span>merged into data/ <b>{p["merged"]:,}</b></span></div>')
    notes = []
    if p["phase"]:
        notes.append(f'<div class="phase"><span class="muted">Phase</span> {esc(p["phase"])}</div>')
    for gate, state in p["gates"].items():
        cls = "gate-bad" if "block" in str(state).lower() else "gate-ok"
        notes.append(f'<div class="gate {cls}"><span class="mono">{esc(gate)}</span> {esc(state)}</div>')
    df = rv.df
    by_src = ""
    if len(df):
        g = df.groupby("source").agg(n=("item_id", "size"), abstracts=("has_abstract", "sum"), films=("film_key", "nunique"),
                                     safe=("eval_safe", lambda s: int((s == True).sum())),  # noqa: E712
                                     related=("gt_related", lambda s: int((s == True).sum()))).sort_values("n", ascending=False)  # noqa: E712
        rows = []
        for s, r in g.iterrows():
            ps = p["per_source"].get(s, {})
            build = (f"<div class='muted small mono'>{esc(ps.get('build_id', ''))} · {esc(ps.get('prompt_version', ''))}</div>"
                     if ps else "")
            rows.append(f"<tr><td>{esc(s)}{build}</td><td class='num'>{int(r['n']):,}</td><td class='num'>{int(r['safe']):,}</td>"
                        f"<td class='num'>{int(r['related']):,}</td><td class='num'>{int(r['abstracts']):,}</td>"
                        f"<td class='num'>{int(r['films']):,}</td></tr>")
        by_src = (f'<table class="tbl compact"><thead><tr><th>Source</th><th class="num">Items</th><th class="num">Eval-safe</th>'
                  f'<th class="num">GT-related</th><th class="num">With abstract</th><th class="num">Films</th></tr></thead>'
                  f'<tbody>{"".join(rows)}</tbody></table>')
    return f"""
<div class="card">
  <div class="card-head"><div class="card-title">Build progress</div><div class="chips">{''.join(chips)}</div></div>
  <div class="prog-figure"><span class="prog-big">{a:.1%}</span>
    <span class="prog-text"><b>{p['with_abstract']:,}</b> of <b>{t:,}</b> items have abstracts
    <span class="muted">(total from {esc(p['target_source'])})</span></span></div>
  <div class="prog-track"><div class="prog-fill" style="width:{m * 100:.2f}%"></div><div class="prog-pend" style="width:{w * 100:.2f}%"></div></div>
  <div class="prog-legend"><span><i class="sw sw-exp"></i>merged into data/</span><span><i class="sw sw-pend"></i>abstract written, not merged</span>
  <span><i class="sw sw-rest"></i>no abstract yet ({max(t - p['with_abstract'], 0):,})</span></div>
  {stages}{''.join(notes)}
  {by_src}
</div>"""


def subsets_html(rv) -> str:
    """eval_safe vs all counts and the gt_related breakdown."""
    df = rv.df
    title = '<div class="card-title">Eval-safe subset &amp; GT relations</div>'
    if df.empty or (df["eval_safe"].isna().all() and df["gt_related"].isna().all()):
        return (f'<div class="card">{title}<div class="muted small">This release has no <span class="mono">gt_related</span> or '
                f'<span class="mono">eval_safe</span> field yet. Once the schema adds them (sequels / same-universe films of a GT '
                f'movie are flagged and kept out of <span class="mono">eval_safe</span>), counts and flags appear here.</div></div>')
    safe = df[df["eval_safe"] == True]  # noqa: E712

    def row(label, frame):
        return (f"<tr><td>{label}</td><td class='num'>{len(frame):,}</td><td class='num'>{int(frame['has_abstract'].sum()):,}</td>"
                f"<td class='num'>{frame['film_key'].nunique():,}</td></tr>")

    rel = df[df["gt_related"] == True]  # noqa: E712
    counts = (f'<table class="tbl compact"><thead><tr><th>Subset</th><th class="num">Items</th><th class="num">With abstract</th>'
              f'<th class="num">Films</th></tr></thead><tbody>{row("All items", df)}{row("Eval-safe", safe)}'
              f'{row("GT-related", rel)}</tbody></table>')
    basis = df["eval_safe_basis"].dropna()
    basis_note = f'<div class="muted small">eval_safe from: {esc(basis.iloc[0])}</div>' if len(basis) else ""
    films = ""
    if len(rel):
        g = (rel.groupby(["film", "gt_rel_type", "gt_rel_movie"], dropna=False).size().reset_index(name="n")
             .sort_values("n", ascending=False).head(12))
        films = "".join(
            f"<tr><td>{esc(r['film'])}</td><td><span class='pill pill-na'>{esc(r['gt_rel_type'] or 'related')}</span></td>"
            f"<td class='muted'>{esc(r['gt_rel_movie'] or '—')}</td><td class='num'>{int(r['n'])}</td></tr>" for _, r in g.iterrows())
        films = (f'<div class="sub-title">GT-related films</div><table class="tbl compact"><thead><tr><th>Film</th><th>Relation</th>'
                 f'<th>GT movie</th><th class="num">Items</th></tr></thead><tbody>{films}</tbody></table>')
    return f'<div class="card">{title}{counts}{basis_note}{films}</div>'


def versions_html(rv) -> str:
    """Item counts per abstract prompt version x content cleaning version."""
    df = rv.df
    if df.empty:
        return ""
    g = (df.assign(prompt=df["prompt_version"].fillna("—"), cleaning=df["normalization"].fillna("—"))
         .groupby(["prompt", "cleaning"]).agg(n=("item_id", "size"), abstracts=("has_abstract", "sum"))
         .reset_index().sort_values("n", ascending=False))
    mixed = len(g) > 1
    rows = "".join(f"<tr><td class='mono'>{esc(r['prompt'])}</td><td class='mono'>{esc(r['cleaning'])}</td>"
                   f"<td class='num'>{int(r['n']):,}</td><td class='num'>{int(r['abstracts']):,}</td></tr>" for _, r in g.iterrows())
    note = ('<div class="banner banner-warn small">Several prompt / cleaning versions are mixed in this release.</div>'
            if mixed else "")
    return (f'<div class="card"><div class="card-title">Pipeline versions per item</div>{note}'
            f'<table class="tbl compact"><thead><tr><th>Abstract prompt</th><th>Content cleaning</th><th class="num">Items</th>'
            f'<th class="num">With abstract</th></tr></thead><tbody>{rows}</tbody></table></div>')


def uploads_html(commits: list[dict]) -> str:
    if not commits:
        return '<div class="card"><div class="card-title">Recent dataset commits</div><div class="muted">No commit history (local folder or empty repo).</div></div>'
    rows = "".join(f"<tr><td class='mono muted'>{esc(c['time'].strftime('%m-%d %H:%M'))}</td><td>{esc(c['title'])}</td></tr>"
                   for c in commits[:10])
    return f'<div class="card"><div class="card-title">Recent dataset commits</div><table class="tbl compact"><tbody>{rows}</tbody></table></div>'


# --- figures ------------------------------------------------------------------------------------


def _style(fig: go.Figure, title: str, height: int = 300, xlabel: str | None = None, ylabel: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=52, r=18, t=56, b=44),
        title=dict(text=f"<b>{title}</b>", x=0.01, xanchor="left", y=0.96, font=dict(size=14, color="#0f172a")),
        font=dict(family=FONT, size=12, color="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.06,
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(title_text=xlabel, gridcolor="#eef2f7", zeroline=False, showline=True, linecolor="#e2e8f0")
    fig.update_yaxes(title_text=ylabel, gridcolor="#eef2f7", zeroline=False)
    return fig


def empty_fig(title: str, msg: str = "No data yet", height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=13, color=C_MUTED), x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, title, height)


def hist_fig(exp: pd.Series, gt: pd.Series | None, title: str, xlabel: str, start: float, end: float, size: float,
             label: str = "Expanded", height: int = 300) -> go.Figure:
    exp = exp.dropna()
    if not len(exp):
        return empty_fig(title)
    fig = go.Figure()
    bins = dict(start=start, end=end, size=size)
    fig.add_trace(go.Histogram(x=exp.clip(start, end - 1e-9), xbins=bins, histnorm="percent", name=f"{label} (n={len(exp):,})",
                               marker=dict(color=C_EXP, line=dict(width=0)), opacity=0.85,
                               hovertemplate="%{x}: %{y:.1f}%<extra>" + label + "</extra>"))
    if gt is not None and len(gt.dropna()):
        g = gt.dropna()
        fig.add_trace(go.Histogram(x=g.clip(start, end - 1e-9), xbins=bins, histnorm="percent", name=f"GT-100 (n={len(g)})",
                                   marker=dict(color=C_GT, line=dict(width=0)), opacity=0.55,
                                   hovertemplate="%{x}: %{y:.1f}%<extra>GT-100</extra>"))
        fig.add_vline(x=float(g.median()), line=dict(color=C_GT, width=2, dash="dot"))
    fig.add_vline(x=float(exp.median()), line=dict(color=C_EXP, width=2, dash="dash"),
                  annotation=dict(text=f"median {exp.median():,.4g}", font=dict(size=11, color=C_EXP), yanchor="bottom"))
    fig.update_layout(barmode="overlay")
    return _style(fig, title, height, xlabel, "% of items")


def genre_fig(df: pd.DataFrame, gt: pd.DataFrame | None, top: int = 14, height: int = 380) -> go.Figure:
    title = "Genres (share of items, multi-label)"
    if df.empty or not df["genres"].map(len).any():
        return empty_fig(title, "No genre metadata", height)
    exp = df["genres"].explode().dropna().value_counts() / len(df) * 100
    exp = exp.head(top)[::-1]
    fig = go.Figure(go.Bar(y=exp.index, x=exp.values, orientation="h", name="Expanded", marker_color=C_EXP,
                           hovertemplate="%{y}: %{x:.1f}%<extra>Expanded</extra>"))
    if gt is not None:
        g = gt["genres"].explode().dropna().value_counts() / len(gt) * 100
        fig.add_trace(go.Bar(y=exp.index, x=[g.get(k, 0) for k in exp.index], orientation="h", name="GT-100", marker_color=C_GT,
                             opacity=0.8, hovertemplate="%{y}: %{x:.1f}%<extra>GT-100</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    return _style(fig, title, height, "% of items", None)


def sources_fig(df: pd.DataFrame, height: int = 300) -> go.Figure:
    title = "Items by source"
    if df.empty:
        return empty_fig(title, height=height)
    g = df.groupby("source").agg(done=("has_abstract", "sum"), n=("item_id", "size")).sort_values("n")
    height = min(height, 150 + 46 * len(g))
    fig = go.Figure()
    fig.add_trace(go.Bar(y=g.index, x=g["done"], orientation="h", name="with abstract", marker_color=C_EXP,
                         hovertemplate="%{y}: %{x:,} with abstract<extra></extra>"))
    fig.add_trace(go.Bar(y=g.index, x=g["n"] - g["done"], orientation="h", name="abstract pending", marker_color=C_EXP_LIGHT,
                         hovertemplate="%{y}: %{x:,} pending<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.45)
    return _style(fig, title, height, "items", None)


def per_film_fig(df: pd.DataFrame, height: int = 300) -> go.Figure:
    title = "Segments per film"
    if df.empty:
        return empty_fig(title, height=height)
    counts = df.groupby("film_key").size()
    fig = go.Figure(go.Histogram(x=counts, xbins=dict(start=0.5, end=max(counts.max(), 1) + 0.5, size=1), marker_color=C_EXP,
                                 hovertemplate="%{x} segments: %{y:,} films<extra></extra>"))
    _style(fig, title, height, "segments from the same film", "films")
    fig.update_xaxes(dtick=1 if counts.max() <= 15 else None, range=[0.4, max(counts.max(), 3) + 0.6])
    return fig


def distribution_figs(df: pd.DataFrame, gt: pd.DataFrame | None, label: str) -> dict[str, go.Figure]:
    g = (lambda c: gt[c]) if gt is not None else (lambda c: None)
    ab = df[df["has_abstract"]] if len(df) else df
    return {
        "tokens": hist_fig(df["script_tokens"] if len(df) else pd.Series(dtype=float), g("script_tokens"),
                           "Content length (cl100k tokens)", "tokens", 0, 16000, 500, label),
        "scenes": hist_fig(df["num_scenes"] if len(df) else pd.Series(dtype=float), g("num_scenes"),
                           "Scenes per segment", "scenes", 0.5, 45.5, 1, label),
        "words": hist_fig(ab["summary_words"] if len(ab) else pd.Series(dtype=float), g("summary_words"),
                          "Abstract length (words)", "words", 40, 400, 10, label),
        "rating": hist_fig(df["imdb_rating"] if len(df) else pd.Series(dtype=float), g("imdb_rating"),
                           "IMDb rating", "rating", 1, 10, 0.25, label),
        "year": hist_fig(df["year"] if len(df) else pd.Series(dtype=float), g("year"),
                         "Release year", "year", 1920, 2030, 5, label),
        "dialogue": hist_fig(df["dialogue_turns"] if len(df) else pd.Series(dtype=float), g("dialogue_turns"),
                             "Dialogue turns per segment", "turns", 0, 320, 10, label),
        "genres": genre_fig(df, gt),
    }


def flags_fig(df: pd.DataFrame, height: int = 320) -> go.Figure:
    title = "Abstract check flags"
    if df.empty or "hard" not in df:
        return empty_fig(title, height=height)
    hard = df["hard"].dropna().explode().dropna().value_counts()
    soft = df["soft"].dropna().explode().dropna().value_counts()
    if not len(hard) and not len(soft):
        return empty_fig(title, "No hard failures or soft flags", height)
    fig = go.Figure()
    if len(hard):
        fig.add_trace(go.Bar(y=[f"hard · {k}" for k in hard.index], x=hard.values, orientation="h", marker_color=C_BAD, name="hard failure"))
    if len(soft):
        fig.add_trace(go.Bar(y=[f"soft · {k}" for k in soft.index], x=soft.values, orientation="h", marker_color=C_GT, name="soft flag"))
    fig.update_layout(bargap=0.35)
    fig.update_yaxes(autorange="reversed")
    return _style(fig, title, height, "items", None)


def box_fig(df: pd.DataFrame, gt: pd.DataFrame | None, label: str, height: int = 360) -> go.Figure:
    if df.empty or gt is None:
        return empty_fig("Expanded vs GT-100", "Needs data and the GT reference", height)
    metrics = [("script_tokens", "Content tokens"), ("num_scenes", "Scenes"), ("dialogue_turns", "Dialogue turns"),
               ("summary_words", "Abstract words")]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[m[1] for m in metrics], horizontal_spacing=0.07)
    ab = df[df["has_abstract"]]
    for i, (col, _) in enumerate(metrics, start=1):
        e = (ab if col == "summary_words" else df)[col].dropna()
        for name, s, color in ((label, e, C_EXP), ("GT-100", gt[col].dropna(), C_GT)):
            fig.add_trace(go.Box(y=s, name=name, marker_color=color, boxmean=True, showlegend=(i == 1), legendgroup=name,
                                 boxpoints=False), row=1, col=i)
    _style(fig, "Distribution summary: expanded vs GT-100", height)
    fig.update_annotations(font=dict(size=12, color="#475569"))
    fig.update_xaxes(showticklabels=False)
    return fig


# --- quality contract ---------------------------------------------------------------------------


def _verdict_pill(v: bool | None, detail: str = "") -> str:
    if v is None:
        return f'<span class="pill pill-na" title="{esc(detail)}">—</span>'
    return (f'<span class="pill pill-ok" title="{esc(detail)}">PASS</span>' if v
            else f'<span class="pill pill-bad" title="{esc(detail)}">FAIL</span>')


def _rate_cell(r: float | None, k: int, n: int) -> str:
    if r is None:
        return '<span class="muted">not item-level</span>'
    cls = "bar-ok" if r >= 0.999 else ("bar-warn" if r >= 0.9 else "bar-bad")
    return (f'<div class="rate"><div class="rate-track"><div class="rate-fill {cls}" style="width:{r * 100:.1f}%"></div></div>'
            f'<span class="rate-num">{r:.1%}</span><span class="rate-n">{k:,}/{n:,}</span></div>')


def contract_html(rv, gt: pd.DataFrame | None) -> str:
    df = rv.df
    report = (rv.aux.get("contract_report.json") or {}).get("results", {})
    report_note = ""
    if report:
        rep_items = (rv.aux.get("contract_report.json") or {}).get("summary", {}).get("items")
        report_note = (f'<div class="muted small">“Build report” = <span class="mono">contract_report.json</span> from the pipeline'
                       f'{f" ({rep_items:,} items)" if rep_items else ""}; it is authoritative for C05, C17, C29, C30 and may lag the live data.</div>')
    body = []
    for gkey, gname in GROUPS.items():
        hint = (" <span class='muted small grp-hint'>informational: the evaluator's FAIL-with-fixes report proposes these; "
                "they do not count toward the automated total</span>") if gkey == "V" else ""
        body.append(f'<tr class="grp"><td colspan="6">{esc(gname)}{hint}</td></tr>')
        for a in [c for c in CONTRACT if c.group == gkey]:
            r, k, n = rate_of(df, a.column) if a.column else (None, 0, 0)
            v, detail = rv.verdicts.get(a.cid, (None, ""))
            rep = report.get(a.cid, {})
            gr, gk, gn = rate_of(gt, a.column) if (gt is not None and a.column) else (None, 0, 0)
            gt_cell = f"{gr:.0%}" if gr is not None else '<span class="muted">—</span>'
            body.append(
                f"<tr><td class='mono cid'>{a.cid}</td><td>{esc(a.text)}<div class='muted small'>{esc(detail)}</div></td>"
                f"<td>{_rate_cell(r, k, n)}</td><td class='center'>{_verdict_pill(v, detail)}</td>"
                f"<td class='center'>{_verdict_pill(rep.get('pass'), rep.get('detail', '')) if rep else '<span class=muted>—</span>'}</td>"
                f"<td class='center'>{gt_cell}</td></tr>")
    return f"""
<div class="card">
  <div class="card-head"><div class="card-title">Quality contract v1.2 · pass rates</div>
  <div class="muted small">Per-item checks reuse the pipeline's <span class="mono">contract_checks.py</span> / <span class="mono">check_abstracts.py</span>.
  Hover a verdict for details.</div></div>
  <table class="tbl contract"><thead><tr><th>ID</th><th>Assertion</th><th style="width:230px">Items passing</th>
  <th class="center">Live verdict</th><th class="center">Build report</th><th class="center">GT-100</th></tr></thead>
  <tbody>{''.join(body)}</tbody></table>
  {report_note}
  <div class="muted small">GT-100 column: the same item-level checks on the original 100 ground-truth items (surrounding whitespace stripped);
  C13 omits the corpus rare-word rate, which needs the whole MovieSum vocabulary.</div>
</div>"""


def abstract_summary_html(rv, gt: pd.DataFrame | None) -> str:
    df = rv.df
    ab = df[df["has_abstract"] & df["c19"].notna()] if len(df) else df

    def stats(frame):
        if frame is None or not len(frame):
            return None
        return {
            "Hard-pass": f"{frame['c19'].astype(bool).mean():.0%}",
            "In target length": f"{frame['c20'].astype(bool).mean():.0%}",
            "Mean words": f"{frame['abs_words'].astype(float).mean():.0f}",
            "Top speaker named": f"{frame['top1'].astype(bool).mean():.0%}",
            "All thirds covered": f"{frame['c22'].astype(bool).mean():.0%}",
            "Lexical grounding (mean)": f"{frame['grounding'].astype(float).mean():.3f}",
        }

    e = stats(ab)
    g = stats(gt[gt["c19"].notna()]) if gt is not None else None
    if not e:
        return '<div class="card"><div class="card-title">Abstract checks</div><div class="muted">No checked abstracts yet.</div></div>'
    rows = "".join(f"<tr><td>{k}</td><td class='num strong'>{v}</td><td class='num muted'>{(g or {}).get(k, '—')}</td></tr>"
                   for k, v in e.items())
    return (f'<div class="card"><div class="card-title">Abstract checks ({len(ab):,} items)</div>'
            f'<table class="tbl compact"><thead><tr><th>Metric</th><th class="num">Expanded</th><th class="num">GT-100</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


# --- GT comparison ------------------------------------------------------------------------------


def gt_table_html(df: pd.DataFrame, gt: pd.DataFrame | None, label: str) -> str:
    if gt is None:
        return '<div class="card"><div class="muted">GT reference (songdj/CML-Bench gt_100.json) could not be loaded.</div></div>'
    if df.empty:
        return '<div class="card"><div class="muted">No items yet.</div></div>'
    ab = df[df["has_abstract"]]

    def q(s):
        s = s.dropna().astype(float)
        if not len(s):
            return "—"
        return (f"<b>{s.median():,.4g}</b> <span class='muted'>({s.quantile(.1):,.4g}–{s.quantile(.9):,.4g}) · mean {s.mean():,.4g}</span>")

    def share(s, pred):
        s = s.dropna()
        return f"<b>{pred(s).mean():.0%}</b>" if len(s) else "—"

    rows = [
        ("Items", f"<b>{len(df):,}</b>", f"<b>{len(gt)}</b>"),
        ("Films", f"<b>{df['film_key'].nunique():,}</b>", f"<b>{gt['film_key'].nunique()}</b>"),
        ("Content tokens", q(df["script_tokens"]), q(gt["script_tokens"])),
        ("Scenes", q(df["num_scenes"]), q(gt["num_scenes"])),
        ("Dialogue turns", q(df["dialogue_turns"]), q(gt["dialogue_turns"])),
        ("Abstract words", q(ab["summary_words"]), q(gt["summary_words"])),
        ("Abstract tokens", q(ab["summary_tokens"]) if "summary_tokens" in ab else "—", q(gt["summary_tokens"])),
        ("IMDb rating", q(df["imdb_rating"]), q(gt["imdb_rating"])),
        ("IMDb ≥ 7.0 (paper's filter)", share(df["imdb_rating"], lambda s: s >= 7.0), share(gt["imdb_rating"], lambda s: s >= 7.0)),
        ("Release year", q(df["year"]), q(gt["year"])),
        ("15–20 scenes", share(df["num_scenes"], lambda s: s.between(15, 20)), share(gt["num_scenes"], lambda s: s.between(15, 20))),
        ("Valid CML (C07)", _pct(df, "c07"), _pct(gt, "c07")),
        ("Abstract hard-pass (C19)", _pct(df, "c19"), _pct(gt, "c19")),
    ]
    body = "".join(f"<tr><td>{a}</td><td class='num'>{b}</td><td class='num'>{c}</td></tr>" for a, b, c in rows)
    return (f'<div class="card"><div class="card-title">{esc(label)} vs original GT-100</div>'
            f'<div class="muted small">Median (p10–p90) · mean. GT-100 = <span class="mono">songdj/CML-Bench ground_truth/gt_100.json</span> '
            f'with <span class="mono">gt_100_info.json</span> metadata.</div>'
            f'<table class="tbl"><thead><tr><th>Metric</th><th class="num">{esc(label)}</th><th class="num">GT-100</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def _pct(df, col):
    r, _, n = rate_of(df, col)
    return f"<b>{r:.0%}</b>" if r is not None else "—"


# --- browser table ------------------------------------------------------------------------------

LENGTH_BUCKETS = {
    "< 4k tokens": (0, 4000),
    "4k–6k": (4000, 6000),
    "6k–8k": (6000, 8000),
    "≥ 8k": (8000, 10**9),
}


SUBSETS = {"all": "All items", "eval_safe": "Eval-safe only", "gt_related": "GT-related only"}


def subset_frame(df: pd.DataFrame, subset: str | None) -> pd.DataFrame:
    if df.empty or subset in (None, "all"):
        return df
    if subset == "eval_safe":
        return df[df["eval_safe"] == True]  # noqa: E712
    if subset == "gt_related":
        return df[df["gt_related"] == True]  # noqa: E712
    return df


def filter_frame(df: pd.DataFrame, sources, film: str | None, lengths, only_failing: bool, query: str,
                 prompts=None, cleanings=None) -> pd.DataFrame:
    if df.empty:
        return df
    m = pd.Series(True, index=df.index)
    if sources:
        m &= df["source"].isin(sources)
    if prompts:
        m &= df["prompt_version"].fillna("—").isin(prompts)
    if cleanings:
        m &= df["normalization"].fillna("—").isin(cleanings)
    if film and film != "All films":
        m &= df["film"] == film
    if lengths:
        lm = pd.Series(False, index=df.index)
        for b in lengths:
            lo, hi = LENGTH_BUCKETS[b]
            lm |= df["script_tokens"].between(lo, hi - 1)
        m &= lm
    if only_failing:
        m &= df["n_fail"] > 0
    if query:
        ql = query.lower().strip()
        m &= df["film"].str.lower().str.contains(ql, regex=False) | df["summary"].str.lower().str.contains(ql, regex=False) | \
            (df["item_id"].str.lower() == ql)
    return df[m].sort_values(["film", "scene_start"], na_position="last")


TABLE_COLUMNS = ["Film", "Item", "Source", "Scenes", "Tokens", "Abstract words", "Versions", "Flags", "Failing checks"]


def _short_version(prompt, cleaning) -> str:
    p = str(prompt or "—").replace("abstract_", "")
    c = str(cleaning or "—").replace("moviesum_clean_", "").replace("detok_", "")
    return f"{p} · {c}"


def _flags(r) -> str:
    out = []
    if r.get("gt_related") is True:
        rel = r.get("gt_rel_type") or "related"
        out.append(f"GT-{rel}" + (f" → {r['gt_rel_movie']}" if r.get("gt_rel_movie") else ""))
    if r.get("eval_safe") is True:
        out.append("eval-safe")
    return "; ".join(out)


def _failing(r) -> str:
    base = r["failed"] or ("none" if r["checked"] else "")
    v2 = f" · v2: {r['failed_v2']}" if r.get("failed_v2") else ""
    pending = "" if r["checked"] else ((" · " if base else "") + "checks pending")
    return f"{base}{pending}{v2}"


def table_frame(df: pd.DataFrame, limit: int = 1000) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    head = df.head(limit)
    return pd.DataFrame({
        "Film": head["film"],
        "Item": head["item_id"],
        "Source": head["source"],
        "Scenes": head["num_scenes"],
        "Tokens": head["script_tokens"],
        "Abstract words": head["summary_words"].map(lambda w: w if w else "pending"),
        "Versions": [_short_version(p, c) for p, c in zip(head["prompt_version"], head["normalization"])],
        "Flags": head.apply(_flags, axis=1),
        "Failing checks": head.apply(_failing, axis=1),
    })
