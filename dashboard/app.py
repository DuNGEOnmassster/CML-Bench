"""Gradio dashboard for the expanded CML dataset (Hugging Face Space entry point).

Environment:
  DATASET_REPO     private dataset repo (default: $HF_ACCOUNT/CML-Dataset-Expanded)
  HF_TOKEN         token that can read it (a Space secret)
  TARGET_ITEMS     planned item count, used when the repo has no build_status.json
  REFRESH_SECONDS  how often the repo head is polled (default 300)
  DATA_DIR         read a local folder with the same layout instead of HF (development)
  GT_DIR           local folder with gt_100.json + gt_100_info.json instead of songdj/CML-Bench

  DATA_DIR=/path/to/release python dashboard/app.py
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import gradio as gr
import pandas as pd

import itemview
import views
from dataio import DataStore, has_subsets, release_label, selected_view

STORE: DataStore | None = None
TICK_SECONDS = 20
HERE = Path(__file__).resolve().parent
ALL_FILMS = "All films"

THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#f5f6fa",
    block_radius="14px",
    block_border_width="1px",
    block_shadow="0 1px 2px rgba(15, 23, 42, 0.04)",
    button_large_radius="10px",
    button_small_radius="8px",
)
HEAD = ('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Courier+Prime:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">')
FORCE_LIGHT = """() => {
  const drop = () => document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
  drop(); new MutationObserver(drop).observe(document.body, {attributes: true, subtree: false});
}"""


# --- selection helpers ---------------------------------------------------------------------------


def release_choices(snap) -> list[tuple[str, str]]:
    return [(release_label(r), r) for r in snap.releases]


def pick_release(snap, current: str | None) -> str | None:
    if current in snap.releases:
        return current
    return next(iter(snap.releases), None)


def effective_subset(snap, release, subset) -> str:
    rv = snap.releases.get(release) if release else None
    return subset if has_subsets(rv) and subset in views.SUBSETS else "all"


# --- dashboard tabs ------------------------------------------------------------------------------

_RENDER_CACHE: dict[tuple, tuple] = {}


def render_dashboard(snap, release: str | None, subset: str | None) -> tuple:
    key = (snap.version, release, subset)
    if key not in _RENDER_CACHE:
        if len(_RENDER_CACHE) >= 8:
            _RENDER_CACHE.pop(next(iter(_RENDER_CACHE)))
        _RENDER_CACHE[key] = _render_dashboard(snap, release, subset)
    return _RENDER_CACHE[key]


def _render_dashboard(snap, release: str | None, subset: str | None) -> tuple:
    gt = STORE.gt.df if STORE.gt else None
    eff = effective_subset(snap, release, subset)
    rv = selected_view(snap, release, eff)
    header = views.header_html(snap, STORE.source.label if STORE.source.kind == "hf" else "local folder", STORE.refresh_seconds)
    banner = views.banner_html(snap, release)
    if subset not in (None, "all") and not has_subsets(snap.releases.get(release) if release else None):
        banner += ('<div class="banner banner-warn">This release has no <span class="mono">eval_safe</span> / '
                   '<span class="mono">gt_related</span> fields yet, so the subset selector shows all items.</div>')
    if rv is None:
        empty = views.empty_fig
        waiting = '<div class="card"><div class="muted">Waiting for the first batch of items.</div></div>'
        return (header, banner, waiting, waiting, waiting, "", views.uploads_html(snap.commits), empty("Items by source"),
                empty("Segments per film"), empty("Content length"), empty("Scenes per segment"), empty("Abstract length"),
                empty("IMDb rating"), empty("Release year"), empty("Dialogue turns"), empty("Genres"), waiting, waiting,
                empty("Abstract check flags"), views.gt_table_html(pd.DataFrame(), gt, "Expanded"), empty("Expanded vs GT-100"))
    label = release_label(release) + ("" if eff == "all" else f" · {views.SUBSETS[eff]}")
    full = snap.releases[release]
    figs = views.distribution_figs(rv.df, gt, label)
    return (
        header,
        banner,
        views.kpis_html(rv, gt),
        views.progress_html(full, STORE.target_items),
        views.subsets_html(full),
        views.versions_html(rv),
        views.uploads_html(snap.commits),
        views.sources_fig(rv.df),
        views.per_film_fig(rv.df),
        figs["tokens"], figs["scenes"], figs["words"], figs["rating"], figs["year"], figs["dialogue"], figs["genres"],
        views.contract_html(rv, gt),
        views.abstract_summary_html(rv, gt),
        views.flags_fig(rv.df),
        views.gt_table_html(rv.df, gt, label),
        views.box_fig(rv.df, gt, label),
    )


# --- sample browser ------------------------------------------------------------------------------


def filter_choices(rv, sources, film, prompts, cleanings):
    df = rv.df if rv is not None else pd.DataFrame(columns=["source", "film", "prompt_version", "normalization"])

    def opts(col):
        return sorted(df[col].fillna("—").unique()) if len(df) else []

    src, prm, cln = opts("source"), opts("prompt_version"), opts("normalization")
    films = [ALL_FILMS] + (sorted(df["film"].unique()) if len(df) else [])
    keep = lambda vals, allowed: [v for v in (vals or []) if v in allowed]  # noqa: E731
    return (gr.update(choices=src, value=keep(sources, src)),
            gr.update(choices=films, value=film if film in films else ALL_FILMS),
            gr.update(choices=prm, value=keep(prompts, prm)),
            gr.update(choices=cln, value=keep(cleanings, cln)))


def browse(release, subset, sources, film, lengths, failing, query, prompts, cleanings, keep_item: str | None = None):
    rv = selected_view(STORE.snapshot, release, subset)
    if rv is None or rv.df.empty:
        return (views.table_frame(pd.DataFrame()), [], '<div class="muted small">No items yet.</div>') + show_item(release, [], 0)
    hits = views.filter_frame(rv.df, sources, film, lengths, failing, query, prompts, cleanings)
    table = views.table_frame(hits)
    keys = table["Item"].tolist()
    count = (f'<div class="muted small">{len(hits):,} matching items' +
             (f" · table shows the first {len(keys):,}" if len(keys) < len(hits) else "") + " · click a row to open it</div>")
    pos = keys.index(keep_item) if keep_item in keys else 0
    return (table, keys, count) + show_item(release, keys, pos)


def show_item(release, keys, pos):
    if not keys:
        return (itemview.screenplay_html(""), itemview.raw_html(""), itemview.abstract_html(None, ""), 0)
    pos = int(pos) % len(keys)
    row, content, summary = STORE.item(release, keys[pos])
    if row is None:
        return (itemview.screenplay_html(""), itemview.raw_html(""), itemview.abstract_html(None, ""), pos)
    nav = f'<div class="sp-nav muted small">Item {pos + 1:,} of {len(keys):,} in the table</div>'
    return (nav + itemview.screenplay_html(content, row.get("scene_start")), itemview.raw_html(content),
            itemview.abstract_html(row, summary), pos)


# --- events --------------------------------------------------------------------------------------


def update_all(snap, release, subset, filters, keep_item=None, reset_filters=False):
    release = pick_release(snap, release)
    choices = release_choices(snap)
    radio = gr.update(choices=choices, value=release, visible=len(choices) > 1)
    subset = subset if subset in views.SUBSETS else "all"
    sources, film, lengths, failing, query, prompts, cleanings = filters
    if reset_filters:
        sources, film, prompts, cleanings = [], ALL_FILMS, [], []
    view_subset = effective_subset(snap, release, subset)
    upd = filter_choices(selected_view(snap, release, view_subset), sources, film, prompts, cleanings)
    sources, film, prompts, cleanings = (u["value"] for u in upd)
    return [snap.version, radio, release, subset, *render_dashboard(snap, release, subset), *upd,
            *browse(release, view_subset, sources, film, lengths, failing, query, prompts, cleanings, keep_item)]


def on_tick(version, release, subset, *rest, force=False):
    *filters, keys, pos = rest
    snap = STORE.snapshot
    if snap.version == version and not force:
        return [gr.skip()] * N_OUTPUTS
    current = keys[int(pos)] if keys and 0 <= int(pos) < len(keys) else None
    return update_all(snap, release, subset, filters, current)


def on_selection(release, subset, *filters):
    return update_all(STORE.snapshot, release, subset, filters, reset_filters=True)


def on_refresh_click():
    STORE.request_refresh()
    gr.Info("Checking the dataset repo for new batches…")


N_OUTPUTS = 0


def build_ui() -> gr.Blocks:
    global N_OUTPUTS
    with gr.Blocks(title="CML dataset dashboard") as demo:
        version = gr.State(-1)
        release = gr.State(None)
        subset = gr.State("all")
        keys = gr.State([])
        pos = gr.State(0)

        header = gr.HTML()
        with gr.Row(equal_height=True, elem_classes="toolbar"):
            with gr.Column(scale=3, min_width=240):
                release_radio = gr.Radio(choices=[], label="Release", visible=False, elem_classes="seg-radio")
            with gr.Column(scale=4, min_width=320):
                # Static on purpose: in Gradio 6.30 an update to this radio from the load event hides it.
                subset_radio = gr.Radio(choices=[(v, k) for k, v in views.SUBSETS.items()], value="all", label="Subset",
                                        elem_classes="seg-radio")
            refresh_btn = gr.Button("↻ Check for new batches", size="sm", variant="secondary", scale=0, min_width=210)
        banner = gr.HTML()

        with gr.Tabs():
            with gr.Tab("Overview"):
                kpis = gr.HTML()
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        progress = gr.HTML()
                        versions = gr.HTML()
                    with gr.Column(scale=2):
                        subsets = gr.HTML()
                        uploads = gr.HTML()
                with gr.Row():
                    fig_sources = gr.Plot(show_label=False)
                    fig_films = gr.Plot(show_label=False)
            with gr.Tab("Distributions"):
                gr.HTML('<div class="muted small tab-note">Indigo = selected release and subset, amber = original GT-100. '
                        'Histograms are normalized to % of items; dashed lines mark medians.</div>')
                with gr.Row():
                    fig_tokens = gr.Plot(show_label=False)
                    fig_scenes = gr.Plot(show_label=False)
                with gr.Row():
                    fig_words = gr.Plot(show_label=False)
                    fig_dialogue = gr.Plot(show_label=False)
                with gr.Row():
                    fig_rating = gr.Plot(show_label=False)
                    fig_year = gr.Plot(show_label=False)
                fig_genres = gr.Plot(show_label=False)
            with gr.Tab("Quality contract"):
                contract = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=2):
                        abs_summary = gr.HTML()
                    with gr.Column(scale=3):
                        fig_flags = gr.Plot(show_label=False)
            with gr.Tab("vs GT-100"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gt_table = gr.HTML()
                    with gr.Column(scale=3):
                        fig_box = gr.Plot(show_label=False)
            with gr.Tab("Sample browser"):
                with gr.Row(equal_height=True):
                    src_filter = gr.Dropdown(label="Source", choices=[], multiselect=True, scale=2)
                    film_filter = gr.Dropdown(label="Film", choices=[ALL_FILMS], value=ALL_FILMS, filterable=True, scale=3)
                    len_filter = gr.CheckboxGroup(label="Content length", choices=list(views.LENGTH_BUCKETS), scale=4)
                with gr.Row(equal_height=True):
                    prompt_filter = gr.Dropdown(label="Abstract prompt version", choices=[], multiselect=True, scale=2)
                    clean_filter = gr.Dropdown(label="Content cleaning version", choices=[], multiselect=True, scale=2)
                    fail_filter = gr.Checkbox(label="Only items failing a contract check", value=False, scale=2)
                with gr.Row(equal_height=True):
                    query = gr.Textbox(label="Search film title, abstract text or item id", placeholder="e.g. heist, Toy Story, tt1704573-s0072-0091",
                                       scale=6, submit_btn=True)
                    prev_btn = gr.Button("◀ Prev", size="sm", scale=0, min_width=90)
                    rand_btn = gr.Button("Random", size="sm", scale=0, min_width=90)
                    next_btn = gr.Button("Next ▶", size="sm", scale=0, min_width=90)
                count = gr.HTML()
                table = gr.Dataframe(interactive=False, max_height=330, wrap=True, show_search="none", elem_classes="items-table",
                                     column_widths=["17%", "14%", "8%", "6%", "7%", "8%", "12%", "13%", "15%"])
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7):
                        with gr.Tabs():
                            with gr.Tab("Screenplay"):
                                sp_html = gr.HTML(elem_classes="sp-scroll")
                            with gr.Tab("Raw CML"):
                                raw_html = gr.HTML(elem_classes="sp-scroll")
                    with gr.Column(scale=5):
                        abs_html = gr.HTML()

        timer = gr.Timer(TICK_SECONDS)

        filters = [src_filter, film_filter, len_filter, fail_filter, query, prompt_filter, clean_filter]
        dash_outputs = [header, banner, kpis, progress, subsets, versions, uploads, fig_sources, fig_films, fig_tokens, fig_scenes,
                        fig_words, fig_rating, fig_year, fig_dialogue, fig_genres, contract, abs_summary, fig_flags, gt_table, fig_box]
        item_outputs = [sp_html, raw_html, abs_html, pos]
        browser_outputs = [table, keys, count, *item_outputs]
        outputs = [version, release_radio, release, subset, *dash_outputs, src_filter, film_filter, prompt_filter,
                   clean_filter, *browser_outputs]
        N_OUTPUTS = len(outputs)
        state_inputs = [version, release, subset, *filters, keys, pos]

        demo.load(lambda *a: on_tick(*a, force=True), state_inputs, outputs)
        timer.tick(on_tick, state_inputs, outputs, show_progress="hidden")
        release_radio.input(on_selection, [release_radio, subset, *filters], outputs)
        subset_radio.input(on_selection, [release, subset_radio, *filters], outputs)
        refresh_btn.click(on_refresh_click, None, None)

        for comp in (src_filter, film_filter, len_filter, fail_filter, prompt_filter, clean_filter):
            comp.input(browse, [release, subset, *filters], browser_outputs, show_progress="hidden")
        query.submit(browse, [release, subset, *filters], browser_outputs)

        def on_select(rel, ks, evt: gr.SelectData):
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            return show_item(rel, ks, row)

        table.select(on_select, [release, keys], item_outputs, show_progress="hidden")
        prev_btn.click(lambda r, k, p: show_item(r, k, p - 1), [release, keys, pos], item_outputs, show_progress="hidden")
        next_btn.click(lambda r, k, p: show_item(r, k, p + 1), [release, keys, pos], item_outputs, show_progress="hidden")
        rand_btn.click(lambda r, k, p: show_item(r, k, random.randrange(len(k)) if k else 0), [release, keys, pos], item_outputs,
                       show_progress="hidden")
    return demo


def main() -> None:
    global STORE
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    STORE = DataStore.from_env()
    STORE.start()
    demo = build_ui()
    demo.queue(default_concurrency_limit=8).launch(
        theme=THEME, css=(HERE / "style.css").read_text(), head=HEAD, js=FORCE_LIGHT, footer_links=[],
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=int(os.environ.get("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
