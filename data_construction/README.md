# CML-Dataset expansion pipeline

Builds more `(script_segment, summary)` items in the CML-Bench format, following the paper's construction
(MovieSum screenplay -> contiguous 15-20 scene excerpt -> AI-written abstract) at the scale of the whole
MovieSum corpus instead of 100 hand-picked films.

**The GitHub repo is public. Never commit screenplay text.** Everything under `data_construction/work/`
is gitignored; releases go only to a private Hugging Face dataset.

## Stages

| Stage | Command | Output |
|---|---|---|
| 0. IMDb metadata | `python data_construction/imdb_meta.py` | `work/sources/imdb_meta.json` (rating, votes, genres, year) |
| 1. Segments | `python data_construction/build_segments.py` | `work/build/segments.jsonl` + rejection logs + `build_stats.json` |
| 2. Batches | `python data_construction/make_abstract_batches.py --run_dir RUN [--sample N --one_per_movie] --batch_size 10` | `RUN/items.jsonl`, `RUN/batches/batch_XXXX/{manifest.json,items/*.xml}` |
| 3. Abstracts | agents, one batch each (see below) | `RUN/abstracts/<item_id>.json` |
| 4. Checks | `python data_construction/check_abstracts.py --run_dir RUN` | `RUN/abstract_checks.jsonl`, `RUN/abstract_checks_summary.json` |
| 5. Assemble | `python data_construction/assemble.py --run_dir RUN --out REL` | `REL/data/*.jsonl`, `REL/info.json`, `REL/stats.json`, `REL/README.md` |
| 6. Upload | `HF_TOKEN=... python data_construction/upload_hf.py --folder REL --repo_id USER/NAME` | private HF dataset |
| Contract | `python data_construction/contract_checks.py --release REL --run_dir RUN --out report.json` | automated assertions C01–C24, C30 |

Stage 1 downloads MovieSum (`rohitsaxena/MovieSum`) and CML-Bench `gt_100.json` if they are missing.
Requirements: Python 3.10+, `tiktoken` (stages 1/4/5), `huggingface_hub` (stage 6).

## Stage 3: abstract writing by agents

No LLM API is used. An agent (or several in parallel, one per batch) is given a batch manifest and:

1. reads `prompt_path` (`prompts/abstract_prompt_v1_2.md`) once;
2. for each item: reads `content_path`, writes `{"item_id", "abstract", "prompt_version", "author"}` to
   `abstract_path`, keeping the word count inside `target_words`;
3. skips items whose `abstract_path` already exists (resumable);
4. runs `python data_construction/check_abstracts.py --batch <batch_dir>` and rewrites only items with
   mechanical hard failures (length, markdown, meta phrases, all-caps names, ungrounded names).

Writers do not judge their own quality; an independent evaluator audits samples against the contract.

## Tests

`python data_construction/tests/test_pipeline.py`
