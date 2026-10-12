# MovieSum-independent literary source expansion

This directory adds **source research, real metadata discovery, and a runnable
rights-reviewed local preparation pipeline** for screenplays, stage plays, and
novels. It does not claim that discovered books are licensed worldwide, or that
prepared text blocks are a completed benchmark. MovieSum is explicitly excluded.

- [Source registry](registry.json): 25 channels, including excluded/metadata-only
  routes, with primary-source evidence reviewed on 2026-10-12.
- [Research and dataset roadmap (中文)](ROADMAP.zh.md): priorities, scope, proposed
  scale, acquisition, annotation, split design, and acceptance criteria.
- [Rights and intake specification](RIGHTS.md): per-asset review and license layers.
- [Survey reports](reports/): observed metadata counts and response checksums;
  candidates must not be described as cleared works or released examples.

No paid model calls, credential lookup, full-text network downloads, automatic
publishing, or correspondence with rightsholders is performed by these commands.

## 1. Discover candidates (Python 3.10+, standard library only)

From the repository root:

```sh
python -m dataset_sources.discover --out dataset_sources/work/survey
```

Implemented connectors:

| Connector | What it actually reads | What it does not establish |
|---|---|---|
| Gutenberg | Official weekly CSV; subjects select fiction/drama/screenplay candidates | Literary form, original year, edition identity, global rights |
| DraCor | API v1 corpus inventory and each corpus's play catalog | Per-work rights; all corpora do **not** share a license |
| ELTeC | Official English/French/German metadata TSVs by default | Rights for every digitization/translation; global PD |
| Wikidata (optional) | Bounded 500-relation sample of films and their source works | Screenplay access or text licenses; not an exhaustive/random film census |

Use `--sources gutenberg dracor eltec wikidata` to include the film/adaptation
sample; it is reported separately as `film_metadata`, never as screenplays.
Other registered sources are documented acquisition routes, **not implemented
crawlers**. Full-text ingestion is deliberately separate from discovery.
Gutenberg is queried through its sanctioned catalog endpoint, not scraped via
ebook pages. `--eltec-languages eng fra deu` selects TSV catalogs;
`--dracor-corpora ger rus fre` limits DraCor. Unknown/missing catalogs are errors,
not empty successful corpora. `--sources gutenberg eltec` avoids the drama API.

Responses are cached in `dataset_sources/work/cache` with URL, timestamp,
SHA-256 and ETag. Re-runs reuse verified bytes. `--refresh` explicitly refreshes
the snapshot. Output directories must be new, so old reports cannot silently
masquerade as current runs. Network failures produce `status: partial`, error
details and exit code 2; successful runs exit 0. Retry to a new output directory
to reuse completed responses. HTTP operations use 30-second timeouts, bounded
response sizes, a project User-Agent, and a minimum interval between requests.

Output: `candidates.jsonl`, small `sample.jsonl`, and `report.json`. Candidate
IDs identify provider records, **not deduplicated intellectual works**.
`release_eligible` remains false and `work_id` stays unset until review. A PG
`Issued` date is a catalog release date; it is never treated as an original
publication date. DraCor's normalized date is kept separate from its print date.

## 2. Review identities, rights and exact assets

Follow [RIGHTS.md](RIGHTS.md). Store the reviewed UTF-8 assets and manifest under
`dataset_sources/work/intake/` (gitignored). An intake record binds the exact file
SHA-256 to the original/edition/translation rights, identity, creator IDs,
story family, evidence and reviewer. No source-wide `CC0` flag clears all texts.

Required identity fields:

| Field | Meaning |
|---|---|
| `record_id`, `edition_id` | Stable source-record and edition IDs |
| `work_id` | Canonical original work; translations/editions share it |
| `story_family_id`, `related_family_ids` | Cross-medium adaptations, remakes, series and links to old GT |
| `source_id`, `source_url` | Registered provider and canonical provenance |
| `kind` | `screenplay`, `stage_play`, `novel`, or `prose_fiction` |
| `language`, `creator_ids`, `genres`, `title` | Explicit bibliographic metadata |
| `identity_reviewed`, `is_translation` | Required booleans; no implicit defaults |
| `format`, `text_path` | `tei`, `cml`, or prose-only `text`; local path relative to manifest |
| `rights` | Hash-bound layered evidence, described in RIGHTS.md |

Use a versioned exclusion ledger even if empty. It has `schema_version: "1.0"`
and string arrays `record_ids`, `work_ids`, `story_family_ids`, `asset_sha256`.
Missing ledgers fail rather than silently disabling exclusions. Before real
evaluation, add the legacy GT-100 identities and all reviewed related families;
the example empty ledger is not decontamination.

## 3. Prepare local texts

```sh
python -m dataset_sources.prepare \
  --manifest dataset_sources/work/intake/manifest.jsonl \
  --exclusions dataset_sources/work/intake/exclusions.json \
  --out dataset_sources/work/prepared-v1
```

The command checks all rights layers, source policy, identity, asset hash and
path confinement before parsing. Missing/NC/ND/territorial/metadata-only rights
fail closed. A rights record is a curator assertion with evidence: the validator
does **not** determine whether its underlying legal assessment is true.

Parsers preserve typed blocks and source locators. TEI preserves explicit
speakers, narration, headings and stage directions. CML is accepted only for
actual screenplays. Plain text is prose-only; prose is never converted into fake
human-authored screenplay. Ambiguous mixed inline stage/speech, unsupported TEI
structures, malformed XML, DTDs and entities are rejected for review. Front/back
matter, cast lists and page/line-layout markers are not narrative body targets.
Whitespace normalization is recorded as parser version `literary-blocks-v1`;
keep the exact original asset privately for fidelity review.

Windows contain whole blocks without overlap. The default 2,000–24,000 bounds
are **Unicode characters, not model tokens**. Oversized blocks and short tails
are logged, never silently truncated. These provisional windows require later
scene/chapter/narrative-boundary review; they are not claimed equivalent to the
legacy 15–20-scene protocol.

Canonical work IDs, transitive story-family links, and exact normalized-text
duplicates create shared split components. Deterministic hashing gives an
approximate 80/10/10 work-family split; small corpora need not populate all three
splits. Exact duplicate editions are logged and emit only one copy. Related
family exclusions propagate transitively, including through excluded editions.
Adding newly discovered family links can change a component: freeze/reversion
the whole split manifest after identity review, never append blindly to a released
test set. Author overlap is reported; author holdout and fuzzy/cross-language
deduplication are **not yet implemented**.

Output files:

- `works.jsonl`: prepared edition metadata and original rights evidence.
- `segments.jsonl`: typed blocks, locators, hashes, split, license partition.
- `attribution.jsonl`: exact rights layers and credits, including ShareAlike.
- `rejections.jsonl`, `duplicate_editions.jsonl`, `window_rejections.jsonl`.
- `report.json`: separate work/family/edition/segment counts, source/kind/language
  coverage, dominant-work share, creator overlap and output/input checksums.

Rejected records or a zero-segment build return exit 2 **after** writing the
report. `release_ready` and `benchmark_ready` remain false. Summaries remain
null with `annotation_status: pending`. The module has no upload command.
The repository's MIT code license does not relicense any source text.

## 4. Reproducible, invented-text demonstration

```sh
python -m unittest discover -s dataset_sources/tests -v
python -m dataset_sources.prepare \
  --manifest dataset_sources/examples/manifest.jsonl \
  --exclusions dataset_sources/examples/exclusions.json \
  --allow-fixtures --min-chars 1 \
  --out dataset_sources/work/demo
```

The example manifest and short play are invented tests, not harvested data or
real rights clearance. They are rejected without `--allow-fixtures`; every
output retains `synthetic_fixture: true` and the report identifies fixture mode.
These examples must never enter corpus size claims.

## Relationship to the existing PRs and benchmark

This change is independent of PRs #1–#4 and targets `main`; it can be reviewed
without merging their MovieSum-dependent pipeline. Reuse their cleaning/audit
ideas only after reconciling schemas and rights. Their IMSDb/DailyScript/
SimplyScripts release claims do not grant the worldwide reuse requested here.

Existing model inference and metrics remain a **legacy screenplay-only** runner.
They are not wired to this heterogeneous corpus. The follow-on benchmark work
includes track-specific tasks, annotated test instances, nine-metric/code
reconciliation, modern-model baselines, human validation and confidence intervals.
See the roadmap for the ordered work and measurable completion gates.
