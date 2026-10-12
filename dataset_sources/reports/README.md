# Observed catalog survey — 2026-10-12

The checked-in JSON report records a **completed metadata survey**, not a
full-text acquisition, global copyright clearance, or new benchmark release.
Raw responses and the full candidate index remain in gitignored `work/`.
`sample-2026-10-12.jsonl` contains only five bibliographic examples per connector.

| Source | Candidate records observed | Coverage |
|---|---:|---|
| Project Gutenberg | 34,729 | Subject-selected prose fiction, drama and one screenplay-labeled candidate |
| DraCor | 4,918 | Play catalogs from all 24 corpora returned by the live API |
| ELTeC | 300 | 100 records each in English, French and German catalogs |
| Wikidata | 472 | Distinct film entities in a bounded 500-adaptation-relation query |
| **Total** | **40,419** | Provider records; not deduplicated works |

By provisional kind: 32,049 prose-fiction candidates, 300 curated novel records,
7,597 stage-play candidates, one screenplay-labeled candidate, and 472
film-metadata records. These are not estimates of every site's complete usable
catalog. PG subject matching may include collections or criticism; every work's
actual form requires review. Wikidata covers a deterministic, non-random subset
of films with `P31=film` and `P144=based on`; it provides no screenplay text.

Important qualifications:

- **Globally cleared works: 0; newly released real-text segments: 0.** This PR
  supplies inventory and preparation infrastructure, not invented rights grants.
- Gutenberg, ELTeC and other editions can overlap. Do not sum these figures as
  independent stories or claim a 40,419-example benchmark.
- DraCor's catalog includes restricted corpora: the observed French corpus
  advertises CC BY-NC-SA 4.0, and ShakeDraCor CC BY-NC 3.0. Their catalog presence
  does not make them eligible for the default open training track.
- The API catalog does not expose play language in these rows; those 4,918 rows
  remain `und` rather than guessing from a corpus name. Other provider language
  codes remain verbatim (including mixed-language strings and 2/3-letter codes).
  Harmonization is an identity-curation task, not evidence of extra languages.
- Modern screenplay acquisition remains a distinct rightsholder/asset review
  task. Neither 472 movie entities nor 32,349 prose/novel records fills that gap.

Reproduce with:

```sh
python -m dataset_sources.discover \
  --sources gutenberg dracor eltec wikidata \
  --out dataset_sources/work/survey-reproduction
```

The report includes each response URL, retrieval time, ETag and SHA-256 plus the
candidate-index checksum. Live catalogs can change. An exact historical rebuild
requires the matching cached responses; these caches are not committed. A rerun
with the retained cache reproduces the index checksum without new requests.
Earlier interrupted requests were retried using completed cached responses;
the final recorded run has `status: complete` and an empty error list.
