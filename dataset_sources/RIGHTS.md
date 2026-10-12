# Rights evidence and intake contract

Target: worldwide public distribution supporting training, evaluation, derivatives
and commercial reuse. This is a **data intake policy**, not a claim that a source
or automatic check can establish legal certainty in every jurisdiction.

## Evidence required for every asset

Keep the original work, translation, edition/transcription and provider-access
conditions separate. A repository's software license, a catalog's CC0 metadata,
an uploader's unsupported label, a US-only public-domain finding, a film's license,
or a permission to read/perform a play does not establish the desired text rights.

Each original and edition layer, plus translation when applicable, must carry:

- An exact license identifier and worldwide scope.
- Evidence URL(s), a concrete assessment (`basis`), and verified licensor authority
  or a documented jurisdictional public-domain assessment.
- Explicit reviewed permission for redistribution, derivatives, commercial use,
  training and evaluation. Boolean strings and missing permissions are rejected.

Accepted intake labels: `CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`,
`LicenseRef-Public-Domain-Reviewed`, `LicenseRef-Worldwide-Grant`. The last two
require documented review; they are **not** auto-generated from author death or
publication dates. CC0 waives only rights the affirmer owns; it cannot clear a
third party's still-protected translation. ShareAlike material is partitioned and
its attribution/conditions retained, not relabeled CC0 or MIT.

Evidence should be archived with its retrieval date and content checksum by the
curator. Private contracts can be kept in restricted storage, with a public
rights statement and public attribution in the eventual dataset. Do not commit
personal contact details, signatures, credentials or confidential agreements.
The current validator requires public HTTPS evidence references and checks their
presence, not their contents; a reviewer must validate authority and scope.

## Record shape

See `examples/manifest.jsonl` for a complete **invented** working example.
The `rights` object contains:

```json
{
  "status": "approved",
  "territory": "worldwide",
  "reviewed_by": "accountable curator ID",
  "reviewed_on": "YYYY-MM-DD",
  "asset_sha256": "64 lowercase hex characters of the exact local asset",
  "attribution": "Author; translator/editor; source; license; changes",
  "access_basis": "Authorized API/export or delivered file; exact applicable conditions",
  "assessment": "Documented explanation covering this work, edition and all rights layers",
  "separate_grant_verified": true,
  "layers": [
    {
      "scope": "original",
      "license": "CC-BY-4.0",
      "territory": "worldwide",
      "authority_verified": true,
      "evidence_urls": ["https://rightsholder.example/this-work-license"],
      "basis": "Describe the grant and why the grantor controls this exact work",
      "permissions": {
        "redistribution": true,
        "derivatives": true,
        "commercial_use": true,
        "training": true,
        "evaluation": true
      }
    }
  ]
}
```

This abbreviated example **will not pass** without an `edition` layer and, for a
translation, a `translation` layer. Do not copy placeholders as real evidence.
Permission-route sources additionally require `separate_grant_verified: true`
covering rightsholder and acquisition conditions. Excluded and metadata-only
sources cannot be overridden through a manifest.

For a reviewed body-only extraction, hash the exact derivative supplied to the
parser, retain the original acquisition hash and transformation log with the
review evidence, and confirm that all layers cover the derivative. Never remove
notices to evade license obligations. The preparation tool rejects unprocessed
Gutenberg wrappers instead of silently stripping them.

## Modern-film acquisition checklist (template, not an executed agreement)

For a voluntary script contribution, confirm:

1. The contributor controls the screenplay, revisions, third-party dialogue and
   underlying adaptation rights; identify coauthors, studio/employer and translator.
2. The exact version can be publicly redistributed worldwide under a suitable
   open license, including machine-readable conversion, training and evaluation.
3. Required credit, license links, change notices, embargo/release dates, and
   any privacy/publicity issues are documented.
4. The grant covers downstream recipients, not merely this research group's
   private reading. Performance rights alone are insufficient.
5. Originals, assets, edition hashes and public grant statements are versioned.

Start with independent filmmakers, film schools, playwrights, literary authors
and open-film projects. Draft outreach or negotiate commissioning only in a
separately authorized step; this PR sends no messages or contracts.

## Release governance

Before a public full-text release, independently review the evidence and notices,
freeze the source/rights manifests, license partitions, exclusion ledger and
checksums, and publish a dataset card distinguishing source licenses from code
licenses. Document a correction/takedown channel and versioned withdrawal ledger;
propagate withdrawals to text, summaries and derived artifacts. Future correction
of a mistaken rights assessment must not be described as revoking a valid CC grant.

Primary policy references: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
[CC0](https://creativecommons.org/publicdomain/zero/1.0/),
[Gutenberg permissions](https://www.gutenberg.org/policy/permission),
[Wikisource copyright](https://wikisource.org/wiki/Help:Copyright),
[Folger restrictions](https://www.folger.edu/copyright-policy/),
[Script Slug terms](https://www.scriptslug.com/legal).
