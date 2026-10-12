import argparse
import copy
import http.client
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_sources.common import digest, read_jsonl, registry, write_json, write_jsonl
from dataset_sources.discover import Client, check_url, dracor, eltec, gutenberg, survey, wikidata
from dataset_sources.parse import parse, windows
from dataset_sources.prepare import prepare
from dataset_sources.rights import PERMISSIONS, rights_errors
from dataset_sources.splits import assign

TEI = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div type="act"><head>ACT I</head>
<sp who="#mara"><speaker>MARA</speaker><p>We will cross the bridge before morning.</p><stage>She closes the door.</stage></sp>
<sp><speaker>OTTO</speaker><p>Then I will carry the lantern.</p></sp></div></body></text></TEI>"""


def fixture(raw=TEI, key="fixture:play", kind="stage_play", fmt="tei", source="rightsholder"):
    return {
        "record_id": key,
        "source_id": source,
        "source_url": "https://example.org/original-fixture",
        "work_id": key,
        "edition_id": key + ":edition",
        "story_family_id": key + ":family",
        "related_family_ids": [],
        "title": "Invented test work",
        "language": "en",
        "kind": kind,
        "creator_ids": ["fixture:author"],
        "genres": ["test"],
        "is_translation": False,
        "identity_reviewed": True,
        "synthetic_fixture": True,
        "format": fmt,
        "text_path": "play.xml",
        "rights": {
            "status": "approved",
            "territory": "worldwide",
            "reviewed_by": "fixture only",
            "reviewed_on": "2026-10-12",
            "asset_sha256": digest(raw),
            "attribution": "Invented fixture, not a harvested work",
            "access_basis": "local test",
            "assessment": "Synthetic test data; not real-world rights clearance",
            "separate_grant_verified": True,
            "layers": [
                {
                    "scope": scope,
                    "license": "CC0-1.0",
                    "territory": "worldwide",
                    "authority_verified": True,
                    "evidence_urls": ["https://example.org/fixture-license"],
                    "basis": "test only",
                    "permissions": {p: True for p in PERMISSIONS},
                }
                for scope in ("original", "edition")
            ],
        },
    }


class RightsTests(unittest.TestCase):
    def setUp(self):
        self.row = fixture()
        self.source = registry()["rightsholder"]

    def errors(self):
        return rights_errors(self.row, self.source, allow_fixtures=True)

    def test_complete_fixture_and_production_separation(self):
        self.assertEqual(self.errors(), [])
        self.assertIn("synthetic_fixture_not_production", rights_errors(self.row, self.source))

    def test_metadata_license_cannot_clear_text(self):
        self.row["rights"]["layers"] = [{"scope": "metadata", "license": "CC0-1.0"}]
        self.assertIn("rights_layers_incomplete_or_duplicate", self.errors())

    def test_territorial_pd_not_worldwide(self):
        self.row["rights"]["layers"][0]["territory"] = "US"
        self.assertTrue(any("territory" in e for e in self.errors()))

    def test_nc_nd_unknown_fail_closed(self):
        for license_id in ["CC-BY-NC-3.0", "CC-BY-ND-4.0", "unknown", "public domain"]:
            self.row["rights"]["layers"][0]["license"] = license_id
            self.assertTrue(any("license_not_open_track" in e for e in self.errors()))

    def test_translation_requires_own_evidence(self):
        self.row["is_translation"] = True
        self.assertIn("rights_layers_incomplete_or_duplicate", self.errors())

    def test_permission_is_not_boolean_string(self):
        self.row["rights"]["layers"][0]["permissions"]["training"] = "true"
        self.assertTrue(any("permissions_incomplete" in e for e in self.errors()))

    def test_source_exclusions(self):
        for source in ["moviesum", "scriptslug", "wikidata", "europeana", "openlibrary"]:
            self.assertIn(
                "source_not_eligible_for_text",
                rights_errors(self.row, registry()[source], allow_fixtures=True),
            )

    def test_missing_grant_and_evidence(self):
        self.row["rights"]["separate_grant_verified"] = False
        self.row["rights"]["layers"][0]["evidence_urls"] = []
        self.assertIn("separate_rightsholder_and_access_grant_required", self.errors())
        self.assertTrue(any("evidence_required" in e for e in self.errors()))


class DiscoveryTests(unittest.TestCase):
    def test_pg_classification_and_dates(self):
        data = "Text#,Type,Issued,Title,Language,Authors,Subjects\n1,Text,2020-01-01,Story,en,Writer,Fiction\n2,Text,2021-01-01,Play,fr,Author,Drama\n3,Sound,2021-01-01,Audio,en,Author,Fiction\n"
        rows = list(gutenberg(data))
        self.assertEqual([r["kind"] for r in rows], ["prose_fiction", "stage_play"])
        self.assertIsNone(rows[0]["original_publication_year"])
        self.assertFalse(rows[0]["release_eligible"])

    def test_dracor_exception_not_relicensed(self):
        corpus = {"name": "shake", "licence": "CC BY-NC 3.0", "commit": "abc"}
        data = {
            "plays": [
                {"name": "invented", "yearNormalized": 1600, "yearPrinted": 1602, "authors": []}
            ]
        }
        row = list(dracor(corpus, data))[0]
        self.assertEqual(row["reported_rights"], "CC BY-NC 3.0")
        self.assertEqual(row["original_publication_year"], 1602)
        self.assertFalse(row["release_eligible"])

    def test_eltec(self):
        data = (
            "xmlid\ttitle\tauthor-name\tlanguage\tfirst-edition\nENG1\tExample\tWriter\teng\t1850\n"
        )
        row = list(eltec(data, "eng", "https://example.org/catalog"))[0]
        self.assertEqual(row["candidate_id"], "eltec:eng/ENG1")
        self.assertFalse(row["release_eligible"])

    def test_url_scope(self):
        for url in [
            "file:///etc/passwd",
            "http://dracor.org/api/v1/corpora",
            "https://example.com/",
            "https://www.gutenberg.org/ebooks/1",
            "https://dracor.org/api/v1/corpora/ger/plays/a/tei",
        ]:
            with self.assertRaises(ValueError):
                check_url(url)

    def test_cache_hash_validation(self):
        with tempfile.TemporaryDirectory() as d:
            url = "https://dracor.org/api/v1/corpora"
            key = digest(url.encode())
            Path(d, key + ".body").write_bytes(b"[]")
            write_json(Path(d, key + ".json"), {"url": url, "sha256": "wrong"})
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                Client(d).get(url)

    def test_transport_failure_has_partial_report(self):
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(
                out=str(Path(d, "out")),
                cache=str(Path(d, "cache")),
                refresh=False,
                timeout=1,
                sources=["gutenberg"],
                dracor_corpora=None,
                eltec_languages=["eng"],
            )
            with patch.object(Client, "get", side_effect=http.client.IncompleteRead(b"partial")):
                self.assertEqual(survey(args), 2)
            report = json.loads(Path(args.out, "report.json").read_text())
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["candidate_records"], 0)

    def test_wrong_catalog_schema_not_empty_success(self):
        with self.assertRaises(ValueError):
            list(gutenberg("<html>Error</html>"))
        with self.assertRaises(ValueError):
            list(eltec("Error", "eng", "https://example.org"))

    def test_film_adaptations_are_metadata_not_screenplays(self):
        bindings = [
            {
                "film": {"value": "http://www.wikidata.org/entity/Q1"},
                "basedOn": {"value": "http://www.wikidata.org/entity/" + key},
            }
            for key in ["Q2", "Q3"]
        ]
        rows = wikidata({"results": {"bindings": bindings}})
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["adaptations"]), 2)
        self.assertEqual(rows[0]["kind"], "film_metadata")
        self.assertFalse(rows[0]["release_eligible"])


class ParserTests(unittest.TestCase):
    def test_speaker_and_stage_fidelity(self):
        blocks = parse(TEI, "tei", "stage_play")
        self.assertEqual(
            [b["speaker"] for b in blocks if b["type"] == "dialogue"], ["MARA", "OTTO"]
        )
        stage = next(b for b in blocks if b["type"] == "stage_direction")
        self.assertEqual(stage["text"], "She closes the door.")
        self.assertIsNone(stage["speaker"])

    def test_prose_is_not_invented_cml(self):
        blocks = parse(b"The train leaves.\n\nA woman waits.", "text", "novel")
        self.assertTrue(all(b["type"] == "narration" and b["speaker"] is None for b in blocks))
        with self.assertRaises(ValueError):
            parse(b"MARA: Hello.", "text", "screenplay")

    def test_unsupported_and_entities_rejected(self):
        for raw in [
            b"<!DOCTYPE TEI><TEI/>",
            b"<broken",
            b"<TEI><text><body><table>Lost</table></body></text></TEI>",
        ]:
            with self.assertRaises(ValueError):
                parse(raw, "tei", "stage_play")

    def test_inline_stage_is_not_speech(self):
        raw = TEI.replace(b"before morning.", b"before <stage>Aside.</stage> morning.")
        with self.assertRaisesRegex(ValueError, "mixed inline"):
            parse(raw, "tei", "stage_play")

    def test_unattributed_cml_and_kind(self):
        with self.assertRaisesRegex(ValueError, "unattributed"):
            parse(
                b"<script><scene><dialogue>Hello</dialogue></scene></script>", "cml", "screenplay"
            )
        with self.assertRaises(ValueError):
            parse(b"<script/>", "cml", "novel")

    def test_windows_nonoverlap_and_oversize_report(self):
        blocks = [{"locator": str(i), "text": "x" * n} for i, n in enumerate([5, 5, 99, 5])]
        wins, drops = windows(blocks, max_chars=12, min_chars=3)
        self.assertEqual([[b["locator"] for b in w] for w in wins], [["0", "1"], ["3"]])
        self.assertEqual(drops[0]["reason"], "oversize_block")


class SplitTests(unittest.TestCase):
    def test_transitive_adaptation_translation_and_duplicate(self):
        rows = [
            {
                "record_id": str(i),
                "work_id": str(i),
                "story_family_id": str(i),
                "normalized_sha256": str(i),
            }
            for i in range(4)
        ]
        rows[0]["related_family_ids"] = ["1"]
        rows[1]["related_family_ids"] = ["2"]
        rows[3]["normalized_sha256"] = "2"
        assignments = assign(rows)
        self.assertEqual(len({x["split_group"] for x in assignments.values()}), 1)
        self.assertEqual(assignments, assign(list(reversed(rows))))

    def test_multiple_editions_do_not_move_family_split(self):
        one = {"record_id": "1", "work_id": "w", "story_family_id": "f", "normalized_sha256": "a"}
        two = {**one, "record_id": "2", "normalized_sha256": "b"}
        self.assertEqual(assign([one])["1"], assign([one, two])["1"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.row = fixture()
        (self.root / "play.xml").write_bytes(TEI)
        self.exclusions = {
            "schema_version": "1.0",
            "record_ids": [],
            "work_ids": [],
            "story_family_ids": [],
            "asset_sha256": [],
        }

    def run_build(self, rows=None, output="out", **kwargs):
        write_jsonl(self.root / "manifest.jsonl", rows or [self.row])
        write_json(self.root / "exclusions.json", self.exclusions)
        return prepare(
            self.root / "manifest.jsonl",
            self.root / "exclusions.json",
            self.root / output,
            allow_fixtures=True,
            min_chars=1,
            **kwargs,
        )

    def test_deterministic_outputs_and_no_fake_benchmark(self):
        a, b = self.run_build(output="a"), self.run_build(output="b")
        self.assertEqual(a, b)
        self.assertEqual(a["distinct_works"], 1)
        self.assertFalse(a["release_ready"])
        rows = list(read_jsonl(self.root / "a/segments.jsonl"))
        self.assertIsNone(rows[0]["summary"])
        self.assertTrue(rows[0]["synthetic_fixture"])

    def test_changed_asset_is_rejected(self):
        (self.root / "play.xml").write_bytes(TEI + b" ")
        self.assertEqual(self.run_build()["rejected_records"], 1)

    def test_path_escape_rejected(self):
        self.row["text_path"] = "../outside.xml"
        self.assertEqual(self.run_build()["rejected_records"], 1)

    def test_missing_exclusions_never_silent(self):
        write_jsonl(self.root / "manifest.jsonl", [self.row])
        with self.assertRaises(FileNotFoundError):
            prepare(self.root / "manifest.jsonl", self.root / "missing.json", self.root / "out")

    def test_cross_source_duplicate_editions(self):
        second = copy.deepcopy(self.row)
        second.update(record_id="fixture:copy", edition_id="copy", source_id="dracor")
        result = self.run_build([self.row, second])
        self.assertEqual(result["duplicate_editions"], 1)
        self.assertEqual(result["segments"], 1)

    def test_exclusion_propagates_through_adaptations(self):
        self.exclusions["story_family_ids"] = ["old-gt-family"]
        self.row["related_family_ids"] = ["middle-family"]
        bridge = copy.deepcopy(self.row)
        bridge.update(
            record_id="bridge",
            story_family_id="middle-family",
            related_family_ids=["old-gt-family"],
        )
        result = self.run_build([self.row, bridge])
        self.assertEqual(result["segments"], 0)
        self.assertEqual(result["rejected_records"], 2)

    def test_sharealike_partition_retained(self):
        self.row["rights"]["layers"][1]["license"] = "CC-BY-SA-4.0"
        report = self.run_build()
        self.assertEqual(report["by_license_partition"], {"sharealike": 1})

    def test_moviesum_is_not_importable(self):
        self.row["source_id"] = "moviesum"
        self.assertEqual(self.run_build()["segments"], 0)

    def test_malformed_rights_yield_rejection_report(self):
        self.row["rights"] = None
        report = self.run_build()
        self.assertEqual(report["rejected_records"], 1)
        self.assertEqual(report["segments"], 0)


if __name__ == "__main__":
    unittest.main()
