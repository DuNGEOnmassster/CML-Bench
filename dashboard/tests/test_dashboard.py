"""Dashboard data-layer tests on synthetic items (no screenplay text).

  python dashboard/tests/test_dashboard.py
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import checks  # noqa: E402
import itemview  # noqa: E402
import views  # noqa: E402
from dataio import DataStore, LocalSource, build_frame  # noqa: E402


def make_content(n_scenes: int, speakers=("ANNA", "BEN")) -> str:
    scenes = []
    for i in range(n_scenes):
        lines = [f"    <stage_direction>INT. ROOM {i} -- DAY</stage_direction>",
                 f"    <scene_description>Anna paces near window {i} while Ben waits.</scene_description>"]
        for j in range(3):
            lines.append(f"    <character>{speakers[j % len(speakers)]}</character>")
            lines.append(f"    <dialogue>Line {j} about the plan for room {i}.</dialogue>")
        scenes.append("  <scene>\n" + "\n".join(lines) + "\n  </scene>")
    return "<script>\n" + "\n".join(scenes) + "\n</script>"


def make_item(imdb: str, start: int, n_scenes: int = 16, summary: str = "Anna and Ben argue about the plan.", **extra) -> dict:
    import hashlib

    content = make_content(n_scenes)
    rec = {
        "movie_name": f"Test Film {imdb[-2:]}_2001",
        "imdb_id": imdb,
        "script_segment": content,
        "summary": summary,
        "item_id": f"{imdb}-s{start:04d}-{start + n_scenes - 1:04d}",
        "scene_start": start,
        "scene_end": start + n_scenes - 1,
        "num_scenes": n_scenes,
        "script_tokens": 5000,
        "imdb_rating": 7.1,
        "genres": ["Drama"],
        "source_dataset": "MovieSum",
        "source_split": "train",
        "source_url": "https://example.org/moviesum",
        "source_file": "train.jsonl",
        "imdb_url": f"https://www.imdb.com/title/{imdb}/",
        "content_sha1": hashlib.sha1(content.encode()).hexdigest(),
        "content_normalization": "moviesum_clean_detok_v2",
        "abstract_prompt_version": "abstract_v1.3",
        "abstract_author": "test",
    }
    rec.update(extra)
    return rec


def write_jsonl(path: str, recs: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def store(self):
        s = DataStore(LocalSource(self.root), gt=None, workers=1)
        s.refresh()
        return s

    def test_missing_and_incremental_batches(self):
        s = DataStore(LocalSource(os.path.join(self.root, "nope")), gt=None, workers=1)
        s.refresh()
        self.assertEqual(s.snapshot.status, "missing")

        write_jsonl(f"{self.root}/data/batch_0001.jsonl", [make_item("tt0000001", 0), make_item("tt0000002", 0, summary="")])
        s = self.store()
        df = s.snapshot.releases["main"].df
        self.assertEqual(len(df), 2)
        self.assertEqual(int(df["has_abstract"].sum()), 1)
        self.assertEqual(s.snapshot.checks_done, 2)
        cached = dict(s._checks)

        write_jsonl(f"{self.root}/data/batch_0002.jsonl", [make_item("tt0000001", 20)])
        self.assertTrue(s.refresh())
        df = s.snapshot.releases["main"].df
        self.assertEqual(len(df), 3)
        self.assertEqual(df["film_key"].nunique(), 2)
        self.assertTrue(set(cached) <= set(s._checks))
        self.assertFalse(s.refresh())  # unchanged head: no work

    def test_pilot_release_and_progress(self):
        write_jsonl(f"{self.root}/pilots/v9/data/p.jsonl", [make_item("tt0000003", 0)])
        with open(f"{self.root}/pilots/v9/pilot.json", "w") as f:
            json.dump({"prompt_version": "abstract_v9", "status": "audited"}, f)
        s = self.store()
        self.assertEqual(list(s.snapshot.releases), ["pilot:v9"])
        banner = views.banner_html(s.snapshot, "pilot:v9")
        self.assertIn("Showing the pilot release", banner)
        self.assertIn("audited", banner)

        write_jsonl(f"{self.root}/content/src/part-00000.jsonl", [make_item("tt0000013", 0, summary=""), make_item("tt0000014", 0, summary="")])
        write_jsonl(f"{self.root}/data/src/src-b0001.safe.jsonl", [make_item("tt0000013", 0)])
        write_jsonl(f"{self.root}/manifests/src/batches.jsonl", [{"batch_id": "src-b0001", "item_ids": ["tt0000013-s0000-0015"]}])
        with open(f"{self.root}/build_status.json", "w") as f:
            json.dump({"status_format": 1, "totals": {"items_total": 10, "items_with_abstract": 3, "items_merged": 1,
                                                       "batches_total": 2, "batches_merged": 1}}, f)
        s.refresh()
        rv = s.snapshot.releases["main"]
        self.assertEqual(len(rv.df), 2)  # merged row replaces its content row; manifests are not items
        self.assertEqual(int(rv.df["has_abstract"].sum()), 1)
        p = views.progress_numbers(rv, env_target=13904)
        self.assertEqual((p["with_abstract"], p["merged"], p["target"], p["batches_total"]), (3, 1, 10, 2))

    def test_item_checks_and_verdicts(self):
        bad = make_item("tt0000004", 0, n_scenes=8, summary="Here is the summary of this excerpt.")
        bad["script_segment"] = bad["script_segment"].replace("Line 0", "Line \\ 0")
        write_jsonl(f"{self.root}/data/a.jsonl", [make_item("tt0000005", 0), bad])
        df = self.store().snapshot.releases["main"].df.set_index("item_id")
        good, b = df.loc["tt0000005-s0000-0015"], df.loc[bad["item_id"]]
        self.assertTrue(good["c07"] and good["c04"] and good["c10"])
        self.assertFalse(b["c10"])
        self.assertFalse(b["c19"])
        self.assertFalse(b["c12b"])
        self.assertIn("C10", b["failed"])
        self.assertIn("C12b", b["failed_v2"])

    def test_overlapping_ranges_and_duplicate_openings(self):
        write_jsonl(f"{self.root}/data/a.jsonl", [make_item("tt0000006", 0), make_item("tt0000006", 10)])
        df = self.store().snapshot.releases["main"].df
        self.assertFalse(df["c18"].any())
        self.assertFalse(df["c24"].any())  # same opening words

    def test_gt_related_and_eval_safe(self):
        rel = make_item("tt0000007", 0, gt_related={"type": "sequel", "gt_movie": "Some GT Film_2019"}, eval_safe=False)
        remake = make_item("tt0000008", 0, gt_related={"type": "remake", "gt_movie": "Other GT Film_1933"})
        plain = make_item("tt0000009", 0, gt_related=None)
        write_jsonl(f"{self.root}/data/a.jsonl", [rel, remake, plain])
        rv = self.store().snapshot.releases["main"]
        df = rv.df.set_index("item_id")
        self.assertEqual(df.loc[rel["item_id"], "gt_rel_type"], "sequel")
        self.assertFalse(df.loc[remake["item_id"], "c15b"])
        self.assertTrue(df.loc[plain["item_id"], "c15b"])
        self.assertEqual(int((rv.df["gt_related"] == True).sum()), 2)  # noqa: E712
        self.assertEqual(len(views.subset_frame(rv.df, "gt_related")), 2)
        self.assertIn("GT-related films", views.subsets_html(rv))
        self.assertIn("GT-sequel", views.table_frame(rv.df)["Flags"].tolist()[0])

    def test_eval_safe_id_list(self):
        write_jsonl(f"{self.root}/data/a.jsonl", [make_item("tt0000010", 0), make_item("tt0000011", 0)])
        with open(f"{self.root}/eval_safe_ids.txt", "w") as f:
            f.write("tt0000010-s0000-0015\n")
        df = self.store().snapshot.releases["main"].df
        self.assertEqual(len(df), 2)  # the id list is not read as items
        self.assertEqual(int((df["eval_safe"] == True).sum()), 1)  # noqa: E712
        self.assertEqual(df["eval_safe_basis"].iloc[0], "id list")

    def test_renderers(self):
        rec = make_item("tt0000012", 40)
        html = itemview.screenplay_html(rec["script_segment"], 40)
        self.assertIn("SCENE 40", html)
        self.assertIn('class="sp-char"', html)
        self.assertIn("not well-formed", itemview.screenplay_html("<script><scene>", 0))
        row = build_frame([{**_row(rec)}], {}).iloc[0].to_dict()
        self.assertIn("moviesum_clean_detok_v2", itemview.abstract_html(row, rec["summary"]))


def _row(rec):
    from dataio import normalize

    row, _ = normalize(rec, 0, "main", frozenset(), frozenset())
    row.update({"file": None, "ref": None, "path": "x"})
    return row


if __name__ == "__main__":
    unittest.main()
