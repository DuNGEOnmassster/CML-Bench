import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit_rules import dedupe_verdicts, sample_batches, stop_rule, wilson_upper  # noqa: E402
from build_segments import CONFIG, choose_windows  # noqa: E402
from check_abstracts import check_one  # noqa: E402
from cml_format import (  # noqa: E402
    Scene,
    clean_heading,
    clean_text,
    drop_front_matter,
    parse_script,
    render,
    retag_orphan_characters,
    segment_stats,
    strip_page_furniture,
    validate_cml,
)
from dataset_schema import make_record, validate_record  # noqa: E402
from make_abstract_batches import target_words  # noqa: E402

RAW = """<script>
  <scene>
    <stage_direction>INT.  WELLES' ROOM -- NIGHT</stage_direction>
    <scene_description>Welles is seated , PROJECTOR RUNNING . He 's watching &amp; waiting .</scene_description>
    <character>EDDIE</character>
    <parenthetical>( V.O . )</parenthetical>
    <dialogue>Do n't get too close , do n't let it get away .</dialogue>
    <character>FADE OUT</character>
    <character>(CONTINUED)</character>
  </scene>
  <scene>
    <stage_direction>EXT. STREET -- DAY</stage_direction>
  </scene>
</script>"""


class FormatTests(unittest.TestCase):
    def test_detokenize(self):
        self.assertEqual(clean_text("( V.O . )"), "(V.O.)")
        self.assertEqual(clean_text("Do n't go , he 's here ."), "Don't go, he's here.")
        self.assertEqual(clean_text("the KING &amp; I"), "the KING & I")
        self.assertEqual(clean_text("INT. ROOM (CONTINUED)"), "INT. ROOM")

    def test_parse_render_roundtrip(self):
        scenes = parse_script(RAW)
        self.assertEqual(len(scenes), 1, "heading-only scene is dropped")
        content = render(scenes)
        self.assertEqual(validate_cml(content), [])
        self.assertIn("&amp;", content)
        self.assertNotIn("CONTINUED", content)
        self.assertIn("<scene_description>FADE OUT</scene_description>", content)
        self.assertIn("<parenthetical>(V.O.)</parenthetical>", content)

    def test_retag(self):
        out = retag_orphan_characters(
            [("character", "SUPERIMPOSE:"), ("dialogue", "APRIL 16, 1992"), ("character", "THROUGH BINOCULARS"),
             ("scene_description", "x"), ("character", "ANNA"), ("dialogue", "Hi.")]
        )
        self.assertEqual(out[0], ("scene_description", "SUPERIMPOSE: APRIL 16, 1992"))
        self.assertEqual(out[1], ("scene_description", "THROUGH BINOCULARS"))
        self.assertEqual(out[-2:], [("character", "ANNA"), ("dialogue", "Hi.")])

    def test_validate_rejects_bad(self):
        self.assertTrue(validate_cml("Here is the segment:\n<script></script>"))
        self.assertTrue(validate_cml("<script><scene><foo>x</foo></scene></script>"))
        self.assertTrue(validate_cml("<script><scene><dialogue></dialogue></scene></script>"))

    def test_front_matter(self):
        scenes = [Scene(0, [("scene_description", "Written by Someone")]), Scene(1, [("stage_direction", "INT. A"), ("dialogue", "x")])]
        self.assertEqual([s.index for s in drop_front_matter(scenes)], [1])

    def test_stats(self):
        st = segment_stats(parse_script(RAW))
        self.assertEqual(st["num_scenes"], 1)
        self.assertEqual(st["dialogue_turns"], 1)
        self.assertEqual(st["num_speakers"], 1)

    def test_v3_cleaning(self):
        self.assertEqual(clean_text("LETO \\* \\[beat\\] \\_\\_"), "LETO [beat] __")
        self.assertEqual(clean_text("( OFF ; CONT 'D . )"), "(OFF)")
        self.assertEqual(clean_text("`` Son , sit . ''"), '"Son, sit."')
        self.assertEqual(clean_text('my father said: " Son , sit'), 'my father said: "Son, sit')
        self.assertEqual(clean_text("does ` nt"), "does ' nt")
        self.assertEqual(clean_text("J\u00b7im"), "Jim")
        self.assertEqual(clean_text("in\u00b7position"), "in position")
        self.assertEqual(clean_text("You\u2022re"), "You're")
        self.assertEqual(clean_heading("64 INT. BAR - NIGHT"), "INT. BAR - NIGHT")
        self.assertEqual(clean_heading("EXT. ROAD - DAY 71-A"), "EXT. ROAD - DAY")
        self.assertEqual(clean_heading("EXT. ROUTE 66"), "EXT. ROUTE 66")

    def test_dialogue_is_never_junk(self):
        script = """<script><scene>
          <stage_direction>INT. BANK - DAY</stage_direction>
          <character>SAM</character><dialogue>926 - 3143.</dialogue>
          <character>ODA MAE</character><dialogue>More.</dialogue>
          <character>SAM</character><dialogue>?!</dialogue>
          <character>ODA MAE</character><parenthetical>(beat)</parenthetical><dialogue />
          <character>SAM</character><dialogue>(MORE)</dialogue>
          <character>ODA MAE</character>
          <scene_description>SAM</scene_description>
          <scene_description>MARK SHIELDS (CNN)</scene_description>
          <character>SAM</character><dialogue>Bye.</dialogue>
        </scene></script>"""
        els = parse_script(script)[0].elements
        self.assertEqual([x for t, x in els if t == "dialogue"], ["926 - 3143.", "More.", "?!", "Bye."])
        self.assertEqual([x for t, x in els if t == "character"], ["SAM", "ODA MAE", "SAM", "SAM"])
        self.assertNotIn(("scene_description", "SAM"), els)
        self.assertIn(("scene_description", "MARK SHIELDS (CNN)"), els)
        self.assertNotIn("(beat)", [x for _, x in els])

    def test_dropped_scenes_recorded(self):
        dropped = {}
        scenes = parse_script(RAW, dropped=dropped)
        self.assertEqual([s.index for s in scenes], [0])
        self.assertEqual(dropped, {1: "no_body"})


class AuditRuleTests(unittest.TestCase):
    def test_stop_rule(self):
        ok = [{"item_id": f"i{i}", "major": 0, "outside": 0} for i in range(100)]
        sample = {f"i{i}" for i in range(100)} | {"m", "m1", "o1"}
        self.assertFalse(stop_rule(ok, sample)["stop"])
        two_close = ok[:20] + [{"item_id": "m1", "major": 1}] + ok[:10] + [{"item_id": "o1", "outside": 1}] + ok[:20]
        self.assertTrue(stop_rule(two_close, sample)["stop"])
        spread = ([{"item_id": "m", "major": 1}] + ok[:59]) * 2
        self.assertFalse(stop_rule(spread, sample)["stop"])
        self.assertTrue(stop_rule(([{"item_id": "m", "major": 1}] + ok[:24]) * 4, sample)["stop"])
        self.assertLess(wilson_upper(0, 261), 0.015)

    def test_orchestrator_pause_and_concentration(self):
        from audit_rules import orchestrator_audit

        ranges = [{"name": "O1", "first": 1, "last": 50}, {"name": "O2", "first": 51, "last": 100}]

        def v(b, item, bad=0):
            return {"item_id": item, "batch_id": f"moviesum-b{b:04d}", "outside": bad, "major": 0}

        two_targeted = [v(18, "t18", 1), v(20, "t20", 1)] + [v(60 + i, f"u{i}") for i in range(5)]
        r = orchestrator_audit(two_targeted, set(), ranges)
        self.assertEqual(r["paused"], [])
        self.assertEqual(r["targeted_concentrated"], [])
        self.assertEqual(r["targeted_by_orchestrator"]["O1"]["revoked"], 2)
        three = two_targeted + [v(22, "t22", 1)]
        self.assertEqual(orchestrator_audit(three, set(), ranges)["targeted_concentrated"], ["O1"])
        spread = three + [v(30, "c30"), v(31, "c31"), v(32, "c32"), v(61, "w1", 1), v(62, "w2", 1)]
        self.assertEqual(orchestrator_audit(spread, set(), ranges)["targeted_concentrated"], [])
        sampled = [v(3, "s3", 1), v(4, "s4"), v(5, "s5", 1)]
        self.assertEqual(orchestrator_audit(sampled, {"s3", "s4", "s5"}, ranges)["paused"], ["O1"])

    def test_targeted_verdicts_alarm_but_never_stop(self):
        ok = [{"item_id": f"t{i}", "major": 0, "outside": 0} for i in range(38)]
        outs = [{"item_id": f"o{i}", "major": 0, "outside": 1} for i in range(3)]
        r = stop_rule(ok + outs, sample_items=set(), targeted_meta={"o0": {"tier": "A", "reasons": ["ungrounded_names"]}})
        self.assertFalse(r["stop"])
        self.assertFalse(r["targeted_alarm"]["fired"])
        self.assertEqual(r["targeted_pool"]["by_tier"]["A"]["outside"], 1)
        majors = [{"item_id": f"m{i}", "major": 1, "outside": 0} for i in range(3)]
        r = stop_rule(ok[:10] + majors, sample_items=set())
        self.assertFalse(r["stop"])
        self.assertTrue(r["targeted_alarm"]["fired"])
        r = stop_rule(ok[:29] + [{"item_id": f"x{i}", "outside": 1} for i in range(11)], sample_items=set())
        self.assertTrue(r["targeted_alarm"]["fired"])
        self.assertFalse(r["stop"])

    def test_dedupe_and_sample_only_completion(self):
        v = [{"item_id": "a", "summary_sha1": "x", "auditor": "claude", "major": 0, "outside": 0, "minor": 1},
             {"item_id": "a", "summary_sha1": "x", "auditor": "gpt", "major": 0, "outside": 1, "minor": 0},
             {"item_id": "b", "summary_sha1": "y", "auditor": "claude", "major": 0, "outside": 0, "minor": 0}]
        d = dedupe_verdicts(v)
        self.assertEqual(len(d), 2)
        self.assertEqual(d[0]["outside"], 1)
        self.assertEqual(d[0]["auditors"], ["claude", "gpt"])
        r = stop_rule(d, sample_items={"b"})
        self.assertEqual(r["random_sample"]["audited"], 1)
        self.assertEqual(r["targeted_pool"]["major_or_outside"], 1)
        self.assertEqual(r["major_or_outside"], 1)

    def test_sample_is_two_per_block_and_stable(self):
        ids = [f"moviesum-b{n:04d}" for n in range(1, 31)] + ["moviesum-hold-b0001"]
        picked = sample_batches(ids)
        self.assertEqual(len(picked), 6)
        self.assertEqual(picked, sample_batches(list(reversed(ids))))
        self.assertNotIn("moviesum-hold-b0001", picked)

    def test_furniture_stripping(self):
        els = [[("scene_description", f"{w} Jane Eyre adapted by Moira Buffini March 2008 {n}.")]
               for w, n in (("He runs.", 24), ("She waits.", 25), ("They talk.", 26))]
        els.append([("scene_description", "©2015 DISNEY PIXAR - PRIVILEGED AND CONFIDENTIAL Arlo smiles.")])
        els.append([("stage_direction", "INT. MOTEL ROOM 12 - NIGHT")])
        out = [t for scene in strip_page_furniture(els) for _, t in scene]
        self.assertEqual(out[:4], ["He runs.", "She waits.", "They talk.", "Arlo smiles."])
        self.assertEqual(out[4], "INT. MOTEL ROOM 12 - NIGHT")

    def test_targeted_tiers_and_budget(self):
        from targeted_rule import h, select

        pool = [{"item_id": f"t{i:03d}", "batch_id": f"b{i // 10:02d}", "reasons": ["writer_source_issue"]} for i in range(200)]
        pool[0]["reasons"] = ["title_mention"]
        notes = {t["item_id"]: ("Page missing in the bar scene." if i % 2 else "OCR typos; cues tagged as action.")
                 for i, t in enumerate(pool)}
        notes["t004"] = "This is a different film's script."
        sel = {r["item_id"]: r["tier"] for r in select(pool, notes)}
        self.assertEqual(sel["t000"], "A")
        self.assertEqual(sel["t004"], "A")
        self.assertTrue(all(sum(1 for i, k in sel.items() if k == "B" and i[1:3] == f"{b:02d}") <= 1 for b in range(20)))
        self.assertTrue(all(h(i) < 1 / 20 for i, k in sel.items() if k == "C"))
        capped = select(pool, notes, merged_items=200)
        self.assertLessEqual(sum(r["tier"] != "A" for r in capped), sum(r["tier"] != "A" for r in select(pool, notes)))
        self.assertFalse(any(r["tier"] == "C" for r in capped))

    def test_c35_content_exclusions(self):
        import json
        import tempfile

        from validate_batch import content_excluded, load_content_exclusions

        self.assertEqual(load_content_exclusions(""), {"items": {}, "films": set()})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c35.json")
            with open(p, "w") as f:
                json.dump({"films": [{"imdb_id": "tt9"}], "items": [{"item_id": "tt1-s0001-0015", "content_sha1": "x"}]}, f)
            c35 = load_content_exclusions(p)
        items = {"tt1-s0001-0015": {"imdb_id": "tt1"}, "tt1-s0016-0030": {"imdb_id": "tt1"},
                 "tt9-s0100-0115": {"imdb_id": "tt9"}, "tt2-s0001-0015": {"imdb_id": "tt2"}}
        self.assertEqual(content_excluded(items, items, c35), {"tt1-s0001-0015", "tt9-s0100-0115"})
        self.assertEqual(content_excluded(["tt9-s0200-0215"], {}, c35), {"tt9-s0200-0215"})

    def test_c34_window_score(self):
        from mislabel_gate import score_items

        cml = ("<script><scene><stage_direction>INT. ROOM - DAY</stage_direction>"
               "<character>ANNA</character><dialogue>(quietly)</dialogue>"
               "<scene_description>ANNA I can't do this.</scene_description>"
               "<character>BOB</character><dialogue>ANNA</dialogue>"
               "<character>ACROSS THE STREET - MOMENTS LATER</character><dialogue>Hello.</dialogue>"
               "<character>BOB</character><dialogue>Fine.</dialogue></scene></script>")
        sig = score_items([{"item_id": "w", "imdb_id": "f", "script_segment": cml}])["w"]
        self.assertEqual((sig["paren_only"], sig["cue_as_dlg"], sig["bad_cue"], sig["fused"], sig["score"]), (1, 1, 1, 1, 4))

    def test_merge_exclusions_bind_to_build_and_content(self):
        import json
        import tempfile

        from hf_sync import load_merge_exclusions

        items = {"a": {"content_sha1": "1"}, "b": {"content_sha1": "2"}}
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_merge_exclusions(d, "build-x", items), ({}, {}))
            spec = {"rule": "C34 mislabel window gate: ...", "build_id": "build-x", "items": [{"item_id": "a", "content_sha1": "1"}]}
            with open(os.path.join(d, "build-x.json"), "w") as f:
                json.dump(spec, f)
            excluded, meta = load_merge_exclusions(d, "build-x", items)
            self.assertEqual(excluded, {"a": "C34"})
            self.assertEqual(meta["listed_items"], 1)
            for bad in ({**spec, "build_id": "build-y"}, {**spec, "items": [{"item_id": "a", "content_sha1": "9"}]}):
                with open(os.path.join(d, "build-x.json"), "w") as f:
                    json.dump(bad, f)
                with self.assertRaises(SystemExit):
                    load_merge_exclusions(d, "build-x", items)


class SchemaTests(unittest.TestCase):
    def test_record_roundtrip(self):
        content = render(parse_script(RAW))
        st = segment_stats(parse_script(RAW), content)
        seg = {"movie_name": "Eight Millimeter_1999", "imdb_id": "tt0134273", "script_segment": content,
               "item_id": "tt0134273-s0000-0000", "source_dataset": "MovieSum", "source_split": "train",
               "source_url": "u", "source_file": "train.jsonl", "imdb_url": "https://www.imdb.com/title/tt0134273/",
               "segment_index": 0, "scene_start": 0, "scene_end": 0, "dropped_scenes": [], "relative_position": 0.0,
               "imdb_rating": 6.5, "imdb_votes": 1, "genres": ["Crime"], "year": 1999, "gt_related": None,
               "content_normalization": "moviesum_clean_detok_v3", "gt_ngram_overlap": 0.0,
               "content_sha1": __import__("hashlib").sha1(content.encode()).hexdigest(), **st}
        rec = make_record(seg, batch_id="moviesum-b0001", build="moviesum-detok_v3-00000000")
        self.assertEqual(validate_record(rec), [])
        self.assertEqual(list(rec)[:4], ["movie_name", "imdb_id", "script_segment", "summary"])
        self.assertTrue(rec["eval_safe"])
        rec2 = make_record(seg, batch_id="moviesum-b0001", build="x", abstract={"abstract": "A b c.", "prompt_version": "v", "author": "a"},
                           count_tokens=lambda s: 3)
        self.assertEqual(validate_record(rec2), [])
        rec2["gt_related"] = {"type": "series", "gt_movie": "M_2000", "gt_imdb_id": "tt0000001"}
        self.assertIn("eval_safe_inconsistent", validate_record(rec2))


class WindowTests(unittest.TestCase):
    def test_windows_respect_bounds(self):
        scenes = [Scene(i, [("stage_direction", f"INT. ROOM {i} - DAY"), ("dialogue", "x")]) for i in range(100)]
        prefix = [0]
        for _ in scenes:
            prefix.append(prefix[-1] + 350)
        windows = choose_windows(scenes, prefix, CONFIG)
        self.assertTrue(windows)
        for a, b in windows:
            self.assertTrue(15 <= b - a <= 20)
            self.assertTrue(CONFIG["tokens_min"] <= prefix[b] - prefix[a] <= CONFIG["tokens_max"])
        self.assertTrue(all(w1[1] <= w2[0] for w1, w2 in zip(windows, windows[1:])), "non-overlapping")


class AbstractCheckTests(unittest.TestCase):
    PLACES = ["harbor", "bakery", "church", "garage", "museum", "stadium", "library", "airport", "cellar"]
    CONTENT = render(
        [Scene(i, [("stage_direction", f"INT. {p.upper()} - DAY"), ("scene_description", f"Welles inspects the {p} carefully."),
                   ("character", "WELLES"), ("dialogue", "Hello."), ("character", "EDDIE"), ("dialogue", "Hi.")]) for i, p in enumerate(PLACES)]
    )

    def test_good_abstract_passes(self):
        text = ("Welles inspects the harbor while Eddie greets him. " * 3 + "Later, Welles inspects the museum and Eddie talks. " * 4
                + "Finally, Welles inspects the cellar and the two part ways. " * 6)
        res = check_one(text, self.CONTENT, target_words(5000))
        self.assertEqual(res["hard"], [])
        self.assertEqual(res["thirds_covered"], [True, True, True])

    def test_bad_abstracts_fail(self):
        self.assertIn("meta_language", check_one("Here is the summary. " + "Welles talks to Eddie. " * 30, self.CONTENT)["hard"])
        self.assertIn("word_count_out_of_bounds", check_one("Welles talks.", self.CONTENT)["hard"])
        self.assertIn("all_caps_character_name", check_one("WELLES talks to Eddie. " * 25, self.CONTENT)["hard"])
        bad = check_one("Welles meets Gandalf, Frodo and Aragorn near Mordor. " * 12, self.CONTENT)
        self.assertIn("ungrounded_proper_nouns", bad["hard"])
        self.assertIn("markdown_or_list", check_one("- Welles talks to Eddie.\n" * 30, self.CONTENT)["hard"])

    def test_target_words(self):
        self.assertEqual(target_words(2000), [90, 155])
        lo, hi = target_words(10000)
        self.assertTrue(90 <= lo < hi <= 300)


if __name__ == "__main__":
    unittest.main()
