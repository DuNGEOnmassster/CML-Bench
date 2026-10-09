import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_segments import CONFIG, choose_windows  # noqa: E402
from check_abstracts import check_one  # noqa: E402
from cml_format import (  # noqa: E402
    Scene,
    clean_text,
    drop_front_matter,
    parse_script,
    render,
    retag_orphan_characters,
    segment_stats,
    validate_cml,
)
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
