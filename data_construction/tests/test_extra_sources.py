import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "extra_sources"))

from cml_format import Scene, render, validate_cml  # noqa: E402
from extra_sources import catalogs  # noqa: E402
from extra_sources.build_extra import NOT_PRODUCED_RE, gt_lookalike, window_noise  # noqa: E402
from dataset_schema import load_gt_related  # noqa: E402
from extra_sources.imdb_index import norm_title, title_variants  # noqa: E402
from extra_sources.quality import script_quality  # noqa: E402
from extra_sources.text_screenplay import extract_text, html_to_text, looks_like_cue, text_to_scenes  # noqa: E402

# Invented text in standard screenplay layout (action at col 15, dialogue 25, cues 37).
PAGE = """
                                                            12.
               INT. KITCHEN - NIGHT                                  12

               Mara stands by the phone, turning a coin over. *

                                     MARA
                         What's the number?

                                     OTTO (CONT'D.)
                         478.
                         0150.

                                     MARA
                              (flat)
                         More.

                                     OTTO
                         ...

                                     MARA

                                     (MORE)
               EXT. STREET - DAY \\*

               Otto walks out into the rain and does not look back at the house.

                                     OTTO
                              (quietly)
                         --
                                                            CUT TO:
"""
GW = ["INT. KITCHEN - NIGHT", "EXT. STREET - DAY"]


class TextParserTests(unittest.TestCase):
    def setUp(self):
        self.scenes, self.diag = text_to_scenes(PAGE * 20)

    def test_layout_and_scenes(self):
        self.assertEqual(self.diag["layout"], "indented")
        self.assertEqual(len(self.scenes), 40)
        self.assertEqual([s.heading for s in self.scenes[:2]], GW, "scene numbers and revision marks stripped")
        self.assertEqual(validate_cml(render(self.scenes)), [])

    def test_real_dialogue_is_kept(self):
        els = self.scenes[0].elements + self.scenes[1].elements
        dialogue = [t for tag, t in els if tag == "dialogue"]
        self.assertEqual(dialogue, ["What's the number?", "478. 0150.", "More.", "...", "--"])
        self.assertIn(("parenthetical", "(flat)"), els)

    def test_page_furniture_and_marks_removed(self):
        content = render(self.scenes)
        for junk in ("(MORE)", "12.", "\\", "*", "CONT'D", ".)"):
            self.assertNotIn(junk, content)
        self.assertIn(("character", "OTTO"), self.scenes[0].elements)

    def test_orphan_cue_dropped(self):
        tags = [(tag, t) for tag, t in self.scenes[0].elements]
        self.assertNotIn(("scene_description", "MARA"), tags)
        self.assertGreater(self.diag["orphan_cues_dropped"], 0)

    def test_flush_layout(self):
        flush = "\n".join(["INT. HALL - DAY", "", "Rain hits the long windows of the hall tonight and nobody moves.", "",
                           "PIA", "We wait.", "", "TOM", "(sighs)", "Fine.", ""]) * 30
        scenes, diag = text_to_scenes(flush)
        self.assertEqual(diag["layout"], "flush")
        self.assertEqual(scenes[0].elements[2:], [("character", "PIA"), ("dialogue", "We wait."), ("character", "TOM"),
                                                  ("parenthetical", "(sighs)"), ("dialogue", "Fine.")])

    def test_running_headers_and_action_page_numbers(self):
        page = ("\n               INT. HALL - DAY\n\n               Rain hits the long windows of the hall, and nobody in the room moves.\n"
                "               12.\n\n               \"Dull Film\" by A. Writer 7/1/99    {n}.\n\n               {n}   WIDE SHOT   {n}\n\n"
                "                                     MARA\n                         478.\n")
        scenes, _ = text_to_scenes("".join(page.format(n=i) for i in range(30)))
        content = render(scenes)
        self.assertNotIn("Dull Film", content, "running header removed")
        self.assertNotIn("moves. 12.", content, "page number inside action removed")
        self.assertIn("WIDE SHOT", content, "numbered shot lines are content, not headers")
        self.assertIn("<dialogue>478.</dialogue>", content, "a number spoken as dialogue stays")

    def test_blank_lines_around_parenthetical_and_glued_header(self):
        flush = "\n".join(["INT. HALL - DAY", "", "Rain hits the long windows of the hall tonight and nobody moves at all.", "",
                           "DAVID", "", "(agonized)", "", "Susan! Duck!", "", "[59]", ""]) * 30
        scenes, _ = text_to_scenes(flush)
        self.assertEqual(scenes[0].elements[2:], [("character", "DAVID"), ("parenthetical", "(agonized)"), ("dialogue", "Susan! Duck!")])
        self.assertNotIn("[59]", render(scenes))
        stamp = "\n          INT. ROOM - DAY\n\n          A quiet room where nothing at all happens for a long while.\n\n                    MARA\n" \
                "          So I-                    Blue Rev. (mm/dd/yy)    {n}.\n\n                       Blue Rev. (mm/dd/yy)   {n}.\n"
        content = render(text_to_scenes("".join(stamp.format(n=i) for i in range(30)))[0])
        self.assertNotIn("Blue Rev", content)
        self.assertIn("<dialogue>So I-</dialogue>", content)

    def test_cue_rules(self):
        for ok in ("MARA", "DR. OTTO", "MARA (V.O.)", "McCLANE"):
            self.assertTrue(looks_like_cue(ok), ok)
        for bad in ("CUT TO:", "FADE OUT.", "INT. HOUSE - DAY", "BANG!", "ANGLE ON MARA", "Mara walks in."):
            self.assertFalse(looks_like_cue(bad), bad)

    def test_html_pre_extraction(self):
        page = "<html><body><pre><b>INT. A - DAY</b>\n" + "x" * 6000 + "</pre></body></html>"
        self.assertTrue(html_to_text(page).startswith("INT. A - DAY"))
        self.assertEqual(extract_text(page.encode(), "html")[0][:12], "INT. A - DAY")


class QualityTests(unittest.TestCase):
    def test_transcript_rejected(self):
        text = "\n".join(f"MARA: line number {i} of the talk." for i in range(3000))
        scenes, diag = text_to_scenes(text)
        reasons, _ = script_quality(text, scenes, diag, {}, None)
        self.assertIn("transcript_like", reasons)

    def test_image_pdf_needs_ocr(self):
        reasons, metrics = script_quality("x" * 500, [], {}, {"pages": 100, "producer": "Image Conversion Plug-in"}, None)
        self.assertEqual(reasons, ["needs_ocr"])

    def test_letter_spaced_ocr(self):
        text = "t a k e s can o f m i l k " * 2000
        reasons, _ = script_quality(text, [], {}, {}, None)
        self.assertIn("ocr_letter_spaced", reasons)


class MatchingTests(unittest.TestCase):
    def test_norm_and_variants(self):
        self.assertEqual(norm_title("The Thing"), norm_title("Thing, The"))
        self.assertEqual(norm_title("Fast & Furious"), "fastandfurious")
        v = [norm_title(t) for t in title_variants('The Wedding Date (was "Something Borrowed")')]
        self.assertIn("weddingdate", v)
        self.assertIn("somethingborrowed", v)
        self.assertIn("jasonx", [norm_title(t) for t in title_variants("Friday the 13th Part 10: Jason X")])

    def test_not_produced(self):
        for t in ("Carnivore%20(Unproduced).txt", "TheCrow3_unproduced.txt", "Some Film (spec script)", "Film X treatment"):
            self.assertTrue(NOT_PRODUCED_RE.search(t), t)
        for t in ("Spectre", "The Specialist", "Inspector Gadget"):
            self.assertFalse(NOT_PRODUCED_RE.search(t), t)

    def test_gt_lookalike(self):
        table = load_gt_related()
        self.assertEqual(gt_lookalike("Toy Story 2", table), "tt1979376")
        self.assertEqual(gt_lookalike("Batman Returns", table), "tt1877830")
        self.assertIsNone(gt_lookalike("Memento", table))

    def test_catalog_parsers(self):
        imsdb = '<a href="/Movie Scripts/Some Film Script.html" title="Some Film Script">Some Film</a> (1997-11 Draft)'
        e = catalogs.imsdb(imsdb)[0]
        self.assertEqual((e["title"], e["year"], e["year_kind"]), ("Some Film", 1997, "draft"))
        self.assertTrue(e["url"].endswith("/scripts/Some-Film.html"))
        daily = '<p><a href="scripts/some_film.txt">Some Film</a>&nbsp;by A. Writer&nbsp;1988 <a href="http://www.imdb.com/Title?0094631">imdb</a>'
        d = catalogs.dailyscript(daily)[0]
        self.assertEqual((d["year"], d["imdb_id"], d["format"]), (1988, "tt0094631", "txt"))


class WindowNoiseTests(unittest.TestCase):
    def test_noise_flags(self):
        seg = [Scene(0, [("stage_direction", "INT. A"), ("character", "BLAKE \\"), ("dialogue", "hi"), ("character", "BLAKE"),
                         ("dialogue", "a ~ b"), ("scene_description", "BLAKE")])]
        self.assertEqual(window_noise(seg), ["backslash_residue", "speaker_fission", "orphan_speaker_line", "ocr_symbols"])
        clean = [Scene(0, [("stage_direction", "INT. A"), ("character", "BLAKE"), ("dialogue", "Well... (beat) it's 4:30.")])]
        self.assertEqual(window_noise(clean), [])

    def test_writer_reported_noise(self):
        talk = [("character", "LOIS"), ("dialogue", "Hi."), ("character", "HENRY"), ("dialogue", "Yes."),
                ("character", "ENRY"), ("dialogue", "No."), ("character", "TAFFORD"), ("dialogue", "So."),
                ("character", "STAFFORD"), ("dialogue", "Go.")]
        fused = [("scene_description", f"LOIS {w}!") for w in ("Superman", "Wait", "Look", "Again", "Now")]
        letters = [("scene_description", c) for c in ("A", "H", "F")]
        reasons = window_noise([Scene(0, [("stage_direction", "INT. A")] + talk + fused + letters)])
        for r in ("speaker_in_action", "drop_cap_names", "margin_letters"):
            self.assertIn(r, reasons)
        action = [("scene_description", "LOIS watches in horror.")] * 6
        self.assertNotIn("speaker_in_action", window_noise([Scene(0, [("stage_direction", "INT. A")] + talk[:2] + action)]))


if __name__ == "__main__":
    unittest.main()
