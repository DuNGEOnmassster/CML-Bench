import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "extra_sources"))

from cml_format import Scene, render, validate_cml  # noqa: E402
from extra_sources import catalogs  # noqa: E402
from extra_sources.build_extra import NOT_PRODUCED_RE, c34_score, gt_lookalike, window_noise  # noqa: E402
from dataset_schema import load_gt_related  # noqa: E402
from extra_sources.imdb_index import norm_title, title_variants  # noqa: E402
from extra_sources.quality import script_quality  # noqa: E402
from extra_sources.text_screenplay import extract_text, html_to_text, looks_like_cue, text_to_scenes  # noqa: E402
from extra_sources.mislabel import signatures  # noqa: E402
from extra_sources.verbatim_check import tag_check  # noqa: E402

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

    def test_dual_dialogue_and_cue_across_page_break(self):
        page = ("\n               INT. DOCK - NIGHT\n\n               Fog rolls off the water and the two radios crackle at once in the dark.\n\n"
                "                         BARNES                    TAYLOR\n"
                "                    We have a visual.         Copy that.\n"
                "                    (beat)                    Hold position.\n\n"
                "                                     OTTO\n                         Who said that?\n\n"
                "                                     MARA\n\n                                                            {n}.\n\n\n"
                "                         I said no and I meant it.\n\n")
        scenes, _ = text_to_scenes("".join(page.format(n=i) for i in range(30)))
        self.assertEqual(scenes[0].elements[2:], [
            ("character", "BARNES"), ("dialogue", "We have a visual."), ("parenthetical", "(beat)"),
            ("character", "TAYLOR"), ("dialogue", "Copy that. Hold position."), ("character", "OTTO"), ("dialogue", "Who said that?"),
            ("character", "MARA"), ("dialogue", "I said no and I meant it.")])

    def test_two_letter_contd_cue_is_not_page_furniture(self):
        page = ("\nINT. TRUCK STOP - DAY\n\nJo sits beside her on the bench and watches the sky turn green.\n"
                "                    JO\n          The first time it is upsetting.\n"
                "Melissa just looks at her and says nothing for a long while.\n"
                "                    JO (CONT'D)\n          You get used to them.\n"
                "                                        12A CONTINUED: (2)\n")
        scenes, _ = text_to_scenes("".join(page for _ in range(30)))
        self.assertEqual(scenes[0].elements[4:], [
            ("scene_description", "Melissa just looks at her and says nothing for a long while."),
            ("character", "JO"), ("dialogue", "You get used to them.")])

    def test_layout_breaks_inside_a_speech(self):
        page = ("\nINT. LAB - NIGHT\n\nThe meters on the console rise slowly while everyone in the room holds still.\n\n"
                "                    CHEN\n          We have a split.\n\n"
                "                    BUCKY\n          (Russian) Truly,\n\n          we can all live like kings.\n"
                "                    MATT\n          No, no -\n"
                "                    LILY                    OTTO\n          I can -                 You might -\n\n"
                "CHEN\n          Output ratio at four.\n\n")
        scenes, _ = text_to_scenes("".join(page for _ in range(30)))
        self.assertEqual(scenes[0].elements[2:], [
            ("character", "CHEN"), ("dialogue", "We have a split."),
            ("character", "BUCKY"), ("parenthetical", "(Russian)"), ("dialogue", "Truly, we can all live like kings."),
            ("character", "MATT"), ("dialogue", "No, no -"),
            ("character", "LILY"), ("dialogue", "I can -"), ("character", "OTTO"), ("dialogue", "You might -"),
            ("character", "CHEN"), ("dialogue", "Output ratio at four.")])

    def test_reply_cue_resumed_speech_fused_cue_and_title_cards(self):
        page = ("\nINT. DECK - NIGHT\n\nThe crew stands in a line on the deck while the wind howls through the rigging.\n\n"
                "                                   BOATS\n                         Corbin, John -\n"
                "                         JOHN\n                         Here.\n\n"
                "                                   WAYNE\n                         So?\n\n"
                "                         (grabbing her shoulders)\n                         I love you.\n\n"
                "                                   JESSICA\n                         Go.\n\n"
                "                         WAYNE Ma'am - he is dead now.\n\n"
                "                                   JOHN\n                         Fine.\n\n"
                "                                   EPILOGUE\n                         A quiet street years later.\n\n")
        scenes, _ = text_to_scenes("".join(page for _ in range(30)))
        self.assertEqual(scenes[0].elements[2:], [
            ("character", "BOATS"), ("dialogue", "Corbin, John -"), ("character", "JOHN"), ("dialogue", "Here."),
            ("character", "WAYNE"), ("dialogue", "So?"), ("parenthetical", "(grabbing her shoulders)"), ("dialogue", "I love you."),
            ("character", "JESSICA"), ("dialogue", "Go."), ("character", "WAYNE"), ("dialogue", "Ma'am - he is dead now."),
            ("character", "JOHN"), ("dialogue", "Fine."), ("scene_description", "EPILOGUE A quiet street years later.")])
        self.assertFalse(looks_like_cue("PART TWO"))
        intro = ("\nINT. CLINIC - DAY\n\nThe waiting room is full and the phones ring without a break all morning.\n\n"
                 "                         SHANNON (30s, a nurse) is on the phone.\n\n"
                 "                                   SHANNON\n                         Hold, please.\n\n")
        scenes, _ = text_to_scenes("".join(intro for _ in range(30)))
        self.assertEqual(scenes[0].elements[2], ("scene_description", "SHANNON (30s, a nurse) is on the phone."))

    def test_c34_score(self):
        talkers = {"MARA", "OTTO"}
        els = [("character", "MARA"), ("dialogue", "(softly)"), ("character", "OTTO"), ("dialogue", "MARA"),
               ("character", "EXT. HOUSE - DAY"), ("dialogue", "Hi."), ("scene_description", "MARA Get down!")]
        self.assertEqual(c34_score(els, talkers), 4)
        self.assertEqual(c34_score([("character", "MARA"), ("dialogue", "Hello there.")], talkers), 0)

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


class SpeakerTagTests(unittest.TestCase):
    RAW = ["               INT. KITCHEN - NIGHT", "",
           "               Mara stands by the phone, turning a coin over and over in her hand.", "",
           "                                     MARA", "                         What is the number you wanted me to call?", "",
           "                                     OTTO", "                              (flat)",
           "                         Four seven eight and then the rest of it.", ""]

    @staticmethod
    def cml(*els):
        return "<script><scene>" + "".join(f"<{t}>{x}</{t}>" for t, x in els) + "</scene></script>"

    def test_raw_cue_speech_must_stay_under_its_speaker(self):
        head = [("stage_direction", "INT. KITCHEN - NIGHT"),
                ("scene_description", "Mara stands by the phone, turning a coin over and over in her hand.")]
        mara = [("character", "MARA"), ("dialogue", "What is the number you wanted me to call?")]
        otto = [("character", "OTTO"), ("parenthetical", "(flat)"), ("dialogue", "Four seven eight and then the rest of it.")]
        n = len(self.RAW) - 1
        ok = tag_check(self.RAW, 0, n, self.cml(*head, *mara, *otto))
        self.assertEqual((ok["cues_checked"], ok["speech_as_action"], ok["wrong_speaker"]), (2, 0, 0))
        fused = tag_check(self.RAW, 0, n, self.cml(*head, ("scene_description", "MARA What is the number you wanted me to call?"), *otto))
        self.assertEqual(fused["speech_as_action"], 1)
        carried = tag_check(self.RAW, 0, n, self.cml(*head, *mara, *otto[1:]))
        self.assertEqual(carried["wrong_speaker"], 1)

    def test_mislabel_signatures(self):
        talkers = {"LOIS", "HENRY", "BARNES", "TAYLOR"}
        els = [("scene_description", "LOIS Superman! Over here!"), ("scene_description", "LOIS watches in horror."),
               ("scene_description", "BARNES TAYLOR We have a visual. Copy that."),
               ("scene_description", "(shaking his head) You're not going."), ("scene_description", "(beat) A door slams."),
               ("dialogue", "Please come with me. HENRY I'm sure it's fine."), ("dialogue", "We leave at dawn.")]
        sig = signatures(els, talkers)
        self.assertEqual((sig["fused_cue"], sig["dual_collapse"], sig["paren_speech"], sig["name_in_dialogue"]), (1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
