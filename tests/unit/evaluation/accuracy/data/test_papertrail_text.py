import unittest

from lsa.evaluation.accuracy.data.papertrail import compose, convert, patch, sentences

PAGE = """<html><body><article>
<h1 class="ltx_title ltx_title_document">Sandpiper decoding</h1>
<div class="ltx_authors"><span class="ltx_personname">A. Author</span></div>
<div class="ltx_abstract"><p>We decode fast.</p></div>
<section class="ltx_section" id="S1"><h2>1 Introduction</h2>
<div class="ltx_para" id="S1.p1"><p id="S1.p1.1">Decoders matter, see Fig. 2 and <a href="#bib.bib1">1</a>. Latency is <math alttext="437\\,\\mathrm{ns}" display="inline">x</math>. We go on.</p></div>
</section>
<section class="ltx_bibliography"><ul>
<li class="ltx_bibitem" id="bib.bib1"><span class="ltx_tag_bibitem">[1]</span><span class="ltx_bibblock">B. Zed, Old decoders</span></li>
<li class="ltx_bibitem" id="bib.bib2"><span class="ltx_tag_bibitem">[2]</span><span class="ltx_bibblock">C. Yan, Newer decoders</span></li>
</ul></section>
</article></body></html>"""


class SentenceTest(unittest.TestCase):
    def test_abbreviations_maths_and_citations_never_end_a_sentence(self):
        text = "See Fig. 2 and et al. here. Maths $a. B$ stays ⟦bib. X⟧ whole. Next one."
        self.assertEqual(
            sentences.split(text),
            ["See Fig. 2 and et al. here.", "Maths $a. B$ stays ⟦bib. X⟧ whole.", "Next one."],
        )

    def test_insertion_goes_after_the_named_sentence(self):
        self.assertEqual(sentences.insert_after("One. Two.", 1, "New."), "One. New. Two.")
        self.assertEqual(sentences.insert_after("One. Two.", 0, "New."), "New. One. Two.")
        with self.assertRaises(ValueError):
            sentences.insert_after("One. Two.", 3, "New.")


class ConvertTest(unittest.TestCase):
    def setUp(self):
        self.paper = convert.parse_html(PAGE, "2609.00001", "2026-09-01")

    def test_a_page_becomes_blocks_with_maths_and_citation_placeholders(self):
        self.assertEqual(self.paper["title"], "Sandpiper decoding")
        self.assertEqual(self.paper["bib_style"], "numeric")
        [block] = self.paper["sections"][0]["blocks"]
        self.assertEqual(block["id"], "S1.p1.1")
        self.assertEqual(
            block["text"],
            "Decoders matter, see Fig. 2 and ⟦bib.bib1⟧. Latency is $437\\,\\mathrm{ns}$. We go on.",
        )

    def test_the_rendered_text_numbers_citations_from_the_bibliography(self):
        text, spans = convert.render(self.paper, 7)
        self.assertTrue(text.startswith("[Paper 0007 | 2026-09-01]\n# Sandpiper decoding\n"))
        self.assertIn("see Fig. 2 and 1. Latency", text)
        self.assertIn("\n## References\n[1] B. Zed, Old decoders.\n[2] C. Yan, Newer decoders.\n", text)
        start, end = spans["S1.p1.1"]
        self.assertTrue(text[start:end].startswith("Decoders matter"))


class PatchTest(unittest.TestCase):
    def setUp(self):
        self.paper = convert.parse_html(PAGE, "2609.00001", "2026-09-01")

    def test_a_patch_inserts_its_sentence_and_its_reference_after_the_last_one_cited_before_it(self):
        # Zed before Yan is not alphabetical => the list is in order of first citation
        snippet = {
            "node": "A3",
            "kind": "patch",
            "host": "2609.00001",
            "ops": [
                {
                    "op": "insert_sentences",
                    "block": "S1.p1.1",
                    "after": 1,
                    "text": "Sandpiper ⟦new.A3⟧ is faster.",
                }
            ],
            "new_bib": [{"key": "new.A3", "text": "A. Able, Sandpiper.", "tag": ""}],
        }
        text, spans = compose.patched_host(self.paper, [snippet])
        self.assertIn("and 1. Sandpiper 2 is faster. Latency", text)
        self.assertIn("[1] B. Zed, Old decoders.\n[2] A. Able, Sandpiper.\n[3] C. Yan", text)
        self.assertEqual([span["role"] for span in spans], [None, "reference"])
        self.assertNotIn(patch.MARK, text)
        excerpt = compose.excerpt(text, spans)
        self.assertEqual(excerpt.split("\n")[2:4], ["", compose.OMITTED])
        self.assertIn("\nSandpiper 2 is faster.\n", excerpt)

    def test_a_patch_anchored_on_a_missing_block_is_refused(self):
        snippet = {
            "node": "A3",
            "host": "x",
            "ops": [{"op": "insert_paragraph", "after_block": "S9", "text": "No."}],
        }
        with self.assertRaises(ValueError):
            patch.apply_patch(self.paper, snippet)

    def test_surnames_sort_by_family_name(self):
        self.assertEqual(patch.surname("J. D. Turner, A. B. Cole (2026). Title."), "turner")
        self.assertEqual(patch.surname("K. van der Berg. Title."), "berg")
        self.assertEqual(patch.surname("Gemma Team (2025). Gemma."), "gemma")


class ContextTest(unittest.TestCase):
    def test_a_context_numbers_its_papers_and_records_their_spans(self):
        chosen = [
            {"id": "qec-A1", "date": "2026-08-25", "kind": "fake"},
            {"id": "2609.1", "date": "2026-09-01", "kind": "host"},
        ]
        text, papers = compose.context("A library.", chosen, lambda paper: f"# {paper['id']}")
        self.assertTrue(text.startswith("A library.\n\nBelow are 2 arXiv preprints"))
        self.assertIn(
            "=== BEGIN PAPERS ===\n\n[Paper 0001 | 2026-08-25]\n# qec-A1\n[Paper 0002 | 2026-09-01]\n# 2609.1\n",
            text,
        )
        self.assertTrue(text.endswith(compose.END))
        self.assertEqual(text[papers[1]["body_start"] : papers[1]["end"]], "# 2609.1\n")
