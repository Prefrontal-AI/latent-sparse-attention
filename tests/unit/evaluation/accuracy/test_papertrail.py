import unittest

from lsa.evaluation.accuracy.data import papertrail as data
from lsa.evaluation.accuracy.data import papertrail_names as names
from lsa.evaluation.accuracy.grading import papertrail as grading

PAPERS = {1: "qec-A1", 2: "2609.17706", 3: "2610.01683", 12: "qec-A2"}


def number(answer, tol=0.01, unit=None):
    return {"answer_type": "number", "answer": answer, "tol": tol, "unit": unit}


QUESTION = {
    "id": "qec.A.1.1",
    "trail": "A",
    "tier": "1",
    "question": "What latency?",
    "answer_type": "number",
    "answer": 437,
    "tol": 0.01,
    "unit": None,
    "answer_papers": ["qec-A1"],
    "check_papers": ["qec-A1"],
    "prompt_suffix": "\n\nQuestion: What latency?",
    "prompt_no_context": "Question about papers not shown: What latency?",
}


class AnswerLineTest(unittest.TestCase):
    def test_the_last_answer_line_is_read(self):
        reply = "ANSWER: 12 is my first guess\nthinking again\nEVIDENCE: Paper 0001\nANSWER: 437"
        self.assertEqual(grading.answer_line(reply), "437")

    def test_a_reply_without_an_answer_line_is_read_by_its_last_line(self):
        self.assertEqual(grading.answer_line("so it is\n\n437 ns  \n\n"), "437 ns")
        self.assertEqual(grading.answer_line(""), "")


class NumberTest(unittest.TestCase):
    def test_scientific_forms_parse(self):
        for written, value in [
            ("1.2e5", 1.2e5),
            ("1.2 x 10^5", 1.2e5),
            ("1.2×10^{5}", 1.2e5),
            ("7×10²⁹", 7e29),
            ("3,180", 3180.0),
            ("10^-3", 1e-3),
            ("−4.5", -4.5),
        ]:
            self.assertAlmostEqual(grading.parse_number(written), value, msg=written)

    def test_a_number_within_its_relative_tolerance_scores(self):
        self.assertEqual(grading.score(number(437), "ANSWER: 440")["score"], 1.0)
        self.assertEqual(grading.score(number(437), "ANSWER: 442")["score"], 0.0)

    def test_a_question_without_a_tolerance_takes_two_percent(self):
        self.assertEqual(
            grading.score({"answer_type": "number", "answer": 100}, "ANSWER: 101.9")["score"], 1.0
        )

    def test_a_percentage_may_be_written_as_a_fraction(self):
        self.assertEqual(grading.score(number(0.612, unit="%"), "ANSWER: 0.00612")["score"], 1.0)

    def test_abstaining_on_a_question_with_an_answer_scores_zero(self):
        self.assertEqual(grading.score(number(437), "ANSWER: NOT IN CORPUS")["score"], 0.0)


class TextAndAbstainTest(unittest.TestCase):
    def test_text_matches_any_accepted_spelling_ignoring_case_and_punctuation(self):
        doc = {"answer_type": "text", "answer": ["Sandpiper-2", "SP2"]}
        self.assertEqual(grading.score(doc, "ANSWER: the sandpiper 2 decoder")["score"], 1.0)
        self.assertEqual(grading.score(doc, "ANSWER: Heron")["score"], 0.0)

    def test_an_abstain_question_scores_any_wording_of_not_in_corpus(self):
        doc = {"answer_type": "abstain", "answer": "NOT IN CORPUS"}
        self.assertEqual(grading.score(doc, "ANSWER: NOT IN CORPUS")["score"], 1.0)
        self.assertEqual(grading.score(doc, "ANSWER: this is not reported in the papers")["score"], 1.0)
        self.assertEqual(grading.score(doc, "ANSWER: 0.3")["score"], 0.0)


class PaperListTest(unittest.TestCase):
    def test_papers_named_by_number_map_through_the_context_order(self):
        doc = {"answer_type": "ids", "answer_ids": ["qec-A1", "2610.01683"]}
        scored = grading.score(doc, "ANSWER: Paper 0001, Paper 0003", PAPERS)
        self.assertEqual(scored["score"], 1.0)

    def test_once_a_paper_is_named_by_number_bare_numbers_count_too(self):
        doc = {"answer_type": "ids", "answer_ids": ["qec-A1", "2610.01683", "qec-A2"]}
        self.assertEqual(grading.score(doc, "ANSWER: Paper 0001, 0003, 12", PAPERS)["score"], 1.0)

    def test_the_score_is_the_f1_of_the_named_set(self):
        doc = {"answer_type": "ids", "answer_ids": ["qec-A1", "2610.01683"]}
        scored = grading.score(doc, "ANSWER: Paper 1, Paper 2", PAPERS)
        self.assertEqual((scored["precision"], scored["recall"]), (0.5, 0.5))
        self.assertAlmostEqual(scored["score"], 0.5)

    def test_process_results_numbers_papers_from_the_row_order(self):
        doc = {"answer_type": "ids", "answer_ids": ["b"], "papers": ["a", "b", "c"]}
        self.assertEqual(grading.process_results(doc, ["ANSWER: Paper 0002"]), {"score": 1.0})


class TaskNameTest(unittest.TestCase):
    def test_a_name_splits_into_sample_condition_and_question_set(self):
        self.assertEqual(names.parse_task("papertrail_ml_longctx_L400000"), ("ml_longctx", "L400000", False))
        self.assertEqual(names.parse_task("papertrail_qec_C3_synthetic_nic"), ("qec", "C3_synthetic", True))
        self.assertEqual(names.task_name("qec", "C3_synthetic", True), "papertrail_qec_C3_synthetic_nic")

    def test_a_condition_the_card_does_not_have_is_refused(self):
        with self.assertRaises(KeyError):
            names.parse_task("papertrail_qec_L123")
        with self.assertRaises(KeyError):
            names.parse_task("papertrail_biology_L400000")

    def test_the_controls_come_first_then_the_lengths_in_order(self):
        self.assertEqual(names.conditions(700_000), [*names.CONTROLS, "L400000", "L512000", "L700000"])
        self.assertEqual(names.condition_length("L10000000"), 10_000_000)
        self.assertIsNone(names.condition_length("C2_hosts"))


class RowTest(unittest.TestCase):
    def test_a_row_closes_the_context_with_its_suffix_and_the_question_alone_has_its_own_wording(self):
        row = data.question_row(QUESTION, "qec", "L400000", ["qec-A1"])
        self.assertEqual(row["closing"], "\n\nQuestion: What latency?")
        self.assertEqual((row["id"], row["papers"]), ("qec/L400000/qec.A.1.1", ["qec-A1"]))
        alone = data.question_row(QUESTION, "qec", "C1", [])
        self.assertEqual(alone["closing"], "Question about papers not shown: What latency?")
