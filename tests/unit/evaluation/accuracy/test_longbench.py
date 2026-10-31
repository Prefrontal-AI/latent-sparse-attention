import unittest

from lsa.evaluation.accuracy.data import longbench as prompts
from lsa.evaluation.accuracy.grading import longbench as answers


class WordTokenizer:
    """Whitespace tokens - enough to test the truncation arithmetic exactly."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


DOC = {
    "_id": "x1",
    "domain": "Single-Document QA",
    "sub_domain": "Legal",
    "difficulty": "hard",
    "length": "short",
    "question": "Who wins?",
    "choice_A": "Ann",
    "choice_B": "Bob",
    "choice_C": "Cid",
    "choice_D": "Dee",
    "answer": "C",
    "context": " ".join(f"w{i}" for i in range(100)),
}


class LongBenchTest(unittest.TestCase):
    def test_extract_answer_matches_official_rules(self):
        self.assertEqual(answers.extract_answer("The correct answer is (B)."), "B")
        self.assertEqual(answers.extract_answer("**The correct answer is (C)**"), "C")
        self.assertEqual(answers.extract_answer("The correct answer is D"), "D")
        self.assertEqual(answers.extract_answer("I think (A) fits.\nThe correct answer is (A)"), "A")
        self.assertIsNone(answers.extract_answer("It is B."))
        self.assertIsNone(answers.extract_answer(""))
        self.assertIsNone(answers.extract_answer("The correct answer is (E)"))

    def test_process_results(self):
        self.assertEqual(answers.process_results(DOC, ["The correct answer is (C)"]), {"acc": 1.0})
        self.assertEqual(answers.process_results(DOC, ["The correct answer is (A)"]), {"acc": 0.0})
        self.assertEqual(answers.process_results(DOC, [""]), {"acc": 0.0})

    def test_prompt_is_the_official_template(self):
        prompt = prompts.prompt_for(DOC, "CONTEXT")
        self.assertTrue(
            prompt.startswith(
                "Please read the following text and answer the question below.\n\n<text>\nCONTEXT\n</text>"
            )
        )
        self.assertIn(
            "What is the correct answer to this question: Who wins?\nChoices:\n(A) Ann\n(B) Bob\n(C) Cid\n(D) Dee",
            prompt,
        )
        self.assertTrue(
            prompt.endswith('Format your response as follows: "The correct answer is (insert answer here)".')
        )

    def test_no_truncation_when_it_fits(self):
        record = prompts.prepare(DOC, WordTokenizer(), max_context_tokens=1000)
        self.assertFalse(record["truncated"])
        self.assertEqual(record["context_tokens"], 100)
        self.assertIn("w0 w1", record["prompt"])
        self.assertIn("w98 w99", record["prompt"])
        self.assertEqual(record["answer"], "C")
        self.assertEqual(record["id"], "x1")

    def test_middle_truncation_keeps_both_ends(self):
        template_tokens = len(prompts.prompt_for(DOC, "").split())
        record = prompts.prepare(DOC, WordTokenizer(), max_context_tokens=template_tokens + 20)
        self.assertTrue(record["truncated"])
        self.assertEqual(record["context_tokens"], 100)
        kept = record["prompt"].split("<text>\n")[1].split("\n</text>")[0].split()
        self.assertEqual(kept, [f"w{i}" for i in range(10)] + [f"w{i}" for i in range(90, 100)])


if __name__ == "__main__":
    unittest.main()
