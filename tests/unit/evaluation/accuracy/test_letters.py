import unittest

from lsa.evaluation.accuracy.grading.letters import MMLU_PRO, letter

# shapes seen in real GLM-5.3-Flash answers on GPQA
REAL = [
    ("## Answer  The correct choice is **(D) -0.7**", "(D)"),
    ("$$\\boxed{E = 3}$$  ## Answer: **(D)**", "(D)"),
    ("The concentration is **6.3 × 10⁻⁷ M** — **Choice (C)** ✓  (Check: the approximation holds)", "(C)"),
    ("(A) x ✗ - (D) Circulation ✓  **The correct answer is (D)**", "(D)"),
    ("Therefore, the highest-density planet is **c**.  **Correct choice: (C) c**", "(C)"),
    ("Product 3: 11 carbons  The answer is **(C) 11**", "(C)"),
    ("Therefore the correct choice is:  \\[ \\boxed{\\text{(D)}\\left(\\cos(\\theta/2)\\right)} \\]", "(D)"),
    ("\\[ \\boxed{\\text{(C) }2.6\\times 10^5\\,\\text{GeV}} \\]", "(C)"),
    # the two cases where the last "(X)" rule fails
    (
        "Therefore, the answer is:  \\[ \\boxed{\\text{(A) CH}_3\\text{C(H)(C}_2\\text{H}_5)\\text{CH}_2\\text{COOH}} \\]",
        "(A)",
    ),
    ("If (E) were an option... Therefore:  \\[ \\boxed{\\text{C}} \\]", "(C)"),
    ("so \\boxed{B}", "(B)"),
]


class LetterTest(unittest.TestCase):
    def test_real_shapes(self):
        for answer, expected in REAL:
            self.assertEqual(letter(answer), expected, answer)

    def test_fallback_and_none(self):
        self.assertEqual(
            letter("I would go with (B) here, though (A) is close."), "(A)"
        )  # no phrase: last (X)
        self.assertEqual(letter("answer is (b)"), "(B)")
        self.assertIsNone(letter("no idea"))
        self.assertIsNone(letter(""))
        self.assertIsNone(letter(None))
        self.assertIsNone(letter("The answer is (E)"))  # not a GPQA letter

    def test_last_phrase_wins(self):
        self.assertEqual(letter("The answer is (A). Wait - no. The answer is (C)."), "(C)")

    def test_mmlu_pro_reads_ten_letters(self):
        self.assertEqual(letter("The answer is (J)", MMLU_PRO), "(J)")
        self.assertEqual(letter("**Answer: (G) 42**", MMLU_PRO), "(G)")
        self.assertEqual(letter("so \\boxed{H}", MMLU_PRO), "(H)")
        self.assertEqual(letter("I lean to (F), then (I)", MMLU_PRO), "(I)")
        self.assertIsNone(letter("The answer is (K)", MMLU_PRO))
        # GPQA keeps its four
        self.assertIsNone(letter("The answer is (J)"))


if __name__ == "__main__":
    unittest.main()
