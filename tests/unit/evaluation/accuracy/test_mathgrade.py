import unittest

from lsa.evaluation.accuracy.grading import mathgrade


class MathGradeTest(unittest.TestCase):
    def test_boxed_integer(self):
        reply = "We compute step by step ... so the answer is $\\boxed{42}$."
        self.assertEqual(mathgrade.grade(reply, "42"), {"math_verify": 1, "boxed_match": 1})
        self.assertEqual(mathgrade.grade(reply, "41"), {"math_verify": 0, "boxed_match": 0})

    def test_last_boxed_wins(self):
        reply = "First I thought \\boxed{7}, but checking again the value is \\boxed{12}."
        self.assertEqual(mathgrade.grade(reply, "12")["boxed_match"], 1)
        self.assertEqual(mathgrade.grade(reply, "12")["math_verify"], 1)
        self.assertEqual(mathgrade.grade(reply, "7")["boxed_match"], 0)

    def test_equivalent_forms(self):
        # symbolic equivalence is what math_verify adds over the string grader
        self.assertEqual(mathgrade.math_verify_match("The answer is $\\boxed{\\frac{1}{2}}$", "0.5"), 1)
        self.assertEqual(mathgrade.math_verify_match("$\\boxed{\\dfrac{3}{4}}$", "\\frac{3}{4}"), 1)
        self.assertEqual(mathgrade.boxed_match("$\\boxed{\\dfrac{3}{4}}$", "\\frac{3}{4}"), 1)
        self.assertEqual(mathgrade.math_verify_match("$\\boxed{x^2+2x+1}$", "(x+1)^2"), 1)

    def test_no_box_and_garbage(self):
        self.assertEqual(mathgrade.grade("I do not know.", "5"), {"math_verify": 0, "boxed_match": 0})
        self.assertEqual(mathgrade.grade("", "5"), {"math_verify": 0, "boxed_match": 0})
        self.assertEqual(mathgrade.grade("\\boxed{", "5")["boxed_match"], 0)

    def test_gold_with_latex(self):
        gold = "\\left( 3, \\frac{\\pi}{2} \\right)"
        reply = "Converting to polar coordinates gives $\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}$."
        self.assertEqual(mathgrade.grade(reply, gold)["math_verify"], 1)
        self.assertEqual(mathgrade.grade(reply, gold)["boxed_match"], 1)


if __name__ == "__main__":
    unittest.main()
