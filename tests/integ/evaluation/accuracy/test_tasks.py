import json
import os
import tempfile
import unittest

from lsa.evaluation.accuracy import data, tasks


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as target:
        for row in rows:
            target.write(json.dumps(row) + "\n")


class RulerFileTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        rows = [
            {
                "index": 0,
                "input": "Some text.\nWhat is the magic number for apple?",
                "outputs": ["1234567"],
                "length": 100,
                "max_length": 4096,
                "gen_prefix": "The special magic number for apple mentioned in the provided text is",
            },
            {
                "index": 1,
                "input": "More text.\nWhat are the magic numbers for pear?",
                "outputs": ["111", "222"],
                "length": 100,
                "max_length": 4096,
                "gen_prefix": "The special magic numbers for pear mentioned in the provided text are",
            },
            {
                "index": 2,
                "input": "Docs.\nQuestion: who?",
                "outputs": ["Ann Lee", "Lee"],
                "length": 100,
                "max_length": 4096,
                "gen_prefix": "Answer:",
            },
        ]
        write_jsonl(data.ruler_path("niah_single_1", 4096, cls.root), rows)
        write_jsonl(data.ruler_path("qa_squad", 4096, cls.root), rows)
        write_jsonl(data.ruler_path("cwe", 4096, cls.root), rows)

    def test_prompt_is_input_plus_answer_prefix(self):
        definition = tasks.resolve("ruler_niah_single_1_4096", data_root=self.root)
        self.assertEqual(definition.length, 4096)
        self.assertEqual(definition.category, "long_context")
        task = definition.build()
        items = tasks.build_items(task)
        self.assertEqual([i.doc_id for i in items], [0, 1, 2])
        self.assertEqual(
            items[0].prompt,
            "Some text.\nWhat is the magic number for apple? The special magic number for apple mentioned in the provided text is",
        )
        self.assertEqual(definition.target(items[1].doc), ["111", "222"])

    def test_every_task_joins_its_answer_prefix_by_one_space_the_common_words_task_too(self):
        # NVIDIA's answer prefix opens with the space the fixed files strip, cwe's as much as any other
        items = tasks.build_items(tasks.resolve("ruler_cwe_4096", data_root=self.root).build())
        self.assertEqual(items[2].prompt, "Docs.\nQuestion: who? Answer:")

    def test_subset_keeps_true_doc_ids_in_ascending_order(self):
        task = tasks.resolve("ruler_niah_single_1_4096", data_root=self.root).build()
        items = tasks.build_items(task, ids=[2, 0])
        self.assertEqual([i.doc_id for i in items], [0, 2])
        self.assertTrue(items[1].prompt.startswith("Docs."))
        self.assertEqual(tasks.build_items(task, ids=[]), [])

    def test_string_match_all_and_part(self):
        task = tasks.resolve("ruler_niah_single_1_4096", data_root=self.root).build()
        items = tasks.build_items(task)
        scores = tasks.grade(task, items, ["It is 1234567.", "I found 111 only", "Ann Lee"])
        self.assertEqual([s["score"] for s in scores], [1.0, 0.5, 1.0])
        qa = tasks.resolve("ruler_qa_squad_4096", data_root=self.root).build()
        qitems = tasks.build_items(qa)
        qscores = tasks.grade(qa, qitems, ["1234567", "111", "lee"])
        self.assertEqual(
            [s["score"] for s in qscores], [1.0, 1.0, 1.0]
        )  # part: any reference counts, case-insensitive
        self.assertEqual(tasks.grade(qa, qitems, ["x", "y", "z"]), [{"score": 0.0}] * 3)

    def test_prompt_set_hash_is_stable_and_subset_dependent(self):
        task = tasks.resolve("ruler_niah_single_1_4096", data_root=self.root).build()
        a = tasks.prompt_set_hash(tasks.build_items(task, ids=[0, 1]))
        b = tasks.prompt_set_hash(tasks.build_items(task, ids=[1, 0]))
        c = tasks.prompt_set_hash(tasks.build_items(task, ids=[1, 2]))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class LongBenchFileTaskTest(unittest.TestCase):
    def test_build_and_grade(self):
        root = tempfile.mkdtemp()
        rows = [
            {
                "id": "a",
                "domain": "Single-Document QA",
                "sub_domain": "Legal",
                "difficulty": "easy",
                "length": "short",
                "answer": "B",
                "context_tokens": 10,
                "truncated": False,
                "prompt": "Please read ... (A) x (B) y ...",
            },
            {
                "id": "b",
                "domain": "Code Repository Understanding",
                "sub_domain": "Code",
                "difficulty": "hard",
                "length": "long",
                "answer": "D",
                "context_tokens": 10,
                "truncated": True,
                "prompt": "Please read ... code ...",
            },
        ]
        write_jsonl(data.longbench_path(data.longbench_card().context, root), rows)
        definition = tasks.resolve("longbench2", data_root=root)
        task = definition.build()
        items = tasks.build_items(task)
        self.assertEqual(items[1].prompt, "Please read ... code ...")
        scores = tasks.grade(task, items, ["The correct answer is (B).", "The correct answer is (A)."])
        self.assertEqual([s["acc"] for s in scores], [1.0, 0.0])
        self.assertEqual(definition.extract("blah The correct answer is (C)"), "C")
        self.assertEqual(definition.target(items[0].doc), "B")


class MmluProTaskTest(unittest.TestCase):
    def test_prompt_lists_the_questions_own_choices_under_gpqas_template(self):
        doc = {
            "question": "2 + 2?",
            "options": ["3", "4", "5"],
            "answer": "B",
            "category": "math",
            "src": "x",
        }
        prompt = tasks._mmlu_pro_text(doc)
        self.assertEqual(
            prompt,
            "What is the correct answer to this question: 2 + 2?\nChoices:\n(A) 3\n(B) 4\n(C) 5\n"
            "Let's think step by step: ",
        )
        definition = tasks.resolve("mmlu_pro")
        self.assertEqual(definition.category, "knowledge")
        self.assertEqual(definition.target(doc), "(B)")
        self.assertEqual(definition.extract("so the answer is (J)"), "(J)")
        self.assertEqual(definition.meta(doc), {"category": "math", "src": "x"})


if __name__ == "__main__":
    unittest.main()
