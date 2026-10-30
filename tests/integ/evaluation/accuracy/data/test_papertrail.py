import json
import tempfile
import unittest
from pathlib import Path

from lsa import io
from lsa.evaluation.accuracy import data, runner, tasks
from lsa.evaluation.accuracy.data import papertrail, suites
from lsa.evaluation.accuracy.data.papertrail.rebuild import Rebuilt
from lsa.evaluation.accuracy.sampling import Sampling
from lsa.evaluation.constants import Suite

CONTEXT = "Setting.\n\n=== BEGIN PAPERS ===\n\n[Paper 0001 | 2026-08-25]\nSandpiper.\n\n[Paper 0002 | 2026-09-01]\nHost.\n"
REBUILT = Rebuilt(CONTEXT, [{"paper_no": 1, "id": "qec-A1"}, {"paper_no": 2, "id": "2609.17706"}], True)


def question(qid, tier, answer_type, answer, **extra):
    return {
        "id": qid,
        "sample": "qec",
        "trail": "A",
        "tier": tier,
        "question": f"question {qid}",
        "answer_type": answer_type,
        "answer": answer,
        "tol": 0.01,
        "unit": None,
        "answer_papers": ["qec-A1"],
        "check_papers": ["qec-A1"],
        "prompt_suffix": f"\n\nQuestion: {qid}?\nANSWER: <your answer>",
        "prompt_no_context": f"Question about papers not shown: {qid}?",
        **extra,
    }


class WordTokenizer:
    """Whitespace tokens under a one-token template - enough to check what the manifest counts."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<user> {messages[0]['content']} <assistant>"

    def encode(self, text, add_special_tokens=False):
        return text.split()


class PreparedConditionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        questions = [
            question("qec.A.1.1", "1", "number", 437),
            question("qec.A.3.1", "3", "ids", None, answer_ids=["qec-A1"]),
            question("qec.A.N.1", "NIC", "abstain", "NOT IN CORPUS"),
        ]
        for condition, rebuilt in (("L400000", REBUILT), ("C1", None)):
            papertrail.write_condition(
                "qec", condition, questions, rebuilt, cls.root, "abc123", WordTokenizer(), "words"
            )

    def test_the_context_is_stored_once_beside_rows_that_do_not_repeat_it(self):
        path = data.papertrail_path("qec", "L400000", self.root)
        self.assertEqual(papertrail.context_path(path).read_text(), CONTEXT)
        rows = io.read_jsonl(path)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(CONTEXT not in json.dumps(row) for row in rows))
        self.assertEqual(rows[0]["papers"], ["qec-A1", "2609.17706"])

    def test_the_manifest_counts_the_longest_prompt_under_the_chat_template(self):
        entry = data.load_manifest(self.root)["papertrail/qec_L400000.jsonl"]
        longest = max(
            len((CONTEXT + row["closing"]).split())
            for row in io.read_jsonl(data.papertrail_path("qec", "L400000", self.root))
        )
        self.assertEqual(entry["prompt_max"], longest + 2)
        self.assertEqual(
            (entry["rows"], entry["scored"], entry["not_in_corpus"], entry["generator"]), (3, 2, 1, "abc123")
        )
        self.assertTrue(entry["verified"])

    def test_a_task_joins_the_context_to_each_question_and_grades_by_paper_number(self):
        definition = tasks.resolve("papertrail_qec_L400000", data_root=self.root)
        self.assertEqual(
            (definition.benchmark, definition.length, definition.metric), (Suite.PAPERTRAIL, 400_000, "score")
        )
        task = definition.build()
        items = tasks.build_items(task)
        self.assertEqual([item.doc["qid"] for item in items], ["qec.A.1.1", "qec.A.3.1"])
        self.assertEqual(items[0].prompt, CONTEXT + "\n\nQuestion: qec.A.1.1?\nANSWER: <your answer>")
        scores = tasks.grade(
            task, items, ["EVIDENCE: Paper 0001\nANSWER: 437", "ANSWER: Paper 0001, Paper 0002"]
        )
        self.assertEqual(scores[0]["score"], 1.0)
        self.assertAlmostEqual(scores[1]["score"], 2 / 3)

    def test_graded_rows_carry_their_tier_and_the_record_scores_each_tier(self):
        definition = tasks.resolve("papertrail_qec_L400000", data_root=self.root)
        task = definition.build()
        items = tasks.build_items(task)
        replies = {0: "ANSWER: 437", 1: "ANSWER: Paper 0002"}
        rows = [
            {"doc_id": doc_id, "repeat": repeat, "answer": reply, "error": None, "finish_reason": "stop"}
            for repeat in (0, 1)
            for doc_id, reply in replies.items()
        ]
        plan = suites.Plan(
            "papertrail_qec_L400000", repeats=2, sampling=Sampling(temperature=1.0, top_p=0.95)
        )
        run_dir = tempfile.mkdtemp()
        entry = runner.grade_rows(plan, definition, task, items, rows, run_dir, {}, None)
        written = io.read_jsonl(Path(run_dir) / entry["samples"])
        self.assertEqual([row["meta"]["tier"] for row in written], ["1", "3", "1", "3"])
        self.assertEqual(entry["breakdown"]["tier"]["1"]["score"], 1.0)
        self.assertEqual(entry["breakdown"]["tier"]["3"]["score"], 0.0)
        self.assertEqual(entry["breakdown"]["tier"]["3"]["n"], 2)
        self.assertEqual(list(entry["breakdown"]["trail"]), ["A"])

    def test_answers_of_every_type_build_one_task(self):
        # A number beside a list of spellings is no Arrow column
        questions = [
            question("qec.A.1.1", "1", "number", 437),
            question("qec.A.2.1", "2", "text", ["Sandpiper-2", "SP2"]),
        ]
        papertrail.write_condition(
            "qec", "L700000", questions, REBUILT, self.root, "abc123", WordTokenizer(), "words"
        )
        task = tasks.resolve("papertrail_qec_L700000", data_root=self.root).build()
        items = tasks.build_items(task)
        scores = tasks.grade(task, items, ["ANSWER: 437", "ANSWER: SP2"])
        self.assertEqual([score["score"] for score in scores], [1.0, 1.0])

    def test_the_not_in_corpus_questions_are_their_own_task(self):
        task = tasks.resolve("papertrail_qec_L400000_nic", data_root=self.root).build()
        items = tasks.build_items(task)
        self.assertEqual([item.doc["tier"] for item in items], ["NIC"])
        self.assertEqual(tasks.grade(task, items, ["ANSWER: NOT IN CORPUS"])[0]["score"], 1.0)

    def test_the_question_alone_is_its_own_prompt(self):
        task = tasks.resolve("papertrail_qec_C1", data_root=self.root).build()
        items = tasks.build_items(task)
        self.assertEqual(items[0].prompt, "Question about papers not shown: qec.A.1.1?")
        self.assertIsNone(tasks.resolve("papertrail_qec_C1", data_root=self.root).length)


class PaperTrailSuiteTest(unittest.TestCase):
    def test_every_sample_runs_every_condition_three_times_and_the_not_in_corpus_control_at_512k(self):
        plans = suites.papertrail()
        self.assertEqual(len(plans), 3 * 11 + 3)
        self.assertTrue(all(plan.repeats == 3 and plan.dataset == Suite.PAPERTRAIL for plan in plans))
        controls = [plan.task for plan in plans if plan.task.endswith("_nic")]
        self.assertEqual(
            controls, [f"papertrail_{sample}_L512000_nic" for sample in ("exoplanets", "ml_longctx", "qec")]
        )

    def test_a_conditions_two_question_sets_share_its_cap_and_its_context(self):
        plans = {plan.task: plan for plan in suites.papertrail()}
        qec, qec_nic = plans["papertrail_qec_L5000000"], suites.Plan("papertrail_qec_L5000000_nic")
        exoplanets = plans["papertrail_exoplanets_L5000000"]
        self.assertEqual(qec.cap_group, qec_nic.cap_group)
        self.assertNotEqual(qec.cap_group, exoplanets.cap_group)
        self.assertEqual(qec.shared_context, qec_nic.shared_context)
        self.assertNotEqual(qec.shared_context, exoplanets.shared_context)
        self.assertIsNone(suites.Plan("ruler_cwe_4096").shared_context)

    def test_as_many_questions_fly_as_lsa_s_bucket_cache_holds_beside_the_context(self):
        concurrency = {plan.task: plan.concurrency for plan in suites.papertrail()}
        self.assertEqual(concurrency["papertrail_qec_C4_background"], 3)
        self.assertEqual(concurrency["papertrail_qec_L512000"], 3)
        self.assertEqual(concurrency["papertrail_qec_L1000000"], 2)
        self.assertEqual(concurrency["papertrail_qec_L5000000"], 1)

    def test_a_length_filter_keeps_that_length_of_every_sample(self):
        plans = suites.select("papertrail", lengths=[10_000_000])
        self.assertEqual(len(plans), 3)
        self.assertEqual({plan.length for plan in plans}, {10_000_000})

    def test_papertrail_runs_after_the_other_long_prompts_and_before_ruler(self):
        papertrail_phase = suites.papertrail()[0].phase
        self.assertEqual(Suite.PHASES[papertrail_phase], (Suite.PAPERTRAIL,))
        self.assertLess(papertrail_phase, Suite.PHASES.index((Suite.RULER,)))
