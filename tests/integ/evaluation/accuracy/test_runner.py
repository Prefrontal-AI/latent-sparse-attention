"""The whole path against the fake server: fixed samples in, record and sample files out, resume."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

from lsa import io
from lsa.evaluation.accuracy import records, regrade, runner
from lsa.evaluation.accuracy.data import hashes, suites
from lsa.evaluation.accuracy.sampling import Sampling
from lsa.evaluation.constants import Records, Suite
from tests.helpers.evaluation.fakes import FakeServer, make_data, respond

GENERATION = {
    "ruler": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "gpqa_diamond": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 163840},
    "aime25": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 81920},
}


def test_each_task_is_sampled_by_its_own_dataset_s_entry():
    gpqa = runner.sampled(suites.Plan("gpqa_diamond"), GENERATION).sampling
    assert (gpqa.temperature, gpqa.top_k, gpqa.max_tokens) == (1.0, 20, 163840)
    aime = runner.sampled(suites.Plan("aime25"), GENERATION).sampling
    assert (aime.temperature, aime.max_tokens) == (0.6, 81920)


def test_every_ruler_task_at_every_length_is_the_one_ruler_dataset():
    for task in ("ruler_vt_131072", "ruler_niah_multikey_3_4096"):
        ruler = runner.sampled(suites.Plan(task), GENERATION).sampling
        assert ruler.chat_template_kwargs == {"enable_thinking": False}, task


def test_a_dataset_the_block_does_not_name_is_refused_by_name():
    with pytest.raises(ValueError, match="'math500' entry"):
        runner.sampled(suites.Plan("math500"), GENERATION)


def test_a_plan_s_own_budget_beats_its_dataset_s_and_nothing_else_does():
    short = runner.sampled(suites.Plan("aime25", max_tokens=2048), GENERATION).sampling
    assert (short.max_tokens, short.temperature) == (2048, 0.6)


def test_a_plan_that_states_its_sampling_keeps_it():
    pinned = suites.Plan("gpqa_diamond", sampling=Sampling(max_tokens=64))
    assert runner.sampled(pinned, GENERATION) is pinned


class RunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # What a stock sglang with the plugin reports - nothing about the indexer it runs
        info = {
            "model_path": "zai-org/Fake-Model",
            "tp_size": 1,
            "max_running_requests": 16,
            "version": "fake",
        }
        cls.server = FakeServer(respond, info=info).start()
        cls.data = tempfile.mkdtemp()
        make_data(cls.data)
        cls.plans = [
            suites.Plan(
                "ruler_niah_single_1_4096",
                n=4,
                repeats=2,
                sampling=Sampling(max_tokens=64, reasoning_effort="low"),
                concurrency=3,
            ),
            suites.Plan("ruler_cwe_4096", n=6, sampling=Sampling(max_tokens=64), concurrency=3),
        ]
        cls.patch = mock.patch.dict(suites.SUITES, {"test": lambda: cls.plans})
        cls.patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls.server.stop()

    def base_url(self):
        return self.server.base_url

    def test_run_writes_record_and_samples_and_resumes(self):
        out = tempfile.mkdtemp()
        logs = []
        run = runner.Runner(
            self.base_url(),
            "test",
            out_root=out,
            note="a note",
            tag="t1",
            data_root=self.data,
            strict_hashes=True,
            log=logs.append,
            name="test",
        )
        header = run.run()
        self.assertRegex(run.run_id, r"^test__\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}-t1$")
        rec = json.loads(Path(run.out, "record.json").read_text())
        self.assertEqual(rec["schema"], Records.SCHEMA)
        self.assertEqual(rec["run"]["note"], "a note")
        self.assertEqual(rec["model"]["name"], "Fake-Model")
        self.assertEqual(rec["model"]["label"], "fake-model")
        # No settings file named, so the server is its own indexer
        self.assertEqual((rec["server"]["indexer"], rec["run"]["name"]), ("native", "test"))
        # A second run of the same kind in the same second gets a counter
        self.assertEqual(runner.unused_run_id(run.run_id, out), run.run_id + "-2")
        self.assertEqual(rec["run"]["plans"][0]["task"], "ruler_niah_single_1_4096")
        self.assertTrue(os.path.exists(os.path.join(run.out, "server_info.json")))

        niah = rec["tasks"]["ruler_niah_single_1_4096"]
        # docs 0..3, two repeats: keys 0,2 right, key1 wrong, key3 empty -> 2/4 per repeat
        self.assertEqual(niah["n"], 8)
        self.assertEqual(niah["scored"], 8)
        self.assertEqual(niah["errors"], 0)
        self.assertEqual(niah["score"], 0.5)
        self.assertEqual(niah["repeat_scores"], [0.5, 0.5])
        self.assertEqual(niah["repeat_std"], 0.0)
        self.assertEqual(niah["empty"], 2)
        self.assertEqual(niah["truncated"], 2)
        self.assertEqual(niah["metric"], "score")
        self.assertEqual(niah["length"], 4096)
        self.assertEqual(niah["category"], "long_context")
        self.assertEqual(
            niah["sampling"],
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 64,
                "reasoning_effort": "low",
                "thinking_budget": None,
                "top_k": None,
                "min_p": None,
                "presence_penalty": None,
                "chat_template_kwargs": None,
                "think": True,
            },
        )
        self.assertEqual(niah["tokens"]["reasoning"]["max"], 20)
        self.assertEqual(niah["tokens"]["prompt"]["mean"], niah["tokens"]["prompt"]["median"])
        self.assertIsNotNone(niah["prompt_set_hash"])
        self.assertGreater(niah["seconds"], 0)

        cwe = rec["tasks"]["ruler_cwe_4096"]
        self.assertEqual(cwe["n"], 6)
        self.assertEqual(cwe["score"], 0.5)  # keys 0,2,4 right out of 6
        self.assertEqual(cwe["repeat_std"], None)

        rows = io.read_jsonl(os.path.join(run.out, niah["samples"]))
        self.assertEqual(
            [(r["repeat"], r["doc_id"]) for r in rows],
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        )
        self.assertEqual(rows[0]["metrics"], {"score": 1.0})
        self.assertEqual(rows[0]["target"], ["1000000"])
        self.assertEqual(rows[0]["reasoning"], "found it")
        self.assertEqual(rows[0]["answer_tokens"], rows[0]["completion_tokens"] - rows[0]["reasoning_tokens"])
        self.assertEqual(rows[3]["answer"], "")
        self.assertEqual(rows[3]["finish_reason"], "length")
        self.assertEqual(rows[3]["metrics"], {"score": 0.0})
        # every request carried the plan's sampling settings
        body = self.server.requests[-1]
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(body["temperature"], 0.0)

        # A second run over the same directory sends nothing and reproduces the record
        before = len(self.server.requests)
        again = runner.Runner(
            self.base_url(), "test", out_root=out, data_root=self.data, log=logs.append, name="test"
        )
        again.run_id = run.run_id
        again.out = run.out
        again.run()
        self.assertEqual(len(self.server.requests), before)
        rec2 = json.loads(Path(run.out, "record.json").read_text())
        self.assertEqual(rec2["tasks"]["ruler_niah_single_1_4096"]["score"], 0.5)
        self.assertEqual(rec2["tasks"]["ruler_niah_single_1_4096"]["n"], 8)
        self.assertEqual(header["tasks"].keys(), rec2["tasks"].keys())

    def test_the_record_says_what_each_task_read_and_what_its_replies_did(self):
        out = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "4242"}):
            run = runner.Runner(
                self.base_url(),
                "test",
                out_root=out,
                data_root=self.data,
                log=lambda *_: None,
                name="test",
                image="stock.img",
            )
        rec = run.run()
        # A stock server reports no context parallel, which is recorded as off rather than left out
        mesh = rec["servers"]["4242"]["mesh"]
        self.assertEqual((mesh["label"], mesh["image"]), ("tp1-cp1-dcp1-pp1-ep1", "stock.img"))
        self.assertEqual(rec["server"]["mesh"], mesh)
        self.assertEqual(rec["run"]["edits"], [])
        # Whether a run finished is worked out where it is read, never stored
        self.assertNotIn("status", rec["run"])
        self.assertEqual((rec["run"]["name"], rec["attention"]["map"]), ("test", None))
        self.assertEqual(rec["model"]["name"], "Fake-Model")
        self.assertEqual((rec["servers"]["4242"]["job"], rec["servers"]["4242"]["tp_size"]), ("4242", 1))
        niah, cwe = rec["tasks"]["ruler_niah_single_1_4096"], rec["tasks"]["ruler_cwe_4096"]
        self.assertEqual(niah["server"], "4242")
        self.assertEqual(
            (niah["data"]["root"], niah["data"]["file"]), (self.data, "ruler/niah_single_1_4096.jsonl")
        )
        # No manifest beside the fake files, so no corpus to name
        self.assertIsNone(niah["data"]["corpus"])
        self.assertEqual((niah["prompt"]["delimiter"], cwe["prompt"]["delimiter"]), (" ", " "))
        self.assertEqual((niah["prompt"]["hash"], niah["prompt"]["strict"]), (niah["prompt_set_hash"], True))
        # "guessing" is one reasoning token, which a forced close of the think block also is
        self.assertEqual(niah["observed"], {"reasoned": 0.75, "cut": 0.25, "budget_enforced": None})
        self.assertEqual(niah["protocol"], records.protocol_key(niah))

    def test_a_continued_run_names_the_server_of_each_task(self):
        out = tempfile.mkdtemp()
        quiet = {"out_root": out, "data_root": self.data, "log": lambda *_: None}
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "1"}):
            first = runner.Runner(
                self.base_url(), "test", only=["ruler_niah_single_1_4096"], **quiet, name="test"
            )
            first.run()
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "2"}):
            rec = runner.Runner(
                self.base_url(), "test", run_id=first.run_id, only=["ruler_cwe_4096"], **quiet
            ).run()
        self.assertEqual(set(rec["servers"]), {"1", "2"})
        self.assertEqual(
            (rec["tasks"]["ruler_niah_single_1_4096"]["server"], rec["tasks"]["ruler_cwe_4096"]["server"]),
            ("1", "2"),
        )
        rebuilt = regrade.regrade(first.out, data_root=self.data, log=lambda *_: None)
        self.assertEqual(rebuilt["tasks"]["ruler_cwe_4096"]["server"], "2")
        self.assertEqual(
            rebuilt["tasks"]["ruler_cwe_4096"]["prompt"], rec["tasks"]["ruler_cwe_4096"]["prompt"]
        )
        self.assertEqual(
            rebuilt["tasks"]["ruler_cwe_4096"]["protocol"], rec["tasks"]["ruler_cwe_4096"]["protocol"]
        )

    def test_a_run_with_neither_a_name_nor_an_id_is_refused(self):
        with self.assertRaisesRegex(ValueError, "name or the run id"):
            runner.Runner(self.base_url(), "test", out_root=tempfile.mkdtemp(), data_root=self.data)

    def test_a_run_continued_by_its_id_alone_is_named_by_it(self):
        run = runner.Runner(
            self.base_url(), "test", out_root=tempfile.mkdtemp(), data_root=self.data, run_id="pinned"
        )
        self.assertEqual((run.run_id, run.label), ("pinned", "pinned"))

    def test_a_record_of_another_schema_is_neither_continued_nor_regraded(self):
        out = tempfile.mkdtemp()
        run = runner.Runner(
            self.base_url(), "test", out_root=out, data_root=self.data, log=lambda *_: None, name="test"
        )
        run.run()
        path = Path(run.out, "record.json")
        path.write_text(json.dumps({**json.loads(path.read_text()), "schema": "lsa-accuracy/1"}))
        with self.assertRaisesRegex(ValueError, "lsa-accuracy/1 record"):
            runner.Runner(
                self.base_url(),
                "test",
                out_root=out,
                data_root=self.data,
                run_id=run.run_id,
                log=lambda *_: None,
            )
        with self.assertRaisesRegex(ValueError, "lsa-accuracy/1 record"):
            regrade.regrade(run.out, data_root=self.data, log=lambda *_: None)

    def test_hash_check_stops_a_run_on_changed_prompts(self):
        out = tempfile.mkdtemp()
        run = runner.Runner(
            self.base_url(),
            "test",
            out_root=out,
            data_root=self.data,
            only=["ruler_cwe_4096"],
            log=lambda *_: None,
            name="test",
        )
        run.expected_hashes = {"ruler_cwe_4096:first6": "0000000000000000"}
        with self.assertRaises(RuntimeError):
            run.run()
        run.strict = False
        run.run()  # warns, continues
        self.assertIn("ruler_cwe_4096", run.results)
        self.assertNotIn("ruler_niah_single_1_4096", run.results)

    def test_continued_run_with_a_task_subset_keeps_the_rest_of_the_record(self):
        out = tempfile.mkdtemp()
        run = runner.Runner(
            self.base_url(), "test", out_root=out, data_root=self.data, log=lambda *_: None, name="test"
        )
        first = run.run()
        again = runner.Runner(
            self.base_url(),
            "test",
            out_root=out,
            data_root=self.data,
            log=lambda *_: None,
            run_id=run.run_id,
            only=["ruler_cwe_4096"],
        )
        rec = again.run()
        self.assertEqual(set(rec["tasks"]), {"ruler_niah_single_1_4096", "ruler_cwe_4096"})
        self.assertEqual(rec["tasks"]["ruler_niah_single_1_4096"], first["tasks"]["ruler_niah_single_1_4096"])
        self.assertEqual(
            {p["task"] for p in rec["run"]["plans"]}, {"ruler_niah_single_1_4096", "ruler_cwe_4096"}
        )
        self.assertEqual(rec["run"]["started"], first["run"]["started"])
        self.assertGreaterEqual(rec["run"]["date"], first["run"]["date"])
        # The record on disk and the one returned are the same record
        on_disk = json.loads(Path(again.out, "record.json").read_text())
        self.assertEqual(on_disk["run"]["started"], rec["run"]["started"])
        # and regrade still covers every task of the record
        rebuilt = regrade.regrade(run.out, data_root=self.data, log=lambda *_: None)
        self.assertEqual(set(rebuilt["tasks"]), {"ruler_niah_single_1_4096", "ruler_cwe_4096"})

    def test_a_continued_run_adds_its_time_to_what_the_earlier_launches_spent(self):
        out = tempfile.mkdtemp()
        run = runner.Runner(
            self.base_url(), "test", out_root=out, data_root=self.data, log=lambda *_: None, name="test"
        )
        run.run()
        path = Path(run.out, "record.json")
        earlier = json.loads(path.read_text())
        earlier["run"]["seconds"] = 1000.0
        path.write_text(json.dumps(earlier))
        again = runner.Runner(
            self.base_url(),
            "test",
            out_root=out,
            data_root=self.data,
            log=lambda *_: None,
            run_id=run.run_id,
            only=["ruler_cwe_4096"],
        )
        rec = again.run()
        self.assertGreaterEqual(rec["run"]["seconds"], 1000.0)
        self.assertLess(rec["run"]["seconds"], 1000.0 + 600)

    def test_started_is_the_moment_the_run_began_and_not_the_moment_it_last_wrote(self):
        """`run` writes the record once before the first task and again after every one of them.

        `started` used to come off the clock in each of those writes, which made it the time of the last one:
        a six-hour run recorded a start a second before its end, `started` and `date` read as duplicates of
        each other, and the field said nothing. The older test for this only failed when two writes happened
        to straddle a second boundary, which is why it stood for so long.
        """
        out = tempfile.mkdtemp()
        run = runner.Runner(
            self.base_url(), "test", out_root=out, data_root=self.data, log=lambda *_: None, name="test"
        )
        pinned = run.started
        self.assertIsNotNone(pinned)
        rec = run.run()
        self.assertEqual(rec["run"]["started"], pinned)
        on_disk = json.loads(Path(run.out, "record.json").read_text())
        self.assertEqual(on_disk["run"]["started"], pinned)

    def test_rebudgeting_moves_the_budget_and_leaves_the_prompt_set_alone(self):
        """A budget override has to be invisible to the prompts, or it is not a comparable run.

        `n`, `subset` and `repeats` are what pick the documents, so they have to survive untouched - a
        rebudgeted run answers the same prompt set and its recorded hash still holds.
        """
        plan = suites.Plan(
            "ruler_cwe_4096", n=6, repeats=2, sampling=Sampling(max_tokens=64, reasoning_effort="low")
        )
        raised = runner.rebudget(plan, 4096)
        self.assertEqual(raised.sampling.max_tokens, 4096)
        self.assertEqual(raised.sampling.reasoning_effort, "low")
        self.assertEqual(
            (raised.task, raised.n, raised.subset, raised.repeats), ("ruler_cwe_4096", 6, None, 2)
        )
        self.assertEqual(raised.key, plan.key)
        # The original is untouched, since a Plan is frozen and the suite hands out the same objects again
        self.assertEqual(plan.sampling.max_tokens, 64)

    def test_max_tokens_rebudgets_every_selected_plan_and_the_record_says_so(self):
        out = tempfile.mkdtemp()
        run = runner.Runner(
            self.base_url(),
            "test",
            out_root=out,
            data_root=self.data,
            max_tokens=256,
            log=lambda *_: None,
            name="test",
        )
        self.assertEqual([plan.sampling.max_tokens for plan in run.plans], [256, 256])
        rec = run.run()
        budgets = {plan["task"]: plan["sampling"]["max_tokens"] for plan in rec["run"]["plans"]}
        self.assertEqual(budgets, {"ruler_niah_single_1_4096": 256, "ruler_cwe_4096": 256})

    def test_no_max_tokens_leaves_the_suite_budgets_as_they_are(self):
        out = tempfile.mkdtemp()
        run = runner.Runner(
            self.base_url(), "test", out_root=out, data_root=self.data, log=lambda *_: None, name="test"
        )
        self.assertEqual([plan.sampling.max_tokens for plan in run.plans], [64, 64])

    def test_a_plan_with_no_sampling_of_its_own_is_sampled_by_its_dataset(self):
        plans = [suites.Plan("ruler_niah_single_1_4096", n=2)]
        out = tempfile.mkdtemp()
        with mock.patch.dict(suites.SUITES, {"generated": lambda: plans}):
            run = runner.Runner(
                self.base_url(),
                "generated",
                out_root=out,
                data_root=self.data,
                generation=GENERATION,
                log=lambda *_: None,
                name="test",
            )
            rec = run.run()
        self.assertEqual(rec["run"]["generation"], GENERATION)
        sampling = rec["tasks"]["ruler_niah_single_1_4096"]["sampling"]
        self.assertEqual(sampling["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual((sampling["temperature"], sampling["max_tokens"]), (0.0, 2048))

    def test_a_plan_with_no_sampling_needs_a_generation_block(self):
        plans = [suites.Plan("ruler_niah_single_1_4096", n=2)]
        with mock.patch.dict(suites.SUITES, {"generated": lambda: plans}):
            with self.assertRaisesRegex(ValueError, "generation block"):
                runner.Runner(
                    self.base_url(),
                    "generated",
                    out_root=tempfile.mkdtemp(),
                    data_root=self.data,
                    name="test",
                )

    def test_resuming_a_task_under_other_sampling_is_refused(self):
        """Samples drawn two ways would pool into one score that is neither protocol."""
        out = tempfile.mkdtemp()
        first = runner.Runner(
            self.base_url(), "test", out_root=out, data_root=self.data, run_id="mixed", log=lambda *_: None
        )
        first.run()
        second = runner.Runner(
            self.base_url(),
            "test",
            out_root=out,
            data_root=self.data,
            run_id="mixed",
            max_tokens=256,
            log=lambda *_: None,
        )
        with self.assertRaisesRegex(ValueError, "--no-resume"):
            second.run()

    def test_a_length_the_server_cannot_hold_is_skipped_and_the_actual_length_recorded(self):
        out = tempfile.mkdtemp()
        # The server holds 20000 tokens - the 32k cell does not fit, the 4k ones do
        measured = {"rows": 6, "sha256": "f", "tokenizer": "t", "date": "2026-09-19"}
        manifest = {
            "ruler/cwe_4096.jsonl": {**measured, "prompt_median": 2999, "prompt_max": 3900},
            "ruler/cwe_32768.jsonl": {**measured, "prompt_median": 31000, "prompt_max": 32000},
            "ruler/niah_single_1_4096.jsonl": {**measured, "prompt_median": 3790, "prompt_max": 3790},
        }
        Path(self.data, "manifest.json").write_text(json.dumps(manifest))
        import shutil

        shutil.copy(
            os.path.join(self.data, "ruler", "cwe_4096.jsonl"),
            os.path.join(self.data, "ruler", "cwe_32768.jsonl"),
        )
        plans = self.plans + [suites.Plan("ruler_cwe_32768", n=2, sampling=Sampling(max_tokens=64))]
        try:
            with mock.patch.dict(suites.SUITES, {"test4": lambda: plans}):
                run = runner.Runner(
                    self.base_url(),
                    "test4",
                    out_root=out,
                    data_root=self.data,
                    log=lambda *_: None,
                    name="test",
                )
                run.info["context_length"] = 20000
                rec = run.run()
        finally:
            os.remove(os.path.join(self.data, "manifest.json"))
            os.remove(os.path.join(self.data, "ruler", "cwe_32768.jsonl"))
        self.assertEqual(set(rec["tasks"]), {"ruler_niah_single_1_4096", "ruler_cwe_4096"})
        self.assertIn(
            "32064 tokens of prompt and answer over the server's 20000", rec["skipped"]["ruler_cwe_32768"]
        )
        self.assertEqual(rec["tasks"]["ruler_niah_single_1_4096"]["length_actual"], 3790)
        self.assertEqual(rec["tasks"]["ruler_cwe_4096"]["length_actual"], 2999)

    def test_a_continued_run_names_the_server_of_every_task(self):
        out = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "111"}):
            run = runner.Runner(
                self.base_url(),
                "test",
                out_root=out,
                data_root=self.data,
                log=lambda *_: None,
                name="test",
            )
            first = run.run()
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "222", "SLURM_JOB_NUM_NODES": "2"}):
            again = runner.Runner(
                self.base_url(),
                "test",
                out_root=out,
                data_root=self.data,
                log=lambda *_: None,
                run_id=run.run_id,
                only=["ruler_cwe_4096"],
            )
            rec = again.run()
        self.assertEqual(sorted(rec["servers"]), ["111", "222"])
        self.assertEqual(rec["run"]["suite"], "test")
        self.assertEqual(rec["tasks"]["ruler_niah_single_1_4096"]["server"], "111")
        self.assertEqual(rec["tasks"]["ruler_cwe_4096"]["server"], "222")
        # The first server stays where it was, and each one says how many nodes it spanned
        self.assertEqual(rec["server"], first["server"])
        self.assertEqual(first["servers"]["111"]["nodes"], 1)
        self.assertEqual(rec["servers"]["222"]["nodes"], 2)
        self.assertEqual(rec["servers"]["222"]["indexer"], "native")

    def test_samples_resize_the_ruler_plans_and_logprobs_land_in_the_rows(self):
        out = tempfile.mkdtemp()
        logs = []
        run = runner.Runner(
            self.base_url(),
            "test",
            out_root=out,
            data_root=self.data,
            samples=5,
            top_logprobs=2,
            log=logs.append,
            name="test",
        )
        self.assertEqual([plan.n for plan in run.plans], [5, 5])
        rec = run.run()
        cwe = rec["tasks"]["ruler_cwe_4096"]
        self.assertEqual((cwe["n_docs"], cwe["key"]), (5, "ruler_cwe_4096:first5"))
        # The grid's hashes are pinned for its own counts, so a resized set is held to none
        self.assertTrue(any("no expected value recorded" in line for line in logs))
        rows = io.read_jsonl(os.path.join(run.out, cwe["samples"]))
        answered = [row for row in rows if row["answer"]]
        self.assertTrue(answered)
        self.assertTrue(all(row["first_token"]["logprob"] == -0.25 for row in answered))

    def test_lengths_pick_the_cells_of_a_suite(self):
        out = tempfile.mkdtemp()
        run = runner.Runner(
            self.base_url(),
            "test",
            out_root=out,
            data_root=self.data,
            lengths=[4096],
            log=lambda *_: None,
            name="test",
        )
        self.assertEqual(len(run.plans), 2)
        with self.assertRaises(KeyError):
            runner.Runner(
                self.base_url(),
                "test",
                out_root=out,
                data_root=self.data,
                lengths=[2000000],
                log=lambda *_: None,
                name="test",
            )

    def test_a_longbench_budget_the_context_cannot_hold_over_its_prompts_is_refused(self):
        plans = [suites.Plan("longbench2", sampling=Sampling(max_tokens=163841))]
        original = self.server.info.get("context_length")
        self.server.info["context_length"] = 1048576
        try:
            with mock.patch.dict(suites.SUITES, {"longbench": lambda: plans}):
                with self.assertRaisesRegex(ValueError, "budget of 163841 run past the server's 1048576"):
                    runner.Runner(
                        self.base_url(),
                        "longbench",
                        out_root=tempfile.mkdtemp(),
                        data_root=self.data,
                        name="test",
                    )
                # The card's 884,736 and the card's 163,840 are exactly a 1,048,576 server
                fits = [suites.Plan("longbench2", sampling=Sampling(max_tokens=163840))]
                with mock.patch.dict(suites.SUITES, {"longbench": lambda: fits}):
                    runner.Runner(
                        self.base_url(),
                        "longbench",
                        out_root=tempfile.mkdtemp(),
                        data_root=self.data,
                        name="test",
                    )
                # Prompts prepared past a shorter server's context are lost whatever the budget - not refused here
                self.server.info["context_length"] = 262144
                runner.Runner(
                    self.base_url(),
                    "longbench",
                    out_root=tempfile.mkdtemp(),
                    data_root=self.data,
                    name="test",
                )
        finally:
            self.server.info["context_length"] = original

    def test_missing_data_is_skipped_and_recorded(self):
        out = tempfile.mkdtemp()
        plans = self.plans + [suites.Plan("ruler_vt_4096", n=2, sampling=Sampling(max_tokens=64))]
        with mock.patch.dict(suites.SUITES, {"test2": lambda: plans}):
            run = runner.Runner(
                self.base_url(),
                "test2",
                out_root=out,
                data_root=self.data,
                log=lambda *_: None,
                name="test",
            )
            rec = run.run()
        self.assertEqual(set(rec["tasks"]), {"ruler_niah_single_1_4096", "ruler_cwe_4096"})
        self.assertIn("ruler_vt_4096", rec["skipped"])
        self.assertIn("vt_4096.jsonl", rec["skipped"]["ruler_vt_4096"])

    def test_the_record_names_the_model_by_the_hub_id_it_was_given(self):
        # The served path is a checkpoint directory, which names the copy rather than the model
        out = tempfile.mkdtemp()
        quiet = {"out_root": out, "data_root": self.data, "log": lambda *_: None}
        first = runner.Runner(self.base_url(), "test", model="org/Other-Model", **quiet, name="test")
        first.run()
        resumed = runner.Runner(self.base_url(), "test", run_id=first.run_id, **quiet)
        rec = resumed.run()
        self.assertEqual(rec["model"]["name"], "org/Other-Model")
        self.assertEqual(rec["model"]["label"], "other-model")
        self.assertEqual(rec["model"]["path"], "zai-org/Fake-Model")

    def served_with(self, override):
        """Run against a fake server whose `/server_info` reports this model config override."""
        info = {
            "model_path": "zai-org/Fake-Model",
            "max_running_requests": 16,
            "version": "fake",
            "json_model_override_args": json.dumps(override),
        }
        server = FakeServer(respond, info=info).start()
        self.addCleanup(server.stop)
        told = runner.Runner(
            server.base_url,
            "test",
            out_root=tempfile.mkdtemp(),
            data_root=self.data,
            log=lambda *_: None,
            name="test",
        )
        told.run()
        return json.loads(Path(told.out, "record.json").read_text())

    def test_the_settings_block_on_the_servers_info_says_the_server_runs_lsa(self):
        body = {"bucket_size": 64, "top_k": 2048, "candidates": 1, "scan": "fp8", "projections": "w.pt"}
        rec = self.served_with(
            {"rope_scaling": {"factor": 4}, "lsa": {**body, "debug": {"check_index": True}}}
        )
        self.assertEqual(rec["server"]["indexer"], "lsa")
        self.assertEqual(rec["run"]["name"], "test")
        # The record quotes the design without the debugging aids
        self.assertEqual(rec["server"]["config"], body)
        self.assertNotIn("config_file", rec["server"])

    def test_a_server_that_reports_no_block_runs_the_checkpoints_own_indexer(self):
        untold = runner.Runner(
            self.base_url(),
            "test",
            out_root=tempfile.mkdtemp(),
            data_root=self.data,
            log=lambda *_: None,
            name="test",
        )
        untold.run()
        plain = json.loads(Path(untold.out, "record.json").read_text())
        self.assertEqual(plain["server"]["indexer"], "native")
        self.assertEqual(plain["run"]["name"], "test")
        self.assertIsNone(plain["server"]["config"])

    def test_a_block_of_debugging_aids_alone_is_the_checkpoints_own_indexer(self):
        rec = self.served_with({"lsa": {"debug": {"check_index": True}}})
        self.assertEqual(rec["server"]["indexer"], "native")
        self.assertIsNone(rec["server"]["config"])

    def test_a_pisa_block_on_the_servers_info_says_the_server_runs_pisa(self):
        rec = self.served_with({"lsa": {"selector": "pisa", "block_size": 64, "keep_count": 72}})
        self.assertEqual(rec["server"]["indexer"], "pisa")
        self.assertEqual(rec["attention"]["indexer"], "pisa")
        # PISA reads no map
        self.assertIsNone(rec["attention"]["map"])

    def test_make_hashes_records_plans_with_data_and_skips_the_rest(self):
        path = os.path.join(tempfile.mkdtemp(), "hashes.json")
        plans = self.plans + [suites.Plan("ruler_vt_4096", n=2)]
        with (
            mock.patch.dict(suites.SUITES, {"test3": lambda: plans}),
            mock.patch.object(suites, "HASHES_PATH", path),
        ):
            out = hashes.make_hashes(suite_names=("test3",), data_root=self.data, log=lambda *_: None)
        self.assertEqual(set(out), {"ruler_niah_single_1_4096:first4", "ruler_cwe_4096:first6"})
        self.assertEqual(json.loads(Path(path).read_text()), out)
        # the recorded value is what a run computes for the same plan
        run = runner.Runner(
            self.base_url(),
            "test",
            out_root=tempfile.mkdtemp(),
            data_root=self.data,
            log=lambda *_: None,
            name="test",
        )
        run.expected_hashes = out
        run.run()  # no RuntimeError: hashes match
        self.assertEqual(run.results["ruler_cwe_4096"]["prompt_set_hash"], out["ruler_cwe_4096:first6"])

    def test_resume_drops_stale_failures_and_regrade_rebuilds_the_record(self):
        out = tempfile.mkdtemp()
        run = runner.Runner(
            self.base_url(), "test", out_root=out, data_root=self.data, log=lambda *_: None, name="test"
        )
        rec = run.run()
        path = os.path.join(run.out, rec["tasks"]["ruler_cwe_4096"]["samples"])
        rows = io.read_jsonl(path)
        # a stale failure of a doc that later succeeded, and a duplicate success
        stale = dict(rows[0], answer="", metrics=None, error="http 500")
        io.write_jsonl(path, [stale] + rows + [rows[1]])
        again = runner.Runner(
            self.base_url(), "test", out_root=out, data_root=self.data, log=lambda *_: None, run_id=run.run_id
        )
        rec2 = again.run()
        self.assertEqual(
            (rec2["tasks"]["ruler_cwe_4096"]["n"], rec2["tasks"]["ruler_cwe_4096"]["errors"]), (6, 0)
        )
        self.assertEqual(len(io.read_jsonl(path)), 6)
        # regrade without a server gives the same scores and keeps the timing
        rebuilt = regrade.regrade(run.out, data_root=self.data, log=lambda *_: None)
        for task in rec2["tasks"]:
            for key in ("score", "n", "empty", "metrics", "tokens", "seconds", "prompt_set_hash"):
                self.assertEqual(rebuilt["tasks"][task][key], rec2["tasks"][task][key], (task, key))
        self.assertIn("regraded", rebuilt["run"])

    def test_summarize_with_errors(self):
        rows = [
            {
                "doc_id": 0,
                "repeat": 0,
                "answer": "x",
                "metrics": {"m": 1.0},
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "reasoning_tokens": 2,
                "answer_tokens": 3,
                "finish_reason": "stop",
                "error": None,
            },
            {"doc_id": 1, "repeat": 0, "answer": "", "error": "http 500"},
            {
                "doc_id": 2,
                "repeat": 0,
                "answer": "",
                "metrics": {"m": 0.0},
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "reasoning_tokens": 7,
                "answer_tokens": 0,
                "finish_reason": "length",
                "error": None,
            },
        ]
        out = records.summarize(rows, "m")
        self.assertEqual(
            (out["n"], out["scored"], out["errors"], out["empty"], out["truncated"]), (3, 2, 1, 1, 1)
        )
        self.assertEqual(out["score"], 0.5)
        # The truncated sample is the only zero, so the score over what finished is 1.0 and the
        # reported 0.5 is the completion budget rather than the model
        self.assertEqual((out["finished"], out["score_finished"]), (1, 1.0))
        self.assertEqual(out["tokens"]["prompt"]["mean"], 11.0)
        self.assertEqual(out["tokens"]["answer"]["sum"], 3)
        self.assertEqual(records.summarize([rows[1]], "m")["score"], None)
        # two repeats, every request failed: no score, no spread, no crash
        failed = [dict(rows[1], repeat=0), dict(rows[1], repeat=1)]
        out = records.summarize(failed, "m")
        self.assertEqual(
            (out["score"], out["repeat_std"], out["repeat_scores"], out["errors"]),
            (None, None, [None, None], 2),
        )

    def test_score_finished_when_every_sample_was_cut(self):
        # `ruler_fwe` on every Qwen rung: 25 of 25 cut, so there is no uncensored score to report and
        # the field says so rather than reading 0.0 and looking like a measurement
        cut = {
            "doc_id": 0,
            "repeat": 0,
            "answer": "",
            "metrics": {"m": 0.0},
            "finish_reason": "length",
            "error": None,
        }
        out = records.summarize([dict(cut, doc_id=i) for i in range(4)], "m")
        self.assertEqual((out["score"], out["truncated"], out["finished"]), (0.0, 4, 0))
        self.assertIsNone(out["score_finished"])

    def test_run_stops_when_the_server_disappears(self):
        out = tempfile.mkdtemp()
        run = runner.Runner(
            self.base_url(), "test", out_root=out, data_root=self.data, log=lambda *_: None, name="test"
        )
        run.base_url = "http://127.0.0.1:9"  # nothing listens here
        with self.assertRaises(RuntimeError) as caught:
            run.run()
        self.assertIn("--run-id", str(caught.exception))
        rec = json.loads(Path(run.out, "record.json").read_text())
        # Stopped by the first task to finish - the other was still retrying
        self.assertEqual(len(rec["tasks"]), 1)
        (stopped,) = rec["tasks"].values()
        self.assertEqual(stopped["errors"], stopped["n"])
        # continued against a live server: the failed rows are retried and do not pile up in the file
        again = runner.Runner(
            self.base_url(), "test", out_root=out, data_root=self.data, log=lambda *_: None, run_id=run.run_id
        )
        rec = again.run()
        niah = rec["tasks"]["ruler_niah_single_1_4096"]
        self.assertEqual((niah["n"], niah["errors"], niah["score"]), (8, 0, 0.5))
        self.assertEqual(len(io.read_jsonl(os.path.join(again.out, niah["samples"]))), 8)


def slow(messages):
    """The usual answers, each held long enough that the requests of a run overlap, grouped by their text."""
    group = "uncapped" if "Uncapped" in messages[-1]["content"] else "capped"
    return {**respond(messages), "delay": 0.2, "group": group}


def run_against(server_cap, plans, data, out_root=None, extra_info=None, **kwargs):
    """Run `plans` against a slow fake server of this request cap; return the record and the server."""
    info = {
        "model_path": "zai-org/Fake-Model",
        "max_running_requests": server_cap,
        "version": "fake",
        **(extra_info or {}),
    }
    server = FakeServer(slow, info=info).start()
    try:
        with mock.patch.dict(suites.SUITES, {"pool": lambda: plans}):
            run = runner.Runner(
                server.base_url,
                "pool",
                out_root=out_root or tempfile.mkdtemp(),
                data_root=data,
                log=lambda *_: None,
                **kwargs,
                name="test",
            )
            rec = run.run()
    finally:
        server.stop()
    return rec, server


@pytest.fixture(scope="module")
def data():
    root = tempfile.mkdtemp()
    make_data(root)
    # A third task at 4096 whose prompts the fake server can tell apart, and one task at another length
    rows = io.read_jsonl(os.path.join(root, "ruler", "niah_single_1_4096.jsonl"))
    uncapped = [{**row, "input": row["input"].replace("Text", "Uncapped text")} for row in rows]
    io.write_jsonl(os.path.join(root, "ruler", "niah_single_2_4096.jsonl"), uncapped)
    io.write_jsonl(
        os.path.join(root, "ruler", "niah_single_1_32768.jsonl"),
        [{**row, "max_length": 32768} for row in rows],
    )
    return root


def capped(task, n, concurrency=None, repeats=1):
    return suites.Plan(task, n=n, repeats=repeats, sampling=Sampling(max_tokens=64), concurrency=concurrency)


def test_ruler_tasks_of_one_length_share_one_cap_and_an_uncapped_task_fills_the_rest(data):
    # Two capped tasks at 4096 hold 2 between them, not 2 each, and the uncapped one takes the other 4 of 6
    plans = [
        capped("ruler_niah_single_1_4096", 6, concurrency=2),
        capped("ruler_cwe_4096", 6, concurrency=2),
        capped("ruler_niah_single_2_4096", 6),
    ]
    rec, server = run_against(6, plans, data)
    assert server.peak_of == {"capped": 2, "uncapped": 4}
    assert server.peak == 6
    assert {task: entry["concurrency"] for task, entry in rec["tasks"].items()} == {
        "ruler_niah_single_1_4096": 2,
        "ruler_cwe_4096": 2,
        "ruler_niah_single_2_4096": 6,
    }


def test_tasks_of_two_groups_are_in_flight_together_under_the_server_s_cap(data):
    # Each group alone holds at most 3; together they fill the server's 4 and never pass it
    plans = [
        capped("ruler_cwe_4096", 6, concurrency=3),
        capped("ruler_niah_single_1_32768", 6, concurrency=3),
    ]
    _, server = run_against(4, plans, data)
    assert server.peak == 4


def test_a_task_s_own_cap_holds_under_a_larger_server_cap(data):
    _, server = run_against(64, [capped("ruler_cwe_4096", 6, concurrency=2)], data)
    assert server.peak == 2


def test_repeats_are_in_flight_together_and_an_uncapped_task_takes_the_server_s_cap(data):
    rec, server = run_against(8, [capped("ruler_niah_single_1_4096", 4, repeats=2)], data)
    assert server.peak == 8
    niah = rec["tasks"]["ruler_niah_single_1_4096"]
    assert (niah["concurrency"], niah["n"], niah["repeat_scores"]) == (8, 8, [0.5, 0.5])


def test_every_task_is_in_the_record_with_its_own_samples_and_time(data):
    plans = [capped("ruler_niah_single_1_4096", 4, repeats=2), capped("ruler_cwe_4096", 6)]
    rec, _ = run_against(4, plans, data)
    assert (rec["tasks"]["ruler_niah_single_1_4096"]["n"], rec["tasks"]["ruler_cwe_4096"]["n"]) == (8, 6)
    for entry in rec["tasks"].values():
        assert entry["seconds"] > 0
        assert entry["output_tokens_per_second"] > 0
        assert entry["samples"] == f"samples/{entry['key'].split(':')[0]}.jsonl"


def test_a_stopped_concurrent_run_resumes_every_task_where_it_was(data):
    # Half of each task's rows are on disk, as a run killed while both were in flight leaves them
    plans = [capped("ruler_niah_single_1_4096", 4, repeats=2), capped("ruler_cwe_4096", 6)]
    out = tempfile.mkdtemp()
    first, _ = run_against(4, plans, data, out_root=out, run_id="stopped")
    for entry in first["tasks"].values():
        path = os.path.join(out, "stopped", entry["samples"])
        rows = io.read_jsonl(path)
        io.write_jsonl(path, rows[: len(rows) // 2])
    rec, server = run_against(4, plans, data, out_root=out, run_id="stopped")
    # Only the missing half was sent, and the record is the one a straight run writes
    assert len(server.requests) == 4 + 3
    for task, entry in rec["tasks"].items():
        assert (entry["n"], entry["score"]) == (first["tasks"][task]["n"], first["tasks"][task]["score"])


def test_every_dataset_runs_in_exactly_one_phase():
    assert sorted(dataset for phase in Suite.PHASES for dataset in phase) == sorted(Suite.DATASETS)


def test_a_phase_sends_nothing_until_the_phase_before_it_has_every_reply(data):
    # The uncapped task stands in for a reasoning task - the reasoning datasets come from the hub, not the fixture
    plans = [capped("ruler_cwe_4096", 6), capped("ruler_niah_single_2_4096", 6)]
    first = property(lambda plan: 0 if plan.task == "ruler_niah_single_2_4096" else 1)
    with mock.patch.object(suites.Plan, "phase", first):
        rec, server = run_against(8, plans, data)
    last_reply = max(i for i, event in enumerate(server.events) if event == ("out", "uncapped"))
    first_send = min(i for i, event in enumerate(server.events) if event == ("in", "capped"))
    assert last_reply < first_send
    # Within a phase the pool still fills - all six of the first phase's requests were in flight together
    assert server.peak_of == {"uncapped": 6, "capped": 6}
    assert set(rec["tasks"]) == {"ruler_cwe_4096", "ruler_niah_single_2_4096"}


def test_a_shared_context_is_prefilled_once_and_one_context_is_in_flight_at_a_time(data):
    # cwe and niah_single_1 open with one context, niah_single_2 - which the fake tells apart - with another
    contexts = {
        "ruler_cwe_4096": "first",
        "ruler_niah_single_1_4096": "first",
        "ruler_niah_single_2_4096": "second",
    }
    plans = [capped(task, 3) for task in contexts]
    with mock.patch.object(suites.Plan, "shared_context", property(lambda plan: contexts[plan.task])):
        rec, server = run_against(8, plans, data)
    events = server.events
    budgets = [body["max_tokens"] for body in server.requests]
    # One prefill of one token per context, and each is alone on the server
    assert budgets.count(1) == 2
    assert events[:2] == [("in", "capped"), ("out", "capped")]
    first_second = events.index(("in", "uncapped"))
    assert events[first_second + 1] == ("out", "uncapped")
    # The second context starts only once the first has every reply, and the first's two plans flew together
    assert max(i for i, event in enumerate(events) if event == ("out", "capped")) < first_second
    assert server.peak_of == {"capped": 6, "uncapped": 3}
    # The prefills are no sample of any task
    assert {task: entry["n"] for task, entry in rec["tasks"].items()} == dict.fromkeys(contexts, 3)


@pytest.mark.parametrize(("attention_kind", "peak"), [("block_sparse", 3), ("lsa", 1), ("full", 1)])
def test_block_sparse_questions_over_one_context_take_what_the_pool_holds(data, attention_kind, peak):
    # A pool of one context and two more questions - each a 64-token budget and the question bound
    each = 64 + runner.Runner.QUESTION_TOKENS
    pool = {"context_length": 100_000, "max_total_num_tokens": 100_000 + 2 * each}
    with mock.patch.object(suites.Plan, "shared_context", property(lambda plan: "one")):
        rec, server = run_against(
            8,
            [capped("ruler_cwe_4096", 6, concurrency=1)],
            data,
            extra_info=pool,
            attention_kind=attention_kind,
        )
    # The prefill still goes alone, then block sparse fills the pool and the others keep the plan's cap
    assert server.events[:2] == [("in", "capped"), ("out", "capped")]
    assert server.peak == peak
    assert rec["tasks"]["ruler_cwe_4096"]["concurrency"] == peak


def test_a_cap_sglang_lowered_at_start_is_the_one_a_run_fills(data):
    # Asked for 8, and the state cache let the server hold 2
    lowered = {"internal_states": [{"effective_max_running_requests_per_dp": 2}]}
    rec, server = run_against(8, [capped("ruler_niah_single_2_4096", 6)], data, extra_info=lowered)
    assert server.peak == 2
    assert rec["tasks"]["ruler_niah_single_2_4096"]["concurrency"] == 2


@pytest.mark.parametrize(
    ("reuse", "radix_off", "peak"), [(True, False, 3), (False, False, 1), (True, True, 1)]
)
def test_lsa_questions_over_one_context_batch_only_under_prefix_reuse_with_the_radix_cache_on(
    data, reuse, radix_off, peak
):
    each = 64 + runner.Runner.QUESTION_TOKENS
    pool = {
        "context_length": 100_000,
        "max_total_num_tokens": 100_000 + 2 * each,
        "disable_radix_cache": radix_off,
    }
    pool["json_model_override_args"] = json.dumps({"lsa": {"prefix_reuse": reuse, "projections": "map.pt"}})
    with mock.patch.object(suites.Plan, "shared_context", property(lambda plan: "one")):
        rec, server = run_against(
            8,
            [capped("ruler_cwe_4096", 6, concurrency=1)],
            data,
            extra_info=pool,
            attention_kind="lsa",
        )
    assert server.events[:2] == [("in", "capped"), ("out", "capped")]
    assert server.peak == peak
    assert rec["tasks"]["ruler_cwe_4096"]["concurrency"] == peak


def test_a_server_that_reports_no_request_cap_is_refused(data):
    server = FakeServer(slow, info={"model_path": "zai-org/Fake-Model", "version": "fake"}).start()
    try:
        with (
            mock.patch.dict(suites.SUITES, {"pool": lambda: [capped("ruler_cwe_4096", 6)]}),
            pytest.raises(ValueError, match="--max-running-requests"),
        ):
            runner.Runner(server.base_url, "pool", out_root=tempfile.mkdtemp(), data_root=data, name="test")
    finally:
        server.stop()


if __name__ == "__main__":
    unittest.main()
