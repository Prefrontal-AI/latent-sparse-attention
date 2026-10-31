"""The record's root and its summaries - no extra needed, `record` is standard library only."""

import os
import tempfile
import unittest
from unittest import mock

from lsa import paths
from lsa.evaluation.accuracy import records
from lsa.evaluation.constants import Evaluation, Records


def shared_root():
    """Return a root holding an evaluation directory, as a shared root does."""
    root = tempfile.mkdtemp()
    os.mkdir(os.path.join(root, "evaluations"))
    return root


class RecordsRootTest(unittest.TestCase):
    """The shared root when it is there, and a private one when it is not."""

    def test_the_shared_root_is_used_when_it_exists(self):
        root = shared_root()
        with mock.patch.dict(os.environ, {paths.ROOT_VARIABLE: root}):
            self.assertEqual(records.records_root(), os.path.join(root, "evaluations"))

    def test_the_working_directory_is_the_fallback(self):
        with mock.patch.dict(os.environ, {paths.ROOT_VARIABLE: "/no/such/place"}):
            expected = os.path.join(os.getcwd(), *Records.LOCAL_ROOT)
            with self.assertLogs(records.logger, "WARNING") as logged:
                self.assertEqual(records.records_root(), expected)
        self.assertIn("/no/such/place/evaluations", logged.output[0])
        self.assertIn(expected, logged.output[0])

    def test_it_reads_the_environment_at_call_time(self):
        first, second = shared_root(), shared_root()
        with mock.patch.dict(os.environ, {paths.ROOT_VARIABLE: first}):
            seen_first = records.records_root()
        with mock.patch.dict(os.environ, {paths.ROOT_VARIABLE: second}):
            seen_second = records.records_root()
        self.assertEqual((seen_first, seen_second), (f"{first}/evaluations", f"{second}/evaluations"))


ENTRY = {
    "prompt_set_hash": "0123456789abcdef",
    "n_docs": 25,
    "sampling": {
        "temperature": 0.0,
        "max_tokens": 2048,
        "thinking_budget": 0,
        "chat_template_kwargs": None,
    },
    "score": 0.9,
}


class ProtocolTest(unittest.TestCase):
    """Two runs of a task share a protocol when their prompts and sampling agree, whatever else differs."""

    def test_unset_sampling_reads_the_same_however_it_was_written(self):
        written_empty = {**ENTRY, "sampling": {**ENTRY["sampling"], "chat_template_kwargs": {}}}
        left_out = {**ENTRY, "sampling": {"temperature": 0.0, "max_tokens": 2048, "thinking_budget": 0}}
        keys = {records.protocol_key(entry) for entry in (ENTRY, written_empty, left_out)}
        self.assertEqual(len(keys), 1)

    def test_the_score_is_not_part_of_it(self):
        self.assertEqual(records.protocol_key(ENTRY), records.protocol_key({**ENTRY, "score": 0.1}))

    def test_prompts_docs_and_every_sampling_setting_are(self):
        changed = [
            {**ENTRY, "prompt_set_hash": "fedcba9876543210"},
            {**ENTRY, "n_docs": 100},
            {**ENTRY, "sampling": {**ENTRY["sampling"], "max_tokens": 512}},
            {**ENTRY, "sampling": {**ENTRY["sampling"], "thinking_budget": None}},
        ]
        keys = {records.protocol_key(entry) for entry in [ENTRY, *changed]}
        self.assertEqual(len(keys), 5)

    def test_no_prompt_hash_means_nothing_to_check(self):
        self.assertIsNone(records.protocol_key({**ENTRY, "prompt_set_hash": None}))
        self.assertEqual(records.protocol_fields(ENTRY)["max_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()


class WhatATaskMeasuredTest(unittest.TestCase):
    """The corpus a file came from, whether a budget was held, and what the replies did."""

    def test_a_corpus_is_named_by_whichever_generator_built_its_file(self):
        self.assertEqual(records.corpus_name({"generator": "c3f5e3b4f87f97e0"}), "ruler-nvidia@c3f5e3b4")
        self.assertEqual(records.corpus_name({"lm_eval": "0.4.13"}), "lm-eval-0.4.13")
        self.assertIsNone(records.corpus_name({"rows": 503}))
        self.assertIsNone(records.corpus_name(None))

    def test_a_budget_is_enforced_only_by_a_server_with_strict_thinking(self):
        budget = {"thinking_budget": 0}
        self.assertTrue(records.budget_enforced(budget, {"enable_strict_thinking": True}))
        self.assertFalse(records.budget_enforced(budget, {"enable_strict_thinking": False}))
        self.assertFalse(records.budget_enforced(budget, None))
        self.assertIsNone(
            records.budget_enforced({"thinking_budget": None}, {"enable_strict_thinking": True})
        )

    def test_a_forced_close_of_the_think_block_is_not_reasoning(self):
        rows = [
            {"reasoning_tokens": 1, "finish_reason": "stop"},
            {"reasoning_tokens": 0, "finish_reason": "stop"},
            {"reasoning_tokens": 900, "finish_reason": "length"},
            {"reasoning_tokens": 2, "finish_reason": "stop"},
            {"error": "timeout"},
        ]
        self.assertEqual(
            records.observed(rows, True), {"reasoned": 0.5, "cut": 0.25, "budget_enforced": True}
        )
        self.assertEqual(records.observed([], None), {"reasoned": None, "cut": None, "budget_enforced": None})


class RecordsDirTest(unittest.TestCase):
    """Each kind's runs sit where the package that wrote them sits."""

    def test_it_defaults_to_the_records_root(self):
        root = shared_root()
        with mock.patch.dict(os.environ, {paths.ROOT_VARIABLE: root}):
            self.assertEqual(records.records_dir(Evaluation.ACCURACY), f"{root}/evaluations/accuracy")
