"""The driver around NVIDIA's RULER generators - the command it builds, the rows it keeps, the manifest."""

import os
import sys
import tempfile
import unittest
from unittest import mock

from lsa.data import manifests, prepare
from lsa.evaluation.accuracy import data
from lsa.evaluation.accuracy.data import ruler


class CheckTest(unittest.TestCase):
    """The table over the manifest - no generator runs, so no clone is needed."""

    @staticmethod
    def cell(length, median, maximum=None):
        return {
            "rows": 20,
            "tokens_to_generate": 128,
            "prompt_median": median,
            "prompt_max": maximum or median,
            "dropped": 0,
            "task_args": {},
        }

    @staticmethod
    def write(root, cells):
        os.makedirs(os.path.join(root, "ruler"), exist_ok=True)
        for (task, length), entry in cells.items():
            open(data.ruler_path(task, length, root), "w").close()
            measured = {"sha256": "", "tokenizer": "t", "date": "2026-09-19", **entry}
            manifests.write_benchmark_file(
                root, f"ruler/{task}_{length}.jsonl", manifests.BenchmarkFile.from_entry(measured)
            )

    def test_passes_when_the_prompts_grow_with_the_length(self):
        root = tempfile.mkdtemp()
        self.write(root, {("vt", 4096): self.cell(4096, 3900), ("vt", 32768): self.cell(32768, 32000)})
        self.assertEqual(ruler.check(["vt"], [4096, 32768], root, log=lambda *_: None), [])

    def test_flags_a_saturated_cell_a_missing_one_and_an_overshoot(self):
        root = tempfile.mkdtemp()
        self.write(
            root,
            {
                ("vt", 4096): self.cell(4096, 3900),
                ("vt", 32768): self.cell(32768, 4100),
                ("cwe", 4096): self.cell(4096, 3900, maximum=9000),
            },
        )
        bad = ruler.check(["vt", "cwe"], [4096, 32768], root, log=lambda *_: None)
        self.assertEqual(
            sorted(bad), ["ruler/cwe_32768.jsonl", "ruler/cwe_4096.jsonl", "ruler/vt_32768.jsonl"]
        )

    def test_prepare_check_reads_the_cells_and_submits_nothing(self):
        root = tempfile.mkdtemp()
        self.write(root, {("vt", 4096): self.cell(4096, 3900), ("vt", 32768): self.cell(32768, 32000)})
        argv = ["lsa data prepare", "ruler", "--check", "--root", root, "--tasks", "vt"]
        argv += ["--lengths", "4096", "32768"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(prepare.slurm, "submit", side_effect=AssertionError("submitted")),
            self.assertRaises(SystemExit) as raised,
        ):
            prepare.main()
        self.assertEqual(raised.exception.code, 0)

    def test_prepare_check_exits_nonzero_on_a_missing_cell(self):
        root = tempfile.mkdtemp()
        self.write(root, {("vt", 4096): self.cell(4096, 3900)})
        argv = ["lsa data prepare", "ruler", "--check", "--root", root, "--tasks", "vt"]
        argv += ["--lengths", "4096", "32768"]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as raised:
            prepare.main()
        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
