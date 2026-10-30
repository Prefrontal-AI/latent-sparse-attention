"""RULER needle filtering and prompt row conversion."""

import unittest

from lsa.evaluation.accuracy.data import ruler

NEEDLE_PREFIX = "The special magic number for {key} mentioned in the provided text is"


def needle_row(key, distractors, prefix_key=None):
    haystack = "\n".join(
        f"One of the special magic numbers for {other} is: 1234567." for other in distractors
    )
    question = f"What is the special magic number for {key} mentioned in the provided text?"
    return {
        "index": 0,
        "input": f"{haystack}\nOne of the special magic numbers for {key} is: 7654321.\n{question}",
        "outputs": ["7654321"],
        "length": 900,
        "answer_prefix": " " + NEEDLE_PREFIX.format(key=prefix_key or key),
    }


class FilterTest(unittest.TestCase):
    def test_reads_one_key_and_several(self):
        self.assertEqual(ruler.queried_keys(NEEDLE_PREFIX.format(key="giddy-otter")), ["giddy-otter"])
        many = "The special magic numbers for used-postfix, squealing-probe, and annoying-doorbell mentioned in the provided text are"
        self.assertEqual(ruler.queried_keys(many), ["used-postfix", "squealing-probe", "annoying-doorbell"])

    def test_keeps_a_clean_prompt_and_drops_a_repeated_key(self):
        self.assertTrue(ruler.one_value_each(needle_row("giddy-otter", ["brave-anchor", "sleepy-melon"])))
        self.assertFalse(ruler.one_value_each(needle_row("giddy-otter", ["brave-anchor", "giddy-otter"])))

    def test_a_key_that_ends_in_another_key_is_not_a_collision(self):
        self.assertTrue(ruler.one_value_each(needle_row("able-cat", ["unable-cat", "able-catapult"])))


class RowsTest(unittest.TestCase):
    def test_renames_the_prefix_renumbers_and_stops_at_the_count(self):
        rows = [needle_row(f"key{i}-x", ["other-thing"]) for i in range(5)]
        out, dropped = ruler.our_rows("niah_multikey_2", rows, 3, 4096)
        self.assertEqual(dropped, 0)
        self.assertEqual([row["index"] for row in out], [0, 1, 2])
        self.assertEqual(out[0]["gen_prefix"], NEEDLE_PREFIX.format(key="key0-x"))
        self.assertEqual(list(out[0]), ["index", "input", "outputs", "gen_prefix", "length", "max_length"])
        self.assertEqual(out[0]["max_length"], 4096)

    def test_counts_what_it_dropped_and_keeps_the_order(self):
        rows = [
            needle_row("aaa-one", ["aaa-one"]),
            needle_row("bbb-two", ["other-thing"]),
            needle_row("ccc-three", ["ccc-three"]),
            needle_row("ddd-four", ["other-thing"]),
        ]
        out, dropped = ruler.our_rows("niah_multikey_2", rows, 2, 4096)
        self.assertEqual(dropped, 2)
        self.assertEqual([ruler.queried_keys(row["gen_prefix"])[0] for row in out], ["bbb-two", "ddd-four"])

    def test_a_task_without_a_filter_keeps_every_row(self):
        rows = [needle_row("aaa-one", ["aaa-one"]) for _ in range(2)]
        out, dropped = ruler.our_rows("niah_single_2", rows, 2, 4096)
        self.assertEqual((len(out), dropped), (2, 0))

    def test_a_prefix_without_its_leading_space_is_refused(self):
        row = dict(needle_row("aaa-one", ["other-thing"]), answer_prefix=NEEDLE_PREFIX.format(key="aaa-one"))
        with self.assertRaises(ValueError):
            ruler.our_rows("niah_single_2", [row], 1, 4096)
