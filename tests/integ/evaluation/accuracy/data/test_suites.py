import unittest

import pytest

from lsa.evaluation.accuracy.data import suites
from lsa.evaluation.constants import Suite


def test_a_filter_that_matches_nothing_is_refused():
    with pytest.raises(KeyError):
        suites.select("full", only=["no_such_task"])


def test_a_task_the_suite_does_not_hold_is_refused_even_beside_one_it_does():
    """The smoke suite holds no RULER task at 131072, and a list naming one used to lose it silently"""
    with pytest.raises(KeyError, match="ruler_vt_131072"):
        suites.select("smoke", only=["gpqa_diamond", "ruler_vt_131072"])


def test_a_length_the_suite_does_not_hold_is_refused():
    with pytest.raises(KeyError, match="262144"):
        suites.select("full", lengths=[4096, 262144])


def test_a_task_is_found_by_its_plan_key_as_well_as_its_name():
    assert [plan.task for plan in suites.select("full", only=["math500:math500_100"])] == ["math500"]


def test_full_runs_papertrail_as_the_papertrail_suite_does_in_its_own_phase():
    papertrail = {plan.key for plan in suites.select("papertrail")}
    full = suites.select("full")
    assert papertrail <= {plan.key for plan in full}
    assert {plan.phase for plan in full if plan.key in papertrail} == {
        Suite.PHASES.index((Suite.PAPERTRAIL,))
    }


def test_full_no_longer_asks_for_graphwalks_which_papertrail_replaced():
    assert not [plan.task for plan in suites.select("full") if plan.task.startswith(Suite.GRAPHWALKS_PREFIX)]


def test_a_length_filter_on_full_keeps_papertrail_beside_ruler_and_nothing_else():
    kept = {plan.task for plan in suites.select("full", lengths=[10_000_000])}
    assert any(task.startswith(Suite.PAPERTRAIL_PREFIX) for task in kept)
    assert all(task.startswith((Suite.PAPERTRAIL_PREFIX, Suite.RULER_PREFIX)) for task in kept)


class DatasetTest(unittest.TestCase):
    def test_every_plan_of_every_suite_is_drawn_from_a_dataset_a_generation_block_can_name(self):
        for name, suite in suites.SUITES.items():
            for plan in suite():
                assert plan.dataset in Suite.DATASETS, (name, plan.task)


def test_full_asks_the_fixed_mmlu_pro_subset_in_the_first_phase():
    plan = suites.select("full", only=["mmlu_pro"])[0]
    assert (plan.key, plan.phase) == ("mmlu_pro:mmlu_pro_2000", 0)
    assert len(plan.ids()) == 2000
