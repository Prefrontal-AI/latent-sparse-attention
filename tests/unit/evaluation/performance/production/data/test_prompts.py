"""The production prompt sets - nested across lengths, every entry its own span of the corpus."""

import itertools

import pytest

from lsa import assets
from lsa.evaluation.performance.production.data import prompts

WORKLOADS = ["production", "production_qwen38"]


@pytest.mark.parametrize("name", WORKLOADS)
def test_every_length_starts_a_prompt_entry_at_the_same_row(name):
    card = assets.workload(name)
    for entry in range(card.entries):
        assert len({prompts.entry_rows(card, entry, length)[0] for length in card.lengths}) == 1


@pytest.mark.parametrize("name", WORKLOADS)
def test_prompt_entries_do_not_overlap_at_the_longest_length(name):
    card = assets.workload(name)
    spans = [prompts.entry_rows(card, entry, max(card.lengths)) for entry in range(card.entries)]
    for (_, end), (start, _) in itertools.pairwise(spans):
        assert end <= start


@pytest.mark.parametrize("name", WORKLOADS)
def test_a_prompt_entry_decodes_a_row_over_its_length(name):
    card = assets.workload(name)
    start, end = prompts.entry_rows(card, 0, 131072)
    assert (end - start) * prompts.row_tokens(card) == 131072 + prompts.row_tokens(card)


def test_qwen38_s_longest_prompt_and_its_answer_fit_the_card_s_yarn():
    # YaRN factor 40 over Qwen 3.8's 262,144 native positions, less the ladder's 256-token answer
    assert max(assets.workload("production_qwen38").lengths) == 40 * 262144 - 256
