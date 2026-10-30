from types import SimpleNamespace

import pytest
import torch

from lsa.training.native.benchmark.lm import (
    Backbone,
    NativeLM,
    Tokenizer,
    batches,
    encode_pair,
    encode_request,
    finish,
    pack_rows,
    padded_length,
    score_continuation,
    score_window,
    stopped,
)

VOCAB = 11
END = 10


def encode(text):
    return [ord(c) - 97 for c in text]


def decode(ids):
    return "".join(chr(97 + i) if i < END else "|" for i in ids)


def bigram_model(seed=0):
    # The next-token logits depend on the previous token alone, so any padding or batching has a known answer
    generator = torch.Generator().manual_seed(seed)
    table = torch.randn(VOCAB, VOCAB, generator=generator)
    backbone = Backbone(
        hidden=lambda tokens: torch.nn.functional.one_hot(tokens, VOCAB).float(),
        project=lambda hidden: hidden @ table,
        device=torch.device("cpu"),
    )
    return NativeLM(backbone, Tokenizer(encode, decode, END), max_length=64, batch_tokens=2048), table


def reference(table, context_ids, continuation_ids):
    tokens = context_ids + continuation_ids
    log_probs = table.log_softmax(-1)
    total = sum(log_probs[tokens[j - 1], tokens[j]].item() for j in range(len(context_ids), len(tokens)))
    greedy = all(
        table[tokens[j - 1]].argmax().item() == tokens[j] for j in range(len(context_ids), len(tokens))
    )
    return total, greedy


def requests(*pairs):
    return [SimpleNamespace(args=pair) for pair in pairs]


def test_a_batch_is_padded_to_the_next_power_of_two_from_the_sparse_block():
    assert [padded_length(n) for n in (1, 512, 513, 1024, 1025, 16384)] == [512, 512, 1024, 1024, 2048, 16384]


def test_trailing_context_whitespace_moves_to_the_continuation():
    # Spaces are not in the toy alphabet, so a stand-in encoder that keeps them shows the move
    context_ids, continuation_ids = encode_pair(lambda text: [ord(c) for c in text], "ab  ", "cd")
    assert context_ids == [ord("a"), ord("b")]
    assert continuation_ids == [ord(" "), ord(" "), ord("c"), ord("d")]


def test_an_empty_context_is_conditioned_on_the_end_token():
    assert encode_request(encode, END, "", "abc") == ([END], [0, 1, 2])
    assert encode_request(encode, END, "ab", "c") == ([0, 1], [2])


def test_a_text_that_opens_with_the_end_token_keeps_it_as_the_context():
    assert encode_request(lambda text: [END, 3, 4], END, "", "x") == ([END], [3, 4])


def test_a_pair_is_cut_to_the_last_tokens_the_model_reads_and_a_long_continuation_is_refused():
    assert score_window([1, 2, 3], [4, 5], max_length=3) == [2, 3, 4, 5]
    with pytest.raises(ValueError, match="cannot be scored"):
        score_window([1], [2, 3, 4], max_length=2)


def test_rows_are_right_padded_and_the_batch_is_shaped_to_its_row_count():
    tokens = pack_rows([[1, 2, 3], [4]], length=4, rows=3, pad_id=9)
    assert tokens.tolist() == [[1, 2, 3, 9], [4, 9, 9, 9], [9, 9, 9, 9]]


def test_requests_group_by_padded_length_longest_first_and_fill_the_token_budget():
    groups = batches([600, 10, 30, 520, 20], batch_tokens=1024)
    assert groups == [([0], 1024, 1), ([3], 1024, 1), ([2, 4], 512, 2), ([1], 512, 2)]


def test_a_continuation_is_scored_the_same_in_chunks_as_whole(monkeypatch):
    monkeypatch.setattr("lsa.training.native.benchmark.lm.PROJECTION_CHUNK", 2)
    generator = torch.Generator().manual_seed(1)
    table = torch.randn(VOCAB, VOCAB, generator=generator)
    hidden = torch.nn.functional.one_hot(torch.tensor([1, 2, 3, 4, 5]), VOCAB).float()
    targets = torch.tensor([2, 3, 4, 5, 6])
    log_likelihood, greedy = score_continuation(lambda rows: rows @ table, hidden, targets)
    expected = (table.log_softmax(-1)[torch.tensor([1, 2, 3, 4, 5]), targets]).sum()
    assert log_likelihood.item() == pytest.approx(expected.item(), rel=1e-5)
    assert greedy.item() == (table[[1, 2, 3, 4, 5]].argmax(-1) == targets).all().item()


def test_loglikelihood_matches_the_bigram_answer_whatever_the_batch_holds():
    lm, table = bigram_model()
    pairs = [("abc", "dd"), ("e", "fgh"), ("abcdefghij" * 4, "a"), ("", "bc")]
    scored = lm.loglikelihood(requests(*pairs))
    for (context, continuation), (log_likelihood, greedy) in zip(pairs, scored, strict=True):
        context_ids, continuation_ids = encode_request(encode, END, context, continuation)
        expected, expected_greedy = reference(table, context_ids, continuation_ids)
        assert log_likelihood == pytest.approx(expected, rel=1e-5)
        assert greedy == expected_greedy


def test_a_request_scores_the_same_alone_as_among_longer_ones():
    lm, _ = bigram_model()
    alone = lm.loglikelihood(requests(("abc", "de")))
    among = lm.loglikelihood(requests(("abcdefghij" * 5, "ab"), ("abc", "de"), ("a", "b")))
    assert alone[0][0] == pytest.approx(among[1][0], rel=1e-6)


def test_rolling_loglikelihood_scores_every_token_once_across_windows():
    lm, table = bigram_model()
    text = "abcdefghij" * 9
    [total] = lm.loglikelihood_rolling(requests((text,)))
    ids = [END, *encode(text)]
    log_probs = table.log_softmax(-1)
    expected = sum(log_probs[ids[j - 1], ids[j]].item() for j in range(1, len(ids)))
    assert total == pytest.approx(expected, rel=1e-5)


def greedy_reply(table, prompt_ids, steps):
    ids = list(prompt_ids)
    for _ in range(steps):
        ids.append(table[ids[-1]].argmax().item())
    return ids[len(prompt_ids) :]


def test_generation_extends_prompts_of_any_length_greedily_in_one_batch():
    lm, table = bigram_model(seed=3)
    kwargs = {"until": [], "max_gen_toks": 4}
    prompts = ["abc", "a", "defghij" * 3]
    replies = lm.generate_until(requests(*((prompt, kwargs) for prompt in prompts)))
    for prompt, reply in zip(prompts, replies, strict=True):
        generated = greedy_reply(table, encode(prompt), 4)
        assert reply == finish(Tokenizer(encode, decode, END), generated, ())


def test_generation_stops_at_a_stop_string_and_the_cap_bounds_the_budget():
    lm, table = bigram_model(seed=3)
    first = greedy_reply(table, encode("abc"), 1)
    stop = decode(first)
    [reply] = lm.generate_until(requests(("abc", {"until": [stop], "max_gen_toks": 6})))
    assert reply == ""
    capped, _ = bigram_model(seed=3)
    capped._gen_cap = 2
    [short] = capped.generate_until(requests(("abc", {"until": [], "max_gen_toks": 6})))
    assert len(short) <= 2


def test_a_reply_is_cut_at_the_end_token_and_at_the_first_stop_string():
    tokenizer = Tokenizer(encode, decode, END)
    assert finish(tokenizer, [0, 1, END, 2], ()) == "ab"
    assert finish(tokenizer, [0, 1, 2, 3], ("c",)) == "ab"
    assert stopped(tokenizer, [0, END], ()) is True
    assert stopped(tokenizer, [0, 1], ("b",)) is True
    assert stopped(tokenizer, [0, 1], ("z",)) is False


def test_time_is_counted_per_request_type():
    lm, _ = bigram_model()
    lm.loglikelihood(requests(("ab", "c")))
    assert set(lm.seconds) == {"loglikelihood"}
