import torch

from lsa.training.constants import Loss
from lsa.training.native.losses import (
    RetrievalScore,
    SequenceScore,
    bucket_names,
    bucket_of_position,
    score_answers,
    score_batch,
    summarise,
    summarise_retrieval,
)


def test_a_position_lands_in_the_bucket_its_edges_put_it_in():
    positions = torch.tensor([0, 511, 512, 1539, 1540, 9000])
    assert bucket_of_position(positions, (512, 1540)).tolist() == [0, 0, 1, 1, 2, 2]


def test_buckets_are_named_by_their_span():
    assert bucket_names((512, 1540), 4096) == ["0_512", "512_1540", "1540_4096"]


def test_a_batch_is_summed_per_sequence_and_bucket_and_ignored_labels_are_not_counted():
    losses = torch.arange(8, dtype=torch.float32)
    labels = torch.tensor([1, 1, 1, 1, 1, 1, Loss.IGNORE_INDEX, Loss.IGNORE_INDEX])
    sums, counts = score_batch(losses, labels, seq_len=4, edges=(2,))
    assert sums.tolist() == [[1.0, 5.0], [9.0, 0.0]]
    assert counts.tolist() == [[2, 2], [2, 0]]


def test_a_mean_is_over_tokens_so_a_long_sequence_weighs_what_it_holds():
    scores = [
        SequenceScore("a", "d1", 0, 0, [2.0, 8.0], [1, 4]),
        SequenceScore("b", "d2", 0, 0, [0.0, 0.0], [1, 0]),
    ]
    metrics = summarise(scores, (2,), 4)
    assert metrics["loss"] == 10.0 / 6
    assert metrics["position/0_2"] == 1.0
    assert metrics["position/2_4"] == 2.0
    assert metrics["source/a"] == 2.0


def test_an_answer_is_scored_by_its_summed_loss_and_whether_every_token_was_the_argmax():
    losses = torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.0, 2.0]])
    answers = torch.tensor([[1, 2, 3], [4, 5, 6]])
    predictions = torch.tensor([[1, 2, 3], [4, 5, 0]])
    sums, exact = score_answers(losses, predictions, answers)
    assert sums.tolist() == [3.0, 6.0]
    assert exact.tolist() == [True, False]


def test_retrieval_is_summarised_per_distance_with_the_gain_over_the_control():
    def score(distance, probe, control, probe_exact, control_exact):
        return RetrievalScore("a", "d", 0, 0, distance, 2, probe, control, probe_exact, control_exact)

    scores = [
        score(256, 0.0, 8.0, True, False),
        score(256, 2.0, 8.0, False, False),
        score(1024, 6.0, 6.0, False, False),
    ]
    metrics = summarise_retrieval(scores)
    assert metrics["retrieval/d256/loss"] == 0.5
    assert metrics["retrieval/d256/control_loss"] == 4.0
    assert metrics["retrieval/d256/gain"] == 3.5
    assert metrics["retrieval/d256/exact"] == 0.5
    assert metrics["retrieval/d256/control_exact"] == 0.0
    assert metrics["retrieval/d1024/gain"] == 0.0
    assert sorted(metrics) == sorted(
        f"retrieval/d{d}/{name}"
        for d in (256, 1024)
        for name in ("loss", "control_loss", "exact", "control_exact", "gain")
    )
