"""The packed score-and-id key - the order every kernel pick and the torch side agree on."""

import torch

from lsa.core.kernels.triton.keys import EMPTY, ordered_int32, pack_keys, unpack_ids, unpack_scores


def test_a_key_unpacks_to_its_score_and_id():
    scores = torch.tensor([-1.0, -0.25, -0.0, 0.0, 1e-7, 0.5, 1.0])
    ids = torch.tensor([0, 7, 99, 12345, 2**31 - 1, 3, 1], dtype=torch.int32)
    keys = pack_keys(scores, ids)
    assert torch.equal(unpack_scores(keys).view(torch.int32), scores.view(torch.int32))
    assert torch.equal(unpack_ids(keys), ids)


def test_higher_scores_make_higher_keys_and_an_equal_score_the_lower_id():
    scores = torch.tensor([-1.0, -0.25, 0.0, 0.5, 0.5, 1.0])
    ids = torch.tensor([4, 5, 6, 7, 2, 1], dtype=torch.int32)
    order = pack_keys(scores, ids).argsort(descending=True)
    assert order.tolist() == [5, 4, 3, 2, 1, 0]


def test_the_ordered_bits_are_monotone_in_the_score():
    scores = torch.sort(torch.randn(10000) * torch.logspace(-6, 6, 10000)).values
    ordered = ordered_int32(scores)
    assert (ordered[1:] >= ordered[:-1]).all()
    assert (ordered[1:] > ordered[:-1]).float().mean() > 0.99


def test_no_id_is_the_empty_key():
    keys = pack_keys(torch.tensor([3.0, 4.0]), torch.tensor([-1, 5], dtype=torch.int32))
    assert keys[0] == EMPTY
    assert keys[1] > EMPTY
    assert unpack_ids(keys).tolist() == [-1, 5]
