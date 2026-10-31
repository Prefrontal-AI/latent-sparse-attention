"""The streams - sharding over ranks and readers, mixing, and exact resumption.

Resumption is the property worth a test: a run long enough to be interrupted is a run whose loss curve has to be
one curve, and a loader that resumes approximately makes it two.
"""

import numpy as np
import pytest

from lsa.training.constants import Loss, Packing, Stream
from lsa.training.data import streams
from lsa.training.data.corpus import TokenizedCorpus, assemble_batch
from lsa.training.data.streams import Layout, open_stream
from tests.helpers.training.data.fake_corpus import write_corpus
from tests.helpers.training.data.fake_sources import FakeEncoder, write_shards, write_text

SEQ_LEN = 8


@pytest.fixture(autouse=True)
def small_slabs_and_a_fake_tokenizer(monkeypatch):
    """Slabs that end inside a shard, so a window can span two of them, and ids from characters"""
    monkeypatch.setattr(Stream, "SLAB_TOKENS", 11)
    monkeypatch.setattr(streams, "TextEncoder", lambda path: FakeEncoder())


def layout(rank=0, world_size=1, readers=1, seq_len=SEQ_LEN, batch=1, seed=0):
    return Layout(rank, world_size, readers, seq_len, batch * seq_len, seed)


def take(stream, batches):
    iterator = iter(stream)
    return [next(iterator) for _ in range(batches)]


def reference_windows(card, seq_len, seed, rank, world_size, workers, worker):
    """The reference training code's ShardedLMDataset, as it reads one worker's windows"""
    paths = [str(path) for path in card.train_paths()]
    reader, readers = rank * workers + worker, world_size * workers
    cycle = 0
    while True:
        order = np.random.default_rng(seed + cycle).permutation(len(paths))
        for index in order[reader::readers]:
            data = np.memmap(paths[index], dtype=np.uint16, mode="r")
            for chunk in range((len(data) - 1) // seq_len):
                yield data[chunk * seq_len : chunk * seq_len + seq_len + 1].astype(np.int64)
        cycle += 1


@pytest.mark.parametrize(("world_size", "workers"), [(1, 1), (2, 1), (2, 2), (1, 4)])
def test_token_shards_read_what_the_reference_training_codes_loader_reads(tmp_path, world_size, workers):
    """The same windows in the same order for every rank, worker and batch, across the shard cycle"""
    card = write_shards(tmp_path / "fw", shards=8, tokens=70)
    batch_size = 3
    for rank in range(world_size):
        mine = open_stream(
            [(card, 1.0)],
            layout(rank, world_size, workers, batch=batch_size, seed=42),
            tokenizer_path="",
        )
        iterator = iter(mine)
        reference = [
            reference_windows(card, SEQ_LEN, 42, rank, world_size, workers, w) for w in range(workers)
        ]
        for batch_index in range(20):
            batch = next(iterator)
            windows = [next(reference[batch_index % workers]) for _ in range(batch_size)]
            assert batch["input"].tolist() == np.concatenate([w[:-1] for w in windows]).tolist()
            assert batch["labels"].tolist() == np.concatenate([w[1:] for w in windows]).tolist()


def test_readers_and_ranks_cover_every_shard_once_a_cycle_and_never_share(tmp_path):
    card = write_shards(tmp_path / "fw", shards=8, tokens=9)
    seen = []
    for rank in range(2):
        mine = open_stream([(card, 1.0)], layout(rank, 2, 2), tokenizer_path="")
        for reader in mine.readers:
            # A shard of 9 ids is one window, and a cycle of two shards per reader is two windows
            seen += [next(reader.children[0])[0] // 1000 for _ in range(2)]
    assert sorted(seen) == list(range(8))


def test_more_readers_than_units_is_refused(tmp_path):
    card = write_shards(tmp_path / "fw", shards=3)
    with pytest.raises(ValueError, match="cannot feed 4 readers"):
        open_stream([(card, 1.0)], layout(0, 2, 2), tokenizer_path="")


def test_a_rank_outside_the_world_is_refused(tmp_path):
    card = write_shards(tmp_path / "fw")
    with pytest.raises(ValueError, match="is outside a world"):
        open_stream([(card, 1.0)], layout(4, 4), tokenizer_path="")


def test_tokens_per_batch_must_be_whole_sequences(tmp_path):
    card = write_shards(tmp_path / "fw")
    with pytest.raises(ValueError, match="seq_len must divide tokens_per_batch"):
        open_stream([(card, 1.0)], Layout(0, 1, 1, 8, 20), tokenizer_path="")


def test_a_pass_that_yields_no_window_is_refused_rather_than_spun_on(tmp_path):
    card = write_shards(tmp_path / "fw", tokens=5)
    with pytest.raises(ValueError, match="yielded no sequence"):
        take(open_stream([(card, 1.0)], layout(), tokenizer_path=""), 1)


def test_a_batch_is_labelled_by_the_next_token_and_holds_whole_sequences(tmp_path):
    card = write_shards(tmp_path / "fw")
    batch = take(open_stream([(card, 1.0)], layout(batch=3), tokenizer_path=""), 1)[0]
    assert batch["input"].shape == (3 * SEQ_LEN,)
    assert (batch["labels"][: SEQ_LEN - 1] == batch["input"][1:SEQ_LEN]).all()
    assert batch["positions"][SEQ_LEN] == 0
    assert not batch["padding_mask"].any()
    assert batch["num_valid_tokens"] == 3 * SEQ_LEN


def test_the_last_label_of_a_row_is_ignored(tmp_path):
    """A packed row's final slice has no next token, so its last label cannot be a real one"""
    asset = write_corpus(tmp_path / "corpus", sample_length=64)
    batch = take(open_stream([(asset, 1.0)], layout(seq_len=64, batch=1), tokenizer_path=""), 1)[0]
    assert batch["labels"][-1] == Loss.IGNORE_INDEX
    assert batch["num_valid_tokens"] == 63


def test_a_batch_assembled_from_samples_is_the_streams_batch(tmp_path):
    """The validator and the stream lay sequences out one way"""
    asset = write_corpus(tmp_path / "corpus", sample_length=64)
    corpus = TokenizedCorpus(asset, sources=["arxiv"], held_out_fraction=0.0)
    stream = open_stream([(asset, 1.0)], layout(batch=2, seq_len=16), tokenizer_path="", sources=["arxiv"])
    streamed = take(stream, 1)[0]
    windows = [corpus.read(sample, 16) for sample in corpus.index(16)]
    # Whichever two samples the order drew, assembling them gives the same layout
    for first in range(len(windows)):
        for second in range(len(windows)):
            if windows[first][0] == streamed["input"][0] and windows[second][0] == streamed["input"][16]:
                assembled = assemble_batch(corpus, [corpus.index(16)[first], corpus.index(16)[second]], 16)
                for name, value in streamed.items():
                    assert (
                        value == assembled[name] if isinstance(value, int) else value.equal(assembled[name])
                    )
                return
    pytest.fail("the batch holds samples the corpus does not")


def tokens_of(batch):
    return batch["input"].tolist()


def source_cases(tmp_path):
    return {
        "shards": ([(write_shards(tmp_path / "fw", shards=4, tokens=80), 1.0)], None),
        "rows": ([(write_corpus(tmp_path / "corpus", sample_length=64), 1.0)], None),
        "text-concatenated": ([(write_text(tmp_path / "text"), 1.0)], Packing.CONCATENATE),
        "text-span": ([(write_text(tmp_path / "text"), 1.0)], Packing.SPAN),
    }


@pytest.mark.parametrize("case", ["shards", "rows", "text-concatenated", "text-span"])
def test_resumption_is_exact(tmp_path, case):
    corpora, packing = source_cases(tmp_path)[case]
    shape = layout(rank=1, world_size=2, readers=2, batch=2, seed=3)
    # `readers` of the rows case have 64 samples a unit, which is plenty, and shards have four units for four readers
    stream = open_stream(corpora, shape, tokenizer_path="", packing=packing, held_out_fraction=0.0)
    iterator = iter(stream)
    for _ in range(7):
        next(iterator)
    state = stream.state_dict()
    expected = [tokens_of(next(iterator)) for _ in range(9)]

    resumed = open_stream(corpora, shape, tokenizer_path="", packing=packing, held_out_fraction=0.0)
    resumed.load_state_dict(state)
    assert [tokens_of(batch) for batch, _ in zip(iter(resumed), range(9), strict=False)] == expected


def test_resumption_is_exact_through_a_cycle_roll(tmp_path):
    card = write_shards(tmp_path / "fw", shards=2, tokens=20)
    stream = open_stream([(card, 1.0)], layout(seed=5), tokenizer_path="")
    iterator = iter(stream)
    for _ in range(9):
        next(iterator)
    assert stream.readers[0].children[0].cycle >= 1
    state = stream.state_dict()
    expected = [tokens_of(next(iterator)) for _ in range(6)]
    resumed = open_stream([(card, 1.0)], layout(seed=5), tokenizer_path="")
    resumed.load_state_dict(state)
    assert [tokens_of(batch) for batch, _ in zip(iter(resumed), range(6), strict=False)] == expected


def test_a_state_saved_under_another_seed_or_layout_is_refused(tmp_path):
    card = write_shards(tmp_path / "fw")
    state = open_stream([(card, 1.0)], layout(seed=1), tokenizer_path="").state_dict()
    with pytest.raises(ValueError, match="another seed"):
        open_stream([(card, 1.0)], layout(seed=2), tokenizer_path="").load_state_dict(state)
    with pytest.raises(ValueError, match="saved with 1 readers"):
        open_stream([(card, 1.0)], layout(readers=2, seed=1), tokenizer_path="").load_state_dict(state)


def test_a_mixture_draws_each_corpus_in_proportion_to_its_weight(tmp_path):
    first = write_shards(tmp_path / "first", shards=2, tokens=400)
    second = write_shards(tmp_path / "second", shards=2, tokens=400)
    # Mark the second corpus's ids so a window says where it came from
    for path in second.train_paths():
        (np.fromfile(path, dtype=np.uint16) + np.uint16(20_000)).tofile(path)
    stream = open_stream([(first, 3.0), (second, 1.0)], layout(seed=1), tokenizer_path="")
    from_second = sum(batch["input"][0] >= 20_000 for batch in take(stream, 200))
    assert 30 <= from_second <= 70


def test_a_mixture_is_a_function_of_its_seed(tmp_path):
    first = write_shards(tmp_path / "first")
    second = write_shards(tmp_path / "second")

    def draw(seed):
        stream = open_stream([(first, 1.0), (second, 1.0)], layout(seed=seed), tokenizer_path="")
        return [tokens_of(batch) for batch in take(stream, 12)]

    assert draw(1) == draw(1)
    assert draw(1) != draw(2)


def test_a_mixture_resumes_exactly(tmp_path):
    corpora = [(write_shards(tmp_path / "a", tokens=90), 2.0), (write_text(tmp_path / "t"), 1.0)]
    stream = open_stream(corpora, layout(readers=2, seed=4), tokenizer_path="", packing=Packing.CONCATENATE)
    iterator = iter(stream)
    for _ in range(5):
        next(iterator)
    state = stream.state_dict()
    expected = [tokens_of(next(iterator)) for _ in range(10)]
    resumed = open_stream(corpora, layout(readers=2, seed=4), tokenizer_path="", packing=Packing.CONCATENATE)
    resumed.load_state_dict(state)
    assert [tokens_of(batch) for batch, _ in zip(iter(resumed), range(10), strict=False)] == expected


def test_a_mixture_weight_must_be_positive(tmp_path):
    card = write_shards(tmp_path / "fw")
    with pytest.raises(ValueError, match="weights must be positive"):
        open_stream([(card, 1.0), (card, 0.0)], layout(), tokenizer_path="")


def test_text_is_tokenized_as_it_is_read_and_documents_are_joined_by_the_end_id(tmp_path):
    card = write_text(tmp_path / "text", files=1, row_groups=1, documents=6)
    stream = open_stream(
        [(card, 1.0)],
        layout(seq_len=4),
        tokenizer_path="",
        packing=Packing.CONCATENATE,
        held_out_fraction=0.0,
    )
    batch = take(stream, 1)[0]
    # The first document is "aaa", then the end id, then "bbbb"
    assert batch["input"].tolist() == [ord("a")] * 3 + [FakeEncoder.eos_id]
    assert batch["labels"].tolist() == [ord("a")] * 2 + [FakeEncoder.eos_id, ord("b")]


def test_a_packing_a_corpus_cannot_do_is_refused(tmp_path):
    card = write_shards(tmp_path / "fw")
    with pytest.raises(ValueError, match="supports packing"):
        open_stream([(card, 1.0)], layout(), tokenizer_path="", packing=Packing.SPAN)


def test_the_held_out_side_of_shards_streams_the_validation_file(tmp_path):
    card = write_shards(tmp_path / "fw", validation_tokens=33)
    batch = take(open_stream([(card, 1.0)], layout(), tokenizer_path="", split="held_out"), 1)[0]
    assert batch["input"][0] == 40_000
