from types import SimpleNamespace

import torch

from lsa.sglang.index import dump
from lsa.sglang.index.dump import Dump, slot_positions
from lsa.sglang.settings.debug import DumpSettings


def mode(decode: bool) -> SimpleNamespace:
    return SimpleNamespace(is_decode=lambda: decode, is_extend=lambda: not decode)


def test_lists_read_back_as_key_positions_through_the_request_table():
    table_row = torch.tensor([10, 11, 12, 40, 41, 0, 0], dtype=torch.int32)
    lists = torch.tensor([[12, 40, -1], [10, 99, 41]], dtype=torch.int32)

    assert slot_positions(lists, table_row, 5).tolist() == [[2, 3, -1], [0, -2, 4]]


def test_the_dump_is_off_unless_a_directory_is_named():
    assert Dump.of(DumpSettings(), 3, 8, 4, "cpu") is None


def test_the_dump_covers_only_the_layers_it_names(tmp_path):
    named = DumpSettings(directory=str(tmp_path), layers=(3, 23))

    assert Dump.of(named, 7, 8, 4, "cpu") is None
    assert Dump.of(named, 23, 8, 4, "cpu") is not None


def test_a_prefill_leaves_its_lists_out_when_the_dump_says_so(tmp_path):
    layer = Dump(tmp_path, 3, 1, False, 8, 3, "cpu")
    positions = torch.arange(4, dtype=torch.int32)
    batch = SimpleNamespace(
        req_pool_indices=torch.tensor([0]),
        seq_lens_cpu=torch.tensor([4]),
        extend_seq_lens_cpu=[4],
        rids=["request"],
    )

    layer.prefill(
        torch.ones(4, 8, dtype=torch.bfloat16), positions, torch.zeros(4, 3, dtype=torch.int32), batch, None
    )

    assert "lists" not in torch.load(tmp_path / "layer3" / "request" / "prefill_0.pt")


def test_a_prefill_keeps_the_stride_rows_and_the_chunk_tail(monkeypatch, tmp_path):
    monkeypatch.setattr(dump, "TAIL", 2)
    layer = Dump(tmp_path, 3, 4, True, 8, 3, "cpu")
    positions = torch.arange(8, 18, dtype=torch.int32)
    x = torch.arange(10, dtype=torch.bfloat16)[:, None].expand(10, 8).contiguous()
    table = torch.arange(100, 132, dtype=torch.int32)[None]
    lists = torch.full((10, 3), -1, dtype=torch.int32)
    lists[:, 0] = table[0, positions.long()]
    batch = SimpleNamespace(
        req_pool_indices=torch.tensor([0]),
        seq_lens_cpu=torch.tensor([18]),
        extend_seq_lens_cpu=[10],
        rids=["request"],
    )

    layer.prefill(x, positions, lists, batch, table)

    saved = torch.load(tmp_path / "layer3" / "request" / "prefill_8.pt")
    assert saved["positions"].tolist() == [8, 12, 16, 17]
    assert saved["x"][:, 0].tolist() == [0, 4, 8, 9]
    assert saved["lists"][:, 0].tolist() == [8, 12, 16, 17]


def test_a_decode_step_is_written_before_the_next_forward_for_real_requests_only(tmp_path):
    layer = Dump(tmp_path, 3, 4096, True, 8, 2, "cpu")
    table = torch.arange(64, dtype=torch.int32).view(2, 32)
    step = SimpleNamespace(forward_mode=mode(True), rids=["a"], seq_lens_cpu=torch.tensor([6, 1]))
    layer.before_forward(step, [1], table)
    # The graph's padded batch of two, one real request at table row 1
    layer.decode(
        torch.ones(2, 8, dtype=torch.bfloat16),
        torch.tensor([5, 0], dtype=torch.int32),
        torch.tensor([[37, 33], [0, 0]], dtype=torch.int32),
    )

    layer.before_forward(SimpleNamespace(forward_mode=mode(False)), [], table)

    files = sorted(path.name for path in (tmp_path / "layer3").rglob("*.pt"))
    assert files == ["decode_5.pt"]
    saved = torch.load(tmp_path / "layer3" / "a" / "decode_5.pt")
    assert saved["lists"].tolist() == [[5, 1]]


def test_a_native_server_dumps_the_x_its_indexer_reads_at_a_prefill(tmp_path):
    from lsa.sglang import plugin

    named = DumpSettings(directory=str(tmp_path), layers=(3,), stride=2)
    indexer = SimpleNamespace()
    batch = SimpleNamespace(
        forward_mode=mode(False),
        req_pool_indices=torch.tensor([0]),
        seq_lens_cpu=torch.tensor([4]),
        extend_seq_lens_cpu=[4],
        rids=["request"],
    )

    plugin.dump_native(
        None,
        indexer,
        x=torch.ones(4, 8, dtype=torch.bfloat16),
        q_lora=None,
        positions=torch.arange(4),
        forward_batch=batch,
        layer_id=3,
        dump=named,
    )

    saved = torch.load(tmp_path / "layer3" / "request" / "write_0.pt")
    assert saved["positions"].tolist() == [0, 1, 2, 3]
    assert "lists" not in saved
