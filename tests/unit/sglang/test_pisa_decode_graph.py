"""A PISA decode step captured in a CUDA graph - replayed over new tokens, it selects what the same step run eagerly does.

The capture freezes every shape and every host value, so a step that read one from the host, or sized a tensor by the
batch's lengths, would replay the capture's selection. Decoding across block boundaries, with requests of different
lengths and a row left by a longer request, catches both.
"""

import pytest
import torch

from lsa.baselines import pisa as reference
from lsa.sglang.index import pisa
from lsa.sglang.settings.design import PisaConfig

pytestmark = pytest.mark.gpu

HEAD_DIM = 16
HEAD_COUNT = 4
LOGIT_SCALE = 0.4
CONTEXT = 400
CONFIG = PisaConfig(block_size=4, keep_count=6, branching=2, beam_width=5)


def positions_of(table, row, slots):
    inverse = {int(slot): position for position, slot in enumerate(table[row].tolist())}
    return [{inverse[int(slot)] for slot in line if slot >= 0} for line in slots]


def reference_set(keys, query, position):
    selection = reference.select(
        keys[: position + 1, None].cpu(),
        query[None].cpu(),
        torch.tensor([position]),
        CONFIG.block_size,
        CONFIG.keep_count,
        CONFIG.branching,
        CONFIG.beam_width,
        LOGIT_SCALE,
        CONFIG.forced_blocks,
    )
    ids = selection.selected_key_ids[0, 0]
    return set(ids[ids >= 0].tolist())


def test_a_captured_decode_step_selects_what_an_eager_one_and_the_reference_do():
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(0)
    slot_count = 3 * CONTEXT
    table = torch.randperm(slot_count, generator=generator).view(3, CONTEXT).int().to(device)
    buffer = torch.zeros(slot_count, 1, HEAD_DIM, device=device)
    index = pisa.PisaIndex(
        CONFIG, HEAD_DIM, 3, CONTEXT, lambda: table, lambda: buffer, device, mean_dtype=torch.float32
    )
    keys = torch.randn(3, CONTEXT, HEAD_DIM, generator=generator).to(device)
    queries = torch.randn(3, CONTEXT, HEAD_COUNT, HEAD_DIM, generator=generator).to(device)

    # A longer request leaves its means in row 0 before the short one takes it
    buffer[table[0, :300].long(), 0] = keys[2, :300]
    index.maintain(["old"], [0], [300])
    rids, rows, starts = ["long", "short"], [2, 0], [150, 37]
    for request, (row, start) in enumerate(zip(rows, starts, strict=True)):
        buffer[table[row, :start].long(), 0] = keys[request, :start]

    static_queries = torch.zeros(2, HEAD_COUNT, HEAD_DIM, device=device)
    static_positions = torch.zeros(2, dtype=torch.long, device=device)
    static_rows = torch.tensor(rows, device=device)

    def prepare(step):
        positions = [start + step for start in starts]
        index.maintain(rids, rows, positions)
        for request, (row, position) in enumerate(zip(rows, positions, strict=True)):
            buffer[table[row, position].long(), 0] = keys[request, position]
            static_queries[request] = queries[request, position]
        static_positions.copy_(torch.tensor(positions, device=device))
        return positions

    record = pisa.DecodeRecord()

    def decode():
        lists = index.decode(static_queries, static_positions, static_rows, LOGIT_SCALE)
        record.record(static_rows, static_positions, static_queries, lists)
        return lists

    prepare(0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            decode()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = decode()

    # Forty steps cross ten leaf blocks of each request and a level of the pyramid
    for step in range(40):
        positions = prepare(step)
        record.expect(rows, rids)
        graph.replay()
        rows_held, positions_held, current, (queries_held, lists_held) = record.take(rows, rids)
        assert (rows_held, positions_held, current) == (rows, positions, [True, True])
        assert torch.equal(queries_held, static_queries) and torch.equal(lists_held, captured)
        assert record.take(rows, rids) is None
        eager = decode()
        assert torch.equal(captured, eager), step
        for request, (row, position) in enumerate(zip(rows, positions, strict=True)):
            served = positions_of(table, row, captured[request : request + 1])[0]
            assert served == reference_set(keys[request], queries[request, position], position), (
                step,
                request,
            )

    # A request that finished on the recorded step may have handed its row to another by the time it is read
    positions = prepare(40)
    record.expect(rows, rids)
    graph.replay()
    _, _, current, _ = record.take(rows[1:], rids[1:])
    assert current == [False, True]
    positions = prepare(41)
    record.expect(rows, rids)
    graph.replay()
    _, _, current, _ = record.take(rows, ["new", rids[1]])
    assert current == [False, True]

    # A replay whose rows are not the ones expected - the warm-up's, say - is not taken
    record.expect(rows[::-1], rids[::-1])
    graph.replay()
    assert record.take(rows, rids) is None
