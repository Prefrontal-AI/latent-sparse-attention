"""A key head on two ranks, each holding half of its group's query heads - the pair select what one rank holding the group does.

The two ranks are real processes on two GPUs, summing their block scores over NCCL as the attention group does, so the sum is
held both eagerly - a prefill's - and replayed from a CUDA graph, as a decode step is.
"""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lsa.sglang.index import pisa
from lsa.sglang.models.qwen3_5.pisa import Replicas
from lsa.sglang.settings.design import PisaConfig

pytestmark = pytest.mark.gpu

KEY_DIM = 256
GROUP_SIZE = 6
CONTEXT = 6144
ROWS = 48
LOGIT_SCALE = KEY_DIM**-0.5
SLOTS = 8192


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def sum_over_group(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor)
    return tensor


def gather_group(tensor: torch.Tensor) -> torch.Tensor:
    stacked = torch.empty(2 * tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(stacked, tensor)
    return stacked.view(2, *tensor.shape)


def build_index(config: PisaConfig, table: torch.Tensor, pool: torch.Tensor, reduce) -> pisa.PisaIndex:
    index = pisa.PisaIndex(
        config, KEY_DIM, 1, CONTEXT, lambda: table, lambda: pool, pool.device, reduce_scores=reduce
    )
    index.maintain(["request"], [0], [CONTEXT])
    return index


def serve_pair(rank: int, port: int, config: PisaConfig) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=2, rank=rank)
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(0)
    table = torch.randperm(SLOTS, generator=generator)[:CONTEXT].view(1, CONTEXT).int().to(device)
    pool = torch.zeros(SLOTS, 1, KEY_DIM, dtype=torch.bfloat16, device=device)
    pool[table[0].long(), 0] = torch.randn(CONTEXT, KEY_DIM, generator=generator).bfloat16().to(device)
    queries = torch.randn(ROWS, GROUP_SIZE, KEY_DIM, generator=generator).bfloat16().to(device)
    positions = torch.arange(CONTEXT - ROWS, CONTEXT, device=device)
    mine = queries[:, rank * 3 : rank * 3 + 3].contiguous()
    replicas = Replicas(0, 1, 2, sum_over_group, gather_group)
    whole = build_index(config, table, pool, None)
    pair = build_index(config, table, pool, replicas.sum_scores)

    wanted = whole.select(queries, positions, [0], [CONTEXT], [ROWS], LOGIT_SCALE)
    served = pair.select(mine, positions, [0], [CONTEXT], [ROWS], LOGIT_SCALE)

    # A decode step of the last row, captured with its sum inside the graph and replayed
    static = mine[-1:].clone()
    static_position = positions[-1:].clone()
    static_row = torch.zeros(1, dtype=torch.long, device=device)

    def decode() -> torch.Tensor:
        return pair.decode(static, static_position, static_row, LOGIT_SCALE)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            decode()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = decode()
    static.copy_(mine[-2:-1])
    static_position.copy_(positions[-2:-1])
    graph.replay()
    replayed = captured.clone()
    wanted_replay = whole.decode(queries[-2:-1], positions[-2:-1], static_row, LOGIT_SCALE)
    assert replicas.agree(served)
    assert torch.equal(served, wanted)
    assert torch.equal(replayed, wanted_replay)
    assert torch.equal(gather_group(replayed)[0], gather_group(replayed)[1])
    # The group is torn down after the graph that holds its collective
    del graph
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "config", [PisaConfig(), PisaConfig(block_size=8, keep_count=576)], ids=["block64", "block8"]
)
def test_the_two_ranks_of_a_key_head_select_the_one_list_a_rank_holding_the_whole_group_selects(config):
    if torch.cuda.device_count() < 2:
        pytest.skip("a pair needs two GPUs")
    mp.spawn(serve_pair, args=(free_port(), config), nprocs=2, join=True)
