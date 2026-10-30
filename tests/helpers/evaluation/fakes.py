"""What the evaluation tests stand in for - a served model, its fixed samples and a kernel harness's files.

`FakeServer` answers chat requests from a Python function and keeps every request. `make_data` writes the
two RULER files the runner tests score, and `respond` answers them right, wrong or empty. `raw` is one file
the attention-layer harness writes, in the shape `lsa.evaluation.performance.kernels.measure.layer` writes it, built here
rather than captured because a captured one is 65 MiB of trials and what is checked is the arithmetic.
"""

import asyncio
import json
import os
import socket
import threading

from aiohttp import web

from lsa import io
from lsa.evaluation.constants import Kernels
from lsa.evaluation.performance.kernels import record


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeServer:
    """respond(messages) -> dict with content, and optionally reasoning, finish_reason, status, delay.

    Token counts are word counts, so tests can check them exactly. With info, /server_info answers
    it too, so the runner can talk to the fake server alone.
    """

    def __init__(self, respond, info=None, slots=None):
        self.respond = respond
        self.info = info
        # With slots, a request past them waits for one to free, as sglang queues past --max-running-requests
        self.slots = slots
        self._slots = None
        self.waiting = 0
        self.peak_waiting = 0
        self.requests = []
        self.running = 0
        self.peak = 0
        self.running_of = {}
        self.peak_of = {}
        # ("in", group) as each request arrives and ("out", group) as its answer leaves, in order
        self.events = []
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    async def _handle(self, request):
        body = await request.json()
        self.requests.append(body)
        out = self.respond(body["messages"])
        if self.slots:
            self._slots = self._slots or asyncio.Semaphore(self.slots)
            if self._slots.locked():
                self.waiting += 1
                self.peak_waiting = max(self.peak_waiting, self.waiting)
                await self._slots.acquire()
                self.waiting -= 1
            else:
                await self._slots.acquire()
        try:
            return await self._answer(body, out)
        finally:
            if self.slots:
                self._slots.release()

    async def _answer(self, body, out):
        # Requests being answered at once, and the most there ever were - in all and per `group` of the answer
        group = out.get("group")
        self.running += 1
        self.running_of[group] = self.running_of.get(group, 0) + 1
        self.peak = max(self.peak, self.running)
        self.peak_of[group] = max(self.peak_of.get(group, 0), self.running_of[group])
        self.events.append(("in", group))
        if out.get("delay"):
            await asyncio.sleep(out["delay"])
        self.events.append(("out", group))
        self.running -= 1
        self.running_of[group] -= 1
        if out.get("status", 200) != 200:
            return web.json_response({"error": out.get("error", "failed")}, status=out["status"])
        content = out.get("content", "")
        reasoning = out.get("reasoning", "")
        prompt_tokens = sum(len(m["content"].split()) for m in body["messages"])
        reply = {
            "id": "fake",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning or None,
                    },
                    "finish_reason": out.get("finish_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(content.split()) + len(reasoning.split()),
                "total_tokens": prompt_tokens + len(content.split()) + len(reasoning.split()),
                "reasoning_tokens": len(reasoning.split()),
            },
        }
        if body.get("logprobs") and content:
            first = content.split()[0]
            alternatives = [{"token": first, "logprob": -0.25}][: body.get("top_logprobs", 0)]
            reply["choices"][0]["logprobs"] = {
                "content": [{"token": first, "logprob": -0.25, "top_logprobs": alternatives}]
            }
        return web.json_response(reply)

    async def _health(self, request):
        return web.json_response({"ok": True})

    async def _info(self, request):
        return web.json_response(self.info)

    def _serve(self):
        asyncio.set_event_loop(self._loop)
        app = web.Application(client_max_size=1 << 30)
        app.router.add_post("/v1/chat/completions", self._handle)
        app.router.add_get("/health", self._health)
        if self.info is not None:
            app.router.add_get("/server_info", self._info)
        runner = web.AppRunner(app)
        self._loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", self.port)
        self._loop.run_until_complete(site.start())
        self._ready.set()
        self._loop.run_forever()
        self._loop.run_until_complete(runner.cleanup())

    def start(self):
        self._thread.start()
        self._ready.wait(10)
        return self

    def stop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)


def echo_target(messages):
    """Default behaviour used by several tests: the 'answer' is the last line of the prompt."""
    text = messages[-1]["content"]
    return {"content": text.strip().splitlines()[-1], "reasoning": "thinking about it"}


def make_data(root):
    rows = []
    for i in range(6):
        rows.append(
            {
                "index": i,
                "input": f"Text number {i}.\nWhat is the magic number for key{i}?",
                "outputs": [str(1000000 + i)],
                "length": 100,
                "max_length": 4096,
                "gen_prefix": f"The special magic number for key{i} mentioned in the provided text is",
            }
        )
    # lm-eval adds no delimiter after an input that already ends in whitespace
    rows[5]["input"] += " "
    io.write_jsonl(os.path.join(root, "ruler", "niah_single_1_4096.jsonl"), rows)
    io.write_jsonl(os.path.join(root, "ruler", "cwe_4096.jsonl"), rows)


def respond(messages):
    """Right for even keys, empty for key3 (reasoning ate the budget), wrong otherwise."""
    text = messages[-1]["content"]
    key = int(text.split("key")[1].split("?")[0])
    if key == 3:
        return {"content": "", "reasoning": "hmm " * 20, "finish_reason": "length"}
    if key % 2 == 0:
        return {"content": f"The number is {1000000 + key}.", "reasoning": "found it"}
    return {"content": "I think it is 42.", "reasoning": "guessing"}


GPU_UUID = "GPU-abc"


def sample(mode, module_ms, metadata_ms=None):
    row = {
        "mode": mode,
        "repeat": 0,
        "module_ms": module_ms,
        "step_gpu_ms": module_ms + 1.0,
        "allocator_retries": 0,
        "allocator_ooms": 0,
        "peak_allocated_bytes": 30 * 2**30,
        "peak_reserved_bytes": 31 * 2**30,
        "allocated_before_bytes": 20 * 2**30,
    }
    if mode == "components":
        exclusive = {name: 1.0 for name in record.COMPONENTS}
        row["components"] = {"exclusive_ms": exclusive}
        row["metadata_components"] = {"exclusive_ms": dict(exclusive)}
    if metadata_ms is not None:
        row["metadata_ms"] = metadata_ms
    return row


def raw(
    context,
    indexer="native",
    phase="prefill",
    module_ms=10.0,
    case=None,
    graph=True,
    growth=False,
    differing=0,
    graph_differing=None,
):
    # Without an index there are no scores and no selection to check - those checks read None
    sparse = indexer in ("native", "lsa")
    validation = {
        "reference_order_output": [{"shape": [1, 2], "dtype": "torch.bfloat16", "exact": True}, None],
        "final_cache_and_index_state_exact": True,
        "checked_score_rows_exact": True if sparse else None,
        "independent_score_oracle": {"passed": True if sparse else None},
        "topk_cutoff_valid": indexer == "lsa" if sparse else None,
    }
    if sparse:
        validation["selection_rows"] = {
            "rows": 8192,
            "differing": differing,
            "widest_side": 12 if differing else 0,
            "bucket_cap": 16,
        }
    out = {
        "schema": Kernels.RAW_SCHEMA,
        "args": {
            "model_path": "/models/glm53flash-real5",
            "prompt_file": "/prompts/p.txt",
            "indexer": indexer,
            "context": context,
            "chunk": 8192,
            "layer": 3,
            "tp_size": 1,
            "warmup": 3,
            "repeats": 2,
            "window_chunks": 32,
            "graph_compare": phase == "decode",
            "decode_offsets": [-1, 0, 1],
        },
        "lsa_commit": "e21842e834f613e370f41b04baef121f3833ec16",
        "lsa_diff": "",
        "source_sha256": {"layer.py": "0" * 64},
        "gpu_uuid": GPU_UUID,
        "driver": "570.211.01",
        "hostname": "node-1",
        "torch": "2.13.0",
        "cuda": "12.9",
        "gpu": "NVIDIA H200",
        "prompt_source_tokens": 299971,
        "prompt_repeated": True,
        "validation": validation,
        "requested_prefill_backend": "flashmla_sparse",
        "memory": {
            "kv_capacity_bytes": 1000,
            "native_index_capacity_bytes": 100,
            "lsa_index_bytes": 50,
        },
        "hierarchy_growth": growth,
        "indexer_config": {"top_level": 0, "top_k": 4096} if indexer == "lsa" else None,
        "samples": [
            sample("plain", module_ms, 0.7),
            sample("components", module_ms + 0.5, 0.8),
            sample("plain", module_ms + 1.0, 0.7),
        ],
        "graph_comparison": None,
    }
    if phase == "decode":
        flags = {name: name == case for name in Kernels.CASES}
        out.update(
            phase="decode",
            requested_decode_backend="fa3",
            decode_case={
                "offset": context - 131072,
                **flags,
                "periods": {"cluster_block_close": 512, "page_allocation": 8192}
                if indexer == "lsa"
                else None,
            },
        )
        if graph:
            out["graph_comparison"] = {
                "status": "validated",
                "reference_order_output_exact": True,
                "samples": [sample("graph", module_ms / 10, 0.7), sample("eager", module_ms, 0.7)],
            }
            if graph_differing is not None:
                out["graph_comparison"]["graph_selection_differing_slots"] = graph_differing
    return out


def write(run_dir, name, payload):
    path = run_dir / "samples" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
