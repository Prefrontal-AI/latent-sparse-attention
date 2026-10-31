"""Wrap everything inside one attention layer so the recorder can charge time to a component.

This is the GPU-coupled half of the accounting: it patches the indexer, the bucket cache and the attention
backend in place for the length of a measurement. The accounting itself is
[recorder](recorder.py).

Three shapes, because the variants are three different layers. The linear layer has its own projections
and one kernel and is done; a layer with no indexer opens the attention ranges and stops there; the sparse
layer opens everything. Qwen's attention is a method of its decoder layer, so it has a branch of its own.

The design's indexer opens most of its ranges itself, through [probe](../../../../sglang/index/probe.py) - the
clustering, the scans and the picks. What is wrapped here is what opens none: the projection, the key write, and
the selector's kernels, which a decode step and an expansion call without a range.
"""

from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch

from lsa.evaluation.constants import Family
from lsa.evaluation.performance.kernels.measure.snapshot import pisa_indexes
from lsa.sglang.index import probe

# The selector's kernels by component. A flat decode step scans through the paged scan, shaped or not,
# and every read expands
SELECTOR_FUNCTIONS = {
    "scan_paged": "index q scan",
    "scan_paged_shaped": "index q scan",
    "expand_in_order": "index k select",
    "expand_by_threshold": "index k select",
}


def wrap_lsa(indexer, wrap, stack) -> None:
    """Charge the design's indexer's parts that open no range of their own."""
    from lsa.core.kernels.cutedsl.decode import DecodeSelect
    from lsa.core.kernels.triton.top_k import TopK
    from lsa.sglang.index import selector

    index = indexer.index
    projection = type(index.projection)
    # The key and query input projects its index queries as they arrive and its index keys from the KV pool's keys
    # wherever they are read, inside the scans and the clustering; the attention input projects both in one call
    for attribute in ("index_queries", "index_keys") if hasattr(projection, "index_keys") else ("__call__",):
        wrap(projection, attribute, "index qk projections", stack)
    wrap(index, "write", "index k prep", stack)
    wrap(TopK, "listed", "index k select", stack)
    # The CuTeDSL decode select lists while it scans
    # => its one launch is the step's scan, and its select is in it
    wrap(DecodeSelect, "listed", "index q scan", stack)
    for name, component in SELECTOR_FUNCTIONS.items():
        wrap(selector, name, component, stack)


def wrap_pisa(indexer, wrap, stack) -> None:
    """Charge PISA's search and its key write, which open no range of their own.

    The search of a prefill and of a decode step is each one call, and it holds the pyramid's levels, the keeps
    and the closing of the blocks a prefill's keys complete - so all of it is `index k select`.
    """
    # Qwen holds one index a key head, and its key write is the backend's own save into the pool
    for index in pisa_indexes(indexer):
        wrap(index, "write", "index k prep", stack)
        wrap(index, "select", "index k select", stack)
        wrap(index, "decode", "index k select", stack)


def wrap_qsa(indexer, wrap, stack) -> None:
    """Charge QSA's indexer - its projections, its block compression, its block scores and its block top-k.

    The decode top-k is `fast_topk` when the budget is 512 blocks, and the indexer imports it at the call,
    so it is wrapped where it is defined.
    """
    from sglang.kernels.ops.elementwise import fast_topk
    from sglang.srt.layers.attention.qsa import qsa_indexer

    wrap(indexer, "project_qk", "index qk projections", stack)
    wrap(indexer, "update_key_state_and_compress", "index k prep", stack)
    for name in ("qsa_mqa_prefill", "qsa_mqa_decode"):
        wrap(qsa_indexer, name, "index q scan", stack)
    for name in ("qsa_fast_topk", "expand_qsa_block_indices"):
        wrap(qsa_indexer, name, "index k select", stack)
    wrap(fast_topk, "fast_topk", "index k select", stack)


@contextmanager
def instrument(variant, recorder):
    """Wrap the layer's components so the recorder charges their time to a component name.

    Args:
        variant: The layer under test, which says what to wrap.
        recorder: Where the ranges go.
    """
    module, backend = variant.module, variant.backend

    def wrap(obj, attribute, name, stack) -> None:
        """Patch `obj.attribute` to run inside a recorder range named `name`, until `stack` closes."""
        original = getattr(obj, attribute)

        @wraps(original)
        def measured(*args, **kwargs):
            """Run the original inside the range, counting backend calls."""
            if name == "topk attention":
                recorder.backends[attribute] += 1
            with recorder.region(name):
                return original(*args, **kwargs)

        stack.enter_context(patch.object(obj, attribute, measured))

    with ExitStack() as stack:
        stack.enter_context(patch.object(probe, "recorder", recorder))
        if variant.linear:
            from lsa.evaluation.performance.kernels.measure.variant import linear_output, linear_projections

            # GDN fuses its two input projections into one GEMM - time the step, not the modules
            inputs = (
                [(module, "_forward_input_proj")]
                if hasattr(module, "_forward_input_proj")
                else [(part, "forward") for part in linear_projections(module).values()]
            )
            for part, attribute in inputs:
                wrap(part, attribute, "attention projections", stack)
            for part in linear_output(module):
                wrap(part, "forward", "attention projections", stack)
            # The short conv, the delta rule and the state update
            wrap(module.attn, "forward", "topk attention", stack)
            yield
            return
        indexer = variant.indexer
        if variant.family == Family.QWEN3_5:
            layer = module.layer
            # The fused QKV GEMM with QK-norm, rotary and the gate split, then the output projection
            wrap(layer, "_prepare_qkv_gate", "attention projections", stack)
            wrap(layer.o_proj, "forward", "attention projections", stack)
            wrap(backend, "forward_extend", "topk attention", stack)
            wrap(backend, "forward_decode", "topk attention", stack)
            # The design's indexer runs inside the backend's call, and its ranges nest in that one's
            if getattr(indexer, "is_lsa", False):
                wrap_lsa(indexer, wrap, stack)
            if variant.pisa:
                wrap_pisa(indexer, wrap, stack)
            # QSA's runs before it, from the layer
            if variant.qsa:
                wrap_qsa(indexer, wrap, stack)
            yield
            return
        # Dense MLA prefills through the MHA paths and decodes through the latent one, each on its own backend
        # => every path's prepare and core are the projections, and the two attention modules the attention
        if variant.kind == "full":
            for path in (
                "forward_normal",
                "forward_normal_chunked_kv",
                "forward_normal_one_shot",
                "forward_absorb",
            ):
                for step in ("prepare", "core"):
                    if hasattr(module, f"{path}_{step}"):
                        wrap(module, f"{path}_{step}", "attention projections", stack)
            wrap(module.attn_mha, "forward", "topk attention", stack)
            wrap(module.attn_mqa, "forward", "topk attention", stack)
            yield
            return
        wrap(module, "forward_absorb_prepare", "attention projections", stack)
        wrap(module, "forward_absorb_core", "attention projections", stack)
        wrap(module.attn_mqa, "forward", "other", stack)
        wrap(backend, "_forward_fa3", "topk attention", stack)
        wrap(backend, "_forward_flashmla_sparse", "topk attention", stack)
        wrap(backend, "_forward_tilelang", "topk attention", stack)
        # A pooled index's tail goes to trtllm in place of flashmla_sparse on Blackwell, as it goes to fa3 on Hopper
        if hasattr(backend, "_forward_trtllm"):
            wrap(backend, "_forward_trtllm", "topk attention", stack)
        if indexer is None:
            yield
            return
        wrap(indexer, "forward", "other", stack)
        if getattr(indexer, "is_pisa", False):
            wrap_pisa(indexer, wrap, stack)
            yield
            return
        if getattr(indexer, "is_lsa", False):
            wrap_lsa(indexer, wrap, stack)
            yield
            return
        from sglang.kernels.ops.attention.dsa import triton_kernel
        from sglang.srt.layers.attention.dsa import (
            dsa_indexer_kpool,
            kpool_fp8_index,
        )

        # A prefill writes the pooled keys on a second stream, and the accounting charges one stream
        # => serial
        stack.enter_context(patch.object(indexer, "_can_overlap_prefill", lambda *args, **kwargs: False))
        wrap(indexer, "_get_q_k_bf16", "index q prep", stack)
        wrap(indexer.wq_b, "forward", "index qk projections", stack)
        wrap(indexer.wk, "forward", "index qk projections", stack)
        wrap(indexer.k_norm, "forward", "index k prep", stack)
        wrap(indexer, "_get_logits_head_gate", "index q prep", stack)
        wrap(indexer, "_compress_write", "index k prep", stack)
        wrap(triton_kernel, "act_quant", "index q prep", stack)
        wrap(
            kpool_fp8_index,
            "gather_index_k_scale_prefix_into",
            "index k prep",
            stack,
        )
        wrap(dsa_indexer_kpool.deep_gemm, "fp8_mqa_logits", "index q scan", stack)
        wrap(
            dsa_indexer_kpool.deep_gemm,
            "fp8_paged_mqa_logits",
            "index q scan",
            stack,
        )
        wrap(indexer, "_topk_from_kpool_logits", "index k select", stack)
        yield
