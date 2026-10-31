"""The layer under test and what belongs to it - module, backend, state, checkpoints, selection checks.

    sparse  the attention layer with the model's indexer or the design's   --indexer native | lsa | pisa
    full    the same layer with no indexer - FA3 over every token  --indexer full
    linear  a linear attention layer of the same model             --indexer kda (GLM) | gdn (Qwen)

Three layers behind one object, because everything the harness does after the model is loaded differs
between them and none of it differs anywhere else. `full` is the sparse layer built without an indexer,
which is a server argument rather than a code path; `linear` is a different module in the same stack, with
its own state, its own projections and its own backend.

Two model families. GLM-5.3-Flash has MLA attention with a native DSA indexer, the design's in its place under
`lsa` or `pisa`, and KDA linear layers. Qwen3.5 and 3.8 have dense GQA attention and Gated DeltaNet linear layers - the design
reads the dense layer's keys and queries through the plugin's attention backend, and `native` is QSA, the block
indexer a QSA checkpoint carries on each attention layer, served by the image's QSA backend.

The state classes live in [snapshot](snapshot.py), because restoring a mutated cache is arithmetic a
CPU can test; what is here is the wiring that needs a loaded model.
"""

import inspect
from contextlib import contextmanager, nullcontext

import torch

from lsa.evaluation.constants import Family
from lsa.evaluation.performance.kernels.family import model_family
from lsa.evaluation.performance.kernels.measure.recorder import REQUIRED
from lsa.evaluation.performance.kernels.measure.selection import (
    LSASelectionCheck,
    NoSelection,
    SelectionCheck,
)
from lsa.evaluation.performance.kernels.measure.snapshot import (
    IndexState,
    LinearSnapshot,
    Snapshot,
    index_buffer,
    kv_buffers,
    linear_layer_id,
    pisa_indexes,
)
from lsa.sglang import settings

# What each `--indexer` is, as a layer. Three of the six are the same kind of layer read three ways
KINDS = {
    "native": "sparse",
    "lsa": "sparse",
    "pisa": "sparse",
    "full": "full",
    "kda": "linear",
    "gdn": "linear",
}

# The variants each family has. Qwen's `native` is QSA, which only a QSA checkpoint carries, and its linear layer is GDN
FAMILY_VARIANTS = {
    Family.GLM5_NEXT: ("native", "lsa", "pisa", "full", "kda"),
    Family.QWEN3_5: ("native", "full", "lsa", "pisa", "gdn"),
}


def hierarchy_bytes(hierarchies):
    """Return the bytes the node buckets of every request's hierarchy hold, grown capacity included."""
    return sum(
        part.nbytes
        for hierarchy in hierarchies or ()
        for rows in (*hierarchy.open_node_buckets, *hierarchy.closed_node_buckets)
        for part in rows.parts.values()
    )


def index_tensors(index):
    """Return the bytes of each tensor the design's index holds for a layer, by name."""
    cache = index.selector.cache
    tensors = {name: getattr(cache, name).nbytes for name in IndexState.TENSORS}
    return {
        "index_keys": index.index_keys.nbytes,
        **tensors,
        "node_buckets": hierarchy_bytes(cache.hierarchies),
    }


def memory(module, pool, parameters):
    """Return the bytes a layer's KV cache, native index, design index and parameters take.

    Args:
        module: The layer under test.
        pool: The KV pool.
        parameters: The layer's parameters.
    """
    design = getattr(module.indexer, "is_lsa", False)
    pisa = getattr(module.indexer, "is_pisa", False)
    index = None if design or pisa else index_buffer(pool, module.layer_id)
    result = {
        "kv_capacity_bytes": sum(buffer.nbytes for buffer in kv_buffers(pool, module.layer_id)),
        "native_index_capacity_bytes": index.nbytes if index is not None else 0,
        "module_parameter_bytes": sum(p.nbytes for p in parameters),
        "lsa_index_bytes": 0,
        "pisa_index_bytes": 0,
    }
    if pisa:
        # The leaf means and the pyramid above are all PISA's index holds - its keys are the KV pool's
        # Qwen holds one index a key head
        leaves = [index.leaf_means for index in pisa_indexes(module.indexer)]
        result["pisa_index_bytes"] = sum(leaf.means.nbytes + leaf.upper.nbytes for leaf in leaves)
    if design:
        result["lsa_index_tensors"] = index_tensors(module.indexer.index)
        result["lsa_index_bytes"] = sum(result["lsa_index_tensors"].values())
    return result


def armed_indexer(layer_id):
    """Return the design's indexer the plugin armed for a dense Qwen layer, or raise when it armed none."""
    from lsa.sglang.models.qwen3_5 import backend

    if layer_id not in backend.ARMED:
        raise ValueError(f"The plugin armed no indexer for layer {layer_id} - armed {sorted(backend.ARMED)}")
    return backend.ARMED[layer_id]


def armed_pisa_indexer(layer_id):
    """Return the PISA indexer the plugin armed for a dense Qwen layer, or raise when it armed none."""
    from lsa.sglang.models.qwen3_5 import pisa

    if layer_id not in pisa.ARMED:
        raise ValueError(
            f"The plugin armed no PISA indexer for layer {layer_id} - armed {sorted(pisa.ARMED)}"
        )
    return pisa.ARMED[layer_id]


def qsa_indexer(layer):
    """Return the QSA indexer a Qwen attention layer built, or raise when its checkpoint carries none."""
    if not getattr(layer, "is_qsa", False):
        raise ValueError(
            "Qwen's native variant is QSA, and this checkpoint has no indexer_* fields to build it"
        )
    return layer.indexer


def linear_projections(module):
    """The linear layer's input projections - GDN's pair, KDA's fused pair or KDA's six"""
    if hasattr(module, "in_proj_qkvz"):
        return {"qkvz_projection": module.in_proj_qkvz, "ba_projection": module.in_proj_ba}
    if module.do_fuse_qkvbfg:
        return {
            "qkv_projection": module.fused_qkvbfg_a_proj,
            "gate_projection": module.fused_fg_b_proj,
        }
    return {
        "qkv_projection": module.qkv_proj,
        "beta_projection": module.b_proj,
        "forget_a_projection": module.f_a_proj,
        "forget_b_projection": module.f_b_proj,
        "gate_a_projection": module.g_a_proj,
        "gate_b_projection": module.g_b_proj,
    }


def linear_output(module):
    """The linear layer's output norm and projection - KDA and GDN name them apart"""
    if hasattr(module, "out_proj"):
        return module.norm, module.out_proj
    return module.o_norm, module.o_proj


def linear_memory(module, req_pool):
    """Return the bytes a linear layer's state and parameters take - one conv window and one recurrent state."""
    cache = req_pool.mamba2_layer_cache(linear_layer_id(module))
    return {
        "kv_capacity_bytes": cache.conv[0][0].nbytes + cache.temporal[0].nbytes,
        "native_index_capacity_bytes": 0,
        "module_parameter_bytes": sum(p.nbytes for p in module.parameters()),
        "lsa_index_bytes": 0,
        "pisa_index_bytes": 0,
    }


class QwenAttention(torch.nn.Module):
    """Qwen3.5's attention is a method of its decoder layer - this module stands in for it, so hooks see it.

    Installed over the layer's `self_attention`, it is what the model calls and what the harness times: the
    projections, the fused norm, rotary and gate, the attention and the output projection. The layer is held
    in a list so the stand-in does not register its own parent as a child.
    """

    def __init__(self, layer, indexer) -> None:
        """Stand in for a decoder layer's attention.

        Args:
            layer: The decoder layer.
            indexer: Its indexer, or None.
        """
        super().__init__()
        self.layers = [layer]
        self.layer_id = layer.attn.layer_id
        self.indexer = indexer
        self.skip_topk = False

    @property
    def layer(self):
        """The decoder layer this stands in for."""
        return self.layers[0]

    def forward(self, positions, hidden_states, forward_batch, **unused):
        """Run the layer's own attention."""
        return type(self.layer).self_attention(self.layer, positions, hidden_states, forward_batch)

    def parts(self):
        """Return the modules whose parameters the attention owns."""
        layer = self.layer
        return [layer.qkv_proj, layer.o_proj, layer.q_norm, layer.k_norm]

    @staticmethod
    def install(layer, indexer):
        """Replace the layer's `self_attention` with a stand-in and return it."""
        module = QwenAttention(layer, indexer)
        object.__setattr__(layer, "self_attention", module)
        return module


class NoHandle:
    """What `register_forward_hook` returns when there is no indexer to hook"""

    def remove(self) -> None:
        """Remove nothing."""


class ReplacedMethod:
    """A method set on one instance in place of the class's, put back by `remove` as it was."""

    def __init__(self, owner, name, function) -> None:
        """Set `function` on `owner` as `name`, keeping what was set there.

        Args:
            owner: The instance whose method is replaced.
            name: The method's name.
            function: The replacement, called without `self`.
        """
        self.owner = owner
        self.name = name
        self.previous = owner.__dict__.get(name)
        setattr(owner, name, function)

    def remove(self) -> None:
        """Put the instance's method back as it was."""
        if self.previous is None:
            delattr(self.owner, self.name)
        else:
            setattr(self.owner, self.name, self.previous)


class Variant:
    """The layer under test and everything the harness asks of it.

    Holds the module, its backend, indexer and pools, and builds the snapshot, the selection check, the
    hooks and the oracle that fit its kind (sparse, full or linear) and model family.
    """

    def __init__(self, runner, args) -> None:
        """Find the layer and check it is the kind the command line asked for.

        Args:
            runner: The model runner.
            args: The command line, whose `indexer` and `layer` pick the variant.

        Raises:
            ValueError: The family has no such variant or the built layer is not the one asked for.
        """
        self.name = args.indexer
        self.kind = KINDS[args.indexer]
        self.linear = self.kind == "linear"
        self.lsa = args.indexer == "lsa"
        self.pisa = args.indexer == "pisa"
        self.design = settings.design_of(runner.model_config.hf_config) if self.lsa else None
        self.family = model_family(runner.model_config.hf_text_config.model_type)
        self.qsa = self.family == Family.QWEN3_5 and args.indexer == "native"
        if args.indexer not in FAMILY_VARIANTS[self.family]:
            raise ValueError(
                f"{self.family} has no {args.indexer} variant - one of {FAMILY_VARIANTS[self.family]}"
            )
        # PISA's selection is the one `index k select` range, since it has no query scan apart from its search
        self.required = REQUIRED["pisa" if self.pisa else self.kind]
        layer = runner.model.model.layers[args.layer]
        holds_linear = (
            hasattr(layer, "linear_attn")
            if self.family == Family.QWEN3_5
            else getattr(layer, "is_linear_attn", False)
        )
        if self.linear != bool(holds_linear):
            raise ValueError(f"Layer {args.layer} does not hold a {self.kind} layer")
        # The linear layer answers to the hybrid dispatcher, which hands it to the linear backend
        self.backend = (
            runner.attn_backend.linear_attn_backend if self.linear else runner.attn_backend.full_attn_backend
        )
        self.dispatch = runner.attn_backend if self.linear else self.backend
        self.pool = runner.token_to_kv_pool
        self.req_pool = runner.req_to_token_pool
        if self.linear:
            self.module = layer.linear_attn if self.family == Family.QWEN3_5 else layer.self_attn
            self.indexer = None
        elif self.family == Family.QWEN3_5:
            # The design's indexer is the one the plugin armed on the backend, PISA's the one it armed beside it, QSA's
            # the one the layer built
            self.indexer = (
                armed_indexer(layer.attn.layer_id)
                if self.lsa
                else armed_pisa_indexer(layer.attn.layer_id)
                if self.pisa
                else qsa_indexer(layer)
                if self.qsa
                else None
            )
            self.module = QwenAttention.install(layer, self.indexer)
        else:
            self.module = layer.self_attn
            self.indexer = self.module.indexer
        if self.kind == "sparse" and (self.indexer is None or self.module.skip_topk):
            raise ValueError("The sparse layer needs its indexer")
        # `armed` does not prove `reached` - a hook that failed to apply leaves the checkpoint's own indexer
        # here and every number would be recorded under the design's label
        if self.kind == "sparse" and self.lsa is not bool(getattr(self.indexer, "is_lsa", False)):
            raise ValueError(
                f"--indexer {args.indexer} but the built indexer is "
                f"{'the design' if getattr(self.indexer, 'is_lsa', False) else 'the checkpoint of its own'}"
            )
        if self.kind == "sparse" and self.pisa is not bool(getattr(self.indexer, "is_pisa", False)):
            raise ValueError(
                f"--indexer {args.indexer} but the built indexer is not PISA's, or is and was not asked for"
            )
        if self.kind == "full" and self.indexer is not None:
            raise ValueError("Full attention must build no indexer")
        # Nothing is selected without an indexer - an empty list stands where the sparse layer's is read
        self.empty_selection = torch.empty(0, 0, dtype=torch.int32, device=runner.device)

    def arguments(self, positional, kwargs):
        """The call's arguments by name - the GDN layer is called positionally"""
        if not positional:
            return dict(kwargs)
        return dict(inspect.signature(self.module.forward).bind(*positional, **kwargs).arguments)

    def checkpoints(self):
        """Return the layer's parts, keyed by name, whose outputs are compared against the reference."""
        module = self.module
        if self.linear:
            norm, output = linear_output(module)
            return {
                **linear_projections(module),
                "linear_attention": module.attn,
                "output_norm": norm,
                "output_projection": output,
            }
        if self.family == Family.QWEN3_5:
            layer = module.layer
            return {
                "qkv_projection": layer.qkv_proj,
                "attention": layer.attn,
                "output_projection": layer.o_proj,
            }
        projections = {
            "qkv_projection": module.fused_qkv_a_proj_with_mqa,
            "q_norm": module.q_a_layernorm,
            "k_norm": module.kv_a_layernorm,
            "q_projection": module.q_b_proj,
        }
        # Dense MLA prefills through the MHA kernels and decodes through the latent one, as sglang dispatches it
        # => no one attention module runs in every pass, and the layer's output is what holds it
        if self.kind == "full":
            return projections
        return {**projections, "attention": module.attn_mqa}

    def snapshot(self, batch):
        """Save the state one forward pass of the layer mutates."""
        if self.linear:
            return LinearSnapshot(self.module, self.req_pool, batch)
        return Snapshot(self.module, self.pool, batch)

    def selection(self, slots, positions=None, rows=None, decode=False):
        """Return the check that a selection is the true top-k, or one that checks nothing.

        Args:
            slots: The request's KV slots in position order.
            positions: Each query row's position.
            rows: The sampled query rows whose scores are kept, or None for all.
            decode: Whether the step is a decode step.
        """
        # PISA's and QSA's lists are timed without holding them to a reference - a served run's strict check holds PISA's
        if self.indexer is None or self.pisa or self.qsa:
            return NoSelection()
        # mrope positions are three rows - the token position is the first
        if positions is not None and positions.dim() == 2:
            positions = positions[0]
        check = LSASelectionCheck if self.lsa else SelectionCheck
        return check(self.module, slots, self.pool.size, positions, rows, decode)

    @property
    def reads_in_backend(self) -> bool:
        """Whether the attention backend reads the selection itself, so it is no module's output.

        Qwen's LSA and PISA are read there. QSA's layer calls its indexer and hands the output to the backend.
        """
        return self.family == Family.QWEN3_5 and not self.qsa

    @property
    def hooked(self) -> bool:
        """Whether `selection_hook` sees the selection, which the harness reads and checks."""
        return self.indexer is not None and not self.reads_in_backend

    def selection_hook(self, hook, with_kwargs=False):
        """Hook the indexer's output, or return a handle that removes nothing where there is no module to hook."""
        # QSA's layer calls its indexer and hands the output to the backend, so the indexer's output is the selection
        # The Qwen designs' selection is read inside the backend and is no module's output, so there is no hook
        if not self.hooked:
            return NoHandle()
        return self.indexer.register_forward_hook(hook, with_kwargs=with_kwargs)

    def listing_hook(self, hook):
        """Register a hook on whatever hands the attention its list, which Qwen's LSA does inside its backend.

        The harness's own checks keep to `selection_hook`, so Qwen's LSA is held to no more than it was.

        Args:
            hook: Called as `hook(indexer, arguments, output)` after each selection, with the list as `output`.

        Returns:
            A handle whose `remove` takes the hook off.
        """
        if not (self.reads_in_backend and self.lsa):
            return self.selection_hook(hook)
        indexer = self.indexer
        select = indexer.select

        def listed(*arguments, **kwargs):
            """Run the select, show the hook its list and return it."""
            output = select(*arguments, **kwargs)
            hook(indexer, arguments, output)
            return output

        return ReplacedMethod(indexer, "select", listed)

    def indexer_call_hook(self, hook):
        """Register a hook that sees the indexer's arguments before each call, for Qwen's LSA its `select`.

        Args:
            hook: Called as `hook(indexer, positional, kwargs)`.

        Returns:
            A handle whose `remove` takes the hook off.
        """
        if not (self.reads_in_backend and self.lsa):
            return self.indexer.register_forward_pre_hook(hook, with_kwargs=True)
        indexer = self.indexer
        select = indexer.select

        def called(*positional, **kwargs):
            """Show the hook the arguments, then run the select."""
            hook(indexer, positional, kwargs)
            return select(*positional, **kwargs)

        return ReplacedMethod(indexer, "select", called)

    def call_indexer(self, *positional, **kwargs):
        """Run the indexer's work alone on the arguments a step handed it, Qwen's LSA through its `select`."""
        if self.reads_in_backend and self.lsa:
            return self.indexer.select(*positional, **kwargs)
        return self.indexer(*positional, **kwargs)

    @contextmanager
    def pinned_selection(self, selected):
        """The indexer answers with this list - the attention runs on a known selection"""
        handle = self.selection_hook(lambda *unused: selected)
        try:
            yield
        finally:
            handle.remove()

    @property
    def frees_indexer(self) -> bool:
        """Whether a stand-in can hand the attention its selection - the indexer's output, or Qwen LSA's backend read."""
        return self.indexer is not None and (not self.reads_in_backend or self.lsa)

    @contextmanager
    def free_indexer(self, selected):
        """The indexer does no work and answers with this list - the layer as it would run if indexing were free.

        For Qwen's LSA the stand-in takes the place of the indexer's `select`, which is the key and query projection,
        the index key write, the scan, the bucket pick and the list expansion.
        The KV write and the attention over the list stay in the backend, and the index's maintenance runs outside the graph.
        Hooks on a module indexer still see the list.

        Args:
            selected: The selection to hand over, a buffer a captured graph keeps reading at replay.

        Yields:
            The times the stand-in answered, one entry each, so the caller can tell the list reached the attention.

        Raises:
            ValueError: The layer reads its selection somewhere other than its indexer's output, and not Qwen's LSA.
        """
        if not self.frees_indexer:
            raise ValueError(
                f"{self.name} reads its selection outside its indexer's output - no free indexer"
            )
        served = []

        def stand_in(*unused, **unused_kwargs):
            served.append(selected)
            return selected

        handle = ReplacedMethod(self.indexer, "select" if self.reads_in_backend else "forward", stand_in)
        try:
            yield served
        finally:
            handle.remove()

    def oracle(self, result, decode, tree=False):
        """Return the context that holds the layer's index scores to an independent reference while it runs.

        Args:
            result: Where the verdict goes.
            decode: Whether the pass is a decode step.
            tree: Whether the read goes through a hierarchy, which has no one score row to hold.
        """
        if self.indexer is None or self.pisa:
            result.update(passed=None, reference="no index scores")
            return nullcontext()
        if self.qsa:
            result.update(passed=None, reference="QSA is timed without a score oracle")
            return nullcontext()
        from lsa.evaluation.performance.kernels.measure import decode as decoding
        from lsa.evaluation.performance.kernels.measure import score_check

        if self.lsa:
            oracle = decoding.lsa_decode_score_oracle if decode else score_check.lsa_score_oracle
            return oracle(self.indexer, result, tree)
        return (decoding.decode_score_oracle if decode else score_check.score_oracle)(self.module, result)

    def call(self, kwargs, batch):
        """The layer as the model calls it"""
        if self.linear or self.family == Family.QWEN3_5:
            accepted = inspect.signature(self.module.forward).parameters
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
                kwargs = {name: value for name, value in kwargs.items() if name in accepted}
            return self.module(**kwargs)
        from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context

        context = get_attn_tp_context()
        with context.maybe_input_scattered(batch):
            context.set_attn_inputs(
                AttentionInputs(kwargs["hidden_states"], batch, self.module.prepare_qkv_latent)
            )
            return self.module(**kwargs)

    def memory(self):
        """Return the bytes the layer's cache, index and parameters take."""
        if self.linear:
            return linear_memory(self.module, self.req_pool)
        if self.family == Family.QWEN3_5:
            parameters = [p for part in self.module.parts() for p in part.parameters()]
            return memory(self.module, self.pool, parameters)
        return memory(self.module, self.pool, self.module.parameters())

    def backends(self):
        """The kernels behind prefill and decode, by name"""
        if self.linear:
            kernels = getattr(self.backend, "kernel_dispatcher", None)
            if kernels is None:
                name = f"{self.name}/{type(self.backend).__name__}"
                return {"prefill": name, "decode": name}
            return {
                "prefill": f"{self.name}/{type(kernels.extend_kernel).__name__}",
                "decode": f"{self.name}/{type(kernels.decode_kernel).__name__}",
            }
        # FA3 on Hopper, and on Blackwell the backend's own class says which kernels ran
        served = (
            type(self.backend).__name__
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
            else "fa3"
        )
        if self.family == Family.QWEN3_5:
            name = (
                f"{served}-lsa-list"
                if self.lsa
                else f"{served}-pisa-list"
                if self.pisa
                else "qsa"
                if self.qsa
                else served
            )
            return {"prefill": name, "decode": name}
        # GLM read densely is no DSA model, so it runs on the plain dense backends
        if self.kind == "full":
            return {"prefill": served, "decode": served}
        return {"prefill": self.backend.dsa_prefill_impl, "decode": self.backend.dsa_decode_impl}
