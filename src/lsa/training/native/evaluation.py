"""The held-out evaluation - a fixed set scored per position bucket and per source, with what the attention read.

`HeldOutEvaluator` is torchtitan's validator for a native run: it scores each sequence's loss in position buckets,
sums the attention's own measurements over layers and ranks, and runs the retrieval probe. A trainer runs it every
so many steps. `lsa native evaluate` runs it once on a checkpoint, with no optimiser and no training data stream:
`build_model` makes the sharded model a training run would, `load_model_weights` fills it from a checkpoint step folder
and `evaluator_config` is the same evaluation a trainer is configured with.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, cast

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch import nn
from torch.distributed._composable.fsdp import FSDPModule
from torchtitan.components.checkpointer import ModelWrapper
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.validate import BaseValidator, ValidationContext, Validator
from torchtitan.config import TORCH_DTYPE_MAP, ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.model import BaseModel
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from lsa import assets
from lsa.data.tokenizer import load_tokenizer
from lsa.io import write_json, write_jsonl
from lsa.training.constants import Corpus, CorpusSplit, Loss
from lsa.training.data.corpus import Sample, TokenizedCorpus, assemble_batch, one_per_document
from lsa.training.native.config import NativeConfig
from lsa.training.native.constants import Evaluation, Retrieval, Runtime
from lsa.training.native.dsa import DsaAttention
from lsa.training.native.layers import MultiresInnerAttention
from lsa.training.native.losses import (
    RetrievalScore,
    SequenceScore,
    score_answers,
    score_batch,
    summarise,
    summarise_retrieval,
)
from lsa.training.native.precision import allow_exact_fp32_matmul, autocast_context
from lsa.training.native.probes import ProbeBuilder, fits


class HeldOutEvaluator(BaseValidator):
    """A fixed held-out set, scored per sequence and per position bucket, with what the attention read.

    torchtitan calls the pass `validate` and its base class is a `Validator`, and those names are its own.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        """Settings of the held-out pass: its corpus, sequences, position-bucket edges and retrieval probe."""

        corpus: str = ""
        autocast_bf16: bool = False
        path: str | None = None
        sources: list[str] | None = None
        held_out_fraction: float = Corpus.HELD_OUT_FRACTION
        sequences: int = Evaluation.SEQUENCES
        seed: int = 0
        edges: list[int] = field(default_factory=lambda: list(Evaluation.EDGES))
        probes: int = Evaluation.PROBES
        retrieval_distances: list[int] = field(default_factory=lambda: list(Retrieval.DISTANCES))
        retrieval_sequences: int = Retrieval.SEQUENCES
        retrieval_distractors: int = Retrieval.DISTRACTORS
        # The directory the retrieval probe's tokenizer is read from
        tokenizer_path: str = ""
        # Where each pass's per-sequence scores are written
        records: str = ""

    def __init__(
        self,
        config: Config,
        *,
        parallelism: ParallelismConfig,
        dp_world_size: int,
        dp_rank: int,
        parallel_dims: ParallelDims,
        validation_context: ValidationContext,
        metrics_processor: MetricsProcessor | None,
        seq_len: int,
        num_tokens_per_batch: int,
        **kwargs: Any,
    ) -> None:
        """Choose the set once and take this rank's share of it.

        Args:
            config: What to read and how to score it.
            parallelism: The run's parallelism, which the model's input preparation reads.
            dp_world_size: Size of the data-parallel group.
            dp_rank: This rank's index within it.
            parallel_dims: The run's mesh.
            validation_context: What the forward runs under.
            metrics_processor: Where the scalars are logged, or None to only print them.
            seq_len: Tokens per sequence.
            num_tokens_per_batch: Tokens this rank reads per forward.
            **kwargs: The rest of what the trainer hands every validator, unused here.

        Raises:
            ValueError: The set does not divide into whole batches on every rank.
        """
        del kwargs
        super().__init__(config=config)
        self.autocast_bf16 = config.autocast_bf16
        self.parallelism = parallelism
        self.parallel_dims = parallel_dims
        self.forward_context = validation_context
        self.metrics_processor = metrics_processor
        self.seq_len = seq_len
        self.probes = config.probes
        # A bucket starting at or past the end would hold nothing
        self.edges = tuple(edge for edge in config.edges if edge < seq_len)
        self.records = Path(config.records) if config.records else None
        self.dp_rank = dp_rank

        sequences_per_batch = num_tokens_per_batch // seq_len
        if config.sequences % (dp_world_size * sequences_per_batch):
            raise ValueError(
                f"{config.sequences} evaluation sequences do not split into whole batches of "
                f"{sequences_per_batch} over {dp_world_size} ranks"
            )
        self.corpus = TokenizedCorpus(
            assets.resolve(config.corpus, config.path),
            sources=config.sources,
            held_out_fraction=config.held_out_fraction,
            split=CorpusSplit.HELD_OUT,
        )
        chosen = one_per_document(self.corpus, seq_len, config.sequences, config.seed)
        mine = chosen[dp_rank::dp_world_size]
        self.batches = [mine[i : i + sequences_per_batch] for i in range(0, len(mine), sequences_per_batch)]
        self._prepare_retrieval(config, dp_world_size, dp_rank, sequences_per_batch)

    def _prepare_retrieval(
        self, config: Config, dp_world_size: int, dp_rank: int, sequences_per_batch: int
    ) -> None:
        """Choose the retrieval probe's distances and sequences, and take this rank's share.

        The probe set is the first sequences of the held-out set's own order, rounded up to whole batches on
        every rank, so the set a rank reads is the same whatever the number of ranks.

        Args:
            config: What to read and how to score it.
            dp_world_size: Size of the data-parallel group.
            dp_rank: This rank's index within it.
            sequences_per_batch: Sequences a rank reads at once.
        """
        self.retrieval_batches: list[list[tuple[int, Sample]]] = []
        self.retrieval_distances: list[int] = []
        if not config.retrieval_distances:
            return
        encode = self._encoder(config.tokenizer_path)
        self.retrieval_distances = [d for d in config.retrieval_distances if fits(d, self.seq_len, encode)]
        if not self.retrieval_distances:
            return
        self.probe_builder = ProbeBuilder(encode, config.seed, config.retrieval_distractors, self.seq_len)
        per_round = dp_world_size * sequences_per_batch
        count = -(-config.retrieval_sequences // per_round) * per_round
        if count != config.retrieval_sequences:
            logger.info(
                f"retrieval probe: {config.retrieval_sequences} sequences rounded up to {count} - "
                f"whole batches of {sequences_per_batch} over {dp_world_size} ranks"
            )
        indexed = list(enumerate(one_per_document(self.corpus, self.seq_len, count, config.seed)))
        mine = indexed[dp_rank::dp_world_size]
        self.retrieval_batches = [
            mine[i : i + sequences_per_batch] for i in range(0, len(mine), sequences_per_batch)
        ]

    @staticmethod
    def _encoder(tokenizer_path: str) -> Callable[[str], list[int]]:
        """Return the tokenizer's encoding of a string, without special tokens.

        Args:
            tokenizer_path: The directory the tokenizer is read from.

        Returns:
            The encoding function.
        """
        tokenizer = load_tokenizer(tokenizer_path)
        return lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        """Score the model without advancing training's random generators.

        Args:
            model_parts: The model, in one part.
            step: The step just taken.
        """
        devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
        with (
            torch.random.fork_rng(devices=devices),
            torch.autocast(
                "cuda" if devices else "cpu",
                dtype=torch.bfloat16,
                enabled=getattr(self, "autocast_bf16", False),
            ),
        ):
            self._validate(model_parts, step)

    def _validate(self, model_parts: list[nn.Module], step: int) -> None:
        """Score the set, log the scalars and write the per-sequence records.

        Args:
            model_parts: The model, in one part.
            step: The step just taken.
        """
        model = model_parts[0]
        attentions = [
            module for module in model.modules() if isinstance(module, (MultiresInnerAttention, DsaAttention))
        ]
        for attention in attentions:
            attention.measure, attention.probes, attention.measurements = True, self.probes, {}
        model.eval()
        # The measurements read values the compiler cannot trace, so a pass that takes them runs eagerly
        with torch.compiler.set_stance("force_eager") if attentions else nullcontext():
            scores = [score for batch in self.batches for score in self._score(model, batch)]
        model.train()
        measurements = self._reduce([attention.measurements for attention in attentions])
        for attention in attentions:
            attention.measure = False

        every_score = self._gather(scores)
        metrics = summarise(every_score, self.edges, self.seq_len)
        loss = metrics.pop("loss")
        metrics.update(measurements)
        retrieval_scores = self._retrieve(model)
        metrics.update(summarise_retrieval(retrieval_scores))
        if self.metrics_processor is not None:
            self.metrics_processor.log_validation(
                loss=loss,
                step=step,
                extra_metrics={f"validation_metrics/{name}": value for name, value in metrics.items()},
            )
        if self.dp_rank == 0:
            logger.info(
                f"evaluation step {step}: "
                + ", ".join(f"{name} {value:.4f}" for name, value in {"loss": loss, **metrics}.items())
            )
        if self.dp_rank == 0 and self.records is not None:
            self._write(self.records, every_score, step)
            self._write_retrieval(self.records, retrieval_scores, step)
            write_json(
                self.records / Evaluation.SUMMARY.format(step=step), {"step": step, "loss": loss, **metrics}
            )

    def _retrieve(self, model: nn.Module) -> list[RetrievalScore]:
        """Run the retrieval probe at every distance and gather every rank's scores.

        Args:
            model: The model, which is put in eval mode for the pass and back.

        Returns:
            One score per probe sequence and distance, over every rank.
        """
        if not self.retrieval_distances:
            return []
        model.eval()
        scores = [
            score
            for distance in self.retrieval_distances
            for batch in self.retrieval_batches
            for score in self._score_probes(model, batch, distance)
        ]
        model.train()
        if not dist.is_initialized():
            return scores
        gathered: list[list[RetrievalScore] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, scores)
        return [score for part in gathered for score in part or []]

    @torch.no_grad()
    def _score_probes(
        self, model: nn.Module, batch: list[tuple[int, Sample]], distance: int
    ) -> list[RetrievalScore]:
        """Score one batch of probe sequences at one distance, with the needle and without.

        Args:
            model: The model.
            batch: Probe sequences as their index in the probe set and the held-out sample they splice into.
            distance: Tokens between the needle's end and the query's start.

        Returns:
            One score per sequence, in batch order.
        """
        probes = [
            self.probe_builder.build(self.corpus.read(sample, self.seq_len)[: self.seq_len], index, distance)
            for index, sample in batch
        ]
        answers = torch.stack([probe.answer for probe in probes])
        # Flat rows [sequences * answer_tokens] of the batch whose outputs predict the answers
        rows = torch.cat([probe.answer_inputs + i * self.seq_len for i, probe in enumerate(probes)])
        results = []
        for spliced in ([probe.tokens for probe in probes], [probe.control for probe in probes]):
            tokens = torch.cat(spliced)
            on_device = {
                "input": tokens,
                "labels": torch.cat([torch.cat([sequence[1:], sequence[-1:]]) for sequence in spliced]),
                "positions": torch.arange(tokens.numel()) % self.seq_len,
                "padding_mask": torch.zeros(tokens.numel(), dtype=torch.bool),
            }
            on_device = {name: value.to(utils.device_type) for name, value in on_device.items()}
            inputs, device_labels, extra_kwargs = cast(BaseModel, model).preprocess_inputs(
                on_device, parallel_dims=self.parallel_dims, parallelism=self.parallelism
            )
            with self.forward_context():
                output = model(inputs, **extra_kwargs)
                losses, predictions = self._token_losses(
                    model, output, cast(torch.Tensor, device_labels), rows.to(utils.device_type)
                )
            results.append(
                score_answers(
                    losses.view(answers.shape).cpu(), predictions.view(answers.shape).cpu(), answers
                )
            )
        (probe_sums, probe_exact), (control_sums, control_exact) = results
        return [
            RetrievalScore(
                source=self.corpus.sources[sample.source],
                document=self.corpus.documents[sample.source][sample.row],
                row=sample.row,
                slice_index=sample.slice_index,
                distance=distance,
                answer_tokens=answers.shape[1],
                probe_loss_sum=float(probe_sums[i]),
                control_loss_sum=float(control_sums[i]),
                probe_exact=bool(probe_exact[i]),
                control_exact=bool(control_exact[i]),
            )
            for i, (_, sample) in enumerate(batch)
        ]

    def _reduce(self, measurements: list[dict[str, float]]) -> dict[str, float]:
        """Average what the attention layers recorded over layers, sequences and ranks.

        Args:
            measurements: One running sum a multi-resolution layer, with its pass count under `passes`.

        Returns:
            The mean of each measurement, or nothing when no layer measured.
        """
        names = sorted({name for measured in measurements for name in measured})
        if not names:
            return {}
        totals = torch.tensor(
            [sum(measured.get(name, 0.0) for measured in measurements) for name in names],
            dtype=torch.float64,
            device=utils.device_type,
        )
        if dist.is_initialized():
            dist.all_reduce(totals)
        sums = dict(zip(names, totals.tolist(), strict=True))
        passes = sums.pop("passes")
        return {f"attention/{name}": value / passes for name, value in sums.items()}

    def _score(self, model: nn.Module, samples: list[Sample]) -> list[SequenceScore]:
        """Run one batch forward and score each of its sequences.

        Args:
            model: The model.
            samples: The batch's sequences.

        Returns:
            One score per sequence, in batch order.
        """
        batch = assemble_batch(self.corpus, samples, self.seq_len)
        batch.pop("num_valid_tokens")
        labels = batch["labels"]
        on_device = {name: value.to(utils.device_type) for name, value in batch.items()}
        inputs, device_labels, extra_kwargs = cast(BaseModel, model).preprocess_inputs(
            on_device, parallel_dims=self.parallel_dims, parallelism=self.parallelism
        )
        with self.forward_context():
            output = model(inputs, **extra_kwargs)
            token_losses, _ = self._token_losses(model, output, cast(torch.Tensor, device_labels))

        loss_sums, token_counts = score_batch(token_losses, labels, self.seq_len, self.edges)
        return [
            SequenceScore(
                source=self.corpus.sources[sample.source],
                document=self.corpus.documents[sample.source][sample.row],
                row=sample.row,
                slice_index=sample.slice_index,
                loss_sums=loss_sums[i].tolist(),
                token_counts=token_counts[i].tolist(),
            )
            for i, sample in enumerate(samples)
        ]

    def _token_losses(
        self, model: nn.Module, output: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn the model's output into one loss and one prediction per token, a chunk of the sequence at a time.

        The trainer's chunked loss sets the model to return hidden states and runs the output projection
        itself, so the output is hidden states whenever that loss is in use and logits otherwise.

        Args:
            model: The model, whose `lm_head` is the output projection.
            output: Hidden states [n, d] or logits [n, V], with any leading batch dim.
            labels: Labels [n] int64.
            rows: Flat positions [r] int64 to score, or None for every token. The output projection then
                runs on those rows alone.

        Returns:
            A tuple of:
                Per-token loss [n] or [r] fp32, zero where the label is ignored.
                The argmax token [n] or [r] int64.
        """
        returns_hidden = bool(getattr(model, "_skip_lm_head", False))
        lm_head = cast(nn.Module, model.lm_head)
        fsdp = returns_hidden and isinstance(lm_head, FSDPModule)
        pieces: list[torch.Tensor] = []
        predictions: list[torch.Tensor] = []
        with spmd.local():
            flat_output = output.reshape(-1, output.shape[-1])
            flat_labels = labels.reshape(-1)
            if rows is not None:
                flat_output, flat_labels = flat_output[rows], flat_labels[rows]
            # Gather the projection once for every chunk, as the chunked loss does
            if fsdp:
                cast(FSDPModule, lm_head).set_reshard_after_forward(False)
                with spmd.no_typecheck():
                    cast(FSDPModule, lm_head).unshard()
            for chunk, chunk_labels in zip(
                flat_output.chunk(Evaluation.CHUNKS), flat_labels.chunk(Evaluation.CHUNKS), strict=True
            ):
                logits = lm_head(chunk) if returns_hidden else chunk
                predictions.append(logits.argmax(dim=-1))
                pieces.append(
                    F.cross_entropy(
                        logits.float(), chunk_labels, reduction="none", ignore_index=Loss.IGNORE_INDEX
                    )
                )
            if fsdp:
                cast(FSDPModule, lm_head).set_reshard_after_forward(True)
                cast(FSDPModule, lm_head).reshard()
        return torch.cat(pieces), torch.cat(predictions)

    def _gather(self, scores: list[SequenceScore]) -> list[SequenceScore]:
        """Collect every rank's scores, in rank order.

        Args:
            scores: This rank's.

        Returns:
            Every rank's, which is every sequence in the set.
        """
        if not dist.is_initialized():
            return scores
        gathered: list[list[SequenceScore] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, scores)
        return [score for part in gathered for score in part or []]

    def _write_retrieval(self, folder: Path, scores: list[RetrievalScore], step: int) -> None:
        """Write one pass's per-sequence probe scores, one JSON line each.

        Args:
            folder: Where the records go.
            scores: Every probe sequence at every distance.
            step: The step the pass followed.
        """
        rows = ({"step": step, **asdict(score)} for score in scores)
        write_jsonl(folder / Evaluation.RETRIEVAL_RECORDS.format(step=step), rows)

    def _write(self, folder: Path, scores: list[SequenceScore], step: int) -> None:
        """Write one pass's per-sequence scores, one JSON line each.

        Args:
            folder: Where the records go.
            scores: Every sequence in the set.
            step: The step the pass followed.
        """
        rows = ({"step": step, "edges": list(self.edges), **asdict(score)} for score in scores)
        write_jsonl(folder / Evaluation.RECORDS.format(step=step), rows)


def evaluator_config(experiment: NativeConfig) -> HeldOutEvaluator.Config:
    """Translate a run's evaluation block into the config the evaluator builds from.

    Args:
        experiment: The experiment.

    Returns:
        The evaluator config, which writes its records under the run's evaluation folder.
    """
    evaluation = experiment.run.evaluation
    return HeldOutEvaluator.Config(
        enable=evaluation.enabled,
        autocast_bf16=experiment.run.runtime.autocast_bf16,
        freq=evaluation.every or Runtime.NEVER,
        corpus=experiment.data.name,
        path=experiment.data.path,
        sources=experiment.data.sources,
        held_out_fraction=experiment.data.held_out_fraction,
        sequences=evaluation.sequences,
        seed=evaluation.seed,
        edges=evaluation.edges,
        probes=evaluation.probes,
        retrieval_distances=evaluation.retrieval_distances,
        retrieval_sequences=evaluation.retrieval_sequences,
        retrieval_distractors=evaluation.retrieval_distractors,
        tokenizer_path=experiment.tokenizer_path,
        records=str(experiment.out / Evaluation.FOLDER),
    )


class EvaluationModel(NamedTuple):
    """A model built for evaluation, and the mesh and context its forward runs under."""

    model: nn.Module
    parallel_dims: ParallelDims
    # Data-parallel ranks, and this process's place among them
    dp_world_size: int
    dp_rank: int
    forward_context: Callable[[], AbstractContextManager[None]]


def build_model(config: Trainer.Config) -> EvaluationModel:
    """Build the sharded model a trainer would, from its config, without the optimiser or the data stream.

    Initialises the process group, applies the run's parallelism and compile to the model and initialises its weights,
    which `load_model_weights` then overwrites. The model returns hidden states and the evaluator projects them in
    chunks, as the trainer's chunked loss does.

    Args:
        config: The trainer config, from `trainer_config`.

    Returns:
        The model in eval mode, with the mesh and the forward context.
    """
    model_spec = config.model_spec
    device = utils.get_local_device()
    utils.device_module.set_device(device)
    allow_exact_fp32_matmul()
    world_size = dist_utils.init_distributed(
        config.comm, enable_cpu_backend=config.training.enable_cpu_offload, base_folder=config.dump_folder
    )
    parallel_dims = ParallelDims.from_config(config.parallelism, world_size)
    dist_utils.set_determinism(parallel_dims, device, config.debug, distinct_seed_mesh_dims=["pp"])
    model_spec.model.update_from_config(config=config)
    with torch.device("meta"), utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
        model = model_spec.model.build()
    model = model_spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=config.training,
        parallelism=config.parallelism,
        compile_config=config.compile,
        ac_config=config.activation_checkpoint,
        dump_folder=config.dump_folder,
    )
    model.to_empty(device=utils.device_type)
    with torch.no_grad():
        cast(BaseModel, model).init_weights(buffer_device=None)
    loss_fn = config.loss.build(compile_config=config.compile)
    if isinstance(loss_fn, ChunkedLossWrapper):
        loss_fn.set_lm_head(model.lm_head)
        model._skip_lm_head = True
    model.eval()
    if parallel_dims.dp_enabled:
        dp_mesh = parallel_dims.get_mesh("batch")
        dp_world_size, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_world_size, dp_rank = 1, 0
    forward_context = dist_utils.get_spmd_context(
        parallel_dims=parallel_dims, spmd_typechecking=config.debug.spmd_typechecking
    )
    forward_context = partial(
        autocast_context,
        context=forward_context,
        device_type=device.type,
        enabled=getattr(config, "autocast_bf16", False),
    )
    return EvaluationModel(model, parallel_dims, dp_world_size, dp_rank, forward_context)


def load_model_weights(model_parts: list[nn.Module], checkpoint: str | Path) -> None:
    """Load a torchtitan checkpoint step folder's model weights into a model, sharded or not.

    The optimiser, the schedule and the data stream in the folder are left unread.

    Args:
        model_parts: The model, in one part.
        checkpoint: A step folder, `.../checkpoint/step-N`.
    """
    wrapper = ModelWrapper(model_parts)
    state_dict = wrapper.state_dict()
    dcp.load(state_dict, checkpoint_id=str(checkpoint))
    wrapper.load_state_dict(state_dict)


def checkpoint_step(checkpoint: str | Path) -> int:
    """Return the step a checkpoint folder holds, from its name.

    Args:
        checkpoint: A step folder, `.../checkpoint/step-N`.

    Returns:
        The step.

    Raises:
        ValueError: The folder is not named `step-N`.
    """
    name = Path(checkpoint).name
    prefix, _, number = name.partition("-")
    if prefix != "step" or not number.isdigit():
        raise ValueError(f"a checkpoint is a step folder named step-N - got {name!r}")
    return int(number)
