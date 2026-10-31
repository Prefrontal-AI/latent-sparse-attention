"""The trainer - torchtitan's, configured from a native run, with a corpus loader of its own.

Native does not reimplement a trainer loop: torchtitan's carries the data-parallel mesh, the optimiser, the schedule,
the chunked loss and the checkpointer, and none of it has anything to do with attention. What it does not carry is
a corpus read by sequence, so that is here, beside `trainer_config`, the translation from a `NativeConfig` into
the config the trainer builds from. Its held-out pass is [the evaluation](evaluation.py).

Every torchtitan import in the package is in this module, [the evaluation](evaluation.py), [the layers](layers.py) and [the model](models.py).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from lsa import assets
from lsa.training.constants import Corpus, CorpusSplit, Loss
from lsa.training.data.streams import Layout, open_stream
from lsa.training.native.config import NativeConfig, RunConfig
from lsa.training.native.constants import Checkpoint, DecayShape, Qwen35, Runtime, Schedule
from lsa.training.native.evaluation import evaluator_config
from lsa.training.native.models import model_spec
from lsa.training.native.optimizer import optimizer_config
from lsa.training.native.precision import allow_exact_fp32_matmul
from lsa.training.native.schedule import decay_step_count, learning_rate_factor


class CorpusDataLoader(BaseDataLoader):
    """The corpora as a torchtitan dataloader, through [the streams](../data/streams.py).

    The state is every reader's position, so a resumed run sees the stream it would have seen.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        """Settings of the dataloader: the corpora and weights, readers, split and sequence length."""

        # The corpora, as the run named them - card names - and their weights, or a path for a tokenized one with no card
        corpora: list[str]
        weights: list[float]
        path: str | None = None
        packing: str | None = None
        readers: int = 1
        tokenizer_path: str = ""
        sources: list[str] | None = None
        held_out_fraction: float = Corpus.HELD_OUT_FRACTION
        split: str = CorpusSplit.TRAIN
        seq_len: int = Qwen35.SEQ_LEN
        seed: int = 0

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        max_context_length: int,
        num_tokens_per_batch: int,
        **kwargs: Any,
    ) -> None:
        """Open the corpus and take this rank's share of it.

        Args:
            config: What to read and how to split it.
            dp_world_size: Size of the data-parallel group.
            dp_rank: This rank's index within it.
            max_context_length: The model's context, which bounds the sequence length.
            num_tokens_per_batch: Tokens this rank yields per micro-batch.
            **kwargs: The rest of what the trainer hands every dataloader, unused here.
        """
        del kwargs
        super().__init__()
        self.dp_world_size = dp_world_size
        self.rank_key = f"dp_rank_{dp_rank}"
        single = len(config.corpora) == 1
        self.stream = open_stream(
            [
                (assets.corpus(name, config.path if single else None), weight)
                for name, weight in zip(config.corpora, config.weights, strict=True)
            ],
            Layout(
                rank=dp_rank,
                world_size=dp_world_size,
                readers=config.readers,
                seq_len=min(config.seq_len, max_context_length),
                tokens_per_batch=num_tokens_per_batch,
                seed=config.seed,
            ),
            tokenizer_path=config.tokenizer_path,
            packing=config.packing,
            split=config.split,
            sources=config.sources,
            held_out_fraction=config.held_out_fraction,
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield this rank's batches forever."""
        return iter(self.stream)

    def state_dict(self) -> dict[str, Any]:
        """Return the stream's position, which is what a resumed run restores."""
        return {"dp_world_size": self.dp_world_size, self.rank_key: self.stream.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the stream's position.

        Args:
            state_dict: What `state_dict` returned.

        Raises:
            ValueError: The saved world size differs or this rank has no saved position.
        """
        if state_dict.get("dp_world_size") != self.dp_world_size:
            raise ValueError(
                f"dataloader checkpoint has dp_world_size {state_dict.get('dp_world_size')} - "
                f"expected {self.dp_world_size}; use run.init_from to branch with a new world size"
            )
        if self.rank_key not in state_dict:
            raise ValueError(
                f"dataloader checkpoint needs state for {self.rank_key} - got {list(state_dict)}"
            )
        self.stream.load_state_dict(state_dict[self.rank_key])


class WSDSchedulers(LRSchedulersContainer):
    """torchtitan's schedulers container, stepping [the schedule](schedule.py) instead of its own."""

    @dataclass(kw_only=True, slots=True)
    class Config(LRSchedulersContainer.Config):
        """Settings of the schedule: its decay shape and kind, over the base's warmup and decay ratio."""

        decay_shape: str = DecayShape.SQRT
        schedule: str = Schedule.WSD

        def build(self, *, optimizers: OptimizersContainer, training_steps: int) -> "WSDSchedulers":
            """Return the container for a run of `training_steps` updates.

            Args:
                optimizers: The optimisers whose learning rates are scheduled.
                training_steps: Optimiser steps of the whole run, unless `total_steps` says otherwise.

            Returns:
                The container.
            """
            steps = training_steps if self.total_steps is None else self.total_steps
            warmup_steps = min(self.warmup_steps, steps)
            decay_steps = (
                decay_step_count(steps, self.decay_ratio)
                if self.decay_ratio is not None
                else steps - warmup_steps
            )

            def factor(step: int) -> float:
                """Return the learning-rate factor of one update."""
                return learning_rate_factor(
                    step,
                    steps=steps,
                    warmup_steps=warmup_steps,
                    decay_steps=decay_steps,
                    min_lr_factor=self.min_lr_factor,
                    decay_shape=self.decay_shape,
                    schedule=self.schedule,
                )

            return WSDSchedulers(optimizers, factor)


class DecayStartCheckpointer(CheckpointManager):
    """torchtitan's checkpointer, which also saves at the decay start and keeps that checkpoint for good."""

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        """Settings of the checkpointer, with the step it also saves at."""

        # The step to save at whatever the interval, or None for no such save
        decay_start_step: int | None = None

    def __init__(self, config: Config, **kwargs: Any) -> None:
        """Build the checkpointer and remember the step that is kept.

        Args:
            config: The checkpoint config.
            **kwargs: What the trainer hands every checkpointer.
        """
        super().__init__(config, **kwargs)
        self.decay_start_step = config.decay_start_step

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        """Say whether `curr_step` is a checkpoint step, the decay start being one."""
        if self.enable and not self.load_only and curr_step == self.decay_start_step:
            return True
        return super()._should_save(curr_step, last_step)

    def _is_purge_exempt(self, step: int) -> bool:
        """Say whether `step` is protected from the retention, the decay start always is."""
        return step == self.decay_start_step or super()._is_purge_exempt(step)


class NativeTrainer(Trainer):
    """Advance attention schedules before torchtitan performs each optimiser update."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        """Build the native trainer with torchtitan's training configuration."""

        autocast_bf16: bool = False

    def _forward_backward_body(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor | tuple[torch.Tensor, ...],
        global_valid_tokens: torch.Tensor,
        extra_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Autocast forward compute while retaining full-precision master parameters.

        Args:
            inputs: The preprocessed microbatch inputs.
            labels: The next-token targets.
            global_valid_tokens: The loss denominator for this optimizer update.
            extra_kwargs: The model's position and attention metadata.

        Returns:
            The microbatch loss divided by the global valid-token count.
        """
        if not getattr(self.config, "autocast_bf16", False):
            return super()._forward_backward_body(inputs, labels, global_valid_tokens, extra_kwargs)
        with self.train_context():
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                predictions = self.model_parts[0](inputs, **extra_kwargs)
                loss_kwargs = {}
                if "positions" in extra_kwargs:
                    loss_kwargs["positions"] = extra_kwargs["positions"]
                loss, _ = self.loss_fn(predictions, labels, global_valid_tokens, **loss_kwargs)
            del predictions
            with spmd.no_typecheck():
                loss.backward()
        return loss

    def train_step(self, data_iterator: Iterator[dict[str, Any]]) -> None:
        """Set the completed update count before all microbatches of this step.

        Args:
            data_iterator: The resumable stream of training batches.
        """
        for model in self.model_parts:
            for module in model.modules():
                set_step = getattr(module, "set_step", None)
                if callable(set_step):
                    set_step(self.step - 1)
        super().train_step(data_iterator)

    def state_dict(self) -> dict[str, Any]:
        """Save counters and this rank's random generators for exact stochastic resumption.

        Returns:
            Training counters and rank-specific CPU and CUDA random states.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        random_state = {"cpu": torch.get_rng_state()}
        if self.device.type == "cuda":
            random_state["cuda"] = torch.cuda.get_rng_state(self.device)
        return {**super().state_dict(), f"rng_rank_{rank}": random_state}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore counters and this rank's random generators before another update.

        Args:
            state_dict: Native training state from the checkpoint.

        Raises:
            ValueError: This rank's random state is missing from the checkpoint.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        rank_key = f"rng_rank_{rank}"
        if rank_key not in state_dict:
            raise ValueError(
                f"native checkpoint needs random state for {rank_key} - "
                "use run.init_from to load weights alone from an older checkpoint"
            )
        random_state = state_dict[rank_key]
        if "cpu" not in random_state or (self.device.type == "cuda" and "cuda" not in random_state):
            raise ValueError(f"native checkpoint needs CPU and device random states for {rank_key}")
        super().load_state_dict(state_dict)
        torch.set_rng_state(random_state["cpu"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(random_state["cuda"].cpu(), self.device)


def provide_process_group_timeout() -> None:
    """Give torch 2.13 the `torch.distributed.set_timeout` that the pinned torchtitan calls after a run's first step.

    The pinned torchtitan commit is newer than torch 2.13.0 and sets the process groups' timeout through a function
    torch publishes later, under the name `_set_pg_timeout` before that.
    Without it every run stops after step 1.
    """
    if not hasattr(torch.distributed, "set_timeout"):
        torch.distributed.set_timeout = torch.distributed.distributed_c10d._set_pg_timeout  # type: ignore[attr-defined]


def trainer_config(experiment: NativeConfig) -> NativeTrainer.Config:
    """Translate one run into the config torchtitan's trainer takes.

    Args:
        experiment: The experiment.

    Returns:
        A `Trainer.Config` ready to build.

    Raises:
        ValueError: A corpus was tokenized for a wider vocabulary than the model has.
    """
    provide_process_group_timeout()
    # The trainer's init_distributed runs after this, in every rank
    allow_exact_fp32_matmul()
    built = model_spec(experiment)
    vocab_size = decoder_vocab_size(built)
    for card, _ in experiment.data.cards:
        # Text is tokenized by the model's own tokenizer as it is read, so only ids already written can be too wide
        if isinstance(card, assets.SourceCard):
            continue
        tokenized_for = card.manifest().vocab_size
        if tokenized_for > vocab_size:
            raise ValueError(
                f"{card.name} is tokenized for a {tokenized_for}-token vocabulary and "
                f"{experiment.model.flavor} has {vocab_size} - tokenize a corpus for this model with `lsa data prepare <card>`"
            )
    run = experiment.run
    return NativeTrainer.Config(
        autocast_bf16=run.runtime.autocast_bf16,
        model_spec=built,
        hf_assets_path=experiment.tokenizer_path,
        dump_folder=str(experiment.out),
        tokenizer=HuggingFaceTokenizer.Config(),
        loss=ChunkedLossWrapper.Config(loss_fn=CrossEntropyLoss.Config(global_vocab_size=vocab_size)),
        metrics=MetricsProcessor.Config(log_freq=run.runtime.log_every, enable_wandb=run.runtime.wandb),
        dataloader=CorpusDataLoader.Config(
            corpora=[name for name, _ in experiment.data.corpora],
            weights=[weight for _, weight in experiment.data.corpora],
            path=experiment.data.path,
            packing=experiment.data.packing,
            readers=experiment.data.readers,
            tokenizer_path=experiment.tokenizer_path,
            sources=experiment.data.sources,
            held_out_fraction=experiment.data.held_out_fraction,
            split=experiment.data.split,
            seq_len=experiment.data.seq_len,
            seed=experiment.data.seed,
        ),
        validator=evaluator_config(experiment),
        optimizer=optimizer_config(run.optim),
        lr_scheduler=WSDSchedulers.Config(
            warmup_steps=run.optim.warmup_steps,
            decay_ratio=run.optim.decay_ratio,
            decay_shape=run.optim.decay_shape,
            schedule=run.optim.schedule,
            min_lr_factor=run.optim.min_lr_factor,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=run.optim.tokens_per_rank,
            num_tokens_per_train_step=run.optim.tokens_per_step or -1,
            max_context_length=experiment.data.seq_len,
            steps=run.optim.steps,
            max_norm=run.optim.grad_clip,
            dtype="float32",
            mixed_precision_param="float32" if run.runtime.autocast_bf16 else "bfloat16",
            mixed_precision_reduce="float32",
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=run.dp_replicate, data_parallel_shard_degree=run.dp_shard
        ),
        checkpoint=checkpoint_config(run),
        activation_checkpoint=_activation_checkpoint(run.runtime.activation_checkpoint),
        compile=_compile(run.runtime.compile),
        # torchtitan takes model initialisation off this and defaults it to None, which makes every run from
        # random init a different random model and every comparison between arms a comparison of two models
        debug=DebugConfig(seed=run.runtime.seed),
    )


def checkpoint_config(run: RunConfig) -> DecayStartCheckpointer.Config:
    """Translate where a run starts from and how it saves into torchtitan's checkpointer.

    Args:
        run: The run block.

    Returns:
        The checkpointer config, enabled when the run saves, saves its decay start or branches off a trunk.
    """
    return DecayStartCheckpointer.Config(
        enable=run.checkpoint_every > 0 or run.checkpoint.decay_start or run.init_from is not None,
        decay_start_step=run.optim.decay_start_step if run.checkpoint.decay_start else None,
        interval=run.checkpoint_every or Runtime.NEVER,
        initial_load_path=run.init_from,
        initial_load_model_only=True,
        async_mode=Checkpoint.ASYNC if run.checkpoint_async else Checkpoint.SYNC,
        last_save_model_only=False,
    )


def _compile(mode: str) -> CompileConfig:
    """Translate a run's compile mode into torchtitan's own config.

    Args:
        mode: One of `Runtime.COMPILE`.

    Returns:
        The compile config, disabled for `off` and naming its components otherwise.
    """
    if mode == Runtime.OFF:
        return CompileConfig(enable=False)
    if mode == Runtime.LOSS:
        return CompileConfig(enable=True, components=["loss"])
    return CompileConfig(enable=True, components=["model", "loss"])


def _activation_checkpoint(mode: str) -> FullAC.Config | SelectiveAC.Config | None:
    """Return the recomputation config a mode names.

    Args:
        mode: One of `Runtime.CHECKPOINTING`.

    Returns:
        The torchtitan activation-checkpointing config, or None for no recomputation.
    """
    if mode == Runtime.FULL:
        return FullAC.Config()
    if mode == Runtime.SELECTIVE:
        return SelectiveAC.Config()
    return None


def check_label_contract() -> None:
    """Fail loudly if torchtitan's ignore label ever stops being the one the loader writes.

    Raises:
        ValueError: The two disagree, which would silently train on padding.
    """
    from torchtitan.components.loss import IGNORE_INDEX as titan_ignore

    if titan_ignore != Loss.IGNORE_INDEX:
        raise ValueError(f"the loader writes {Loss.IGNORE_INDEX} and the loss skips {titan_ignore}")
