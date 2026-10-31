"""The reference training code's learning-rate schedule, transcribed, which a run's own is held to."""

import math


def reference_factor(step, warmup_steps, total_steps, decay_steps, floor, shape):
    # The reference training code's wsd_lr_factor, transcribed
    if step < warmup_steps:
        return max(1e-8, step / warmup_steps)
    decay_start = total_steps - decay_steps
    if step < decay_start:
        return 1.0
    progress = min(1.0, (step - decay_start) / max(1, decay_steps))
    if shape == "linear":
        remaining = 1.0 - progress
    elif shape == "sqrt":
        remaining = 1.0 - math.sqrt(progress)
    else:
        remaining = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (1.0 - floor) * remaining


def reference_decay_steps(decay_fraction, total_steps):
    # The reference training code's wsd_decay_steps
    return math.ceil(decay_fraction * total_steps)


class CheckpointRun:
    """Train a tiny model through native's stream, schedule and real DCP checkpoints."""

    def __init__(
        self,
        folder,
        card,
        *,
        seq_len=8,
        world_size=1,
        rank=0,
        tokens_per_batch=16,
        steps=10,
        warmup_steps=2,
        decay_ratio=0.5,
        initial_load_path=None,
        interval=2,
        keep_latest_k=0,
        shard=False,
    ):
        from unittest.mock import patch

        import torch
        from torchtitan.components.optimizer import default_adamw

        from lsa.training.native.evaluation import HeldOutEvaluator
        from lsa.training.native.trainer import (
            CorpusDataLoader,
            DecayStartCheckpointer,
            NativeTrainer,
            WSDSchedulers,
        )

        class RandomValidator(HeldOutEvaluator):
            def _validate(self, model_parts, step):
                torch.rand(17, device=next(model_parts[0].parameters()).device)

        self.validator = object.__new__(RandomValidator)
        devices = [torch.cuda.current_device()] if shard else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(7)
            self.model = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        if shard:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard

            self.model.cuda()
            fully_shard(self.model, mesh=init_device_mesh("cuda", (world_size,)))
        optimizer_config = default_adamw(lr=0.01, weight_decay=0.1)
        optimizer_config.implementation = "for-loop"
        self.optimizers = optimizer_config.build(model_parts=[self.model])
        self.schedulers = WSDSchedulers.Config(
            warmup_steps=warmup_steps,
            decay_ratio=decay_ratio,
            decay_shape="sqrt",
            min_lr_factor=0.1,
        ).build(optimizers=self.optimizers, training_steps=steps)
        with patch("lsa.training.native.trainer.assets.corpus", return_value=card):
            self.loader = CorpusDataLoader.Config(
                corpora=[card.name],
                weights=[1.0],
                seq_len=seq_len,
                seed=42,
                held_out_fraction=0.0,
            ).build(
                dp_world_size=world_size,
                dp_rank=rank,
                max_context_length=seq_len,
                num_tokens_per_batch=tokens_per_batch,
            )
        self.stream = self.loader.stream
        self.training_state = object.__new__(NativeTrainer)
        self.training_state.step = 0
        self.training_state.ntokens_seen = 0
        self.training_state.sdc_replayer = None
        self.training_state.device = next(self.model.parameters()).device
        self.rates = []
        self.checkpointer = DecayStartCheckpointer.Config(
            enable=True,
            interval=interval,
            decay_start_step=steps - math.ceil(steps * decay_ratio),
            async_mode="disabled",
            keep_latest_k=keep_latest_k,
            initial_load_path=initial_load_path,
            initial_load_model_only=initial_load_path is not None,
            last_save_model_only=False,
        ).build(
            dataloader=self.loader,
            model_parts=[self.model],
            optimizers=self.optimizers,
            lr_schedulers=self.schedulers,
            states={"train_state": self.training_state},
            sd_adapter=None,
            base_folder=str(folder),
        )

    @property
    def step(self):
        return self.training_state.step

    @step.setter
    def step(self, value):
        self.training_state.step = value

    def advance(self, count, *, accumulation=1, save=False, stochastic=False, validate=False):
        import torch

        iterator = iter(self.stream)
        for _ in range(count):
            self.rates.append(self.optimizers.optimizers[0].param_groups[0]["lr"])
            self.optimizers.zero_grad()
            for _ in range(accumulation):
                batch = next(iterator)
                device = next(self.model.parameters()).device
                inputs = batch["input"].to(device).float().unsqueeze(-1) / 1000
                labels = batch["labels"].to(device).float().unsqueeze(-1) / 1000
                if stochastic:
                    sampled = torch.randperm(inputs.shape[0], device=device)[: max(1, inputs.shape[0] // 2)]
                    inputs, labels = inputs[sampled], labels[sampled]
                loss = torch.nn.functional.mse_loss(self.model(inputs), labels) / accumulation
                loss.backward()
            self.optimizers.step()
            self.schedulers.step()
            self.step += 1
            if save:
                self.checkpointer.save(self.step)
            if validate and self.step % 2 == 0:
                self.validator.validate([self.model], self.step)
