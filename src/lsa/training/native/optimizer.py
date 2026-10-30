"""Group native AdamW parameters with the upstream language-model decay rule."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn
from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig, default_adamw

from lsa.training.native.config import OptimConfig
from lsa.training.native.constants import WeightDecayScope


def parameter_name(name: str) -> str:
    """Remove checkpoint and compilation wrappers from a parameter name.

    Args:
        name: The parameter's module-qualified name.

    Returns:
        The corresponding name before wrappers were installed.
    """
    return canonical_fqn(name, prefixes=("_checkpoint_wrapped_module", "_orig_mod"))


def preserve_parameter_flags(
    model: nn.Module, *, parallelize: Callable[..., nn.Module], **kwargs: Any
) -> nn.Module:
    """Keep constructor decay exclusions across parameter replacement by parallelism.

    Args:
        model: The model before parallelism replaces its parameters.
        parallelize: Torchtitan's model parallelization function.
        **kwargs: The trainer's parallelization arguments.

    Returns:
        The parallelized model with its original decay exclusions retained by name.
    """
    names = frozenset(
        parameter_name(name)
        for name, parameter in model.named_parameters()
        if getattr(parameter, "_no_weight_decay", False)
    )
    parallelized = parallelize(model, **kwargs)
    parallelized.__dict__["no_weight_decay_names"] = names
    return parallelized


class NativeOptimizers(OptimizersContainer):
    """Keep torchtitan's optimizer and checkpoint handling with explicit decay groups."""

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        """Settings of the optimizers: the base's, with the scope of the weight decay."""

        weight_decay_scope: str = WeightDecayScope.ALL_PARAMETERS

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        """Build torchtitan's AdamW container using the selected decay scope.

        Args:
            config: AdamW settings and decay scope.
            model_parts: The parallelized model parts.

        Raises:
            ValueError: The scope is unknown or the native optimizer is not one AdamW setup.
        """
        if config.weight_decay_scope not in WeightDecayScope.ALL:
            raise ValueError(
                f"weight_decay_scope must be one of {WeightDecayScope.ALL} - got {config.weight_decay_scope!r}"
            )
        if len(config.param_groups) != 1 or config.param_groups[0].optimizer_name != "AdamW":
            names = [group.optimizer_name for group in config.param_groups]
            raise ValueError(
                f"native optimizer needs one AdamW setup before splitting decay groups - got {names}"
            )
        self.weight_decay_scope = config.weight_decay_scope
        super().__init__(config, model_parts=model_parts)

    def _build_param_groups(
        self,
        model: nn.Module,
        param_group_configs: list[ParamGroupConfig],
        impl_kwargs: dict[str, Any],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
        """Split trainable parameters by dimensionality and preserved decay exclusions.

        Args:
            model: The model part whose parameters the optimizer owns.
            param_group_configs: The native AdamW setup.
            impl_kwargs: Torchtitan's fused or foreach implementation settings.

        Returns:
            AdamW groups and their logging labels, with decayed parameters first.
        """
        if self.weight_decay_scope == WeightDecayScope.ALL_PARAMETERS:
            return super()._build_param_groups(model, param_group_configs, impl_kwargs)
        excluded = getattr(model, "no_weight_decay_names", None)
        if excluded is None:
            excluded = {
                parameter_name(name)
                for name, parameter in model.named_parameters()
                if getattr(parameter, "_no_weight_decay", False)
            }
        parameters: dict[bool, list[nn.Parameter]] = {True: [], False: []}
        names: dict[bool, list[str]] = {True: [], False: []}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            decays = parameter.ndim >= 2 and parameter_name(name) not in excluded
            parameters[decays].append(parameter)
            names[decays].append(canonical_fqn(name))
        groups = []
        labels = []
        for decays in (True, False):
            if not parameters[decays]:
                continue
            settings = dict(param_group_configs[0].optimizer_kwargs)
            if not decays:
                settings["weight_decay"] = 0.0
            groups.append(
                {"params": parameters[decays], "param_names": names[decays], **impl_kwargs, **settings}
            )
            labels.append("decay" if decays else "no_decay")
        return {"AdamW": groups}, {"AdamW": labels}


def optimizer_config(optim: OptimConfig) -> NativeOptimizers.Config:
    """Build the native AdamW setup with torchtitan's language-model defaults.

    Args:
        optim: The run's optimizer settings.

    Returns:
        A torchtitan optimizer container config with explicit decay scope.
    """
    default = default_adamw(lr=optim.lr, weight_decay=optim.weight_decay)
    return NativeOptimizers.Config(
        param_groups=default.param_groups,
        implementation=default.implementation,
        weight_decay_scope=optim.weight_decay_scope,
    )
