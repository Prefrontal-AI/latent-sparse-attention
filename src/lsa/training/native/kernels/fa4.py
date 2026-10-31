"""FA4's forward and backward, trimmed to the outputs torch's FA4 shim unpacks.

torch's FA4 shim unpacks two values from the forward and three from the backward, and flash-attn-4 4.0.0b36
returns five from each (`out, lse` and then its extras, `dq, dk, dv` and then `dqv, dsink`).
No FA4 release matches what torch expects, so the shim is pointed at this module, which drops the extras.
"""

import functools
from typing import Any

from torch.nn.attention import _fa4, register_flash_attention_impl


def _flash_attn_fwd(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Run FA4's forward and keep the output and the log-sum-exp.

    Args:
        *args: Positional arguments of FA4's forward.
        **kwargs: Keyword arguments of FA4's forward.

    Returns:
        The output and the log-sum-exp.
    """
    # Optional extra => imported when first called
    from flash_attn.cute import interface

    return interface._flash_attn_fwd(*args, **kwargs)[:2]


def _flash_attn_bwd(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Run FA4's backward and keep the query, key and value gradients.

    Args:
        *args: Positional arguments of FA4's backward.
        **kwargs: Keyword arguments of FA4's backward.

    Returns:
        The query, key and value gradients.
    """
    from flash_attn.cute import interface

    return interface._flash_attn_bwd(*args, **kwargs)[:3]


def register() -> None:
    """Make torch's "FA4" implementation run through this module, so activating it unpacks what it expects."""
    register_flash_attention_impl(
        "FA4", register_fn=functools.partial(_fa4.register_flash_attention_fa4, module_path=__name__)
    )
