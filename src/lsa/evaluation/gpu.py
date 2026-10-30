"""The GPU a run ran on - read from the device, kept in the record as `run.gpu`, and named in the run's name.

A record states the card model it ran on under `run.gpu`, as the device reports it (`NVIDIA H200`,
`NVIDIA B300 SXM6 AC`), and a run's name ends in the card's short token (`glm53_flash_lsa_flat_tp8_h200`).
`TOKENS` is the one place that maps the first to the second, and a device it does not list is refused, so a
new card is a line here and a row in the glossary rather than a name nobody chose.
"""

import subprocess
from typing import Any

# The device name as nvidia-smi and torch report it, to the token a run's name ends in
TOKENS: dict[str, str] = {
    "NVIDIA H200": "h200",
    "NVIDIA B300 SXM6 AC": "b300",
}


def token(device: str) -> str:
    """Return the short token a device's name is written as in a run's name.

    Args:
        device: The device name, as `nvidia-smi` reports it.

    Returns:
        The token, lowercase.

    Raises:
        ValueError: The device is not one `TOKENS` lists.
    """
    if device not in TOKENS:
        raise ValueError(
            f"no name token for the GPU {device!r} - add it to gpu.TOKENS, known: {sorted(TOKENS)}"
        )
    return TOKENS[device]


def resolve(value: str) -> str:
    """Return the device name a token or a device name stands for.

    Args:
        value: A token such as `h200`, or a device name, or empty.

    Returns:
        The device name, empty for an empty value.

    Raises:
        ValueError: The value is neither a known token nor a known device name.
    """
    if not value:
        return ""
    for device, short in TOKENS.items():
        if value in (device, short):
            return device
    raise ValueError(f"{value!r} is neither a GPU token nor a device name - known: {sorted(TOKENS.items())}")


def detect() -> str:
    """Read the card model from the device through nvidia-smi.

    Returns:
        The device name every visible GPU shares.

    Raises:
        RuntimeError: nvidia-smi is missing, fails, lists no GPU, or lists more than one model.
    """
    try:
        finished = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot read the GPU model - nvidia-smi failed: {error}") from error
    names = sorted({line.strip() for line in finished.stdout.splitlines() if line.strip()})
    if len(names) != 1:
        raise RuntimeError(f"expected one GPU model on this node, nvidia-smi lists {names or 'none'}")
    return names[0]


def settle(requested: str) -> str:
    """Return the device a launch runs on, read from the node and checked against what was asked for.

    Args:
        requested: A token or device name the launch was submitted for, or empty.

    Returns:
        The device name read from the node.

    Raises:
        ValueError: The node's device is not the one requested.
    """
    device = detect()
    wanted = resolve(requested)
    if wanted and wanted != device:
        raise ValueError(f"the run was submitted for {wanted!r} and this node has {device!r}")
    return device


def named(name: str, device: str) -> str:
    """Return a run's name with its card's token last.

    Args:
        name: The name the config gives the run.
        device: The device name, empty when none is known.

    Returns:
        `<name>_<token>`, and the name itself when it already ends in the token.

    Raises:
        ValueError: The device is not one `TOKENS` lists.
    """
    short = token(device)
    return name if name.endswith(f"_{short}") else f"{name}_{short}"


def card_of(record: dict[str, Any]) -> str:
    """Return the card a record states under `run.gpu`, empty when it states none."""
    return str((record.get("run") or {}).get("gpu") or "")


def refusals(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    """Say where two records ran on different cards, which no join may span.

    A record that states no card is no evidence of another card, and passes.

    Args:
        first: The base record.
        second: The record joined onto it.

    Returns:
        One line naming both cards, or none when they are the same or one is unknown.
    """
    cards = [card_of(record) for record in (first, second)]
    if not all(cards) or cards[0] == cards[1]:
        return []
    return [f"run.gpu {cards[0]!r} vs {cards[1]!r}"]
