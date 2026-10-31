"""What a debugging run adds to a server's settings - checks and a dump, all off unless the block turns them on.

They are the `debug` sub-block of the model config's settings block, nested as the table says and refused the way the design's own keys are.
Each check costs the server something a served run must not pay, which is why none is a default.

| key | what it does | cost |
|---|---|---|
| `check_index` | checks after every forward that every decode context-parallel rank holds the same index | two sums across the group and a host sync a layer |
| `check_pisa` | holds rows of each PISA layer's lists against the reference search and tallies a verdict | the reference runs on the host a row at a time |
| `dump` | writes what the dumped layers read - the input, positions and lists - for an offline read | rank 0 writes a file per prefill chunk and decode step, and syncs the device before every forward |

This module runs on the job host as well as in the server, so it imports no sglang.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from lsa.sglang.settings.design import refuse_unknown, require_whole

# Prefill forwards, and as many captured decode steps, each PISA layer checks before it stops
CHECK_STEPS: int = 6

# The layers a dump covers when the block names none - GLM-5.3-Flash's first sparse layer, its middle and its last
DUMP_LAYERS: tuple[int, ...] = (3, 23, 43)

# The position stride a prefill keeps rows at in a dump
DUMP_STRIDE: int = 4096


@dataclass(frozen=True)
class PisaCheck:
    """The check of PISA's lists against the reference search, off while `rows` is 0."""

    # Rows of each of a batch's first requests, on every PISA layer, held against the reference
    rows: int = 0
    # A request is checked only once it holds this many keys, so a long prefix is checked rather than a prompt's first chunks
    from_keys: int = 0
    # Prefill forwards, and as many captured decode steps, each layer checks before it stops
    steps: int = CHECK_STEPS

    def __post_init__(self) -> None:
        """Reject a count the check cannot use.

        Raises:
            ValueError: `rows`, `from_keys` or `steps` is not a whole number, or `steps` is under 1.
        """
        require_whole("check_pisa.rows", self.rows, 0)
        require_whole("check_pisa.from_keys", self.from_keys, 0)
        require_whole("check_pisa.steps", self.steps, 1)

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "PisaCheck":
        """Build from the `debug.check_pisa` block.

        Args:
            body: The block's contents.

        Returns:
            The check, validated.

        Raises:
            ValueError: A key is not one of the check's, or a value is refused.
        """
        refuse_unknown("debug.check_pisa", body, {item.name for item in fields(cls)})
        return cls(**body)


@dataclass(frozen=True)
class DumpSettings:
    """The dump of what a layer reads, off while `directory` is None."""

    # Where tensor-parallel rank 0 writes, one torch file per prefill chunk and per decode step of a request
    directory: str | None = None
    # The layers dumped
    layers: tuple[int, ...] = DUMP_LAYERS
    # A prefill keeps every row at a multiple of this position, and the chunk's last rows
    stride: int = DUMP_STRIDE
    # False leaves a prefill's lists out - a stride of 1 keeps the input whole, and its lists would be 18 KB a row
    lists: bool = True
    # True has every rank dump under `rank<r>/`, since the key and query input selects a list a rank
    all_ranks: bool = False

    def __post_init__(self) -> None:
        """Reject a dump the writer cannot make.

        Raises:
            ValueError: `directory` is not a path or None, `layers` is not whole numbers, `stride` is not a positive
                whole number, or `lists` or `all_ranks` is not true or false.
        """
        object.__setattr__(self, "layers", tuple(self.layers))
        if self.directory is not None and not (isinstance(self.directory, str) and self.directory):
            raise ValueError(f"dump.directory must be a path or null - got {self.directory!r}")
        for layer in self.layers:
            require_whole("dump.layers", layer, 0)
        require_whole("dump.stride", self.stride, 1)
        for name in ("lists", "all_ranks"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"dump.{name} must be true or false - got {getattr(self, name)!r}")

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "DumpSettings":
        """Build from the `debug.dump` block.

        Args:
            body: The block's contents.

        Returns:
            The dump, validated.

        Raises:
            ValueError: A key is not one of the dump's, or a value is refused.
        """
        refuse_unknown("debug.dump", body, {item.name for item in fields(cls)})
        return cls(**body)


@dataclass(frozen=True)
class DebugSettings:
    """Every debugging aid a server can be asked for."""

    # Check after every forward that every decode context-parallel rank holds the same index
    check_index: bool = False
    check_pisa: PisaCheck = field(default_factory=PisaCheck)
    dump: DumpSettings = field(default_factory=DumpSettings)

    def __post_init__(self) -> None:
        """Reject a check that is not a switch.

        Raises:
            ValueError: `check_index` is not true or false.
        """
        if not isinstance(self.check_index, bool):
            raise ValueError(f"debug.check_index must be true or false - got {self.check_index!r}")

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "DebugSettings":
        """Build from the settings block's `debug` sub-block.

        Args:
            body: The sub-block's contents.

        Returns:
            The settings, validated.

        Raises:
            ValueError: The sub-block is not an object, a key is not one of its own, or a value is refused.
        """
        if not isinstance(body, Mapping):
            raise ValueError(
                f"debug must be an object of {sorted(item.name for item in fields(cls))} - got {body!r}"
            )
        refuse_unknown("debug", body, {item.name for item in fields(cls)})
        parts = dict(body)
        for name, part in (("check_pisa", PisaCheck), ("dump", DumpSettings)):
            if name in parts:
                if not isinstance(parts[name], Mapping):
                    raise ValueError(f"debug.{name} must be an object - got {parts[name]!r}")
                parts[name] = part.from_mapping(parts[name])
        return cls(**parts)

    @property
    def dumps(self) -> bool:
        """Whether a directory is named for the dump."""
        return self.dump.directory is not None
