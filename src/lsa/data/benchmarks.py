"""Read prepared benchmark files through their dataset cards."""

import json
import os
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lsa.assets import BenchmarkCard
    from lsa.data.manifests import BenchmarkManifest


class BenchmarkFiles:
    """Locate a benchmark's files and read the rows its manifest describes."""

    def __init__(self, card: "BenchmarkCard", root: str | None = None) -> None:
        """Bind a benchmark card to its own data root or an alternate root."""
        self.card = card
        self.root = root or card.data_root()

    @cached_property
    def manifest(self) -> "BenchmarkManifest":
        """Read the benchmark manifest and validate its declared card fields."""
        return self.card.manifest(self.root)

    def path(self, **coordinates: int | str) -> Path:
        """Locate a file at the card's grid coordinates, whether prepared yet or not."""
        return Path(self.card.file_path(self.root, **coordinates))

    def row_count(self, **coordinates: int | str) -> int | None:
        """Return the measured row count, or None when a file has not been prepared."""
        key = os.path.relpath(self.path(**coordinates), self.root)
        record = self.manifest.files.get(key)
        return None if record is None else record.rows

    def sample(self, index: int, **coordinates: int | str) -> dict[str, Any]:
        """Read one prepared row without loading the whole prompt file into memory."""
        count = self.row_count(**coordinates)
        if count is None:
            raise FileNotFoundError(f"no prepared file at {self.path(**coordinates)}")
        if not 0 <= index < count:
            raise ValueError(f"{self.card.name} holds {count} samples - asked for {index}")
        with self.path(**coordinates).open() as handle:
            for position, line in enumerate(handle):
                if position == index:
                    loaded: dict[str, Any] = json.loads(line)
                    return loaded
        raise ValueError(f"{self.path(**coordinates)} has fewer rows than its manifest's {count}")

    def prompt(self, sample: dict[str, Any], join: str | None = None) -> str:
        """Compose a row's prompt from the fields declared by its card, joined by its separator or by `join`."""
        return (self.card.join if join is None else join).join(str(sample[name]) for name in self.card.prompt)
