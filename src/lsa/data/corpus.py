"""Read tokenized source rows and document extents without training policy."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lsa.data.constants import CorpusFiles

if TYPE_CHECKING:
    from lsa.assets import TokenizedCard


def document_extents(root: Path, source: str) -> dict[str, tuple[int, int]]:
    """Read where every document of one tokenized source sits, off its manifest alone.

    Args:
        root: The corpus directory.
        source: Source name.

    Returns:
        Document id to its first tokenized row and how many rows it was tokenized into, in manifest order.
    """
    import pyarrow.parquet as pq

    manifest = pq.read_table(
        root / f"{source}{CorpusFiles.MANIFEST_SUFFIX}",
        columns=[CorpusFiles.DOC_ID, CorpusFiles.ROW_START, CorpusFiles.N_SAMPLES],
    ).to_pydict()
    return {
        str(document): (int(start), int(rows))
        for document, start, rows in zip(
            manifest[CorpusFiles.DOC_ID],
            manifest[CorpusFiles.ROW_START],
            manifest[CorpusFiles.N_SAMPLES],
            strict=True,
        )
    }


class SourceRows:
    """The tokenized sources on disk - where each document sits in its bin, and the bins themselves.

    numpy and pyarrow are reached inside the calls that need them, so importing this costs nothing and a
    missing one reads as the missing extra.
    """

    def __init__(self, corpus: "TokenizedCard", sources: list[str] | None = None) -> None:
        """Open every source whose bin exists under the corpus.

        Args:
            corpus: The corpus to read, which carries its own root and row length.
            sources: Sources to open, or None for every one the card names that is present.

        Raises:
            FileNotFoundError: No source bin exists under the root.
        """
        self.root = Path(corpus.path)
        self.sample_length = corpus.sample_length
        wanted = list(sources or corpus.sources)
        self.sources = [name for name in wanted if (self.root / f"{name}{CorpusFiles.BIN_SUFFIX}").exists()]
        if not self.sources:
            raise FileNotFoundError(f"no tokenized sources under {self.root} - looked for {wanted}")
        self._bins: dict[str, Any] = {}

    def source_rows(self, source: str) -> Any:
        """Return one source's bin, mapped rather than read.

        Args:
            source: Source name.

        Returns:
            The rows [rows, sample_length] uint32, as a numpy memory map.
        """
        import numpy as np

        if source not in self._bins:
            self._bins[source] = np.memmap(
                self.root / f"{source}{CorpusFiles.BIN_SUFFIX}", dtype=np.uint32, mode="r"
            ).reshape(-1, self.sample_length)
        return self._bins[source]
