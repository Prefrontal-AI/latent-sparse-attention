"""Evaluate projections and assemble reports of the exact read and the served read.

An evaluation set is a few hundred gigabytes per layer, so a report streams it: each document is read, scored
on the card and let go, a few reads ahead of the one being scored.

The exact read is the ceiling a bucketed read sits under. The served read is what a server reads: the design's
bucket cache built over the whole document and read by the serving selector page by page - Ward's buckets, the
top-p pick, the expansion, the window and the sinks, and the beam for fixed and tree. It runs on a GPU
only, so a report on the host has the exact read alone.
"""

import itertools
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lsa.sglang.index.selector import read_sampled
from lsa.sglang.settings.design import DesignSettings, IndexerConfig
from lsa.training.distillation.constants import Report, Rows
from lsa.training.distillation.data.cache import Document, layer_path, listing, set_root
from lsa.training.distillation.data.pool import Entry, stream
from lsa.training.distillation.metrics import key_read, mean_reports, qsa_read, read_metrics
from lsa.training.distillation.model import Indexer, QSAIndexer

# The config module reaches omegaconf and hydra, so importing it here would fail before `require_extra` runs
# => the guard names the extra and the import happens once past it
if TYPE_CHECKING:
    from lsa.training.distillation.config import DistillConfig, DocumentSet
    from lsa.training.distillation.data.online import OnlineTeacher


@dataclass(frozen=True)
class Scored:
    """One evaluation set at one layer - the files it scores, or the documents the online teacher read for it."""

    document_set: "DocumentSet"
    paths: list[Path]
    # The set's documents on the host, read once by the online teacher and kept, or None to stream `paths`
    documents: list[Document] | None = None

    def stream(self, indexer: IndexerConfig) -> Iterable[Document]:
        """Return the set's documents one at a time, on the host."""
        if self.documents is not None:
            return self.documents
        return stream(self.paths, indexer)


def load_scoring(
    experiment: "DistillConfig", layers: list[int], teacher: "OnlineTeacher | None" = None
) -> dict[str, dict[int, Scored]]:
    """Resolve every evaluation set to the files it scores, and check that they are all written.

    Nothing is read here off a cache, because a set is streamed each time it is scored. The online teacher reads a
    set's documents here instead, once, and they are kept for every report.

    Args:
        experiment: The experiment.
        layers: The layers to score.
        teacher: The online teacher, or None to score the cache's files.

    Returns:
        The files or the documents of each evaluation set, by set name and layer.

    Raises:
        FileNotFoundError: An evaluation document has no file for a layer, which a report would find late.
    """
    if teacher is not None:
        return {
            document_set.name: {
                layer: Scored(document_set, [], documents)
                for layer, documents in teacher.read_now(
                    scored_entries(document_set, teacher), layers
                ).items()
            }
            for document_set in experiment.data.evaluate
        }
    scoring: dict[str, dict[int, Scored]] = {}
    for document_set in experiment.data.evaluate:
        root = set_root(experiment.data.root, document_set)
        entries = list(itertools.islice(listing(experiment.data.root, document_set), document_set.limit))
        by_layer = {}
        for layer in layers:
            paths = [
                layer_path(root, document_set.source, document_set.split, entry["name"], layer)
                for entry in entries
            ]
            absent = [path for path in paths if not os.path.exists(path)]
            if absent:
                raise FileNotFoundError(
                    f"{document_set.name}: {len(absent)} of {len(paths)} documents have no file for layer {layer}, "
                    f"the first being {absent[0]}"
                )
            by_layer[layer] = Scored(document_set, paths)
        scoring[document_set.name] = by_layer
    return scoring


def scored_entries(document_set: "DocumentSet", teacher: "OnlineTeacher") -> list[Entry]:
    """Return the documents one evaluation set scores, as the online teacher's catalog holds them.

    Args:
        document_set: The set.
        teacher: The online teacher.

    Returns:
        The set's first `limit` documents, by name.
    """
    return list(itertools.islice(teacher.catalog([document_set]), document_set.limit))


def scored_rows(document: Document, which: str) -> torch.Tensor:
    """Return the query rows a set scores of one document.

    Training rows are sorted by position, so they are taken evenly over the document rather than first.

    Args:
        document: The document.
        which: One of `Rows.BOTH`.

    Returns:
        Query rows [R] int64, on the document's device.
    """
    if which == Rows.TAIL:
        return document.eval_rows()
    return spread(document.train_rows(), Rows.TRAIN_SCORED)


def spread(rows: torch.Tensor, most: int) -> torch.Tensor:
    """Keep at most `most` of some rows, evenly over them with both ends kept, or every one when few enough.

    Args:
        rows: Query rows [R] int64.
        most: Rows to keep at most.

    Returns:
        The kept rows [min(R, most)] int64, in their order.
    """
    picked = torch.linspace(0, len(rows) - 1, min(most, len(rows)))
    return rows[picked.round().long().unique().to(rows.device)]


def served_designs(experiment: "DistillConfig", device: str | torch.device) -> dict[str, DesignSettings]:
    """Return the designs whose served read a report scores, by name - none off a GPU."""
    if torch.device(device).type != "cuda":
        return {}
    return experiment.run.report.served.settings(experiment.model.indexer)


def reads_own_space(settings: DesignSettings, run_indexer: IndexerConfig) -> bool:
    """Return whether a design reads a map or an input other than the run's - `lsa distill served-read` scores it alone."""
    return bool(settings.serving.projections) or settings.indexer.input != run_indexer.input


def served_read(
    document: Document,
    rows: torch.Tensor,
    index_queries: torch.Tensor,
    index_keys: torch.Tensor,
    settings: DesignSettings,
) -> dict[str, float]:
    """Score a design's served read of some rows of a document.

    Args:
        document: The cached document, on the GPU.
        rows: Query rows [R] int64, ascending.
        index_queries: Their index queries [R, index_dim] fp32, unit rows.
        index_keys: Every position's index key [T, index_dim], unit rows - read at bf16, as a server holds them.
        settings: The design.

    Returns:
        One number per metric.
    """
    positions = document.query_positions[rows].to(torch.int32)
    read = read_sampled(settings, index_queries, index_keys.to(torch.bfloat16), positions)
    return read_metrics(document, rows, read)


def report(
    experiment: "DistillConfig",
    maps: Mapping[int, Indexer],
    scoring: dict[str, dict[int, Scored]],
    step: int,
    reader: Callable[[Document], Document] | None = None,
) -> dict[str, Any]:
    """Score the exact read, and on a GPU the served read, on every evaluation set, averaged over layers and documents.

    Each document is read onto the host, staged to the device the maps are on for as long as it is scored,
    and dropped. A QSA indexer's read is its own selection, and has no served read beside it.

    Args:
        experiment: The experiment.
        maps: The current maps, or QSA indexers, by layer.
        scoring: The evaluation files by set name and layer.
        step: The step being reported, for the printed line.
        reader: Reads the run's input off a staged document's `x` - the key and query - or None to project `x`.

    Returns:
        The report, keyed by set name - the exact read's metrics, and each served design's under `served`.
    """
    out: dict[str, Any] = {"step": step, "tau": {str(layer): item.tau for layer, item in maps.items()}}
    device = next(iter(maps.values())).log_tau.device
    designs = served_designs(experiment, device)
    own = sorted(
        design for design, settings in designs.items() if reads_own_space(settings, experiment.model.indexer)
    )
    if own:
        raise ValueError(
            f"served designs {own} read their own map or input - the trainer scores its own maps on its own input, "
            "and `lsa distill served-read` scores those"
        )
    for name, by_layer in scoring.items():
        read: list[dict[str, Any]] = []
        layer_reads: dict[int, list[dict[str, Any]]] = {}
        served_seconds = 0.0
        for layer, scored in by_layer.items():
            index_map = maps[layer]
            for waiting in scored.stream(experiment.model.indexer):
                document = waiting.to(device)
                if reader is not None:
                    document = reader(document)
                rows = scored_rows(document, scored.document_set.rows)
                if isinstance(index_map, QSAIndexer):
                    read.append(qsa_scored(document, rows, index_map))
                    layer_reads.setdefault(layer, []).append(read[-1])
                    continue
                with torch.no_grad():
                    index_queries = index_map.index_queries(document.query_inputs(rows))
                    index_keys = index_map.index_keys(document.key_inputs())
                    measured: dict[str, Any] = key_read(
                        document,
                        index_queries,
                        index_keys,
                        experiment.model.indexer.top_k,
                        rows,
                        depths=experiment.run.report.depths,
                    )
                    began = time.monotonic()
                    kept = spread(torch.arange(len(rows), device=device), experiment.run.report.served.rows)
                    served = {
                        design: served_read(document, rows[kept], index_queries[kept], index_keys, settings)
                        for design, settings in designs.items()
                    }
                    served_seconds += time.monotonic() - began
                read.append({**measured, Report.SERVED: served} if served else measured)
                layer_reads.setdefault(layer, []).append(read[-1])
        out[name] = mean_reports(read)
        if experiment.run.report.depths:
            out[name][Report.BY_LAYER] = {
                str(layer): mean_reports(items) for layer, items in layer_reads.items()
            }
        timing = f" ({served_seconds:.1f} s served)" if designs else ""
        print(f"{experiment.run.name}: step {step} {name}{timing} {_line(out[name])}", flush=True)
    return out


@torch.no_grad()
def qsa_scored(document: Document, rows: torch.Tensor, indexer: QSAIndexer) -> dict[str, float]:
    """Score a QSA indexer's read of some rows of a document.

    Args:
        document: The document, read for a QSA indexer, on the indexer's device.
        rows: Query rows [R] int64.
        indexer: The layer's indexer.

    Returns:
        One number per metric.
    """
    index_queries = indexer.index_queries(document.query_inputs(rows), document.query_positions[rows])
    index_keys = indexer.index_keys(document.key_inputs())
    return qsa_read(document, index_queries, index_keys, indexer.config, rows)


def _line(metrics: dict[str, Any], prefix: str = "") -> str:
    """Render one evaluation set's report as a line.

    Args:
        metrics: The set's report, a tree of numbers.
        prefix: What each name is printed after.

    Returns:
        The line, without a newline.
    """
    return " ".join(
        _line(value, f"{prefix}{name}/") if isinstance(value, dict) else f"{prefix}{name} {value:.3f}"
        for name, value in metrics.items()
        if name != Report.BY_LAYER
    )
