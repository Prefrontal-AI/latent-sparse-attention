"""What an extraction reads - its documents numbered by their place in the plan, its query positions, and the manifests that record them."""

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from lsa import assets, io
from lsa.core import RotaryLayout
from lsa.training.constants import Targets
from lsa.training.data.targets import write_manifest
from lsa.training.distillation.constants import Cache, Probe, Split
from lsa.training.distillation.data import selection as document_selection
from lsa.training.distillation.data.documents import CorpusDocuments, Documents, PreparedDocuments, Selected
from lsa.training.distillation.data.teacher_config import TeacherConfig


def training_query_count(run: TeacherConfig, length: int) -> int:
    """Return how many training queries a document of this length is sampled at.

    Args:
        run: The extraction.
        length: Tokens in the document.

    Returns:
        The count, which fits in the positions past the sinks and the window and before the tail.

    Raises:
        ValueError: The document has fewer such positions than the count.
    """
    span = length - run.data.queries.tail - (run.model.indexer.sink_size + run.model.indexer.window_size)
    count = max(run.data.queries.train, round(run.data.queries.density * span))
    if span < count:
        raise ValueError(
            f"a {length}-token document has {max(span, 0)} positions past its sinks and window and before "
            f"its tail - fewer than {count} training queries"
        )
    return count


def query_positions(run: TeacherConfig, length: int, document_index: int) -> torch.Tensor:
    """Sample a document's query positions - training rows first, then the tail.

    Args:
        run: The extraction.
        length: Tokens in the document.
        document_index: Which document, which is what makes the draw differ between them.

    Returns:
        Absolute positions [train + tail] int64, training rows sorted and the tail last.
    """
    first = run.model.indexer.sink_size + run.model.indexer.window_size
    count = training_query_count(run, length)
    generator = torch.Generator().manual_seed(run.data.queries.seed + document_index)
    drawn = first + torch.randperm(length - run.data.queries.tail - first, generator=generator)
    training = drawn[:count].sort().values
    return torch.cat([training, torch.arange(length - run.data.queries.tail, length)])


@dataclass(frozen=True)
class Probes:
    """The queries a prepared prompt names for itself, and what each is meant to find.

    Rows are in position order, so the `tail` question queries come last and a list's items before them.
    A row's targets are padded with `Cache.NO_KEY`, which is all of a row that has none.
    """

    positions: torch.Tensor
    kinds: torch.Tensor
    target_key_ids: torch.Tensor
    line_key_ids: torch.Tensor | None

    def fields(self) -> dict[str, torch.Tensor]:
        """Return what a layer file keeps of these beside the attention at the positions.

        Returns:
            Each `Cache` name and its tensor on the host - the kinds [Q] int8, the targets [Q, N] int32, and the
            needle's line [Q, L] int32 for a task that has one.
        """
        fields = {Cache.QUERY_KIND: self.kinds, Cache.TARGET_KEY_IDS: self.target_key_ids}
        if self.line_key_ids is not None:
            fields[Cache.LINE_KEY_IDS] = self.line_key_ids
        return fields


def document_queries(
    run: TeacherConfig, documents: Documents, selected: Selected, length: int, document_index: int
) -> tuple[torch.Tensor, torch.Tensor, Probes | None]:
    """Return a document's token ids, its query positions and, for a prepared prompt under `queries.targets`, its probes.

    A probe set still plans the tokenized side every extraction plans, which is not a prepared prompt and has no
    row to aim at, so it is queried as any other document is.

    Args:
        run: The extraction.
        documents: Where the document comes from.
        selected: Which document.
        length: Tokens to keep.
        document_index: Which document, for the draw of its query positions.

    Returns:
        Token ids [1, T] int64.
        Query positions [Q] int64.
        The probes, or None for a document queried by the draw.
    """
    if run.data.queries.targets and isinstance(documents, PreparedDocuments):
        ids, probes = probe_prompt(run, documents, selected, length)
        return ids, probes.positions, probes
    ids = documents.tokens(selected, length)
    return ids, query_positions(run, ids.shape[1], document_index), None


def probe_prompt(
    run: TeacherConfig, documents: Documents, selected: Selected, length: int
) -> tuple[torch.Tensor, Probes]:
    """Read a prepared prompt and the queries its row names for itself.

    A needle task is queried at the last `tail` positions - the question and the assistant's opening - and aimed at
    the tokens of each answer's first occurrence in the prompt and at the whole line of text it sits on - its sentence
    where the haystack has no line breaks. A counting
    task is queried at the list's items instead, and at its last `tail` positions with nothing to find.

    Args:
        run: The extraction, whose `queries.tail` is the question's length.
        documents: The prepared prompts.
        selected: Which prompt.
        length: Tokens to keep, which is the front of a longer prompt cut off.

    Returns:
        ids: Token ids [1, T] int64.
        probes: The queries and their targets, in `ids`' positions.

    Raises:
        ValueError: The documents are not prepared prompts, or the row's answer is not in its prompt, or the
            front a long prompt loses holds a query or a target.
    """
    if not isinstance(documents, PreparedDocuments):
        raise ValueError("queries.targets reads a prepared prompt's row, so it needs `prepared` documents")
    prompt = documents.prompt(selected, offsets=True)
    if prompt.spans is None:
        raise ValueError("the tokenizer reported no offsets, which queries.targets needs")
    answers = prompt.sample["outputs"]
    answers = [answers] if isinstance(answers, str) else list(answers)
    total = prompt.ids.shape[1]
    question = torch.arange(total - run.data.queries.tail, total)
    if selected.source == Probe.LIST_TASK:
        probes = list_probes(prompt.text, prompt.spans, answers, question, selected.start)
    else:
        probes = needle_probes(prompt.text, prompt.spans, answers, question)
    cut = max(total - length, 0)
    return prompt.ids[:, cut:], shifted(probes, cut)


def needle_probes(text: str, spans: torch.Tensor, answers: list[str], question: torch.Tensor) -> Probes:
    """Aim every question position at the needle - its value tokens, and separately the line it sits on.

    Args:
        text: The prompt as the tokenizer saw it.
        spans: Where each leading token sits in `text` [n, 2] int64, as [start, end) characters.
        answers: The strings the row expects, each of which occurs in `text`.
        question: The question's positions [tail] int64.

    Returns:
        The queries, one row per question position, every row aimed at the same keys.

    Raises:
        ValueError: An answer does not occur in the prompt.
    """
    values, lines = [], []
    for answer in answers:
        start = text.find(answer)
        if start < 0:
            raise ValueError(f"the answer {answer!r} does not occur in its prompt")
        line_start, line_end = line_around(text, start, start + len(answer))
        values.append(tokens_over(spans, start, start + len(answer)))
        lines.append(tokens_over(spans, line_start, line_end))
    count = len(question)
    return Probes(
        positions=question,
        kinds=torch.full((count,), Cache.QUERY_KINDS.index("question"), dtype=torch.int8),
        target_key_ids=torch.cat(values).unique().int().expand(count, -1).contiguous(),
        line_key_ids=torch.cat(lines).unique().int().expand(count, -1).contiguous(),
    )


def list_probes(
    text: str, spans: torch.Tensor, answers: list[str], question: torch.Tensor, seed: int
) -> Probes:
    """Query a numbered word list at its repeated items, each aimed at the word's earlier occurrences.

    Every occurrence of an answer word but its first is an `answer` query. As many `control` queries are drawn
    from the occurrences of the other words that have one before them, by a generator seeded with `seed`.
    A query sits at the first token of its item's word, and its targets are the first tokens of that word's
    earlier occurrences. The question's positions come last, with nothing to find.

    Args:
        text: The prompt as the tokenizer saw it, holding a list of `1. word 2. word ...`.
        spans: Where each leading token sits in `text` [n, 2] int64, as [start, end) characters.
        answers: The words the task counts.
        question: The question's positions [tail] int64.
        seed: What draws the control items, which is the row.

    Returns:
        The queries, in position order.

    Raises:
        ValueError: The prompt holds no numbered list.
    """
    items = list(re.finditer(r"\d+\. (\S+)", text))
    if not items:
        raise ValueError("the prompt holds no numbered word list")
    words = [item.group(1) for item in items]
    firsts = torch.searchsorted(
        spans[:, 1].contiguous(), torch.tensor([item.start(1) for item in items]), right=True
    ).tolist()
    occurrences: dict[str, list[int]] = {}
    before = []
    for word, first in zip(words, firsts, strict=True):
        seen = occurrences.setdefault(word, [])
        before.append(len(seen))
        seen.append(first)

    counted = set(answers)
    asked = [i for i, word in enumerate(words) if word in counted and before[i] > 0]
    pool = [i for i, word in enumerate(words) if word not in counted and before[i] > 0]
    generator = torch.Generator().manual_seed(seed)
    drawn = torch.randperm(len(pool), generator=generator)[: len(asked)].tolist()
    controls = [pool[i] for i in drawn]

    kinds_of = {i: Cache.QUERY_KINDS.index("answer") for i in asked}
    kinds_of.update({i: Cache.QUERY_KINDS.index("control") for i in controls})
    chosen = sorted(kinds_of)
    targets = [occurrences[words[i]][: before[i]] for i in chosen]
    question_kind = Cache.QUERY_KINDS.index("question")
    return Probes(
        positions=torch.cat([torch.tensor([firsts[i] for i in chosen], dtype=torch.int64), question]),
        kinds=torch.tensor([kinds_of[i] for i in chosen] + [question_kind] * len(question), dtype=torch.int8),
        target_key_ids=padded(targets, len(question)),
        line_key_ids=None,
    )


def line_around(text: str, start: int, stop: int) -> tuple[int, int]:
    """Return the line of text a needle sits on - its own sentence, or its own row where the haystack breaks lines.

    RULER's single-needle tasks hide the needle as one sentence inside an essay that has no line breaks at all,
    and its multi-key tasks put one needle a line among its distractors, so a line is bounded by a break or by a
    sentence's end, whichever is nearer.

    Args:
        text: The prompt.
        start: The needle value's first character.
        stop: One past its last.

    Returns:
        The line's first character and one past its last, the closing period included.
    """
    first = text.rfind("\n", 0, start) + 1
    sentence = text.rfind(". ", 0, start)
    if sentence >= 0:
        first = max(first, sentence + 2)
    ends = [len(text)]
    if (row := text.find("\n", stop)) >= 0:
        ends.append(row)
    if (sentence := text.find(". ", stop)) >= 0:
        ends.append(sentence + 1)
    return first, min(ends)


def tokens_over(spans: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Return the tokens that overlap a stretch of the text.

    Args:
        spans: Where each token sits in the text [n, 2] int64, as [start, end) characters.
        start: The stretch's first character.
        stop: One past its last.

    Returns:
        Token positions [m] int64.
    """
    return ((spans[:, 1] > start) & (spans[:, 0] < stop)).nonzero(as_tuple=True)[0]


def padded(rows: list[list[int]], blank: int) -> torch.Tensor:
    """Stack ragged rows, then `blank` rows with nothing in them, into one padded tensor.

    Args:
        rows: The positions each row holds.
        blank: Rows with nothing to hold, after the others.

    Returns:
        Positions [len(rows) + blank, widest] int32, `Cache.NO_KEY` past a row's end and at least one wide.
    """
    width = max([len(row) for row in rows] + [1])
    out = torch.full((len(rows) + blank, width), Cache.NO_KEY, dtype=torch.int32)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.int32)
    return out


def shifted(probes: Probes, cut: int) -> Probes:
    """Move a prompt's queries and targets back by the front that was cut off it.

    Args:
        probes: The queries in the whole prompt's positions.
        cut: Tokens lost off the front.

    Returns:
        The queries in the kept tokens' positions.

    Raises:
        ValueError: A query or a target sat in the part that was cut.
    """
    if cut == 0:
        return probes
    cutting = [probes.target_key_ids] + ([probes.line_key_ids] if probes.line_key_ids is not None else [])
    if int(probes.positions.min()) < cut or any(bool(((ids >= 0) & (ids < cut)).any()) for ids in cutting):
        raise ValueError(f"the {cut} tokens a prompt loses off its front hold a query or a target")

    def move(ids: torch.Tensor) -> torch.Tensor:
        return torch.where(ids >= 0, ids - cut, ids)

    return Probes(
        probes.positions - cut,
        probes.kinds,
        move(probes.target_key_ids),
        move(probes.line_key_ids) if probes.line_key_ids is not None else None,
    )


@dataclass(frozen=True)
class Planned:
    """One document an extraction will read, and where it sits in the whole plan.

    `index` is the document's place in the unsharded plan, which is what seeds its query positions. It is
    carried rather than counted so that a document caches the same bytes whichever shard reads it.
    """

    documents: Documents
    selected: Selected
    index: int


def flatten_plan(planned: list[tuple[Documents, list[Selected]]]) -> list[Planned]:
    """Flatten a plan, numbering each document by its place in the whole of it.

    Args:
        planned: One `(documents, selection)` per dataset, as `plan` returns.

    Returns:
        Every document the extraction reads, in plan order.
    """
    flat = [(source, selected) for source, selection in planned for selected in selection]
    return [Planned(source, selected, index) for index, (source, selected) in enumerate(flat, start=1)]


def extracted_now(run: TeacherConfig, documents: list[Planned]) -> list[Planned]:
    """Keep the planned documents this extraction reads, which is every one unless it names lengths or sides.

    Args:
        run: The extraction.
        documents: The whole plan, numbered.

    Returns:
        The documents to read, still numbered by their place in the whole plan.
    """
    lengths = set(run.data.tokenized.lengths) if run.data.tokenized.lengths else None
    splits = set(run.data.tokenized.splits) if run.data.tokenized.splits else None
    return [
        document
        for document in documents
        if (lengths is None or window_length(run, document.selected) in lengths)
        and (splits is None or document.selected.split in splits)
    ]


def window_length(run: TeacherConfig, selected: Selected) -> int:
    """Return how many tokens one window is read at - its own length, else what the run takes.

    Args:
        run: The extraction.
        selected: The window.

    Returns:
        Tokens to read.
    """
    return selected.length or run.data.tokenized.tokens_per_document


def read_documents(path: str | Path, dataset: str) -> list[dict[str, Any]]:
    """Return the entries of a document-selection manifest that name one dataset.

    Args:
        path: The document-selection manifest.
        dataset: The tokenized dataset card's name.

    Returns:
        The entries, in the file's order.

    Raises:
        ValueError: The manifest names no documents in this dataset.
    """
    entries = [entry for entry in io.read_json(path)[Targets.DOCUMENTS] if entry["dataset"] == dataset]
    if not entries:
        raise ValueError(f"the manifest at {path} names no document of {dataset}")
    return entries


def nested_entries(
    entries: list[dict[str, Any]], splits: list[str], lengths: list[int]
) -> list[dict[str, Any]]:
    """Add to a selection each window of the sides named at every shorter length, from the same first token.

    A window of length `L` at slot `w` starts at token `w * L`, so at a length `n` that divides `L` it starts at slot
    `w * L / n` and the shorter window is the front of the longer.

    Args:
        entries: The selection's entries, each carrying its split, length and window.
        splits: The sides to nest.
        lengths: The lengths to nest into.

    Returns:
        The entries, each followed by its shorter windows from the longest down.

    Raises:
        ValueError: A shorter length does not divide the length of a window it nests into.
    """
    nested = []
    for entry in entries:
        nested.append(entry)
        if entry["split"] not in splits:
            continue
        for length in sorted(lengths, reverse=True):
            if length >= entry["length"]:
                continue
            if entry["length"] % length:
                raise ValueError(
                    f"length {length} does not divide the {entry['length']}-token window of {entry['document']}"
                )
            nested.append({**entry, "length": length, "window": entry["window"] * entry["length"] // length})
    return nested


def selection_datasets(run: TeacherConfig) -> dict[str, assets.TokenizedCard]:
    """Resolve inline selection datasets, including one explicitly path-based corpus."""
    settings = run.data.selection
    if settings is None:
        raise ValueError("this extraction has no data.selection")
    override = run.data.tokenized.asset if run.data.tokenized.path else None
    if override is not None and override.name not in settings.datasets:
        raise ValueError(f"tokenized.path holds {override.name}, absent from selection.datasets")
    return {
        name: override if override is not None and name == override.name else assets.tokenized(name)
        for name in settings.datasets
    }


def resolve_selection(run: TeacherConfig) -> dict[str, Any]:
    """Resolve and publish deterministic selection provenance before loading model weights.

    Refuse changes to a collection's allocation or extraction identity, including on forced reruns.
    Atomic exclusive publication makes all workers and stages agree on one complete JSON document.
    """
    payload = selection_payload(run)
    folder = Path(run.run.output) / run.run.collection
    path = folder / "selection.json"
    if path.exists():
        if io.read_json(path) != payload:
            raise ValueError(
                f"{path}: resolved selection or extraction settings differ - use a new collection"
            )
        return payload
    if (folder / Targets.MANIFEST).exists() or any(folder.rglob("*.pt")):
        if path.exists() and io.read_json(path) == payload:
            return payload
        raise ValueError(
            f"{folder}: existing targets have no inline selection provenance - copy and migrate them first"
        )
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".selection.tmp", delete=False) as target:
        temporary = Path(target.name)
    try:
        io.write_json(temporary, payload)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if io.read_json(path) != payload:
                raise ValueError(f"{path}: another worker resolved a different selection") from None
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def selection_payload(run: TeacherConfig) -> dict[str, Any]:
    """Resolve an extraction's inline selection and the settings it is extracted under, without writing anything.

    Args:
        run: The extraction.

    Returns:
        What `selection.json` records - the allocation, the datasets' manifests, the extraction settings, the full
        selection and the windows taken.

    Raises:
        ValueError: The extraction has no selection, a dataset is tokenized for another model, or nothing trains.
    """
    settings = run.data.selection
    if settings is None:
        raise ValueError("this extraction has no data.selection")
    datasets = selection_datasets(run)
    measured = {}
    for name, card in datasets.items():
        if card.tokenizer != run.model.name:
            raise ValueError(
                f"{name} is tokenized for {card.tokenizer}, not extraction model {run.model.name}"
            )
        measured[name] = asdict(card.manifest())
    full = document_selection.build_full(settings, datasets)
    entries = (
        document_selection.tier(settings, full, settings.tokens_per_tier)
        if settings.tokens_per_tier
        else full
    )
    if not entries or not any(entry["split"] == Split.TRAIN for entry in entries):
        raise ValueError("data.selection provides no training documents")
    payload = json.loads(
        json.dumps(
            {
                "collection": run.run.collection,
                "config": asdict(settings),
                "datasets": measured,
                "extraction": {
                    "model": asdict(run.model),
                    "queries": asdict(run.data.queries),
                    "layers": run.run.layers,
                    "prefill": run.run.prefill,
                    "stages": run.run.stages,
                },
                "full_documents": full,
                Targets.DOCUMENTS: entries,
            }
        )
    )
    return dict(payload)


def plan(
    run: TeacherConfig, tokenizer: Any, resolved: dict[str, Any] | None = None
) -> list[tuple[Documents, list[Selected]]]:
    """Return every dataset this extraction reads and the documents it takes from each.

    Args:
        run: The extraction.
        tokenizer: The model's tokenizer, which the prepared prompts are read through.
        resolved: An inline selection already published before model loading.

    Returns:
        One `(documents, selection)` per dataset the run names, the tokenized one first.
    """
    if run.data.selection is not None:
        resolved = resolved if resolved is not None else resolve_selection(run)
        planned: list[tuple[Documents, list[Selected]]] = []
        for name, card in selection_datasets(run).items():
            corpus = CorpusDocuments(card, collection=run.run.collection)
            entries = [entry for entry in resolved[Targets.DOCUMENTS] if entry["dataset"] == name]
            chosen = corpus.select_documents(entries)
            if len(run.data.selection.datasets) > 1:
                chosen = [replace(selected, name=f"{name}_{selected.name}") for selected in chosen]
            planned.append((corpus, chosen))
    else:
        planned = plan_tokenized(run)
    if run.data.prepared.name:
        config = run.data.prepared
        prepared = PreparedDocuments(
            config.asset,
            config.sources,
            config.length,
            tokenizer,
            config.root,
            config.join,
            config.chat,
            config.chat_template_kwargs,
        )
        if any(documents.collection == prepared.collection for documents, _ in planned):
            raise ValueError("prepared and tokenized documents must use different collections")
        planned.append((prepared, prepared.select(config.samples, config.rows, config.source_rows)))
    return planned


def plan_tokenized(run: TeacherConfig) -> list[tuple[Documents, list[Selected]]]:
    """Resolve the existing counts-based or explicit-manifest corpus selection."""
    corpus = CorpusDocuments(
        run.data.tokenized.asset,
        run.data.tokenized.sources,
        run.data.tokenized.held_out_fraction,
        run.data.tokenized.eval_share,
    )
    if run.data.tokenized.documents:
        entries = read_documents(run.data.tokenized.documents, corpus.collection)
        if run.data.tokenized.nest:
            entries = nested_entries(
                entries, run.data.tokenized.splits or [], run.data.tokenized.lengths or []
            )
        selection = corpus.select_documents(entries)
    else:
        selection = corpus.select(
            run.data.tokenized.train, run.data.tokenized.evaluate, run.data.tokenized.windows
        )
    return [(corpus, selection)]


def collection_entries(
    run: TeacherConfig, planned: list[tuple[Documents, list[Selected]]]
) -> dict[str, list[dict[str, Any]]]:
    """Return the document index of each collection an extraction writes, as its manifest records it.

    A name is an identifier, so each entry carries what a set may select on beside it - the source, the domain, the
    side, the document it stands for and its length.

    Args:
        run: The extraction.
        planned: Every dataset and the documents it contributes.

    Returns:
        Each collection's entries, in plan order.
    """
    collections: dict[str, list[dict[str, Any]]] = {}
    for documents, selected_documents in planned:
        entries = collections.setdefault(documents.collection, [])
        for selected in selected_documents:
            entry: dict[str, Any] = {
                "name": selected.name,
                "source": selected.source,
                "document": selected.document,
                "window": selected.window,
                "split": selected.split,
                "length": window_length(run, selected),
            }
            if isinstance(documents, CorpusDocuments) and run.data.selection is not None:
                entry["dataset"] = documents.asset.name
                entry["domain"] = run.data.selection.domain_of[selected.source]
            entries.append(entry)
    return collections


def write_collection_manifests(
    root: Path,
    run: TeacherConfig,
    planned: list[tuple[Documents, list[Selected]]],
    layers: list[int],
    rotary: RotaryLayout,
) -> None:
    """Record what produced each collection inside the collection itself.

    The document list is the index a run's selection is resolved against, so it carries what a set may
    select on - the domain, the side and the document id the name stands for. A name is an identifier and
    nothing is parsed back out of it.

    Args:
        root: The cache root.
        run: The extraction.
        planned: Every dataset and the documents it contributes.
        layers: The layers extracted.
        rotary: Where the rotation sits in a head vector, recorded in the manifest.
    """
    collections = collection_entries(run, planned)
    recorded_run = asdict(run)
    if run.data.selection is not None:
        recorded_run["data"]["tokenized"]["lengths"] = None
        recorded_run["data"]["tokenized"]["splits"] = None
    for collection, entries in collections.items():
        provenance = {}
        if run.data.selection is not None and collection == run.run.collection:
            path = root / collection / "selection.json"
            provenance["selection"] = {
                "file": path.name,
                "sha256": io.file_sha256(path),
                "config": asdict(run.data.selection),
            }
        write_manifest(
            root / collection,
            {
                "run": recorded_run,
                "collection": collection,
                "rotary": {"end": rotary.end, "dim": rotary.dim},
                "layers": layers,
                Targets.DOCUMENTS: entries,
                **provenance,
                "torch": torch.__version__,
            },
        )
