"""Render the RULER files the native benchmark reads, the way the paper's evaluation rendered them.

    lsa native prepare-ruler --lengths 4096 8192 16384

The paper scored the native models on lm-eval's port of RULER, sized by the models' own GPT-2 tokenizer, 100 documents a
cell and one seed a cell. These files are that rendering, a card-named task at a time, written under
`<root>/data/benchmarks-gpt2/ruler/<task>_<length>.jsonl` where the benchmark reads them.
They are reproducible under the lm-eval version `uv.lock` pins, whose generators the rendering calls.
They are not the files `lsa data prepare ruler` makes - NVIDIA's own generators draw other needles and haystacks, and a
checkpoint scores about 5 points lower on their files over the seven needle tasks other than niah_multikey_3.

Generation is CPU work and runs as one slurm job. A compute node has no egress, so the sources are fetched where the
command is typed: the essays and the tokenizer from the hub, SQuAD and HotpotQA from the release the generators' own
sources are mirrored in, and the sentence tokenizer.
"""

import argparse
import os
import random
import sys
from typing import Any

from lsa import assets, env, io, paths, slurm
from lsa.evaluation.accuracy.data import ruler as nvidia
from lsa.evaluation.accuracy.data import ruler_card, ruler_path
from lsa.training.native.benchmark.tasks import RULER_TOKENIZER, ruler_root

ESSAYS = "baber/paul_graham_essays"
# The sources lm-eval reads over http, and the file each is kept in under the root
SOURCES = {"squad.json": "sources/squad.json", "hotpotqa.json": "sources/hotpotqa.json"}
SEED = 1234
# The seed lm-eval's cwe_utils gives its own generator at import
CWE_SEED = 42
SAMPLES = 100
LENGTHS: tuple[int, ...] = (4096, 8192, 16384)
FIELDS = ("input", "outputs", "gen_prefix", "length", "max_length")

# What a job asks for - one node's CPUs and no GPU
PARAMETERS: dict[str, Any] = {
    "gpus_per_node": 0,
    "tasks_per_node": 1,
    "cpus_per_task": 4,
    "mem_gb": 16,
    "timeout_min": 60,
}

# The needle tasks - the haystack, the kind of key and value, and how many keys, values and queries
NEEDLES: dict[str, dict[str, Any]] = {
    "niah_single_1": {"haystack": "repeat", "key": "words", "value": "numbers"},
    "niah_single_2": {"haystack": "essay", "key": "words", "value": "numbers"},
    "niah_single_3": {"haystack": "essay", "key": "words", "value": "uuids"},
    "niah_multikey_1": {"haystack": "essay", "key": "words", "value": "numbers", "num_needle_k": 4},
    "niah_multikey_2": {"haystack": "needle", "key": "words", "value": "numbers"},
    "niah_multikey_3": {"haystack": "needle", "key": "uuids", "value": "uuids"},
    "niah_multiquery": {"haystack": "essay", "key": "words", "value": "numbers", "num_needle_q": 4},
    "niah_multivalue": {"haystack": "essay", "key": "words", "value": "numbers", "num_needle_v": 4},
}

# Tokens the QA prompts leave for the answer and the question's own framing
QA_ROOM = 40
# Tokens a fwe prompt leaves for the words it asks for
FWE_ANSWER = 50


def tokenizer_path() -> str:
    """Return the pinned snapshot of the tokenizer the files are counted in."""
    return assets.model(RULER_TOKENIZER).checkpoint


def root_default() -> str:
    """Return the root the rendering lives under unless the command line names one."""
    return ruler_root()


def source_path(root: str, name: str) -> str:
    """Return where one downloaded source is kept under a root."""
    return os.path.join(root, SOURCES[name])


def seeded() -> None:
    """Seed every generator the lm-eval ports draw from, so a cell is the same file whatever was rendered before it."""
    import numpy
    from lm_eval.tasks.ruler import cwe_utils

    random.seed(SEED)
    numpy.random.seed(SEED)
    # cwe_utils keeps a generator of its own, seeded once at import and then used to shuffle the word list.
    # Put it, and the list, back in the state a fresh import leaves them in.
    words = sorted(cwe_utils.WORDS)
    fresh = random.Random(CWE_SEED)
    fresh.shuffle(words)
    cwe_utils.WORDS[:] = words
    cwe_utils.RNG.setstate(fresh.getstate())


def needle_rows(task: str, length: int, samples: int, tokenizer: Any) -> list[dict]:
    """Generate one needle cell.

    Args:
        task: A key of `NEEDLES`.
        length: Context length in tokens.
        samples: Rows.
        tokenizer: The tokenizer the length is counted in.

    Returns:
        The rows, as lm-eval's port makes them.
    """
    from lm_eval.tasks.ruler.niah_utils import TEMPLATE
    from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack

    options = dict(NEEDLES[task])
    haystack = options.pop("haystack")
    return generate_samples(
        get_haystack(type_haystack=haystack),
        max_seq_length=length,
        template=TEMPLATE,
        type_haystack=haystack,
        type_needle_k=options.pop("key"),
        type_needle_v=options.pop("value"),
        num_samples=samples,
        TOKENIZER=tokenizer,
        **options,
    )


def variable_rows(length: int, samples: int, tokenizer: Any) -> list[dict]:
    """Generate the variable tracking cell, one short worked example ahead of each prompt."""
    from lm_eval.tasks.ruler import vt_utils

    example = vt_utils.sys_vartrack_w_noise_random(
        tokenizer=tokenizer, num_samples=1, max_seq_length=500, incremental=5
    )[0]
    return vt_utils.sys_vartrack_w_noise_random(
        tokenizer=tokenizer, num_samples=samples, max_seq_length=length, icl_example=example
    )


def common_words_rows(length: int, samples: int, tokenizer: Any) -> list[dict]:
    """Generate the common words extraction cell."""
    from lm_eval.tasks.ruler import cwe_utils

    return cwe_utils.sys_word_pair_random(num_samples=samples, max_seq_length=length, tokenizer=tokenizer)


def frequent_words_rows(length: int, samples: int, tokenizer: Any) -> list[dict]:
    """Generate the frequent words extraction cell, with room in the context for the words it asks for."""
    # Unlike lm-eval's own `get_dataset`, which sizes the prompt at `length`, this leaves `FWE_ANSWER` tokens so the
    # prompt and the words it asks for fit in the `max_length` the row records
    from lm_eval.tasks.ruler import fwe_utils

    rows = fwe_utils.sys_kwext(tokenizer=tokenizer, max_seq_length=length - FWE_ANSWER, num_samples=samples)
    for row in rows:
        row["max_length"] = length
    return rows


def fitting_questions(
    questions: list[dict], documents: list[str], length: int, samples: int, tokenizer: Any
) -> list:
    """Keep the first questions whose own paragraphs fit in the length.

    lm-eval's sampler drops distractors to fit a prompt but never a question's own paragraphs.
    Keeping only questions whose own paragraphs fit, with `QA_ROOM` tokens to spare, is this module's choice,
    so a cell never holds a question its context cannot carry.

    Args:
        questions: lm-eval's question records.
        documents: The paragraphs they index into.
        length: Context length in tokens.
        samples: How many to keep.
        tokenizer: The tokenizer the length is counted in.

    Returns:
        `samples` questions.

    Raises:
        ValueError: Fewer questions fit.
    """
    from lm_eval.tasks.ruler import qa_utils

    kept = []
    for question in questions:
        text, answer = qa_utils.generate_input_output(0, len(question["context"]), [question], documents)
        if len(tokenizer(f"{text} {answer}").input_ids) + QA_ROOM > length:
            continue
        kept.append(question)
        if len(kept) == samples:
            return kept
    raise ValueError(f"only {len(kept)} questions fit in {length} tokens, {samples} wanted")


def question_rows(task: str, length: int, samples: int, tokenizer: Any, root: str) -> list[dict]:
    """Generate one question answering cell from the SQuAD or HotpotQA file under the root."""
    from lm_eval.tasks.ruler import qa_utils

    reader, name = (
        (qa_utils.read_squad, "squad.json")
        if task == "qa_squad"
        else (qa_utils.read_hotpotqa, "hotpotqa.json")
    )
    # lm-eval reads these over http, which a compute node cannot
    patched: Any = qa_utils
    patched.download_json = lambda url: io.read_json(source_path(root, name))
    questions, documents = reader()
    # The generators draw from the global state, so seed after the reads
    seeded()
    return qa_utils.generate_samples(
        tokenizer=tokenizer,
        # lm-eval shuffles the list it is given in place once a prompt takes every paragraph, and the reader's copy is cached
        docs=list(documents),
        qas=fitting_questions(questions, documents, length, samples, tokenizer),
        max_seq_length=length,
        num_samples=samples,
        tokens_to_generate=32,
    )


def render(task: str, length: int, samples: int, tokenizer: Any, root: str) -> list[dict]:
    """Generate one cell's rows in the layout the benchmark reads.

    Args:
        task: One of the RULER card's tasks.
        length: Context length in tokens.
        samples: Rows.
        tokenizer: The tokenizer the length is counted in.
        root: The root the QA sources were fetched under.

    Returns:
        Rows numbered from 0, each with `input`, `outputs`, `gen_prefix`, `length` and `max_length`.

    Raises:
        ValueError: The task is not a card task, or a row lacks a field the benchmark reads.
    """
    if task not in ruler_card().tasks:
        raise ValueError(f"{task} is not a RULER task - expected one of {ruler_card().tasks}")
    seeded()
    if task in NEEDLES:
        rows = needle_rows(task, length, samples, tokenizer)
    elif task == "vt":
        rows = variable_rows(length, samples, tokenizer)
    elif task == "cwe":
        rows = common_words_rows(length, samples, tokenizer)
    elif task == "fwe":
        rows = frequent_words_rows(length, samples, tokenizer)
    else:
        rows = question_rows(task, length, samples, tokenizer, root)
    for i, row in enumerate(rows):
        missing = [field for field in FIELDS if field not in row]
        if missing:
            raise ValueError(f"{task} {length}: row {i} has no {missing}")
        row["index"] = i
    return rows


def prepare(root: str, tasks: list[str], lengths: list[int], samples: int, force: bool) -> None:
    """Write every cell of the grid that is not already on disk.

    Args:
        root: The root the files go under.
        tasks: RULER task names.
        lengths: Context lengths in tokens.
        samples: Rows per cell.
        force: Regenerate a cell that exists.
    """
    from lm_eval.tasks.ruler.common_utils import get_tokenizer

    tokenizer = get_tokenizer(tokenizer=tokenizer_path())
    for length in lengths:
        for task in tasks:
            path = ruler_path(task, length, root)
            if os.path.exists(path) and not force:
                print(f"{path}: kept", flush=True)
                continue
            rows = render(task, length, samples, tokenizer, root)
            io.write_jsonl(path, rows)
            sizes = sorted(row["length"] for row in rows)
            print(f"{path}: {len(rows)} rows, median {sizes[len(sizes) // 2]} tokens", flush=True)


def ensure(root: str) -> None:
    """Fetch what the generators read, once, where the network is.

    Args:
        root: The root the files go under.

    Raises:
        ValueError: A downloaded source is not the file the release pins.
    """
    # The package reads the hub offline by default, and this is the one step that must reach it
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "0"
    import datasets
    from huggingface_hub import constants, snapshot_download

    constants.HF_HUB_OFFLINE = False
    datasets.config.HF_DATASETS_OFFLINE = False
    os.makedirs(os.path.join(root, "sources"), exist_ok=True)
    for name, urls in nvidia.ASSET_URLS.items():
        if name not in SOURCES:
            continue
        path = source_path(root, name)
        if not os.path.exists(path):
            nvidia.fetch(path, tuple(url.format(revision=ruler_card().revision) for url in urls))
        if io.file_sha256(path) != nvidia.ASSETS[name]:
            raise ValueError(f"{path} is not the file the release pins - delete it and prepare again")
    nvidia.ensure_nltk()
    model = assets.model(RULER_TOKENIZER)
    snapshot_download(model.repo, revision=model.revision, allow_patterns=["*.json", "merges.txt"])
    cache = os.environ.get("HF_DATASETS_CACHE") or str(paths.data() / "hf")
    datasets.load_dataset(ESSAYS, split="train", cache_dir=cache)


def add_options(parser: argparse.ArgumentParser) -> None:
    """Add what the rendering is made with to the command's parser.

    Args:
        parser: The command's parser.
    """
    parser.add_argument(
        "--root", default=root_default(), help="where the files go - `LSA_RULER_ROOT` by default"
    )
    parser.add_argument("--tasks", nargs="+", default=list(ruler_card().tasks))
    parser.add_argument("--lengths", nargs="+", type=int, default=list(LENGTHS))
    parser.add_argument("--samples", type=int, default=SAMPLES, help="rows per cell")
    parser.add_argument("--force", action="store_true", help="regenerate cells that exist")
    slurm.add_flags(parser)


def main(argv: list[str] | None = None) -> None:
    """Fetch the sources and submit the job that renders the cells the command line names.

    Args:
        argv: The command line, or None for `sys.argv`.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_options(parser)
    args = parser.parse_args(argv)
    root = os.path.abspath(args.root)
    if not args.dry_run:
        ensure(root)
    environment = [*env.OFFLINE_HF, *env.EVAL_DATA, *env.exports({"PYTHONUNBUFFERED": "1"})]
    slurm.submit(
        args,
        "lsa-prepare-ruler-native",
        environment,
        PARAMETERS,
        prepare,
        root,
        args.tasks,
        args.lengths,
        args.samples,
        args.force,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
