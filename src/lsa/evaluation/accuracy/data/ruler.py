"""RULER from NVIDIA's own generators - every file the suites run, at every length of the grid.

lm-eval's port of RULER stops where its corpus does: the essays hold 643k tokens, SQuAD 204k and the
common-words vocabulary 143k, so at 1M six of thirteen tasks were repeats of a shorter point. NVIDIA's
current generators repeat the essays and the documents, draw cwe from a 370101-word list and size a prompt
by estimate plus bisection, so every length is reached.
They run as they are from a pinned clone. This module is the driver around them: it builds each task
command the way their scripts/data/prepare.py does, renames the output into this corpus's layout and
records what was made.

  lsa data prepare ruler --jobs 32                            # the card's grid - 100 rows to 1M, 20 above
  lsa data prepare ruler --tasks cwe --lengths 2000000 --samples 10
  lsa data prepare ruler --check                         # one line per cell

Files: <root>/ruler/<task>_<length>.jsonl, and one <root>/manifest/ruler_<task>_<length>.jsonl.json each.

The grid is CPU work - a cell is one generator process, which counts a prompt's tokens in cached pieces on its
share of the node's cores (ruler.patch) - so
`lsa data prepare ruler` sends it to slurm as one job on one node's CPUs and no GPU, with the options
`add_options` declares. Preparation first verifies or installs the pinned generators and haystacks where
the command is typed, since a compute node may have no egress - a clone and a download, a few minutes on a shared host.
`--fetch-in-job` moves that into the submitted job, for a site whose compute nodes can reach the network.
A dry run only prints the submission, and `--check` only reads.
"""

import argparse
import datetime
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any

from lsa import assets, io
from lsa.data import manifests
from lsa.data.tokenizer import tokenizer_repo
from lsa.evaluation.accuracy.data import DATA_ROOT, load_manifest, ruler_card, ruler_path

# What the generating job asks for when no flag says otherwise - 64 CPUs and 512G of one node for twelve hours, and no GPU
# The full grid to 10M is what this is sized for, and a short one asks for less: 16 CPUs, 64 GB and 2 hours generated
# the 26 cells of 4,096 and 32,768 in about 32 minutes
PARAMETERS: dict[str, Any] = {
    "gpus_per_node": 0,
    "tasks_per_node": 1,
    "cpus_per_task": 64,
    "mem_gb": 512,
    "timeout_min": 12 * 60,
}

# Our edits to their generators
PATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruler.patch")


def patched_files(patch: str) -> list[str]:
    """Return the paths a patch edits, as `git diff` names them."""
    with open(patch) as source:
        return [line[len("+++ b/") :].strip() for line in source if line.startswith("+++ b/")]


# The generators' clone and the sentence tokenizer they load, beside the corpus they made. The commit the
# clone is pinned to and the seed every cell is generated with are the card's. An installed clone is never
# repaired in place, so the default clone is named by the patch it carries - a changed patch installs anew
RULER_NVIDIA = os.environ.get(
    "RULER_NVIDIA", os.path.join(DATA_ROOT, f"ruler-nvidia-{(io.file_sha256(PATCH) or '')[:8]}")
)
NLTK_DATA = os.environ.setdefault("NLTK_DATA", os.path.join(DATA_ROOT, "nltk_data"))
GENERATORS = os.path.join(RULER_NVIDIA, "scripts", "data")
# What their scripts import beyond this environment - kept inside the clone, at the versions that made the files
DEPS = os.path.join(RULER_NVIDIA, ".deps")
DEPENDENCIES = ("tenacity==9.1.4",)

# The haystack sources, as the pinned clone holds them.
# english_words.json is the commit's own LFS object, and the essays and SQuAD are release assets of the exact files.
# HotpotQA is fetched unpinned upstream, so a fresh fetch that differs is a different corpus and setup refuses it
ASSETS = {
    "english_words.json": "affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca",
    "squad.json": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    "hotpotqa.json": "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8",
    "PaulGrahamEssays.json": "58e352531a80cef2d22c205dbebfbfd64a8afe55a32434de845f200718756c65",
}
# Tried in order - the CMU host for HotpotQA is often down, and the mirror is the file it serves.
# `{revision}` is the card's pinned commit
ASSET_URLS = {
    "english_words.json": (
        "https://media.githubusercontent.com/media/NVIDIA/RULER/{revision}/scripts/data/synthetic/json/english_words.json",
    ),
    "squad.json": (
        "https://github.com/Prefrontal-AI/latent-sparse-attention/releases/download/ruler-sources-v1/squad.json",
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    ),
    "hotpotqa.json": (
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json",
    ),
    "PaulGrahamEssays.json": (
        "https://github.com/Prefrontal-AI/latent-sparse-attention/releases/download/ruler-sources-v1/PaulGrahamEssays.json",
    ),
}

# This corpus's task name -> the name in their scripts/synthetic.yaml
YAML_NAME = {"qa_squad": "qa_1", "qa_hotpot": "qa_2"}
# Their word list carries 9.2M tokens at the official 30/3 frequencies - the 10M cell needs 40/4
CWE_ARGS = {10000000: {"freq_cw": 40, "freq_ucw": 4}}
# The needle haystack draws its distractor keys with replacement - a prompt whose queried key carries two
# values has no single answer, so these tasks are generated with slack and the clean prompts are kept
FILTERED = ("niah_multikey_2", "niah_multikey_3")
QUERIED = re.compile(r" for (.+?) mentioned in the provided text")


def load_module(path: str, name: str) -> ModuleType:
    """Import one of their files by path - the clone is not a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_config(ruler_task: str) -> dict:
    """Return their settings for one task - the yaml entry, then the defaults of the generator it names."""
    import yaml

    with open(os.path.join(RULER_NVIDIA, "scripts", "synthetic.yaml")) as source:
        customized = yaml.safe_load(source)
    config = dict(customized[YAML_NAME.get(ruler_task, ruler_task)])
    constants = load_module(os.path.join(GENERATORS, "synthetic", "constants.py"), "ruler_constants")
    config.update(constants.TASKS[config["task"]])
    return config


def task_args(ruler_task: str, length: int) -> dict:
    """Return the generator arguments of one cell - their task's, with the cwe override at 10M."""
    args = dict(task_config(ruler_task)["args"])
    if ruler_task == "cwe":
        args.update(CWE_ARGS.get(length, {}))
    return args


def task_command(
    ruler_task: str, length: int, samples: int, save_dir: str, tokenizer_name: str | None = None
) -> list[str]:
    """Build the command their prepare.py builds for one cell - model template `base`, so no wrapper."""
    tokenizer_name = tokenizer_name or tokenizer_repo(ruler_card())
    config = task_config(ruler_task)
    command = [
        sys.executable,
        os.path.join(GENERATORS, "synthetic", f"{config['task']}.py"),
        "--save_dir",
        str(save_dir),
        "--save_name",
        ruler_task,
        "--subset",
        "validation",
        "--tokenizer_path",
        tokenizer_name,
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(length),
        "--tokens_to_generate",
        str(config["tokens_to_generate"]),
        "--num_samples",
        str(samples),
        "--random_seed",
        str(ruler_card().seed),
    ]
    for key, value in task_args(ruler_task, length).items():
        command += [f"--{key}", str(value)]
    if config["task"] == "qa":
        command += ["--pre_samples", "0"]
    return command + ["--template", config["template"] + config.get("answer_prefix", "")]


def queried_keys(gen_prefix: str) -> list[str]:
    """Return the keys the needle question asks about - 'X, Y, and Z' when there are several.

    Raises:
        ValueError: The prefix asks about no key.
    """
    match = QUERIED.search(gen_prefix)
    if not match:
        raise ValueError(f"no query in {gen_prefix[:120]!r}")
    return [key.strip() for key in match.group(1).replace(", and ", ", ").split(", ")]


def one_value_each(row: dict) -> bool:
    """Say whether every queried key appears once in the haystack - the needle - and once in the question."""
    return all(
        len(re.findall(rf"(?<![\w-]){re.escape(key)}(?![\w-])", row["input"])) == 2
        for key in queried_keys(row["answer_prefix"])
    )


def our_rows(ruler_task: str, rows: list[dict], samples: int, length: int) -> tuple[list[dict], int]:
    """Convert their rows to this corpus's layout - the answer prefix becomes the generation prefix.

    Returns:
        The first `samples` usable rows, renumbered.
        How many ambiguous rows were dropped on the way.

    Raises:
        ValueError: A row's answer prefix does not open with the space the prompt joins it by.
    """
    out: list[dict] = []
    dropped = 0
    for row in rows:
        if len(out) == samples:
            break
        if not row["answer_prefix"].startswith(" "):
            raise ValueError(
                f"{ruler_task}: answer prefix without its leading space - {row['answer_prefix'][:40]!r}"
            )
        if ruler_task in FILTERED and not one_value_each(row):
            dropped += 1
            continue
        out.append(
            {
                "index": len(out),
                "input": row["input"],
                "outputs": row["outputs"],
                "gen_prefix": row["answer_prefix"].strip(),
                "length": row["length"],
                "max_length": length,
            }
        )
    return out, dropped


def generate_ruler(
    ruler_task: str,
    length: int,
    samples: int,
    root: str,
    tokenizer_name: str | None = None,
    threads: int | None = None,
) -> tuple[list[dict], int]:
    """Run their generator for one cell.

    Args:
        threads: The cores the patched tokenizer counts a prompt's pieces on - every core when unset.

    Returns:
        The cell's rows in this corpus's layout.
        How many ambiguous rows were dropped.

    Raises:
        RuntimeError: Fewer clean rows came out than were asked for.
    """
    # Their rows are a prefix of a longer run, so the slack never changes the rows kept
    asked = samples + max(3, round(0.5 * samples)) if ruler_task in FILTERED else samples
    os.makedirs(os.path.join(root, ".work"), exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"{ruler_task}_{length}_", dir=os.path.join(root, ".work"))
    try:
        subprocess.run(
            task_command(ruler_task, length, asked, work, tokenizer_name),
            check=True,
            env=dict(
                os.environ,
                PYTHONPATH=DEPS,
                TOKENIZERS_PARALLELISM="true",
                RAYON_NUM_THREADS=str(threads or len(os.sched_getaffinity(0))),
            ),
        )
        rows = io.read_jsonl(os.path.join(work, ruler_task, "validation.jsonl"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out, dropped = our_rows(ruler_task, rows, samples, length)
    if len(out) < samples:
        raise RuntimeError(f"{ruler_task} {length}: {len(out)} clean of {asked} generated - raise the slack")
    return out, dropped


def generator_commit() -> str:
    """Return the commit the clone is at."""
    return subprocess.run(
        ["git", "-C", RULER_NVIDIA, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def manifest_entry(
    ruler_task: str,
    length: int,
    rows: list[dict],
    path: str,
    dropped: int,
    seconds: float,
    tokenizer_name: str,
) -> dict:
    """Describe one file - the prompt sizes are their `length` less the task's generation allowance."""
    generate = task_config(ruler_task)["tokens_to_generate"]
    sizes = sorted(row["length"] - generate for row in rows)
    return {
        "rows": len(rows),
        "sha256": io.file_sha256(path),
        "seed": ruler_card().seed,
        "tokenizer": tokenizer_name,
        "generator": generator_commit(),
        "task_args": task_args(ruler_task, length),
        "tokens_to_generate": generate,
        "max_length": length,
        "prompt_min": sizes[0],
        "prompt_median": sizes[len(sizes) // 2],
        "prompt_max": sizes[-1],
        "dropped": dropped,
        "date": datetime.date.today().isoformat(),
        "seconds": round(seconds, 1),
    }


def prepare_cell(
    ruler_task: str, length: int, samples: int, root: str, tokenizer_name: str, threads: int | None = None
) -> None:
    """Generate one (task, length) file and write its manifest entry."""
    path = ruler_path(ruler_task, length, root)
    start = time.perf_counter()
    rows, dropped = generate_ruler(ruler_task, length, samples, root, tokenizer_name, threads)
    io.write_jsonl(path, rows)
    entry = manifest_entry(
        ruler_task, length, rows, path, dropped, time.perf_counter() - start, tokenizer_name
    )
    record = manifests.BenchmarkFile.from_entry({**entry, "declared": manifests.declared(ruler_card())})
    manifests.write_benchmark_file(root, os.path.relpath(path, root), record)
    print(
        f"{path}: {len(rows)} rows, median {entry['prompt_median']} tokens, {entry['seconds']} s", flush=True
    )


def prepare_ruler(
    tasks: list[str],
    lengths: list[int],
    samples: int | dict[int, int],
    root: str = DATA_ROOT,
    tokenizer_name: str | None = None,
    force: bool = False,
    jobs: int = 1,
    fetch_first: bool = False,
) -> None:
    """Generate every (task, length) cell that the manifest does not already hold at its row count.

    Every cell is its own process with its own seed, so skipping one changes nothing about the others, and
    cells run side by side without sharing anything but the root. The longest start first, so a pool is
    not left waiting on one 1M cell at the end.

    Args:
        samples: Rows per cell - one count for every length, or one count per length.
        tokenizer_name: The tokenizer to count in, or None for the one the corpus names.
        jobs: Cells generated at once.
        fetch_first: Verify or install the pinned generators and sources first, as `ensure` does - for a job that
            runs where the command was not typed.
    """
    if fetch_first:
        ensure()
    tokenizer_name = tokenizer_name or tokenizer_repo(ruler_card())
    manifest = load_manifest(root)
    cells = []
    for length in lengths:
        rows_wanted = samples if isinstance(samples, int) else samples[length]
        for ruler_task in tasks:
            path = ruler_path(ruler_task, length, root)
            key = os.path.relpath(path, root)
            if (
                not force
                and key in manifest
                and os.path.exists(path)
                and manifest[key]["rows"] == rows_wanted
            ):
                print(f"{path}: kept ({rows_wanted} rows)", flush=True)
                continue
            cells.append((ruler_task, length, rows_wanted))
    cells.sort(key=lambda cell: cell[1], reverse=True)
    # The cells share the node's cores - each counts its tokens on its share
    threads = max(1, len(os.sched_getaffinity(0)) // jobs)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(prepare_cell, *cell, root, tokenizer_name, threads) for cell in cells]
        for future in futures:
            future.result()


def check(
    tasks: list[str], lengths: list[int], root: str = DATA_ROOT, growth: float = 1.5, log: Callable = print
) -> list[str]:
    """Print one line per cell - rows, how close the prompts come to the length, what was dropped.

    Their generators quantize the haystack coarsely, so a prompt sits below its length by a task-specific
    margin - the essay haystack grows 500 words at a time, the QA tasks a document at a time. What must hold
    is that nothing overshoots and that no task has run out of corpus: every step of the grid at least
    doubles the length, so the prompts have to grow with it.

    Returns:
        The manifest keys of the cells that are missing, overshoot, or stopped growing.
    """
    manifest = load_manifest(root)
    bad = []
    log(
        f"{'task':<18}{'length':>10}{'rows':>6}{'prompt tokens':>15}{'of length':>11}{'growth':>8}"
        f"{'dropped':>9}  args"
    )
    for ruler_task in tasks:
        previous: int | None = None
        for length in lengths:
            key = os.path.relpath(ruler_path(ruler_task, length, root), root)
            entry = manifest.get(key)
            if not entry or not os.path.exists(os.path.join(root, key)):
                log(f"{ruler_task:<18}{length:>10}{'-':>6}{'missing':>15}")
                bad.append(key)
                continue
            target = length - entry["tokens_to_generate"]
            share = entry["prompt_median"] / target
            grew: float | None = entry["prompt_median"] / previous if previous else None
            grew_text = f"{grew:.2f}" if grew else "-"
            extra = " ".join(f"{name} {value}" for name, value in sorted(entry["task_args"].items()))
            log(
                f"{ruler_task:<18}{length:>10}{entry['rows']:>6}{entry['prompt_median']:>15}{share:>10.1%}"
                f"{grew_text:>8}{entry['dropped']:>9}  {extra}"
            )
            if entry["prompt_max"] > target:
                log(f"  {key}: a prompt of {entry['prompt_max']} tokens is over the target {target}")
                bad.append(key)
            if grew is not None and grew < growth:
                log(f"  {key}: prompts grew {grew:.2f}x while the length grew - the corpus is running out")
                bad.append(key)
            previous = entry["prompt_median"]
    log(f"{len(tasks) * len(lengths)} cells, {len(bad)} bad")
    return bad


def fetch(path: str, urls: tuple[str, ...]) -> None:
    """Download one source from the first of its urls that answers.

    Raises:
        RuntimeError: None of them did.
    """
    for url in urls:
        if subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", path, url]).returncode == 0:
            return
    raise RuntimeError(f"{os.path.basename(path)}: no url answered - {urls}")


def verify(directory: str) -> None:
    """Verify the generator pin, exact patch, dependency versions and haystack contents.

    Args:
        directory: The existing or staged generator checkout.

    Raises:
        ValueError: An installed prerequisite differs from the corpus's declared inputs.
    """

    def git(*arguments: str) -> str:
        """Read the checkout's revision or tracked files."""
        return subprocess.run(
            ["git", "-C", directory, *arguments], capture_output=True, text=True, check=True
        ).stdout

    revision = git("rev-parse", "HEAD").strip()
    if revision != ruler_card().revision:
        raise ValueError(f"generator revision {revision}, expected {ruler_card().revision}")
    patched = patched_files(PATCH)
    with tempfile.TemporaryDirectory(prefix="lsa-ruler-patch-") as scratch:
        for path in patched:
            original = os.path.join(scratch, path)
            os.makedirs(os.path.dirname(original), exist_ok=True)
            with open(original, "w") as output:
                output.write(git("show", f"HEAD:{path}"))
        subprocess.run(["git", "apply", "--unsafe-paths", "--directory", scratch, PATCH], check=True)
        for path in patched:
            with open(os.path.join(scratch, path), "rb") as expected:
                with open(os.path.join(directory, path), "rb") as found:
                    if expected.read() != found.read():
                        raise ValueError(f"{path}: the installed generator does not match the pinned patch")
    allowed = {*patched, *(f"scripts/data/synthetic/json/{name}" for name in ASSETS)}
    changed = set(git("diff", "--name-only", "HEAD").splitlines())
    if changed - allowed:
        raise ValueError(f"unexpected generator edits: {sorted(changed - allowed)}")
    installed = {
        distribution.metadata["Name"].lower(): distribution.version
        for distribution in importlib.metadata.distributions(path=[os.path.join(directory, ".deps")])
    }
    for dependency in DEPENDENCIES:
        name, version = dependency.split("==")
        if installed.get(name) != version:
            raise ValueError(f"{dependency}: pinned generator dependency missing or changed")
    for name, expected_hash in ASSETS.items():
        path = os.path.join(directory, "scripts", "data", "synthetic", "json", name)
        if not os.path.isfile(path) or io.file_sha256(path) != expected_hash:
            raise ValueError(f"{name}: haystack source missing or changed")


def verify_nltk(directory: str | None = None) -> None:
    """Check the sentence tokenizers in an installed or staged directory.

    Args:
        directory: Resource root, or the configured NLTK directory when unset.
    """
    import nltk

    for resource in (
        "tokenizers/punkt/english.pickle",
        "tokenizers/punkt_tab/english/collocations.tab",
        "tokenizers/punkt_tab/english/sent_starters.txt",
        "tokenizers/punkt_tab/english/abbrev_types.txt",
        "tokenizers/punkt_tab/english/ortho_context.tab",
    ):
        nltk.data.find(resource, paths=[NLTK_DATA if directory is None else directory])


def install(directory: str) -> None:
    """Fetch the pinned generator and its inputs into a new staging directory.

    Args:
        directory: An absent staging path, never an existing installation.
    """
    revision = ruler_card().revision
    environment = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    subprocess.run(
        ["git", "clone", "--quiet", "https://github.com/NVIDIA/RULER", directory],
        check=True,
        env=environment,
    )
    subprocess.run(["git", "-C", directory, "checkout", "--quiet", revision], check=True)
    subprocess.run(["git", "-C", directory, "apply", PATCH], check=True)
    dependencies = os.path.join(directory, ".deps")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            sys.executable,
            "--target",
            dependencies,
            *DEPENDENCIES,
        ],
        check=True,
    )
    haystacks = os.path.join(directory, "scripts", "data", "synthetic", "json")
    for name, urls in ASSET_URLS.items():
        fetch(os.path.join(haystacks, name), tuple(url.format(revision=revision) for url in urls))


def publish(source: str, destination: str) -> None:
    """Atomically publish staged prerequisites without replacing a concurrent installation.

    Args:
        source: The verified staging directory.
        destination: The final path, which another preparer may have published first.
    """
    try:
        subprocess.run(["mv", "--no-clobber", "--no-target-directory", source, destination], check=True)
    except subprocess.CalledProcessError:
        if not os.path.lexists(destination):
            raise


def ensure_nltk() -> None:
    """Reuse verified tokenizers or publish a verified installation at an absent resource root."""
    if os.path.lexists(NLTK_DATA):
        verify_nltk()
        return
    parent = os.path.dirname(os.path.abspath(NLTK_DATA))
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lsa-nltk-", dir=parent) as staging:
        resources = os.path.join(staging, "resources")
        subprocess.run(
            [sys.executable, "-m", "nltk.downloader", "--quiet", "-d", resources, "punkt", "punkt_tab"],
            check=True,
        )
        verify_nltk(resources)
        publish(resources, NLTK_DATA)
    verify_nltk()


def ensure() -> None:
    """Reuse verified prerequisites or install them before the CPU job is submitted.

    Existing inputs are never replaced, including when preparation regenerates cells with `--force`.

    Raises:
        ValueError: Existing prerequisites are incomplete or differ from the declared inputs.
    """
    try:
        if os.path.lexists(RULER_NVIDIA):
            verify(RULER_NVIDIA)
            ensure_nltk()
            print(f"{RULER_NVIDIA}: pinned generators and sources verified", flush=True)
            return
        parent = os.path.dirname(os.path.abspath(RULER_NVIDIA))
        os.makedirs(parent, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".lsa-ruler-", dir=parent) as staging:
            checkout = os.path.join(staging, "checkout")
            install(checkout)
            verify(checkout)
            ensure_nltk()
            publish(checkout, RULER_NVIDIA)
            verify(RULER_NVIDIA)
    except (ValueError, LookupError, OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"RULER prerequisites could not be verified: {error}. Existing inputs were not replaced; "
            "set RULER_NVIDIA and NLTK_DATA to fresh private paths to install clean prerequisites."
        ) from error
    print(f"{RULER_NVIDIA}: pinned generators and sources installed", flush=True)


def add_options(parser: argparse.ArgumentParser) -> None:
    """Add what the RULER grid is generated with to `lsa data prepare`'s parser.

    Args:
        parser: The command's parser.
    """
    parser.add_argument("--root", default=DATA_ROOT)
    parser.add_argument(
        "--tokenizer", default=None, help="the tokenizer to count in - GLM-5.3-Flash's when unset"
    )
    parser.add_argument("--tasks", nargs="+", default=list(ruler_card().tasks))
    parser.add_argument("--lengths", nargs="+", type=int, default=ruler_card().length_values)
    parser.add_argument(
        "--samples", type=int, default=None, help="rows per cell - the grid's own count per length when unset"
    )
    parser.add_argument("--force", action="store_true", help="regenerate cells the manifest holds")
    parser.add_argument("--jobs", type=int, default=1, help="cells generated at once")
    parser.add_argument(
        "--fetch-in-job",
        action="store_true",
        help="clone the generators and download the sentence tokenizer in the submitted job rather than here - "
        "needs compute nodes that can reach the network",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="read the prepared cells and print one line each, submitting nothing",
    )


def check_prepared(args: argparse.Namespace) -> int:
    """Check the cells the command line names and return the exit status.

    Args:
        args: The parsed command line, of which `root`, `tasks` and `lengths` are read.

    Returns:
        1 when a cell is missing, overshoots or stopped growing, else 0.
    """
    return 1 if check(args.tasks, args.lengths, args.root) else 0


def job(card: assets.DatasetCard, args: argparse.Namespace) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Return what the job runs to generate the cells the command line names, and what it is called with.

    Args:
        card: The RULER card, which the generator reads for itself.
        args: The parsed command line.

    Returns:
        `prepare_ruler`, and the tasks, lengths, rows per cell, absolute root, tokenizer, force flag, jobs and
        whether the job fetches the prerequisites itself.

    Raises:
        ValueError: The prerequisites the generators need are incomplete, which a submitted job would only meet later.
    """
    samples = args.samples or {length: ruler_card().rows_at(length) for length in args.lengths}
    root = os.path.abspath(args.root)
    if not (args.dry_run or args.fetch_in_job):
        ensure()
    return prepare_ruler, (
        args.tasks,
        args.lengths,
        samples,
        root,
        args.tokenizer,
        args.force,
        args.jobs,
        args.fetch_in_job,
    )
