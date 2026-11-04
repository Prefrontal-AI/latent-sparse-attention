# Data

Every dataset a run reads has one card, named for what it is, under [`src/lsa/assets/datasets`](../../src/lsa/assets/datasets) - one directory per kind.

```bash
python -c 'from lsa import assets; print(assets.dataset_names())'
python -c 'from lsa import assets; print(assets.tokenized("glm53_flash").manifest().rows)'
```

## Five kinds of dataset

| kind | cards | what it is |
|---|---|---|
| `source` | arxiv, code, doab, edgebench, games, gutenberg, hansard, swe, usgpo, wikimedia | text as collected - where it came from, its subsets and split, its licence |
| `tokenized` | glm53_flash, gpt2, qwen3_5, qwen38_27b | some sources as one tokenizer's ids, in rows of `sample_length` |
| `shards` | fineweb_edu_100bt | flat token files another pipeline wrote, read where they are - nothing prepares them |
| `benchmark` | aime25, gpqa_diamond, graphwalks, longbench2, math500, mmlu_pro, ruler | what an accuracy run asks - a hub dataset at a pinned revision, or a generator and its grid |
| `workload` | production | what a performance run sends, built from a tokenized dataset |

A card's `kind:` picks its schema and has to match the directory it sits in.
A name is one card across all five, so `assets.dataset(name)` finds it wherever it is, and `assets.source`, `assets.tokenized`, `assets.shards`, `assets.benchmark` and `assets.workload` read one kind.

## Declared on the card, measured in the manifest

A card holds only what is declared before anything is prepared - the sources and the tokenizer, the hub dataset and its revision, the grid.
What preparing measured lives in the dataset's manifest on disk beside the data:

| kind | manifest | measured |
|---|---|---|
| `tokenized` | `corpus.json` in its directory | vocabulary, rows and documents per source, the tokenizer snapshot and its revision, git sha, date |
| `shards` | `metadata.json`, written by the shards' own pipeline and read in place | vocabulary, dtype, shard counts, the validation tokens |
| `benchmark` | one entry per prepared file under the data root's `manifest/` | rows, sha256, tokenizer, date, and whatever its generator records |
| `workload` | `manifest.json` in its directory | each file's sha256 and every entry's rows, tokens and characters |

Every manifest repeats the card's declared fields verbatim as `declared` - every field but `path`, `about` and `preparer`, since where a dataset sits, how it is described and which generator wrote it do not change its bytes.
`card.manifest()` reads it and refuses one prepared under a card that differs, naming the fields that do.
So a card edited after its data was made fails the run that reads it, rather than being trusted by name.

The prepare step writes the manifest - `prepare` for a tokenized dataset, the generators for a benchmark, the prompt builder for a workload.
A source has no manifest of its own, and a hub benchmark none either, because nothing here prepares them.

## Preparing a card

```bash
lsa data prepare <card>           # one CPU slurm job, whatever the card's kind needs
lsa data prepare <card> --help    # what that card's generator takes
lsa data prepare <card> --dry-run # what would be submitted
```

One verb prepares every dataset, and it always submits - nothing is generated where the command is typed.
The card's kind says what runs: a source card with a `collector` is streamed from the hub, a tokenized card uses the shared tokenizer preparation, a workload is the production prompt builder, and a benchmark is the generator its card's `preparer` names, because RULER and LongBench v2 are both benchmarks and the kind alone does not tell them apart.
A generator's own options - lengths, tasks, parallel jobs - are flags of `lsa data prepare`, and `--cpus`, `--mem-gb`, `--hours`, `--partition` and `--reservation` change what the job asks for.

## Sources

A source card is its own origin: `dataset` is the hub dataset it was streamed from at `revision`, or `collected_from` the directory of a source that never left the cluster.
Its text is one folder of parquet shards, each carrying a `doc_id` and a `text` column, with a `_DONE` marker written when collecting finished.
The nine collected sources are `data/sources/<card>` under the shared root.
The collector streamed `main` and recorded no revision, so each card's `revision` is the one that was its dataset's head that day, read off the commit history afterwards - except `code`, whose dataset is gated and whose revision nobody could read.

`licence` is an SPDX id, `internal` or `per-document`.
A shard's own `license` column is the collector's one label per source, and arxiv's is wrong - upstream holds CC BY-SA and public-domain papers beside CC BY and CC0.
So a `per-document` source has a `manifest.parquet` beside its shards, one row per document with its id, the upstream dataset and revision, the licence as upstream states it and its url, restored once from upstream by `doc_id`.
`card.document_licences()` reads its `licence` column.
A manifest says what upstream says and no more - hansard's calls every document the Open Parliament Licence, although 780 are from the Scottish Parliament, the Northern Ireland Assembly and the Senedd, whose terms differ, and their `doc_id` prefixes `sp`, `ni` and `senedd` are what tells them apart.

`games` and `edgebench` are agent traces, released scrubbed as the text they were rendered to in the `traces-v1` release of the public repository, under CC BY 4.0.
Their cards still say `internal`, the label of a source this repository does not collect.
`code` is The Stack, which is gated, so its revision and its per-file licences are unrecorded, and its licence is `other`, as the dataset itself declares, until someone with access pins them.

The agent traces are a source too, and are not collected from the hub.
Download them from the `traces-v1` release into the card's folder, dropping the source's prefix from each file name:

```bash
mkdir -p "$LSA_ROOT/data/sources/edgebench"
gh release download traces-v1 -R Prefrontal-AI/latent-sparse-attention --pattern 'edgebench-*' --dir "$LSA_ROOT/data/sources/edgebench"
(cd "$LSA_ROOT/data/sources/edgebench" && for f in edgebench-*; do mv "$f" "${f#edgebench-}"; done)
```

`games` is the same with `games-*` and `$LSA_ROOT/data/sources/games`.
The release holds the text each session was rendered to through GLM-5.3-Flash's chat template, scrubbed, so there is nothing to render.
The text differs slightly from the corpus of the paper, as the release notes say.

## Collecting a source from the hub

```bash
lsa data prepare gutenberg                        # one source, as a CPU job that reads the hub
lsa data prepare gutenberg --limit-docs 200       # the first 200 documents the stream gives, to see that it runs
lsa data prepare gutenberg --verify <lock file>   # after it finishes - compare what was collected with a lock file
lsa data prepare gutenberg --write-lock <out>     # write the lock file of shards that exist
```

Eight sources have a `collector`: arxiv, doab, gutenberg, hansard, usgpo and wikimedia from the Common Pile, swe from Open-SWE-Traces, and code from The Stack.
Each is the first collector's rules, ported and not changed, because a change to one is a different corpus.
The job is the one that reads the hub, so unlike every other preparing job it is not offline.
It never replaces a collection that is already there - it refuses, and the folder is moved aside by hand - and it writes into a staging folder that is moved into place with the `_DONE` marker last, so a killed job leaves nothing that looks collected.

**What a document is** depends on the card's `collector`.
A `text` source is one row of its hub dataset, named by its id.
A `repositories` source is the files of one repository joined as `# <path>` and the file, in the order of the paths, named `<language>/<owner>/<repo>`.
The Stack stores a repository's files scattered, so each language is read in the stream's order until its files hold the card's `buffer_chars`, and a repository's files beyond that slice are not in its document.
A `traces` source is one agent trajectory, rendered as each turn's role, reasoning, content and tool calls.

**What is kept** is a document of at least the card's `min_tokens` - 64k - counted with no special tokens by the card's tokenizer at its pinned revision, SmolLM3-3B-Base, which is not the tokenizer a corpus is later tokenized with.
A document shorter than 2.5 characters for each of those 64k tokens could never reach them, so it is dropped before it is counted.
Collection stops at the first batch that takes the kept tokens past the card's `max_tokens`, which is 3B for usgpo, swe and code, so the last batch can pass it.

Two things in the first collector are not what its labels say, and both are kept:

- A file whose repository lists no licence passes the filter, and only the first licence it does list is looked at.
- The permitted list holds ISC, MPL-2.0, CC0-1.0, CC-BY-4.0 and others, but every `code` document carries the label `mit/apache/bsd`.

**What cannot be read off the card** is the order a hub dataset streams in, and for `code` the revision.
Where the revision is pinned the order follows from it.
Where it is not, a lock file is how a reader tells a collection that reproduced the corpus from one that did not.
It holds the sha256 of every document the corpus was collected with, in order, and `--verify` names each document that is missing, extra or changed, and says when the documents they share are in another order - which is a different corpus, since a tokenized row lands where the documents before it put it.
A collection that matches its lock is the corpus, and one that does not is a reader's own, which is reproducible only as far as its lock says.

## Tokenized datasets

```bash
eval "$(lsa env)"
lsa data prepare glm53_flash                       # every source the card names
lsa data prepare glm53_flash --sources edgebench   # one, the rest left as they are
```

The card names the sources, the model card whose tokenizer writes the ids and the row length, and `prepare` reads the tokenizer from your own hub cache at the revision the model card pins, so no path in a config or a command is anyone's personal copy.
Each source becomes `<source>.bin`, `uint32` rows of `sample_length` ids, and `<source>.manifest.parquet`, where each document starts in that bin and how many rows it gave.
A row index is relative to its own source's bin, so the sources of one dataset are bound by the card's `sources` order and nothing else.

The manifest is counted off every source on disk rather than the ones a job just tokenized, so tokenizing one source per job, which is how the whole of a dataset is made, leaves one record of all of it.

| card | tokenizer | sources |
|---|---|---|
| [glm53_flash](../../src/lsa/assets/datasets/tokenized/glm53_flash.yaml) | GLM-5.3-Flash | the nine collected sources and the trajectories |
| [qwen3_5](../../src/lsa/assets/datasets/tokenized/qwen3_5.yaml) | Qwen3.5-4B | the nine collected sources |
| [qwen38_27b](../../src/lsa/assets/datasets/tokenized/qwen38_27b.yaml) | Qwen3.8-27B | the nine collected sources and the trajectories |

A tokenized row is token ids and nothing else, so a tokenized dataset belongs to the one tokenizer that wrote it.
A run reads the vocabulary off the manifest and compares it with the model's before it builds anything, and reading it through the card refuses a dataset tokenized under another card - so a mismatched corpus and a card that has drifted from its data are both sentences at config time rather than a device-side assert eight processes in.
The held-out split is a hash of the document id, so two datasets of the same documents hold out the same documents whatever tokenizer produced them.

Naming one in a training run is [training.md](training.md#a-card-is-named-not-pathed).

## Extract distillation targets

```bash
lsa distill targets configs/data/targets/glm53_flash/long_16k_1m.yaml --dry-run
lsa distill targets configs/data/targets/glm53_flash/long_16k_1m.yaml --workers 2
```

The extraction config declares the frozen `model`, the input `data` and how `run` writes its targets.
GLM recipes include `data.selection`, which reads tokenized dataset cards and balances document windows across lengths and domains.
It declares the datasets, lengths, token budgets, domains and source grouping in the same file that extracts the targets.
`data.selection.tokens_per_tier` optionally takes a smaller training selection from the full allocation while retaining its scoring side.
The Qwen recipe chooses fixed-length windows directly under `data.tokenized` instead.

`run.output` names the target-cache root and `run.collection` names the collection when using inline selection.
Before loading weights, extraction writes `<run.output>/<run.collection>/selection.json`, recording the allocation and extraction settings, full allocation and final document windows.
Each window records its dataset, source, document, split, length and window index.
Workers read this same selection, and a resume rejects changed allocation or extraction settings rather than mixing targets from different selections.
`data.tokenized.lengths` and `data.tokenized.splits` limit which part of the selection a worker extracts without changing the recorded allocation.
No additional YAML is generated.
An existing cache without inline selection provenance is not adopted automatically; use a new `run.output`, such as a private tree under your own data directory, and keep the original cache unchanged.

After extraction finishes, a trainer under `configs/training/distillation/` reads the cache as `data.root`.
See [training](training.md#on-glm-53-flash) for staged extraction and the complete sequence.

## Benchmarks

```bash
eval "$(lsa env)"           # LSA_EVAL_DATA defaults to data/benchmarks under the root
lsa data prepare ruler --jobs 32 # the card's grid, as one CPU slurm job
lsa data prepare ruler --check        # after the preparation job finishes - reads the cells, submits nothing
lsa data prepare longbench2      # at the card's context
```

`mmlu_pro`, `gpqa_diamond`, `math500`, `aime25` and `graphwalks` are read from the hub cache at the revision their cards pin.
GraphWalks is two parquet files of one hub dataset rather than two configs, so its card's `subsets` names the files - `graphwalks_128k` reads the short one, and `graphwalks_256k` and `graphwalks_1m` a tier each of the long one, since a 1M server holds the 256k tier and not the 1.14M-token top one.
Its files are fetched by name from the hub cache and read as local parquet, because `load_dataset` keys the cache of a dataset of files by what it can only resolve online, and no compute node then finds it.
A compute node may have no egress, as on many Slurm clusters, so download them once into `HF_HOME` from a machine that has it:

```bash
HF_HUB_OFFLINE=0 hf download openai/graphwalks --repo-type dataset --revision be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7
```
`ruler` and `longbench2` are prepared once into fixed prompt files, so two runs over different task sets see the same prompts.
Files land at `$LSA_EVAL_DATA/ruler/<task>_<length>.jsonl` and `$LSA_EVAL_DATA/longbench2/longbench2_<context>.jsonl` - the card's `file` under a root of the same layout as its own.
The shared reader and the benchmark generators are separate: [`data/benchmarks`](../../src/lsa/data/benchmarks.py) resolves shared prepared-file paths, [`ruler`](../../src/lsa/evaluation/accuracy/data/ruler.py) drives the generators and [`longbench`](../../src/lsa/evaluation/accuracy/data/longbench.py) prepares LongBench v2.
`lsa data prepare ruler` asks for a node's 64 CPUs, 512G and no GPU for twelve hours.
Preparation first verifies the pinned generators, dependency versions, haystack hashes and sentence tokenizer resources on the submitting host.
It installs absent prerequisites there before submitting the CPU job, since a compute node may have none.
Existing mismatched inputs are never replaced; use fresh `RULER_NVIDIA` and `NLTK_DATA` paths for a clean installation.
`--fetch-in-job` moves the RULER clone and the NLTK download into the job, for a site whose submitting machine has no network access but whose compute nodes do.
`--dry-run` prints the submission without installing prerequisites, and `--force` regenerates prompt cells without replacing the generator.

The [RULER card](../../src/lsa/assets/datasets/benchmarks/ruler.yaml) holds the grid - thirteen tasks, eight needle variants, `vt`, `cwe`, `fwe`, `qa_squad` and `qa_hotpot`, each generated at 4096, 32768, 131072, 524288 and 1000000 with 100 rows, and at 2000000, 5000000 and 10000000 with 20.
A smaller grid of 20 and 10 rows is each file's first 20 and 10, since a cell's rows are drawn in order from one seeded stream, and `--samples` takes them.
A run picks `data.suite` - `full`, `ruler` or `smoke` - and optionally `data.tasks` or `data.lengths` to take only some of that suite, and [`suites`](../../src/lsa/evaluation/accuracy/data/suites.py) is where the lists and their sample counts live.
`full` is a fixed 2,000 of MMLU-Pro, GPQA Diamond, a fixed 100 of MATH-500, AIME25 four times, a fixed 60 of LongBench v2 and every RULER cell.
MMLU-Pro is zero-shot under GPQA's chain-of-thought template with the question's own up to ten choices, and scored by the same letter filter - lm-eval's own `mmlu_pro` is five-shot for a base model.
Its draw keeps each of the 14 categories at its share of the 12,032, so it scores the mixture the published number does.
`ruler` is the RULER part of it alone, and `smoke` is a few prompts of each benchmark, to check that an architecture runs.
Each asks for every row a RULER file holds, and a length the server cannot hold is skipped with the reason in the record - which is how a two-node configuration runs `full` or `ruler` restricted by `data.lengths` to the lengths past 1M, continued into the one-node run's id.
The fixed 2,000, 100 and 60 are drawn with their cards' seeds, and the ids are in [`subsets.json`](../../src/lsa/evaluation/accuracy/data/subsets.json).

### NVIDIA's generators, not lm-eval's port

lm-eval's port of RULER stops where its corpus does: the essays hold 643k tokens, SQuAD 205k and the common-words vocabulary 143k.
So at a nominal 1M five of its tasks served 643k-token prompts, and `cwe` and `qa_squad` at 512k repeated a shorter point.
NVIDIA's current generators repeat the essays and the documents and switch `cwe` to a 370101-word list, so every cell reaches its length - `check` prints how close, and fails a cell that overshoots or whose prompts stopped growing with the length.
The RULER card joins its `input` haystack and `gen_prefix` answer prefix with a newline.
Evaluation and target extraction use that same declared join.

The clone sits at `$LSA_EVAL_DATA/ruler-nvidia-<first 8 of the patch's sha256>`, so a changed patch installs a fresh clone rather than repairing one, pinned to the card's NVIDIA/RULER revision `c3f5e3b4` with [`ruler.patch`](../../src/lsa/evaluation/accuracy/data/ruler.patch), and its sentence tokenizer at `$LSA_EVAL_DATA/nltk_data`.
The patch lets the common-words search run past 4M tokens, and counts a prompt's GLM tokens in pieces cut at spaces its pre-tokenizer never joins to the character before - batched on the cell's share of the cores and cached across the search and the samples - which gives the count one tokenization gives, so every prompt is byte-identical to the unpatched tokenizer's.
`setup` refuses a haystack source whose sha256 is not the one the files were made from.
The essays and SQuAD are fetched from release assets of the exact files, with SQuAD's upstream url as a fallback, and HotpotQA from its upstream urls, which can drift.
A cell is one generator process with the card's seed, and `--jobs` runs that many at once.
A 1M cell takes about five minutes, a 2M cell about three and a 10M cell fifteen to twenty-seven.

These files were generated on 2026-09-19 by the driver that predates this repository, and the port reproduces every prompt-set pin it gave.
The files lm-eval's port generated on 2026-09-13 are kept under `$LSA_EVAL_DATA/ruler-lmeval-0.4.13`, the root to regrade a record made on them against.

The hub datasets are read from a HuggingFace cache, which matters on a site whose compute nodes have no egress; `lsa env` prints the exports, and [evaluation.md](evaluation.md#on-slurm) says what they are.

### What the manifest records

Generation writes one manifest entry per file under `$LSA_EVAL_DATA/manifest/`, since the shared filesystem does not serialise a lock and one file per entry needs none.
A run reads its entries from there, and the data root is the only copy.
A RULER entry carries its sha256, its row count, the seed, tokenizer `zai-org/GLM-5.3-Flash`, the generator commit and task arguments, the target length, the minimum, median and maximum prompt length and how many ambiguous prompts were dropped, and the card it was prepared under.
So a RULER length is a count in GLM tokens, and the record's `length_actual` is that median.
An accuracy record's `data` block carries the entry without the card.

[`accuracy/data/hashes.json`](../../src/lsa/evaluation/accuracy/data/hashes.json) holds the prompt-set hash of each `task:subset` key.
A run recomputes it and fails on a mismatch, or warns under `--allow-hash-mismatch`, which is what makes two records comparable.

A pin goes stale when the corpus is regenerated or a hub dataset's rendering changes.
Delete that key and rerun `--hashes`, which fills only the keys that are missing, after checking that nothing else moved.

```bash
python -m lsa.evaluation.accuracy --hashes
```

A new pin matches the records made after it, not the ones before - whether an older record saw the same prompts is its own `prompt_set_hash` to answer.

The hub revisions on the benchmark cards are the ones the shared datasets cache holds, read off its folder names after the files were made.
The LongBench v2 file of 2026-09-21 was built from whatever snapshot that cache held then, which is the pinned one only because the cache holds no other.

## Workloads

```bash
lsa data prepare production      # the card's seven files, about four minutes
```

The [production card](../../src/lsa/assets/datasets/workloads/production.yaml) names the tokenized dataset and the source its text is decoded from, the lengths and the entries per file, and a production run names the card as `data.workload`.
Entry i starts at the same row at every length, so the first L tokens of stream i are a prefix of every longer prompt of it and the ladder is nested.
