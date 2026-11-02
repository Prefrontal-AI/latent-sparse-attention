# PaperTrail

PaperTrail is a long-context benchmark on real, recent scientific literature, designed and built by [Sukrit Sumant](https://github.com/SukritSumant).
The benchmark, its synthetic text, its questions and its answer keys are his work.
This directory holds the benchmark's definition and the instructions to rebuild it.

In each of three domains - exoplanets, long-context ML and quantum error correction - two fictional result trails are hidden in a pile of real arXiv papers.
A fabricated paper introduces a value, real papers we insert a few sentences into cite and use it, and a second fabricated paper revises and renames it.
Later papers still use the old form.
Each sample asks 30 scored questions, 10 per tier - recall from one paper, a link between two, a trace across three or more - and 6 not-in-corpus questions.
The lengths run from 400k to 10M tokens, and four controls sit beside them.

## Licence

The kit under [`kit/`](kit) - the synthetic text, the questions, the answer keys and the checksums - is licensed under [CC BY 4.0](LICENSE).
Credit Sukrit Sumant as the author when you use or redistribute it, and say if you changed it.

The arXiv papers are not part of the kit and keep their own licences.
Nothing here redistributes them, which is why the benchmark is rebuilt rather than downloaded.

## Rebuilding

```bash
lsa data prepare papertrail                      # every sample, every length and control, as a CPU job
lsa data prepare papertrail --max-length 1000000 # the lengths a one-node server holds
```

The preparer fetches the 1,246 papers from arXiv one request at a time, 3 seconds apart, into a cache under the data root.
It needs internet access on the machine you submit from, because a compute node has none, so run it from a login node or a machine with egress.
It converts each paper, inserts the kit's text, assembles every context and checks each against the kit's checksum.
It refuses any paper or context that differs - the usual cause is arXiv rendering a paper's HTML again.
`--keep-mismatched` writes such contexts marked unverified instead of refusing.

The files land under `<data root>/papertrail/`, one questions file and one context file per sample and condition.
The code is in [`src/lsa/evaluation/accuracy/data/papertrail`](../../src/lsa/evaluation/accuracy/data/papertrail), and the benchmark's card is [papertrail.yaml](../../src/lsa/assets/datasets/benchmarks/papertrail.yaml).

## Layout of the kit

| Path | Contents |
|---|---|
| `kit/papers.jsonl` | every real paper - arXiv id, version, v1 date, label, title, sha256 of its converted text and of its text with our insertions |
| `kit/<sample>/sample.json` | the setting text, the host papers and the nodes inserted into each, the fake papers' dates, and every context's papers in order with its sha256 and token count |
| `kit/<sample>/snippets/<node>.json` | our synthetic text - two fabricated papers per trail, and insertion-only patches |
| `kit/<sample>/questions.jsonl` | the 30 scored questions and their answer keys |
| `kit/<sample>/nic.jsonl` | the 6 not-in-corpus questions, whose answer is that the papers do not hold it |

## Contexts

| Context | What it is |
|---|---|
| `L400000`, `L512000`, `L700000`, `L1000000`, `L2000000`, `L5000000`, `L10000000` | the benchmark at each length, in GLM-5.3-Flash tokens - every length holds every paper of the one below it |
| `C2_hosts` | the host papers of both trails, unpatched - no question should be answerable |
| `C3_synthetic` | our text alone - both fake papers in full and our inserted passages under their hosts' headers - every question should be answerable |
| `C4_background` | the 700k length without its fake and host papers - no question should be answerable |
| `R400k_real` | real papers alone at 400k, the control of the realism audit |

The control `C1` is the question alone and has no context.

## Verifying the kit

The benchmark's card pins the digest of the kit, and the preparer refuses a kit that differs.
You can recompute it without this code, from inside `kit/`:

```bash
find . -type f | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum | sha256sum
```

## Provenance

The kit is Sukrit Sumant's rebuild kit from the PaperTrail repository at `55ca88cf`, after the two fix rounds of 8 October 2026, unchanged but for one length.

`L512000` was added so the ladder meets the RULER grid at 512k.
It is the 400k papers, then the papers the 700k length adds, in the sample's background rank order, each added if it fits and skipped if not - so it sits inside 700k and leaves every other length as it was.
Putting 512k into the ladder's own rule instead would have changed every length above it, because each is built from the one below.
Its contexts were built and checked against the 700k ones paper by paper, every paper carrying the same text.

## Please keep the answer keys out of training data

The kit holds the questions and their answer keys.
A model trained on them cannot be measured with this benchmark, so keep this directory out of training corpora and out of crawled web copies.
