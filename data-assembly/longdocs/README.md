# LongDocs

LongDocs is the long-document corpus the distillation targets and the native 128k decay are trained on.
It is documents of 64k tokens and over from eight public sources and from the in-house agent traces, which the paper describes in its training data appendix.
Its text carries the licences of its sources, so it is not redistributed here, and you assemble it from the hub instead.

| Source | What a document is | Upstream |
|---|---|---|
| gutenberg | a book | Common Pile, Project Gutenberg |
| doab | a book | Common Pile, DOAB |
| arxiv | a paper | Common Pile, arXiv |
| usgpo | a government publication | Common Pile, USGPO |
| hansard | a day of parliamentary debate | Common Pile, UK Hansard |
| wikimedia | a page | Common Pile, Wikimedia |
| code | a repository, its files joined in path order | The Stack, Python and Java |
| swe | an agent trajectory | Open-SWE-Traces |

The licence of each is in `THIRD_PARTY_NOTICES.md` at the root of the repository, and the collection rules are in [the data guide](../../docs/guide/data.md#collecting-a-source-from-the-hub).

## Before you start

`code` is read from The Stack, which is gated.
Accept its terms with your own Hugging Face account at `huggingface.co/datasets/bigcode/the-stack-dedup`, and make a token available to the job as `HF_TOKEN`.
The other sources need no account.

A collection is a CPU job that reads the hub, so the machine it runs on needs internet access.
All eight come to about 19 GB of parquet shards.

## Collecting

One job per source, which can run side by side.

```bash
for source in arxiv doab gutenberg hansard usgpo wikimedia swe code; do
  lsa data prepare $source
done
```

Each job asks for 32 CPUs and 128 GB for up to 24 hours, which `--cpus`, `--mem-gb` and `--hours` change.
`--dry-run` prints what would be submitted, and `--limit-docs 200` reads only the first 200 documents the stream gives, which is enough to see that a source runs.
A collection never replaces one that is already there.
It refuses, and you move the old folder aside.

## What is and is not the same as the paper's

The rules are the first collector's, ported without change, and a pinned source should stream the same documents in the same order.
Two of them are not what a label says, and they are kept so that the corpus is the paper's: a `code` file whose repository lists no licence is kept, and the permitted licences go beyond MIT, Apache and BSD.
The data guide lists both.

What a collection cannot promise is the order a hub dataset streams in.
Every card pins its dataset's revision except `code`, which was read from the head of a gated dataset without recording it, and whose repositories are the ones that fell in a 20 GB slice of each language.
The collector for text sources was checked on the first documents of gutenberg, and the one for trajectories on the first documents of swe, against the corpus, and every document matched exactly.
The collector for `code` was not checked, because its slice is read before its first document appears.

A lock file settles it for any source.
It holds the sha256 of every document of the corpus, in order, and `lsa data prepare <source> --verify <lock file>` names each document your collection is missing, has extra, or has differently.
The lock files are published in the release `longdocs-locks-v1` of the public repository, and the in-house traces, scrubbed, in `traces-v1`.

## Tokenizing

```bash
lsa data prepare glm53_flash --sources arxiv doab gutenberg hansard usgpo wikimedia swe code
```

A tokenized card names the in-house `games` and `edgebench` sources as well, and the paper's distillation mix draws on them.
They are not published yet, so tokenize the sources you have, as above.
The native 128k decay does not use the agent sessions, so for it this is the whole corpus.
