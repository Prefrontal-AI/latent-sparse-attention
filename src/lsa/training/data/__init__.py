"""The tokenized corpus every stage reads, written and read.

Two halves of one format, which is why they are one submodule rather than one beside the trainer and one
beside the collection pipeline. [`tokenize`](../../data/tokenize.py) does what a dataset card needs - tokenizes a
tokenized card's sources into rows for its tokenizer and measures them into its manifest, or collects agent
sessions into a source's documents; [`corpus`](corpus.py) reads the rows back by rank, resumably. A change to
the layout is a change to both, and finding that out by breaking a training run is the outcome this
directory exists to avoid.

| module | what it does |
|---|---|
| [tokenize](../../data/tokenize.py) | one dataset card made readable - tokenized rows and their manifest |
| [corpus](corpus.py) | the tokenized corpus, read by rank, resumable exactly |

Neither imports a training framework. `make check` gates both on a CPU with nothing installed, which is what
keeps the format's rules testable without eight processes and a node.
"""
