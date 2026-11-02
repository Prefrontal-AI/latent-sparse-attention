# Third-party notices

The NSA training kernels under `src/lsa/training/native/kernels/nsa` are adapted from the NSA kernels of [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), `fla/ops/nsa`, under the MIT License below.

```text
MIT License

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Datasets

This repository ships no text of the datasets below, except the PaperTrail kit.
It holds the scripts and lock files a reader assembles them with, and what a reader assembles keeps the licence its owner gave it.
Each source's card in `src/lsa/assets/datasets/sources` names the hub dataset and the revision it is read at.

| Source | Upstream | Licence |
|---|---|---|
| gutenberg | [Common Pile](https://huggingface.co/datasets/common-pile/project_gutenberg_filtered) filtered release, Kandpal et al. 2025, with PG19 (Rae et al. 2019) | Public domain, as the release's metadata marks English Project Gutenberg books. The label is the release maintainers' and is not verified here |
| usgpo | [Common Pile](https://huggingface.co/datasets/common-pile/usgpo_filtered) filtered release, from GovInfo | Public domain, as the release marks every record |
| doab | [Common Pile](https://huggingface.co/datasets/common-pile/doab_filtered) filtered release | CC BY and CC BY-SA books only, kept by the release after it checked each book for a whitelisted licence statement |
| wikimedia | [Common Pile](https://huggingface.co/datasets/common-pile/wikimedia_filtered) filtered release, March 2025 dumps, talk pages included | CC BY-SA 4.0 |
| hansard | [Common Pile](https://huggingface.co/datasets/common-pile/uk_hansard_filtered) filtered release | Open Parliament Licence, as the release states. Attribution to the parliaments is required by that licence |
| arxiv | [Common Pile](https://huggingface.co/datasets/common-pile/arxiv_papers_filtered) filtered release | Each paper's own: public domain, CC BY, CC BY-SA or CC0, as the release's per-record metadata states it |
| code | [The Stack](https://huggingface.co/datasets/bigcode/the-stack-dedup), Kocetkov et al. 2023, Python and Java | The Stack is gated, so a reader accepts its terms of use with their own account first. Use must follow each file's repository licence, attribution included, and the Stack honours removal requests. Its licence detection can be wrong. The first collector did not record a licence per file, kept a file whose repository lists no licence, and only looked at the first licence a repository lists |
| swe | [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces), NVIDIA | CC BY 4.0 from NVIDIA, on top of the licence of each repository, which is the one in the source's manifest. The label does not cover the terms of the models that generated the trajectories, which NVIDIA's card leaves unstated |
| PaperTrail | The kit in `data-assembly/papertrail`, by Sukrit Sumant | CC BY 4.0, see `data-assembly/papertrail/LICENSE`. The arXiv papers it is rebuilt from keep the licences their authors gave them and are fetched from arXiv, never redistributed |

The Common Pile releases carry a warning that licence metadata can be wrong ("license laundering").
Cite Kandpal et al. 2025 (arXiv:2506.05209) when using them, and keep each record's `authors`, `title` and `url` when redistributing CC BY or CC BY-SA text.
Text under CC BY-SA (wikimedia, and the CC BY-SA part of arxiv and doab) must keep that licence when adapted.
This notice does not say whether tokenized or other derived copies count as adaptations, and does not settle it.
