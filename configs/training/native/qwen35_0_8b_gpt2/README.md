# Native 0.8B hybrids at the GPT-2 vocabulary

The runs of the paper's native comparison (Section 4.4), and the configs that score the released checkpoints.
Every file here overlays `base.yaml`, and `phase/` holds the pre-train and decay overlays the arms share.
The README at the repository root lists which file trains which arm.

## Presets

`lsa native evaluate --suite benchmark --tasks <preset>` takes a preset, a task, or several of either joined by commas.

| Preset | Tasks |
|---|---|
| `core` | hellaswag, arc_easy, arc_challenge, piqa, winogrande, lambada_openai, wikitext, sciq |
| `ruler` | the 13 RULER tasks at 4,096, 8,192 and 16,384 tokens |
| `ruler-32k` | the same 13 at 32,768, past the length the models trained at |

A RULER cell is also a task of its own, `ruler_<task>_<length>`, for example `ruler_niah_single_3_4096`.
The 13 tasks are `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`, `niah_multiquery`, `niah_multivalue`, `vt`, `cwe`, `fwe`, `qa_hotpot` and `qa_squad`, 100 documents each.
The paper's RULER column is the mean of all 13 at 4,096 tokens.

## RULER data

The benchmark reads fixed prompt files, so prepare them once.
Pass nothing to choose a tokenizer: `lsa native prepare-ruler` counts every prompt in the models' own GPT-2 tokenizer, which is the tokenizer the paper's evaluation sized them by.

```bash
lsa --site mycluster native prepare-ruler --lengths 4096 8192 16384
```

The command fetches its sources where it is typed, because a compute node has no egress, and then submits one CPU job.
It needs lm-eval's `ruler` extra, which the `evaluation` extra carries: `uv sync --extra training --extra evaluation`.

The files land at `$LSA_ROOT/data/benchmarks-gpt2/ruler/<task>_<length>.jsonl`.
Set `LSA_RULER_ROOT` to read and write another root, which holds the same `ruler/` folder.
An existing file is kept, and `--force` regenerates it.

Do not use `lsa data prepare ruler` for this.
It renders NVIDIA's own generators, which draw other needles and haystacks, and the same checkpoint scores about 5 points lower on the seven needle tasks they share, mostly on `niah_single_3`.
Those files are what the served accuracy suite reads, under the tokenizer of the model it serves.

Grading is RULER's string match: every reference must appear for the needle, `vt`, `cwe` and `fwe` tasks, and any one of them for the two QA tasks.
The paper's QA cells took the fraction of references found instead, which scores `qa_squad` about 4 points lower and leaves the mean 0.4 points lower.

## Reproducing the paper's RULER column

```bash
lsa --site mycluster native prepare-ruler --lengths 4096
lsa --site mycluster native evaluate configs/training/native/qwen35_0_8b_gpt2/evaluate_released_lsa_k2048.yaml \
  --suite benchmark --tasks ruler_niah_single_1_4096,ruler_niah_single_2_4096,ruler_niah_single_3_4096,ruler_niah_multikey_1_4096,ruler_niah_multikey_2_4096,ruler_niah_multikey_3_4096,ruler_niah_multiquery_4096,ruler_niah_multivalue_4096,ruler_vt_4096,ruler_cwe_4096,ruler_fwe_4096,ruler_qa_hotpot_4096,ruler_qa_squad_4096 \
  --gpus 1
```

The released `decay128k-lsa-2048` scores 52.8 on the 13 tasks, against the paper's 52.2.
A cell holds 100 documents, so its standard error is up to 5 points, and only the mean is worth comparing.
A run folder is named after the task list as typed, and the file system caps a name at 255 characters, so 13 explicit cells do not fit in one run.
Split them over two runs of `--tasks` lists, or name a preset, whose folder is its short name.

## Not provided

SWDE and FDA, the two extraction tasks of the paper's Table 4, have no task or preset here, and their prompt files are not part of this repository.
