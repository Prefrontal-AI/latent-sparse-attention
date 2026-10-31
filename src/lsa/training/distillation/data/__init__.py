"""The targets distillation trains on - the teacher's own attention, extracted once and read back by every run.

Generating them runs the model; loading them does not, which is what lets a run train thousands of steps
in minutes.

| module | what it does |
|---|---|
| [selection](selection.py) | choosing an extraction's documents, balanced over lengths and domains |
| [documents](documents.py) | where an extraction's documents come from, and which side of the split each falls |
| [teacher](teacher.py) | extracting the targets, one document and layer at a time, and `lsa distill targets` |
| [teacher_config](teacher_config.py) | the yaml schema of one extraction |
| [teacher_plan](teacher_plan.py) | which documents and query positions an extraction reads, and the manifests that record them |
| [teacher_capture](teacher_capture.py) | the hooks that keep what each layer's attention is handed |
| [teacher_attention](teacher_attention.py) | the teacher's attention over the keys a selection could take |
| [stage](stage.py) | an extraction a span of the stack at a time, one card to a span |
| [cache](cache.py) | what one extracted document holds and how a run reads it |

The manifest an extraction writes beside its targets is [targets](../../data/targets.py)'s, because more than
one stage reads it.
"""
