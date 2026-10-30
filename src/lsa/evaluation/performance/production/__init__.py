"""The whole served model, timed as production serves it - first token and every token after, one record per run.

| module | what it does |
|---|---|
| [measure](measure.py) | drive sglang's serving benchmark against a live server, one client process per length |
| [record](record.py) | the record - the client's result files folded into one json - and its table |
| [data](data/__init__.py) | the prompt set a run sends |
"""
