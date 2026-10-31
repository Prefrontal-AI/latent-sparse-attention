"""Model accuracy evaluation of a served model - lm-eval tasks, our request path, one record format.

    data/       what a run reads - the prompt files, the suites and their pinned prompt-set hashes
    grading/    what turns a reply into a score where lm-eval's own falls short
    tasks       each benchmark as an lm-eval task over the prompt files, graded by `grading`
    sampling    what a request is sampled with
    client      the requests to a server, many at once
    runner      one suite against one server into one record
    regrade     a record rebuilt from its sample files, with no server
    records/    the record - its format, and the readers that print it

Dependencies point one way - `grading` imports nothing of the package, `tasks` imports it and the data
paths, `runner` everything below it, and `regrade` the runner's grading.
The viewer reads the records from outside, since it reads the kernel records as well, and nothing here imports it.
"""

from lsa import env

# Before any module of the package imports lm-eval, datasets or transformers - the hub reads its offline
# switches at import
env.prefer_offline_hub()
