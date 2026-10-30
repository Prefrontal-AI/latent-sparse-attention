"""Evaluate a served model.

    python -m lsa.evaluation.accuracy --host HOST --port PORT --suite full --name NAME [--tasks ...] [--note ...]

The server must serve /v1/chat/completions and /server_info (SGLang). Results go to
<out>/<run_id>/ - record.json, samples/<task>.jsonl, server_info.json.
With --pool the process is one worker of several that drain one run's requests, see `lsa.evaluation.accuracy.worker`.

The prompts are fixed files rather than synthesised per run, so two runs over different task sets see the
same prompts. `LSA_EVAL_DATA` says where they are and `lsa.evaluation.accuracy.data` is what fills it.
"""

import argparse
from typing import Any

from lsa import io
from lsa.evaluation.accuracy import records, regrade, runner, worker
from lsa.evaluation.accuracy.data import DATA_ROOT, hashes, suites
from lsa.evaluation.constants import Attention, Evaluation


def main() -> None:
    """Parse the command line and score a server, or print a suite, pin its hashes or regrade a run."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--suite", default="full", choices=sorted(suites.SUITES))
    parser.add_argument("--tasks", nargs="*", help="only these tasks of the suite (names or plan keys)")
    parser.add_argument("--lengths", nargs="*", type=int, help="only these RULER lengths of the suite")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="the first this many rows of every RULER task rather than the grid's count - the files under "
        "--data-root must hold them",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=None,
        help="keep each reply's first token with its logprob and this many alternatives - a parity signal "
        "finer than the answer text",
    )
    parser.add_argument("--note", default="", help="free text stored in the record")
    parser.add_argument("--tag", default="", help="short label appended to the run id")
    parser.add_argument(
        "--name",
        default="",
        help="what the config calls the run, which the record is grouped by and a new run id starts with - "
        "required unless --run-id continues a run",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="continue this run (its directory under --out) instead of starting a new one",
    )
    parser.add_argument(
        "--out", default=records.records_dir(Evaluation.ACCURACY), help="where run directories land"
    )
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="LongBench v2 prompts prepared for this limit",
    )
    parser.add_argument(
        "--pins",
        default=None,
        help="the set of prompt-set pins --data-root's files are held to, hashes_<PINS>.json - the default "
        "corpus's hashes.json when unset",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="ignore samples already in the run directory"
    )
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="warn instead of stop when prompts differ from the recorded set",
    )
    parser.add_argument(
        "--pool",
        action="store_true",
        help="be one worker of the run's pool - claim only the requests no other worker holds, write this "
        "worker's own shard, and leave when none is left to claim",
    )
    parser.add_argument(
        "--worker",
        default=None,
        help="this worker's name under --pool, unique among the run's workers (default its slurm job)",
    )
    parser.add_argument("--concurrency-scale", type=float, default=1.0)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=14400.0,
        help="seconds one request may take before it is an error row",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="override every selected task's completion budget - for a model that reasons past the "
        "suite's own ceiling and so scores an unfinished answer as a wrong one",
    )
    parser.add_argument(
        "--generation",
        default=None,
        help="json file holding one request-settings mapping per dataset (gpqa_diamond, ruler, ...) - how the "
        "served model is sampled, which the experiment yaml states",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="the hub id the record names the model by - the served path is a local checkpoint "
        "whose directory name says which copy was loaded rather than which model it is",
    )
    parser.add_argument(
        "--attention-kind",
        default=None,
        choices=Attention.RECORDED,
        help="the method the record files the run under - the server cannot tell a checkpoint's own sparse "
        "attention from full attention",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="the container the server runs in - the server cannot say, and the record keeps it with the mesh",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="the card model the server runs on, as the device reports it - the record keeps it under run.gpu",
    )
    parser.add_argument("--list", action="store_true", help="print the suite and exit")
    parser.add_argument(
        "--hashes",
        action="store_true",
        help="record the prompt-set hashes of the suites and exit (no server)",
    )
    parser.add_argument(
        "--regrade",
        metavar="RUN_DIR",
        help="rebuild this run's record from its sample files and exit (no server)",
    )
    args = parser.parse_args()
    generation = None
    if args.generation:
        generation = io.read_json(args.generation)

    if args.list:
        plans = [p for p in suites.SUITES[args.suite]() if not args.lengths or p.length in args.lengths]
        for plan in [runner.sampled(plan, generation) for plan in plans] if generation else plans:
            print(f"{plan.key:<44} repeats {plan.repeats}  {plan.sampling}")
        return
    if args.hashes:
        hashes.make_hashes(data_root=args.data_root, max_context=args.max_context, pins=args.pins)
        return
    if args.regrade:
        regrade.regrade(args.regrade, data_root=args.data_root, max_context=args.max_context)
        return
    if args.pool and (args.no_resume or not args.run_id):
        parser.error("--pool continues the run --run-id names, so it takes no --no-resume and needs --run-id")
    arguments: dict[str, Any] = {"worker": args.worker} if args.pool else {}
    run = (worker.PoolRunner if args.pool else runner.Runner)(
        f"http://{args.host}:{args.port}",
        args.suite,
        out_root=args.out,
        note=args.note,
        tag=args.tag,
        name=args.name,
        only=args.tasks,
        lengths=args.lengths,
        samples=args.samples,
        top_logprobs=args.top_logprobs,
        data_root=args.data_root,
        max_context=args.max_context,
        resume=not args.no_resume,
        strict_hashes=not args.allow_hash_mismatch,
        concurrency_scale=args.concurrency_scale,
        timeout=args.request_timeout,
        run_id=args.run_id,
        max_tokens=args.max_tokens,
        generation=generation,
        model=args.model,
        attention_kind=args.attention_kind,
        image=args.image,
        gpu=args.gpu,
        pins=args.pins,
        **arguments,
    )
    run.run()


if __name__ == "__main__":
    main()
