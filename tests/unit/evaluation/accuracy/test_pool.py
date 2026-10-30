"""The queue several workers share - a claim is taken once, a lease expires, and the files say what is done."""

import json
import os
import threading
import time
from pathlib import Path

from lsa import io
from lsa.evaluation.accuracy import pool
from lsa.evaluation.constants import Pool

TASK = "ruler_cwe_4096"


def units(count, task=TASK, repeat=0):
    return [pool.Unit(task, doc, repeat) for doc in range(count)]


def age(queue, seconds):
    """Make every claim a queue holds look as if nothing had touched it for this long."""
    for path in queue.held.values():
        then = time.time() - seconds
        os.utime(path, (then, then))


def test_a_unit_is_claimed_by_exactly_one_of_many_workers_at_once(tmp_path):
    claimed = []
    barrier = threading.Barrier(8)

    def work(name):
        queue = pool.Queue(tmp_path, name)
        barrier.wait()
        claimed.extend(queue.claim(units(40), 40))

    threads = [threading.Thread(target=work, args=(f"w{i}",)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(claimed, key=lambda unit: unit.doc_id) == units(40)


def test_a_claim_takes_candidates_in_order_up_to_the_limit(tmp_path):
    queue = pool.Queue(tmp_path, "w1")
    assert queue.claim(units(10), 3) == units(3)
    # Held units are skipped, so the next call carries on from where this one stopped
    assert queue.claim(units(10), 3) == units(6)[3:]


def test_a_unit_another_worker_holds_is_not_taken(tmp_path):
    pool.Queue(tmp_path, "w1").claim(units(2), 2)
    assert pool.Queue(tmp_path, "w2").claim(units(4), 4) == units(4)[2:]


def test_a_claim_nobody_touched_for_the_lease_is_taken_under_the_next_epoch(tmp_path):
    dead = pool.Queue(tmp_path, "dead", lease=900)
    dead.claim(units(2), 2)
    age(dead, 901)
    survivor = pool.Queue(tmp_path, "survivor", lease=900)
    assert survivor.claim(units(2), 2) == units(2)
    names = sorted(os.listdir(tmp_path / Pool.DIR / Pool.CLAIMS / TASK))
    assert names == ["0-0.0", "0-0.1", "1-0.0", "1-0.1"]
    # And the takeover is itself a live claim
    assert pool.Queue(tmp_path, "third", lease=900).claim(units(2), 2) == []


def test_two_workers_finding_a_lease_expired_both_cannot_take_it(tmp_path):
    dead = pool.Queue(tmp_path, "dead", lease=900)
    dead.claim(units(1), 1)
    age(dead, 5000)
    taken = []
    barrier = threading.Barrier(6)

    def work(name):
        queue = pool.Queue(tmp_path, name, lease=900)
        barrier.wait()
        taken.extend(queue.claim(units(1), 1))

    threads = [threading.Thread(target=work, args=(f"w{i}",)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(taken) == 1


def test_a_released_unit_can_be_taken_at_once(tmp_path):
    first = pool.Queue(tmp_path, "w1")
    (unit,) = first.claim(units(1), 1)
    first.release(unit)
    assert pool.Queue(tmp_path, "w2").claim(units(1), 1) == [unit]


def test_a_finished_unit_is_no_longer_kept_alive_and_stop_releases_the_rest(tmp_path):
    queue = pool.Queue(tmp_path, "w1")
    done, open_ = queue.claim(units(2), 2)
    queue.finish(done)
    assert list(queue.held) == [open_]
    queue.stop()
    assert pool.Queue(tmp_path, "w2").claim(units(2), 2) == [open_]


def test_the_heartbeat_keeps_a_long_unit_claimed(tmp_path):
    queue = pool.Queue(tmp_path, "w1", lease=0.6, heartbeat=0.1)
    queue.claim(units(1), 1)
    queue.start()
    try:
        time.sleep(1.2)
        # Twice the lease on, and still nobody else may take it
        assert pool.Queue(tmp_path, "w2", lease=0.6).claim(units(1), 1) == []
    finally:
        queue.stop()


def test_a_worker_with_no_heartbeat_loses_its_claims_to_the_lease(tmp_path):
    queue = pool.Queue(tmp_path, "w1", lease=0.3)
    queue.claim(units(1), 1)
    time.sleep(0.5)
    assert pool.Queue(tmp_path, "w2", lease=0.3).claim(units(1), 1) == units(1)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def row(doc, repeat=0, error=None, seconds=1.0):
    return {"doc_id": doc, "repeat": repeat, "error": error, "seconds": seconds}


def test_what_is_done_is_what_the_graded_file_and_every_shard_hold(tmp_path):
    write_rows(pool.samples_file(tmp_path, TASK), [row(0), row(1)])
    write_rows(pool.shard_file(tmp_path, "w1", TASK), [row(2), row(3, error="boom")])
    write_rows(pool.shard_file(tmp_path, "w2", TASK), [row(3), row(4, error="boom")])
    standing = pool.Ledger(tmp_path).standing(TASK)
    assert standing.finished == {(0, 0), (1, 0), (2, 0), (3, 0)}
    assert standing.graded == {(0, 0), (1, 0)}
    # A failure in a shard counts against its unit, until another worker finishes it
    assert standing.failures == {(3, 0): 1, (4, 0): 1}
    assert not standing.exhausted((3, 0))


def test_a_unit_that_failed_as_often_as_allowed_is_out_of_attempts(tmp_path):
    for i in range(Pool.MAX_ATTEMPTS):
        write_rows(pool.shard_file(tmp_path, f"w{i}", TASK), [row(0, error="refused")])
    standing = pool.Ledger(tmp_path).standing(TASK)
    assert standing.exhausted((0, 0))
    assert not standing.exhausted((1, 0))


def test_an_error_row_in_the_graded_file_is_retried_as_a_single_job_would(tmp_path):
    write_rows(pool.samples_file(tmp_path, TASK), [row(0, error="boom")])
    standing = pool.Ledger(tmp_path).standing(TASK)
    assert standing.finished == set()
    assert not standing.failures


def test_the_ledger_reads_only_what_was_appended_and_ignores_a_line_still_being_written(tmp_path):
    shard = pool.shard_file(tmp_path, "w1", TASK)
    write_rows(shard, [row(0)])
    ledger = pool.Ledger(tmp_path)
    assert ledger.standing(TASK).finished == {(0, 0)}
    with shard.open("a") as out:
        out.write(json.dumps(row(1)) + "\n" + '{"doc_id": 2, "repe')
    assert ledger.standing(TASK).finished == {(0, 0), (1, 0)}
    with shard.open("a") as out:
        out.write('at": 0, "error": null}\n')
    assert ledger.standing(TASK).finished == {(0, 0), (1, 0), (2, 0)}


def test_an_exact_ledger_sees_rows_a_cached_stat_still_hides(tmp_path, monkeypatch):
    shard = pool.shard_file(tmp_path, "w1", TASK)
    write_rows(shard, [row(0)])
    loose, exact = pool.Ledger(tmp_path), pool.Ledger(tmp_path, exact=True)
    assert loose.standing(TASK).finished == exact.standing(TASK).finished == {(0, 0)}
    cached = shard.stat()
    with shard.open("a") as out:
        out.write(json.dumps(row(1)) + "\n")
    # This node's client still answers a stat with the size before another node's append
    real_stat = Path.stat
    monkeypatch.setattr(
        Path, "stat", lambda path, **kwargs: cached if path == shard else real_stat(path, **kwargs)
    )
    assert loose.standing(TASK).finished == {(0, 0)}
    assert exact.standing(TASK).finished == {(0, 0), (1, 0)}


def test_the_ledger_follows_a_graded_file_rewritten_whole(tmp_path):
    graded = pool.samples_file(tmp_path, TASK)
    write_rows(graded, [row(0), row(1), row(2)])
    ledger = pool.Ledger(tmp_path)
    assert len(ledger.standing(TASK).finished) == 3
    io.write_jsonl(graded, [row(0)])
    assert ledger.standing(TASK).finished == {(0, 0)}


def test_a_reader_skips_a_row_cut_in_the_middle(tmp_path):
    path = tmp_path / "cut.jsonl"
    path.write_text(json.dumps(row(0)) + "\n" + '{"doc_id": 1, "rep')
    assert pool.read_rows(path) == [row(0)]


def test_one_row_stands_for_a_unit_a_success_over_a_failure_and_the_later_of_two_alike(tmp_path):
    rows = [row(0, error="a"), row(0, error="b"), row(1), row(1, error="c"), row(2), row(2, seconds=9.0)]
    kept = {(r["doc_id"], r["repeat"]): r for r in pool.one_per_unit(rows)}
    assert kept[(0, 0)]["error"] == "b"
    assert kept[(1, 0)]["error"] is None
    assert kept[(2, 0)]["seconds"] == 9.0


def test_a_task_s_rows_are_the_graded_file_and_every_shard(tmp_path):
    write_rows(pool.samples_file(tmp_path, TASK), [row(0)])
    write_rows(pool.shard_file(tmp_path, "w1", TASK), [row(1)])
    write_rows(pool.shard_file(tmp_path, "w2", TASK), [row(2)])
    write_rows(pool.shard_file(tmp_path, "w2", "other"), [row(9)])
    assert [r["doc_id"] for r in pool.task_rows(tmp_path, TASK)] == [0, 1, 2]


def test_the_first_writer_of_a_file_stands(tmp_path):
    path = tmp_path / "pool" / "prior.json"
    assert pool.publish_once(path, {"seconds": 5})
    assert not pool.publish_once(path, {"seconds": 99})
    assert io.read_json(path) == {"seconds": 5}
    assert [p.name for p in path.parent.iterdir()] == ["prior.json"]


def test_a_worker_is_live_while_its_heartbeat_is_fresh_and_it_has_not_ended(tmp_path):
    now = time.time()
    pool.write_worker(tmp_path, "fresh", {"heartbeat": now - 10})
    pool.write_worker(tmp_path, "stale", {"heartbeat": now - 1000})
    pool.write_worker(tmp_path, "done", {"heartbeat": now - 1, "ended": True})
    assert pool.live_workers(tmp_path, 180, now) == ["fresh"]
