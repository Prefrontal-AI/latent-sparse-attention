from pathlib import Path

from lsa.evaluation.performance.production import waves


def stamp(second):
    return f"2026-10-18 00:{second // 60:02d}:{second % 60:02d}"


def decode(second, running, ranks="TP0"):
    return f"[{stamp(second)} {ranks}] Decode batch, #running-req: {running}, #full token: 0, #queue-req: 0"


def flush(second, ranks=("TP0",)):
    return [f"[{stamp(second)} {rank}] Reset HybridReqToTokenPool" for rank in ranks]


def write(tmp_path, lines) -> Path:
    path = tmp_path / "server.log"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_warmup_before_a_flush_is_not_counted(tmp_path):
    # A warmup of two requests, then the flush, then one measured request
    log = write(tmp_path, [decode(1, 2), decode(2, 2), *flush(3), decode(4, 1), decode(5, 1)])
    assert waves.wave_peaks(log, [3.0]) == [1]


def test_a_full_wave_peaks_at_the_concurrency(tmp_path):
    log = write(tmp_path, [*flush(0), decode(1, 3), decode(2, 4), decode(3, 4), decode(4, 2)])
    assert waves.wave_peaks(log, [4.0]) == [4]


def test_a_short_wave_peaks_below_the_concurrency(tmp_path):
    log = write(tmp_path, [*flush(0), decode(1, 3), decode(2, 3), decode(3, 1)])
    assert waves.wave_peaks(log, [3.0]) == [3]


def test_the_next_contexts_warmup_is_outside_a_window(tmp_path):
    log = write(tmp_path, [*flush(0), decode(1, 1), decode(20, 5), *flush(40), decode(41, 1)])
    assert waves.wave_peaks(log, [5.0, 5.0]) == [1, 1]


def test_each_context_takes_its_own_window_in_the_order_they_ran(tmp_path):
    log = write(tmp_path, [*flush(0), decode(1, 2), *flush(30), decode(31, 4)])
    assert waves.wave_peaks(log, [10.0, 10.0]) == [2, 4]


def test_ranks_count_the_highest_any_reports_at_a_second(tmp_path):
    lines = [
        *flush(0, ("TP0", "TP1")),
        decode(1, 3, "ATTN_CP0 TP0"),
        decode(1, 4, "ATTN_CP1 TP1"),
        decode(2, 2, "PP0 TP0"),
    ]
    assert waves.wave_peaks(write(tmp_path, lines), [3.0]) == [4]


def test_a_flush_logged_by_every_rank_is_one_flush(tmp_path):
    log = write(tmp_path, [*flush(0, ("TP0", "TP1", "TP2", "TP3")), decode(1, 1)])
    assert waves.read_log(log)[0] and len(waves.read_log(log)[0]) == 1


def test_a_different_number_of_flushes_than_contexts_is_none(tmp_path):
    log = write(tmp_path, [*flush(0), decode(1, 1)])
    assert waves.wave_peaks(log, [3.0, 3.0]) == [None, None]


def test_a_flush_before_the_previous_context_could_finish_is_none(tmp_path):
    log = write(tmp_path, [*flush(0), decode(1, 1), *flush(5), decode(6, 1)])
    assert waves.wave_peaks(log, [60.0, 3.0]) == [None, None]


def test_no_log_and_no_markers_and_no_decode_line_are_none(tmp_path):
    assert waves.wave_peaks(tmp_path / "missing.log", [3.0]) == [None]
    assert waves.wave_peaks(write(tmp_path, [decode(1, 1)]), [3.0]) == [None]
    assert waves.wave_peaks(write(tmp_path, [*flush(0)]), [3.0]) == [None]
