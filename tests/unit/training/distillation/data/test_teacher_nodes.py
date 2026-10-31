"""The staging between the teacher's nodes - requests in order, one reader a document, and room given back."""

import torch

from lsa.training.distillation.data.teacher_nodes import Request, Staging


def a_staging(tmp_path, node=0, budget=1e12):
    return Staging(tmp_path / "staging" / "1-0", node, 2, budget)


def test_a_serving_node_reads_the_trainers_requests_in_order_and_each_once(tmp_path):
    trainer, server = a_staging(tmp_path), a_staging(tmp_path, node=1)
    assert server.requests() == []
    trainer.open()

    trainer.publish([Request(0, "c", "a", [3, 7]), Request(1, "c", "b", [3])])
    first = server.requests()
    trainer.publish([Request(2, "c", "a", [7])])

    assert [request.name for request in first] == ["a", "b"]
    assert server.requests() == [Request(2, "c", "a", [7])]
    assert server.requests() == []


def test_a_document_is_claimed_by_one_node_only(tmp_path):
    trainer, server = a_staging(tmp_path), a_staging(tmp_path, node=1)
    trainer.open()

    assert server.claim(4)
    assert not trainer.claim(4)
    assert trainer.claim(5)


def test_a_staged_document_is_whole_once_marked_and_gone_once_every_layer_is_taken(tmp_path):
    trainer, server = a_staging(tmp_path), a_staging(tmp_path, node=1)
    trainer.open()
    blobs = {layer: {"top_key_ids": torch.arange(4) + layer, "meta": {"layer": layer}} for layer in (3, 7)}

    assert not trainer.ready(0)
    server.put(0, blobs)
    assert trainer.ready(0)
    assert server.footprint() > 0

    taken = trainer.take(0, 7)
    assert torch.equal(taken["top_key_ids"], blobs[7]["top_key_ids"]) and taken["meta"] == {"layer": 7}
    assert trainer.ready(0)
    trainer.take(0, 3)
    assert trainer.ready(0)
    trainer.release(0)
    assert not trainer.ready(0)
    assert server.footprint() == 0


def test_a_serving_node_stops_claiming_past_its_budget_but_never_with_nothing_waiting(tmp_path):
    trainer, server = a_staging(tmp_path), a_staging(tmp_path, node=1, budget=1)
    trainer.open()

    assert server.has_room()
    server.put(0, {3: {"top_key_ids": torch.arange(64)}})
    assert not server.has_room()
    trainer.take(0, 3)
    assert server.has_room()


def test_a_launch_starts_clean_and_the_trainer_says_when_it_is_done(tmp_path):
    stale = Staging(tmp_path / "staging" / "1-0", 0, 2, 1e12)
    stale.open()
    stale.publish([Request(0, "c", "a", [3])])
    requeued = Staging(tmp_path / "staging" / "1-1", 0, 2, 1e12)

    requeued.open()

    assert not stale.root.exists()
    assert not requeued.finished()
    requeued.finish()
    assert requeued.finished()


def test_a_job_of_one_node_has_no_staging(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "1")
    assert Staging.of_job(tmp_path, 400) is None

    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    staging = Staging.of_job(tmp_path, 400)

    assert staging is not None and staging.serving
    assert staging.root == tmp_path / "teacher_staging" / "77-2"
    assert staging.budget == 400e9
