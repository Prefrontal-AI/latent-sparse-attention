"""The spans an extraction's stack is cut into, and the residual streams that cross a cut."""

import torch

from lsa.training.distillation.data.stage import (
    Boundary,
    BoundaryWriter,
    boundary_path,
    boundary_ready,
    remove_boundary,
)


def streams(length, start=0):
    return (torch.arange(length * 4 * 8, dtype=torch.float32).view(1, length, 4, 8) + start).bfloat16()


def test_streams_written_a_chunk_at_a_time_read_back_a_chunk_at_a_time(tmp_path):
    path = boundary_path(tmp_path, 16, "natural", "train", "doc")
    writer = BoundaryWriter(path)
    whole = torch.cat([streams(5), streams(7, start=1000)], dim=1)
    writer.append(whole[:, :5])
    writer.append(whole[:, 5:])
    assert not boundary_ready(path)

    writer.publish()
    assert boundary_ready(path)
    boundary = Boundary(path)
    assert boundary.length == 12
    assert torch.equal(boundary.read(0, 12), whole)
    assert torch.equal(boundary.read(3, 9), whole[:, 3:9])


def test_a_discarded_writer_leaves_nothing(tmp_path):
    path = boundary_path(tmp_path, 16, "natural", "train", "doc")
    writer = BoundaryWriter(path)
    writer.append(streams(4))
    writer.discard()
    assert not boundary_ready(path)
    assert list(path.parent.iterdir()) == []


def test_removed_streams_are_gone_and_no_longer_ready(tmp_path):
    path = boundary_path(tmp_path, 16, "natural", "train", "doc")
    writer = BoundaryWriter(path)
    writer.append(streams(4))
    writer.publish()

    remove_boundary(path)
    assert not boundary_ready(path)
    assert not path.exists()
