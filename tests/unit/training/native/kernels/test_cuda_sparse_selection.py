import pytest

from lsa.training.native.kernels import cuda_sparse

GROUP, HEAD_DIM = 8, 128


@pytest.mark.parametrize(
    ("tokens", "forward", "backward"),
    [
        (4096, "fwd_hard", "bwd_v14"),
        (16384, "fwd_hard", "bwd_v14"),
        (65536, "fwd_hard", "bwd_hybrid"),
        (65536 + 64, "fwd_r7", None),
        (131072, "fwd_r7", None),
    ],
)
def test_v2_reads_by_its_kernels_up_to_their_limit_and_by_the_plain_ones_past_it(tokens, forward, backward):
    assert cuda_sparse.forward_kernel(tokens, GROUP, HEAD_DIM, v2=True) == forward
    assert cuda_sparse.backward_kernel(tokens, GROUP, HEAD_DIM, grouped=True, v2=True) == backward


def test_v2_reads_by_the_plain_kernels_for_other_groups():
    assert cuda_sparse.forward_kernel(65536, 1, HEAD_DIM, v2=True) == "fwd_r7"
    assert cuda_sparse.backward_kernel(65536, 1, HEAD_DIM, grouped=True, v2=True) is None


def test_the_plain_read_never_takes_the_v2_kernels():
    assert cuda_sparse.forward_kernel(65536, GROUP, HEAD_DIM) == "fwd_r7"
    assert cuda_sparse.backward_kernel(65536, GROUP, HEAD_DIM, grouped=True) is None
