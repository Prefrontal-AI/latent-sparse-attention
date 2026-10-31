import pytest
import torch

from lsa.training.native.kernels import pisa
from lsa.training.native.kernels.nsa import autograd as kernels
from tests.helpers.training.kernels.inputs import attention_inputs, relative_error, run_reference

pytestmark = [pytest.mark.gpu, pytest.mark.cluster]


@pytest.mark.parametrize("shape", [(1, 256, 8, 1, 128), (2, 257, 4, 2, 64)])
def test_compressed_and_selected_reads_match_reference_training_code_with_gradients(shape):
    q, k, v, _ = attention_inputs(*shape, 0)
    block, stride, selected_size, selected_count = 32, 16, 32, 4
    scale = shape[-1] ** -0.5
    compressed_k = k[:, :15].detach().contiguous().requires_grad_()
    compressed_v = v[:, :15].detach().contiguous().requires_grad_()
    args = [q, compressed_k, compressed_v, block, stride, scale]
    gradient = torch.randn_like(q)
    expected, gradients = run_reference(
        "nsa.autograd.compressed_attention", args, leaves=(0, 1, 2), grad_output=gradient
    )
    output, lse = kernels.compressed_attention(*args)
    actual = torch.autograd.grad(output, (q, compressed_k, compressed_v), gradient)
    for got, want in zip((output, lse, *actual), (*expected, *gradients), strict=True):
        assert relative_error(got, want) < 1e-3
    indices = kernels.select_blocks(q, compressed_k, lse, block, stride, selected_size, selected_count, scale)
    expected, _ = run_reference(
        "nsa.autograd.select_blocks",
        [q, compressed_k, lse, block, stride, selected_size, selected_count, scale, 0, shape[1]],
    )
    assert torch.equal(indices.cpu(), expected[0])
    args = [q, k, v, indices, selected_size, scale]
    expected, gradients = run_reference(
        "nsa.autograd.selected_attention", args, leaves=(0, 1, 2), grad_output=gradient
    )
    output, lse = kernels.selected_attention(*args)
    actual = torch.autograd.grad(output, (q, k, v), gradient)
    for got, want in zip((output, lse, *actual), (*expected, *gradients), strict=True):
        assert relative_error(got, want) < 1e-3


@pytest.mark.parametrize("length,local_blocks", [(257, 2), (1024, 8)])
def test_pisa_selection_and_read_match_reference_training_code(length, local_blocks):
    q, k, v, _ = attention_inputs(1, length, 8, 1, 128, 2)
    args = [q, k, 32, 2, 12, 128**-0.5, 0, local_blocks]
    indices = pisa.select_blocks(*args)
    expected, _ = run_reference("pisa.select_blocks", args)
    assert torch.equal(indices.cpu(), expected[0])
    gradient = torch.randn_like(q)
    args = [q, k, v, indices, 32, 128**-0.5]
    expected, gradients = run_reference(
        "nsa.autograd.selected_attention", args, leaves=(0, 1, 2), grad_output=gradient
    )
    output, lse = kernels.selected_attention(*args)
    actual = torch.autograd.grad(output, (q, k, v), gradient)
    for got, want in zip((output, lse, *actual), (*expected, *gradients), strict=True):
        assert relative_error(got, want) < 1e-3
