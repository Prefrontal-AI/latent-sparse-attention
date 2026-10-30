"""How a record names the mesh a server ran on - every axis, at every size, whatever the server reports."""

from lsa.evaluation.accuracy.records import mesh, mesh_label_of

# What the fork reports serving GLM-5.3-Flash under context parallel on one node
FORK_CP = {
    "tp_size": 8,
    "attn_cp_size": 8,
    "dcp_size": 8,
    "enable_prefill_cp": True,
    "cp_strategy": "contiguous",
    "pp_size": 1,
    "ep_size": 8,
    "dp_size": 1,
    "enable_dp_attention": False,
    "nnodes": 1,
    "moe_runner_backend": "deep_gemm",
}


def test_a_context_parallel_and_a_tensor_parallel_mesh_of_one_size_are_told_apart():
    context_parallel = mesh(FORK_CP, "fork.img")
    tensor_parallel = mesh({"tp_size": 8, "pp_size": 1, "ep_size": 8, "nnodes": 1}, "stock.img")
    assert context_parallel["label"] == "tp1-cp8-dcp8-pp1-ep8"
    assert tensor_parallel["label"] == "tp8-cp1-dcp1-pp1-ep8"
    assert context_parallel["tp_size"] == tensor_parallel["tp_size"] == 8


def test_attention_data_parallel_takes_its_share_of_attention_s_width():
    assert mesh({"tp_size": 8, "dp_size": 2, "enable_dp_attention": True}, None)["attn_tp_size"] == 4
    assert mesh({"tp_size": 8, "dp_size": 2}, None)["attn_tp_size"] == 8


def test_a_server_is_named_by_its_recorded_mesh():
    assert mesh_label_of({"tp_size": 8, "mesh": mesh(FORK_CP, None)}) == "tp1-cp8-dcp8-pp1-ep8"
