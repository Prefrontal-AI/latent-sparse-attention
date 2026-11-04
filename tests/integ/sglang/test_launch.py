"""What this repo hands the server - an environment, a model config override and a command line.

Nothing here starts one. What is worth testing is the handover, because everything past it is either
sglang's or the suite's and is tested there.
"""

import json
import shlex
import socket

import pytest

from lsa.sglang.constants import Banner, Server
from lsa.sglang.launch import (
    SglangConfig,
    container_command,
    derived_ports,
    free_port,
    launch_line,
    plugin_armed,
    serve,
    serve_environment,
    server_command,
    wait_ready,
    with_model_override,
    with_rotation,
    with_settings,
    write_settings,
)


def backend_at(root, **overrides):
    return SglangConfig(image=IMAGE, **overrides)


# A docker tag rather than a built squashfs - enroot caches the pull
IMAGE = "lmsysorg/sglang:latest"

MODEL = "zai-org/GLM-5.3-Flash"


def test_no_environment_variable_names_the_plugin(tmp_path):
    # sglang loads every installed entry point, so the allow-list is left unset on LSA, native and pipeline runs alike
    assert "SGLANG_PLUGINS" not in serve_environment()
    assert "SGLANG_PLUGINS" not in serve_environment(pipeline_stages=2)


def test_a_server_spanning_nodes_allocates_in_expanding_segments():
    assert serve_environment(pipeline_stages=2)["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in serve_environment(pipeline_stages=1)


def test_a_configuration_s_environment_is_exported_over_the_launcher_s():
    environment = serve_environment(environment={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    overridden = serve_environment(environment={"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "0"})
    assert overridden["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] == "0"


def test_the_environment_keeps_what_was_already_exported(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/caches/hf")
    assert serve_environment()["HF_HOME"] == "/caches/hf"


def test_the_server_command_names_the_model_and_the_shape(tmp_path):
    argv = server_command(backend_at(tmp_path, tp=8, port=61248), MODEL)
    assert argv[argv.index("--model-path") + 1] == MODEL
    assert argv[argv.index("--tp") + 1] == "8"
    assert argv[argv.index("--port") + 1] == "61248"


def test_the_list_width_never_travels_as_an_argument(tmp_path):
    # The plugin derives it from the design and sets it on the text config the model reads
    # => no argument carries it
    assert "--json-model-override-args" not in server_command(backend_at(tmp_path), MODEL)


def test_a_card_rotation_reaches_the_server_in_the_nested_text_config():
    rope = {"rope_type": "yarn", "factor": 40.0, "original_max_position_embeddings": 262144}
    argv = with_rotation(["--chunked-prefill-size", "8192"], rope)
    assert argv[:2] == ["--chunked-prefill-size", "8192"]
    assert argv[2] == Server.MODEL_OVERRIDE_FLAG
    assert json.loads(argv[3]) == {"text_config": {"rope_parameters": rope}}


def test_no_rotation_leaves_the_arguments_as_they_are():
    assert with_rotation(["--tp", "1"], None) == ["--tp", "1"]
    assert with_rotation(["--tp", "1"], {}) == ["--tp", "1"]


def test_a_run_restating_the_rotation_is_refused():
    own = json.dumps({"text_config": {"rope_parameters": {"factor": 1.0}}})
    with pytest.raises(ValueError, match="text_config.rope_parameters.factor"):
        with_rotation([Server.MODEL_OVERRIDE_FLAG, own], {"factor": 40.0})


def test_a_run_s_own_override_and_the_rotation_travel_as_one_json():
    own = json.dumps({"text_config": {"index_topk": 512}})
    argv = with_rotation(["--tp", "1", Server.MODEL_OVERRIDE_FLAG, own], {"factor": 40.0})
    assert argv.count(Server.MODEL_OVERRIDE_FLAG) == 1
    assert argv[:2] == ["--tp", "1"]
    assert json.loads(argv[argv.index(Server.MODEL_OVERRIDE_FLAG) + 1]) == {
        "text_config": {"index_topk": 512, "rope_parameters": {"factor": 40.0}}
    }


def test_an_override_given_with_an_equals_sign_is_merged_too():
    argv = with_model_override([f"{Server.MODEL_OVERRIDE_FLAG}={json.dumps({'a': 1})}"], {"b": 2})
    assert argv == [Server.MODEL_OVERRIDE_FLAG, json.dumps({"a": 1, "b": 2})]


@pytest.mark.parametrize("own", ["not json", "[1, 2]", ""])
def test_an_override_that_is_not_an_object_is_refused_with_what_arrived(own):
    with pytest.raises(ValueError, match="json object"):
        with_model_override([Server.MODEL_OVERRIDE_FLAG, own], {"b": 2})


def test_the_settings_block_reaches_the_server_in_the_model_config_override():
    body = {
        "top_k": 2048,
        "projections": "/maps/x.pt",
        "early_budget": {"position": 1, "top_k": 8, "top_p": 4},
    }
    argv = with_settings(["--tp", "8"], body)
    assert argv[:2] == ["--tp", "8"]
    assert argv[2] == Server.MODEL_OVERRIDE_FLAG
    # The projection file stays the path it is, and nothing else rides along
    assert json.loads(argv[3]) == {"lsa": body}


def test_the_settings_block_and_the_rotation_share_one_override_with_the_nested_text_config_intact():
    rope = {"rope_type": "yarn", "factor": 40.0}
    argv = with_settings(with_rotation([], rope), {"top_k": 2048})
    assert argv.count(Server.MODEL_OVERRIDE_FLAG) == 1
    assert json.loads(argv[1]) == {"text_config": {"rope_parameters": rope}, "lsa": {"top_k": 2048}}


def test_a_run_may_add_its_debug_sub_block_to_the_settings_block():
    own = json.dumps({"lsa": {"debug": {"check_index": True}}})
    argv = with_settings([Server.MODEL_OVERRIDE_FLAG, own], {"top_k": 2048})
    assert json.loads(argv[1]) == {"lsa": {"debug": {"check_index": True}, "top_k": 2048}}


def test_a_run_setting_a_key_of_the_block_to_another_value_is_refused():
    own = json.dumps({"lsa": {"top_k": 1024}})
    with pytest.raises(ValueError, match="lsa.top_k"):
        with_settings([Server.MODEL_OVERRIDE_FLAG, own], {"top_k": 2048})


def test_a_run_s_serve_mapping_may_carry_the_model_override_and_it_merges_with_the_settings_block():
    from lsa.evaluation.config import serve_argv

    own = json.dumps({"lsa": {"debug": {"check_index": True}}})
    argv = with_settings(serve_argv({"tp-size": 8, "json-model-override-args": own}), {"top_k": 2048})
    assert argv.count(Server.MODEL_OVERRIDE_FLAG) == 1
    assert json.loads(argv[argv.index(Server.MODEL_OVERRIDE_FLAG) + 1]) == {
        "lsa": {"debug": {"check_index": True}, "top_k": 2048}
    }


def test_serve_args_reach_the_server_verbatim(tmp_path):
    backend = backend_at(tmp_path, serve_args=["--attention-backend", "dsa", "--disable-radix-cache"])
    argv = server_command(backend, None)
    assert argv[-3:] == ["--attention-backend", "dsa", "--disable-radix-cache"]


def test_the_srun_step_asks_for_its_gpus_and_overlaps_the_launcher(tmp_path):
    """The launcher holds the allocation, so a step that does not overlap it never starts at all"""
    argv = container_command(backend_at(tmp_path, runner="srun", tp=8), "echo hi")

    assert "--gres=gpu:8" in argv
    assert "--overlap" in argv


def test_one_node_serves_without_a_pipeline(tmp_path):
    argv = server_command(backend_at(tmp_path), MODEL)
    assert "--pp-size" not in argv
    assert "--node-rank" not in launch_line(backend_at(tmp_path), MODEL)
    assert "--nodes=1" in container_command(backend_at(tmp_path), "echo hi")


def test_a_pipeline_meets_at_the_launcher_s_host(tmp_path, monkeypatch):
    monkeypatch.setattr("socket.gethostname", lambda: "node-11")
    argv = server_command(backend_at(tmp_path, pp=2), MODEL)
    assert argv[argv.index("--pp-size") + 1] == "2"
    assert argv[argv.index("--nnodes") + 1] == "2"
    assert argv[argv.index("--dist-init-addr") + 1] == "node-11:29548"


def test_each_node_reads_its_own_rank(tmp_path):
    """Every node runs one line, so a rank written into it would give both nodes the same one"""
    line = launch_line(backend_at(tmp_path, pp=2), MODEL)
    assert line.endswith('--node-rank "$SLURM_NODEID"')


def test_a_pipeline_step_runs_one_task_on_every_node(tmp_path):
    argv = container_command(backend_at(tmp_path, pp=2, tp=8), "echo hi")
    assert {"--nodes=2", "--ntasks=2", "--ntasks-per-node=1", "--gres=gpu:8"} <= set(argv)


def test_a_context_past_the_checkpoint_s_own_is_allowed(tmp_path):
    assert serve_environment()["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] == "1"


def test_a_launch_with_no_image_refuses(tmp_path):
    from lsa import site

    with pytest.raises(site.NotNamed, match="image is not named.*configs/site"):
        container_command(SglangConfig(image=""), "true")
    with pytest.raises(TypeError, match="image"):
        SglangConfig()


def test_a_server_that_never_armed_is_not_ready(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("some sglang output\nno banner here\n")
    assert not plugin_armed(log)


def test_armed_alone_is_not_ready(tmp_path):
    log = tmp_path / "server.log"
    # The hooks registered and the model still asked for another class - this is what the first run did
    log.write_text(f"loading\n{Banner.READY} lsa-top_level0 width 2560\nserving\n")
    assert not plugin_armed(log)


def test_both_banners_mean_lsa_was_reached(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(f"{Banner.READY} lsa-top_level0 width 2560\n{Banner.BUILT} layer 3\n")
    assert plugin_armed(log)


def test_a_missing_log_has_not_armed(tmp_path):
    assert not plugin_armed(tmp_path / "absent.log")


def test_settings_are_written_where_they_were_asked_for(tmp_path):
    path = write_settings({"top_k": 2048}, tmp_path / "runs" / "indexer.json")
    assert json.loads(path.read_text()) == {"top_k": 2048}


class FakeProcess:
    """A process that is still running, which is all `wait_ready` asks of it."""

    returncode = None

    def poll(self):
        return None


def ready_at(monkeypatch, answered):
    """Make the port healthy and have it report this checkpoint."""
    monkeypatch.setattr("lsa.sglang.launch.healthy", lambda _: True)
    monkeypatch.setattr("lsa.sglang.launch.serving_checkpoint", lambda _: answered)


def test_a_port_held_by_another_runs_server_is_refused_rather_than_scored(tmp_path, monkeypatch):
    """`port` defaults to one number, so two runs on a node answer at the same place.

    An LSA configuration escapes this by accident, because the banner check reads its own log and fails.
    Two native configurations do not, and would quietly score one server twice under two names - so the
    checkpoint is compared.
    """
    ready_at(monkeypatch, "/srv/other/Qwen3.5-9B")
    with pytest.raises(RuntimeError, match="rather than"):
        wait_ready(FakeProcess(), backend_at(tmp_path), tmp_path / "server.log", model=MODEL)


def test_the_right_checkpoint_on_the_port_is_ready(tmp_path, monkeypatch):
    ready_at(monkeypatch, MODEL)
    wait_ready(FakeProcess(), backend_at(tmp_path), tmp_path / "server.log", model=MODEL)


def test_a_trailing_slash_is_not_a_different_checkpoint(tmp_path, monkeypatch):
    ready_at(monkeypatch, MODEL + "/")
    wait_ready(FakeProcess(), backend_at(tmp_path), tmp_path / "server.log", model=MODEL)


def test_a_server_that_reports_no_checkpoint_does_not_block_the_run(tmp_path, monkeypatch):
    """The check can only refuse on a positive disagreement - an older server reporting nothing is served."""
    ready_at(monkeypatch, "")
    wait_ready(FakeProcess(), backend_at(tmp_path), tmp_path / "server.log", model=MODEL)


def test_no_checkpoint_asked_for_means_no_comparison(tmp_path, monkeypatch):
    ready_at(monkeypatch, "/srv/other/Qwen3.5-9B")
    wait_ready(FakeProcess(), backend_at(tmp_path), tmp_path / "server.log")


def test_the_banner_check_still_applies_to_the_right_checkpoint(tmp_path, monkeypatch):
    ready_at(monkeypatch, MODEL)
    log = tmp_path / "server.log"
    log.write_text("no banner here\n")
    with pytest.raises(RuntimeError, match="hooks did not apply"):
        wait_ready(FakeProcess(), backend_at(tmp_path), log, armed=True, model=MODEL)


@pytest.mark.parametrize(
    ("field", "value"), [("runner", "podman"), ("tp", 0), ("startup_minutes", 0), ("port", -1)]
)
def test_launch_block_refuses_a_value_the_server_has_no_meaning_for(field, value):
    with pytest.raises(ValueError):
        SglangConfig(image=IMAGE, **{field: value})


def test_launch_block_carries_no_part_of_the_design():
    with pytest.raises(TypeError, match="index_dim"):
        SglangConfig(image=IMAGE, index_dim=96)


def test_a_pipeline_across_nodes_needs_srun():
    with pytest.raises(ValueError, match="srun"):
        SglangConfig(image=IMAGE, pp=2, runner="docker")
    with pytest.raises(ValueError, match="pp"):
        SglangConfig(image=IMAGE, pp=0)


def test_an_image_the_run_names_is_kept_under_context_parallel():
    assert SglangConfig(image=IMAGE, serve_args=["--dcp-size", "8"]).image == IMAGE


def started_with(tmp_path, monkeypatch, backend, settings=None):
    """Serve `backend` against a fake process, and return the url and the shell line it was started with."""
    seen = {}

    class Started(FakeProcess):
        def __init__(self, argv, **_):
            seen["argv"] = argv

    monkeypatch.setattr("lsa.sglang.launch.subprocess.Popen", Started)
    monkeypatch.setattr("lsa.sglang.launch.stop", lambda _: None)
    ready_at(monkeypatch, MODEL)
    log = tmp_path / "server.log"
    # A server that runs the plugin has printed both banners by the time it answers
    monkeypatch.setattr("lsa.sglang.launch.plugin_armed", lambda _: True)
    with serve(backend, MODEL, settings, log) as url:
        pass
    return url, seen["argv"][-1]


def test_a_server_run_under_settings_is_started_with_them_in_its_model_config(tmp_path, monkeypatch):
    path = write_settings({"top_k": 2048, "projections": "/maps/x.pt"}, tmp_path / "indexer.json")
    backend = backend_at(tmp_path, port=61248, serve_args=with_rotation([], {"factor": 40.0}))
    _, line = started_with(tmp_path, monkeypatch, backend, path)
    override = shlex.split(line.split(" && exec ")[1])
    assert override.count(Server.MODEL_OVERRIDE_FLAG) == 1
    assert json.loads(override[override.index(Server.MODEL_OVERRIDE_FLAG) + 1]) == {
        "text_config": {"rope_parameters": {"factor": 40.0}},
        "lsa": {"top_k": 2048, "projections": "/maps/x.pt"},
    }


def test_a_native_server_is_started_without_a_model_config_override(tmp_path, monkeypatch):
    _, line = started_with(tmp_path, monkeypatch, backend_at(tmp_path, port=61248))
    assert Server.MODEL_OVERRIDE_FLAG not in line


def test_the_default_port_is_a_free_one_resolved_when_the_server_starts(tmp_path, monkeypatch):
    assert SglangConfig(image=IMAGE).port == 0
    monkeypatch.setattr("lsa.sglang.launch.free_port", lambda avoid=(): 54321)
    url, line = started_with(tmp_path, monkeypatch, backend_at(tmp_path))
    assert url == "http://127.0.0.1:54321"
    assert "--port 54321" in line


def test_a_named_port_is_used_as_given(tmp_path, monkeypatch):
    monkeypatch.setattr("lsa.sglang.launch.free_port", lambda avoid=(): 54321)
    url, line = started_with(tmp_path, monkeypatch, backend_at(tmp_path, port=61300))
    assert url == "http://127.0.0.1:61300"
    assert "--port 61300" in line


def test_port_zero_is_a_free_port_the_server_is_started_on_and_the_url_names(tmp_path, monkeypatch):
    """Workers packed on one node share the default port, so 0 is resolved when the server starts."""
    assert SglangConfig(image=IMAGE, port=0).port == 0
    monkeypatch.setattr("lsa.sglang.launch.free_port", lambda avoid=(): 54321)
    url, line = started_with(tmp_path, monkeypatch, backend_at(tmp_path, port=0))
    assert url == "http://127.0.0.1:54321"
    assert "--port 54321" in line


def test_a_server_on_one_node_takes_a_rendezvous_port_clear_of_its_own(tmp_path, monkeypatch):
    """sglang draws torch's rendezvous port from the ephemeral range, where a server beside it took it (234831)."""
    url, line = started_with(tmp_path, monkeypatch, backend_at(tmp_path, port=0))
    port = int(url.rsplit(":", 1)[1])
    nccl = int(line.split("--nccl-port ")[1].split()[0])
    assert nccl not in derived_ports(port)
    assert port + 233 in derived_ports(port)


@pytest.mark.parametrize("named", [["--nccl-port", "41000"], ["--nccl-port=41000"]])
def test_a_rendezvous_port_the_run_names_is_kept(tmp_path, monkeypatch, named):
    _, line = started_with(tmp_path, monkeypatch, backend_at(tmp_path, port=0, serve_args=named))
    assert line.count("--nccl-port") == 1
    assert "41000" in line


def test_a_free_port_is_bindable_and_outside_the_range_the_kernel_hands_out(tmp_path, monkeypatch):
    ephemeral = tmp_path / "ip_local_port_range"
    ephemeral.write_text("32768\t60999\n")
    monkeypatch.setattr("lsa.sglang.launch.EPHEMERAL_RANGE", ephemeral)
    for _ in range(20):
        port = free_port()
        assert 20000 <= port < 32768
        with socket.socket() as again:
            again.bind(("", port))


def test_a_free_port_skips_one_already_bound_or_avoided(monkeypatch):
    with socket.socket() as held:
        held.bind(("", 0))
        taken = held.getsockname()[1]
        draws = iter([taken, 23456, 23457])
        monkeypatch.setattr("lsa.sglang.launch.random.randrange", lambda *_: next(draws))
        assert free_port(avoid={23456}) == 23457


def test_the_container_mounts_the_root_the_checkout_and_the_sites_mounts(tmp_path, monkeypatch):
    from lsa import paths, site
    from lsa.sglang.launch import checkout_root, mount_list

    users = tmp_path / "users" / "ada"
    users.mkdir(parents=True)
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "here.yaml").write_text(
        f"root: {root}\nmounts:\n  - {tmp_path}/users/$USER\n  - {tmp_path}/missing\n"
    )
    monkeypatch.setattr(site, "SITES", tmp_path)
    monkeypatch.setattr(site, "DEFAULT_SITE", "here")
    monkeypatch.setattr(site, "_selected", None)
    monkeypatch.setenv("USER", "ada")
    monkeypatch.delenv(paths.ROOT_VARIABLE, raising=False)
    monkeypatch.delenv("LSA_EXTRA_MOUNTS", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "no_hub"))
    checkout = str(checkout_root())
    assert mount_list() == [f"{users}:{users}", f"{root}:{root}", f"{checkout}:{checkout}"]


def test_a_site_without_mounts_mounts_only_the_root_and_the_checkout(tmp_path, monkeypatch):
    from lsa import paths, site
    from lsa.sglang.launch import checkout_root, mount_list

    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "here.yaml").write_text(f"root: {root}\n")
    monkeypatch.setattr(site, "SITES", tmp_path)
    monkeypatch.setattr(site, "DEFAULT_SITE", "here")
    monkeypatch.setattr(site, "_selected", None)
    monkeypatch.delenv(paths.ROOT_VARIABLE, raising=False)
    monkeypatch.delenv("LSA_EXTRA_MOUNTS", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "no_hub"))
    checkout = str(checkout_root())
    assert mount_list() == [f"{root}:{root}", f"{checkout}:{checkout}"]


def test_the_container_mounts_the_sites_hub_cache_at_its_own_path(tmp_path, monkeypatch):
    from lsa import paths, site
    from lsa.sglang.launch import checkout_root, mount_list

    root = tmp_path / "root"
    root.mkdir()
    hub = tmp_path / "hub_cache"
    hub.mkdir()
    (tmp_path / "here.yaml").write_text(f"root: {root}\nhf_home: {hub}\nmounts: []\n")
    monkeypatch.setattr(site, "SITES", tmp_path)
    monkeypatch.setattr(site, "DEFAULT_SITE", "here")
    monkeypatch.setattr(site, "_selected", None)
    monkeypatch.delenv(paths.ROOT_VARIABLE, raising=False)
    monkeypatch.delenv("LSA_EXTRA_MOUNTS", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    checkout = str(checkout_root())
    assert mount_list() == [f"{hub}:{hub}", f"{root}:{root}", f"{checkout}:{checkout}"]


def site_with_hub(tmp_path, monkeypatch, mounts=""):
    """Point the site at a root and a hub cache that both exist, and return them."""
    from lsa import paths, site

    root = tmp_path / "root"
    root.mkdir()
    hub = tmp_path / "site_hub"
    hub.mkdir()
    (tmp_path / "here.yaml").write_text(f"root: {root}\nhf_home: {hub}\nmounts: [{mounts}]\n")
    monkeypatch.setattr(site, "SITES", tmp_path)
    monkeypatch.setattr(site, "DEFAULT_SITE", "here")
    monkeypatch.setattr(site, "_selected", None)
    monkeypatch.delenv(paths.ROOT_VARIABLE, raising=False)
    monkeypatch.delenv("LSA_EXTRA_MOUNTS", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    return root, hub


def test_the_hub_cache_the_environment_names_beats_the_sites(tmp_path, monkeypatch):
    from lsa.sglang.launch import checkout_root, mount_list

    root, site_hub = site_with_hub(tmp_path, monkeypatch)
    env_hub = tmp_path / "env_hub"
    env_hub.mkdir()
    monkeypatch.setenv("HF_HOME", str(env_hub))
    checkout = str(checkout_root())
    # Both caches exist, so only the precedence decides which one is mounted
    assert mount_list() == [f"{env_hub}:{env_hub}", f"{root}:{root}", f"{checkout}:{checkout}"]
    assert f"{site_hub}:{site_hub}" not in mount_list()


def test_a_hub_cache_the_site_also_lists_is_mounted_once(tmp_path, monkeypatch):
    from lsa.sglang.launch import checkout_root, mount_list

    root, hub = site_with_hub(tmp_path, monkeypatch)
    (tmp_path / "here.yaml").write_text(f"root: {root}\nhf_home: {hub}\nmounts:\n  - {hub}\n")
    checkout = str(checkout_root())
    assert mount_list() == [f"{hub}:{hub}", f"{root}:{root}", f"{checkout}:{checkout}"]


def test_a_hub_cache_in_the_extra_mounts_is_mounted_once(tmp_path, monkeypatch):
    from lsa.sglang.launch import checkout_root, mount_list

    root, hub = site_with_hub(tmp_path, monkeypatch)
    monkeypatch.setenv("LSA_EXTRA_MOUNTS", f"{hub},{hub}")
    checkout = str(checkout_root())
    assert mount_list() == [f"{hub}:{hub}", f"{root}:{root}", f"{checkout}:{checkout}"]


def test_a_trailing_slash_does_not_mount_a_directory_twice(tmp_path, monkeypatch):
    from lsa.sglang.launch import checkout_root, mount_list

    root, hub = site_with_hub(tmp_path, monkeypatch, mounts=f"{tmp_path}/site_hub/")
    monkeypatch.setenv("HF_HOME", f"{hub}//")
    monkeypatch.setenv("LSA_EXTRA_MOUNTS", f" {hub}/ ")
    checkout = str(checkout_root())
    assert mount_list() == [f"{hub}:{hub}", f"{root}:{root}", f"{checkout}:{checkout}"]


def test_a_link_is_mounted_as_named_and_not_as_resolved(tmp_path, monkeypatch):
    from lsa.sglang.launch import mount_list

    _, hub = site_with_hub(tmp_path, monkeypatch)
    link = tmp_path / "link"
    link.symlink_to(hub)
    monkeypatch.setenv("LSA_EXTRA_MOUNTS", f"{link}/")
    assert f"{link}:{link}" in mount_list()
    assert f"{hub}:{hub}" in mount_list()
