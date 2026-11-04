"""Start a served model, wait until it is ready, and stop it - what every served experiment shares.

The server is a stock sglang in a pinned container image with this package installed beside it, so what
this module builds is a container invocation and a command line, and the plugin is armed by a block of the model
config that the command line carries.
It runs on the job host rather than in the container, so it imports no sglang.

What runs against the server is the experiment's own - `lsa.evaluation.accuracy.job` scores it, the production
ladder times it. This module owns the steps around that - the `sglang:` block that says how a server starts,
writing the [settings](settings/design.py) where the record can quote them, starting the server, and waiting
until it says LSA is armed.

A healthy port is not enough to start scoring. sglang's plugin loader logs a failed hook rather than raising
it, so a server whose hooks did not apply would answer `/health` and quietly run the checkpoint's own
indexer under an LSA label. `wait_ready` requires the plugin's own banner in the log before it returns.

The server and everything it starts go down together: it is launched in a session of its own, and the
session is signalled on the way out whether the suite finished or raised. A server torn down by its pid
alone leaves tensor-parallel workers holding the GPUs, and the next experiment then has nowhere to load.
"""

import json
import os
import random
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from lsa import env, io, paths, site
from lsa.sglang.constants import Banner, Plugin, Server


@dataclass(frozen=True)
class SglangConfig:
    """Where the server runs and how it starts.

    The server is a stock sglang in a container image, with `lsa.sglang` installed beside it as a plugin.
    So what this block carries is the image and the server's own command line, which is this repository's to
    own now that no flag of it belongs elsewhere. The design it runs is not here - it is the settings block
    the plugin is armed with, written from `model.indexer` and `model.serving` and carried on the command line
    by `with_settings`.

    What it does not carry is anything the cluster already knows. `image` is a docker tag and enroot caches
    the pull under `/var/cache/enroot-container-images/$UID`, the mounts are the user's own data root and
    this checkout, and `HF_HOME` and the JIT caches come from the environment - see `serve_environment`.

    The checkpoint is not here either. It is `model.name`, said once, because a block naming a second one
    would let a config serve one model and record another.

    `serve_args` reaches the server verbatim, which is where the attention backend, the memory fraction, the
    running-request cap and `--disable-radix-cache` go. The list width is not among them - the plugin sets
    it on the config the model reads. A `--json-model-override-args` of its own is merged with the card's rotation
    and the settings block into the one the server is given, and a run that says what they say is refused.

    `pp` is how many nodes the server spans, one pipeline stage per node with `tp` GPUs each. Past 1M
    tokens the weights and the KV cache no longer fit on one node, so the 2M to 10M lengths are served at
    two stages.

    `environment` is exported to the server over everything `serve_environment` sets, for a setting one
    configuration needs and the rest do not - QSA's 1M server allocating in expanding segments, say.
    """

    # No default, the image is named by the config through its site
    # => empty where the site names none, and a launch or a submission refuses it
    image: str
    tp: int = 8
    pp: int = 1
    port: int = 0  # 0 is a free port the host picks when the server starts
    runner: str = "srun"
    serve_args: list[str] = field(default_factory=list)
    startup_minutes: int = 60
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a launch block that would be refused after a model has already loaded.

        Raises:
            NotNamed: `image` is empty.
            ValueError: `runner` is not srun or docker.
            ValueError: `tp`, `pp` or `startup_minutes` is not positive, or `port` is negative.
            ValueError: `pp` spans more than one node under `docker`, which starts on one host.
        """
        if self.runner not in ("srun", "docker"):
            raise ValueError(f"runner must be srun or docker - got {self.runner!r}")
        for name in ("tp", "pp", "startup_minutes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive - got {getattr(self, name)}")
        if self.port < 0:
            raise ValueError(f"port must be 0 for a free one or positive - got {self.port}")
        if self.pp > 1 and self.runner != "srun":
            raise ValueError(f"pp {self.pp} spans {self.pp} nodes and only srun starts on more than one")


def require_image(image: str) -> None:
    """Refuse a launch or a submission whose image the site leaves empty.

    Args:
        image: The container image the server runs in.

    Raises:
        NotNamed: `image` is empty.
    """
    if not image:
        raise site.NotNamed(
            "run.image is not named - a config names it `${site.images.<name>}`, so fill that name in "
            "configs/site/<site>.yaml, or pick another site with `site=<name>`"
        )


def merge_override(base: Mapping[str, Any], additions: Mapping[str, Any], where: str = "") -> dict[str, Any]:
    """Return two config overrides as one, refusing a key the two say differently.

    Args:
        base: What the run's own override says.
        additions: What the launcher adds.
        where: The dotted path the two are being merged at, for the message.

    Returns:
        The merged override - objects merged key by key, every other value kept.

    Raises:
        ValueError: A key holds a different value in each.
    """
    merged = dict(base)
    for key, value in additions.items():
        if key not in merged or merged[key] == value:
            merged[key] = value
        elif isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_override(merged[key], value, f"{where}{key}.")
        else:
            raise ValueError(
                f"serve_args passes {Server.MODEL_OVERRIDE_FLAG} with {where}{key} = {merged[key]!r} and the launch "
                f"says {value!r} - the run says it once, so drop it from the run's override"
            )
    return merged


def with_model_override(serve_args: Sequence[str], additions: Mapping[str, Any]) -> list[str]:
    """Return a server's arguments with `additions` merged into the one model config override they carry.

    sglang takes the flag once, so a run's own override and everything the launcher adds travel as one json.

    Args:
        serve_args: The server's command line, as a run gives it.
        additions: What the launcher adds to the model config.

    Returns:
        The command line with the merged override last.

    Raises:
        ValueError: The run's override is not a json object, or says what `additions` say with another value.
    """
    arguments = list(serve_args)
    own: Any = {}
    for position, argument in enumerate(arguments):
        flag, equals, value = argument.partition("=")
        if flag != Server.MODEL_OVERRIDE_FLAG:
            continue
        spelled = 1 if equals else 2
        if not equals:
            value = arguments[position + 1] if position + 1 < len(arguments) else ""
        try:
            own = json.loads(value)
        except json.JSONDecodeError as refused:
            raise ValueError(
                f"{Server.MODEL_OVERRIDE_FLAG} must be a json object - got {value!r}"
            ) from refused
        del arguments[position : position + spelled]
        break
    if not isinstance(own, dict):
        raise ValueError(f"{Server.MODEL_OVERRIDE_FLAG} must be a json object - got {own!r}")
    return [*arguments, Server.MODEL_OVERRIDE_FLAG, json.dumps(merge_override(own, additions))]


def with_rotation(serve_args: Sequence[str], rope_parameters: dict[str, Any] | None) -> list[str]:
    """Return a server's arguments with the model card's rotation passed to it, the one place it is said.

    The card's `rope_parameters` replace the checkpoint's own - YaRN past the native context - in every harness
    that loads the model, and the server reads them through sglang's config override, merged into the nested
    text config the model builds its rotary from.

    Args:
        serve_args: The server's command line, as a run gives it.
        rope_parameters: The card's rotation, or None for the checkpoint's own.

    Returns:
        The command line, with the override appended when there is a rotation to pass.

    Raises:
        ValueError: The run passes a config override of its own that sets the rotation too.
    """
    if not rope_parameters:
        return list(serve_args)
    return with_model_override(serve_args, {Server.TEXT_CONFIG: {"rope_parameters": dict(rope_parameters)}})


def with_settings(serve_args: Sequence[str], settings: Mapping[str, Any]) -> list[str]:
    """Return a server's arguments with the plugin's settings block in the model config override.

    The block is the design's flat body, and a run's own override may add to it - its `debug` sub-block, say.
    A projection file stays the path it is, which the container mounts where the job host has it.

    Args:
        serve_args: The server's command line, as a run gives it.
        settings: The block, as `DesignSettings.body` or `PisaConfig.body` returns it.

    Returns:
        The command line with the block in the override.

    Raises:
        ValueError: The run's own override sets a key of the block to another value.
    """
    return with_model_override(serve_args, {Plugin.BLOCK: dict(settings)})


def checkout_root() -> Path:
    """Return the checkout this process is running from.

    The server installs this tree rather than a wheel staged somewhere, so what is scored is what is checked
    out. A staged wheel is a second artifact to rebuild, and a run that forgets scores the last one built.

    Returns:
        The repository root holding `src/lsa`.
    """
    import lsa

    return Path(lsa.__file__).resolve().parents[2]


# Comma-separated directories to mount as well, each at its own path
EXTRA_MOUNTS = "LSA_EXTRA_MOUNTS"


def _as_named(path: str | Path) -> str:
    """Return a path as it is named, minus trailing slashes, so two spellings of one directory compare equal.

    Args:
        path: A directory, as a mount source.

    Returns:
        The path without its trailing slashes. Links are not resolved, because a mount has to be named as given.
    """
    named = str(path)
    return named.rstrip("/") or named


def mount_list() -> list[str]:
    """Return what the container needs to see.

    The project root carries the checkpoints, the teacher trees and the prompt corpus, the checkout carries
    the plugin, the hub cache carries the snapshots a model card names, and the site's `mounts` carry what
    else its machine keeps weights in - a user's own cache root, say. Nothing else is mounted - no other user's tree.

    The hub cache is the one `env.hf_home` returns, so `HF_HOME`, the site's `hf_home` and the hub's own default
    are mounted by the same precedence the checkpoint's path was built with. It goes in at its own path, because a
    model card's snapshot is named by that host path and the server opens it as given - a cache left unmounted
    makes the server fail on a path it cannot open. A cache that is not a directory on this host is skipped.

    Both are needed rather than one. A served checkpoint lives under the project root and may
    symlink its weight blobs into the user's `HF_HOME`, so mounting either alone leaves a dangling link.

    The project root has to be named rather than reached through a link into it.
    A symlink resolves inside the container to a path that was never mounted, and the plugin was handed a projection file it could not open that way once already.

    Returns:
        One `source:target` per mount.
    """
    root = _as_named(checkout_root())
    mounts = [f"{root}:{root}"]
    declared = [os.path.expandvars(mount) for mount in site.load().get("mounts") or []]
    # A checkpoint that lives in neither root - another user's hub cache, say - is named by the operator,
    # each directory mounted at its own path, so its snapshot's links into `blobs/` still resolve
    extra = [mount for mount in os.environ.get(EXTRA_MOUNTS, "").split(",") if mount.strip()]
    for shared in (str(paths.root()), str(env.hf_home()), *declared, *extra):
        shared = _as_named(shared.strip())
        if Path(shared).is_dir() and f"{shared}:{shared}" not in mounts:
            mounts.insert(0, f"{shared}:{shared}")
    return mounts


def cache_variables() -> dict[str, str]:
    """Return the one JIT cache the environment must not be left to decide.

    Triton is the exception. Eight tensor-parallel ranks compile the same kernel at once and a cache on a
    shared mount loses that race, so it is node-local. Every other cache root comes from the environment,
    where `source_env.sh` already puts it.

    Returns:
        The node-local Triton cache.
    """
    return {"TRITON_CACHE_DIR": f"/tmp/{os.environ.get('USER', 'lsa')}/lsa_triton"}


def serve_environment(
    pipeline_stages: int = 1,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment the server runs under.

    Nothing here names the plugin - sglang loads every installed `sglang.srt.plugins` entry point, and what the plugin
    is told rides in the model config on the command line, which is what makes the native baseline the absence of a
    setting rather than a different build.

    Args:
        pipeline_stages: How many nodes the server spans, one stage each.
        environment: What the configuration exports to its server, over everything set here.

    Returns:
        The environment for the child, under anything already exported.
    """
    variables = dict(cache_variables())
    # The checkpoint's config says 1M, and sglang refuses a longer `--context-length` unless told
    # => the 2M to 10M lengths need it, and a shorter context never reads it
    variables["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    # A server sized for 10M fragments its pool, native's 10M index workspace failing on 23.5 GiB reserved and unused
    # => one expanding segment
    if pipeline_stages > 1:
        variables["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    variables.update(environment or {})
    return {**os.environ, **variables}


def server_command(backend: SglangConfig, model: str) -> list[str]:
    """Build the sglang command line for one experiment.

    The list width is not here. The plugin derives it from the design's settings and sets it on the config the
    model reads, so a run states the design once and the width cannot drift from it.

    A server spanning nodes meets at this host, because the launcher runs as the job's first task and the
    first pipeline stage is the one that answers http. Each node's rank is not here - it differs per node,
    so `launch_line` has the shell read it.

    Args:
        backend: The launch block.
        model: The checkpoint, which is `model.name` - this block does not carry a second copy of it.

    Returns:
        The argv for the server.
    """
    argv = [
        "sglang",
        "serve",
        "--model-path",
        model,
        "--trust-remote-code",
        "--tp",
        str(backend.tp),
        "--host",
        "0.0.0.0",
        "--port",
        str(backend.port),
    ]
    if backend.pp > 1:
        argv += [
            "--pp-size",
            str(backend.pp),
            "--nnodes",
            str(backend.pp),
            "--dist-init-addr",
            f"{socket.gethostname()}:{Server.DIST_INIT_PORT}",
        ]
    return argv + list(backend.serve_args)


def container_command(backend: SglangConfig, inner: str) -> list[str]:
    """Wrap a shell line so it runs inside the pinned image.

    The image is not ours and neither is what is in it, so this says where it is and nothing about how
    sglang is built. `image` is a docker tag that enroot pulls and caches, or a path to a squashfs already
    built - a pinned path is what a model the published tags cannot parse has to use.

    Both runners name the GPUs: `docker` takes every one on the host, and the `srun` step asks for `tp` of
    them on each of its `pp` nodes, one task per node.

    Args:
        backend: The launch block.
        inner: The shell line to run inside.

    Returns:
        The argv for `subprocess.Popen`.

    Raises:
        NotNamed: No image was named.
        ValueError: The runner is not one that can start one.
    """
    require_image(backend.image)
    mounts = ",".join(mount_list())
    if backend.runner == "srun":
        return [
            "srun",
            # The launcher already holds the allocation as a step, so this one must overlap it
            # => see `Server.CPUS_PER_GPU` for the cores a `tp` of 8 needs
            "--overlap",
            f"--nodes={backend.pp}",
            f"--ntasks={backend.pp}",
            "--ntasks-per-node=1",
            "--mem=0",
            "--cpu-bind=none",
            f"--gres=gpu:{backend.tp}",
            f"--container-image={backend.image}",
            f"--container-mounts={mounts}",
            "bash",
            "-lc",
            inner,
        ]
    if backend.runner == "docker":
        argv = ["docker", "run", "--rm", "--ipc=host", "--network", "host", "--gpus", "all"]
        for mount in mount_list():
            argv += ["-v", mount]
        return argv + [backend.image, "bash", "-lc", inner]
    raise ValueError(f"runner must be srun or docker - got {backend.runner!r}")


def launch_line(backend: SglangConfig, model: str) -> str:
    """Return the shell line the container runs - install the plugin, then serve.

    The checkout is installed without its dependencies because the image already has torch and every other
    thing the kernels reach, and resolving them again would fight the image's own pins.

    Every node runs this same line, so a pipeline stage's rank is read from the node's own srun task.

    Args:
        backend: The launch block.
        model: The checkpoint to serve.

    Returns:
        One shell line.
    """
    serve = " ".join(shlex.quote(part) for part in server_command(backend, model))
    if backend.pp > 1:
        serve += ' --node-rank "$SLURM_NODEID"'
    root = shlex.quote(str(checkout_root()))
    return f"pip install --no-deps --quiet {root} && exec {serve}"


def write_settings(settings: dict[str, Any], path: Path) -> Path:
    """Write the indexer settings where the record can quote them.

    The server is given the same settings on its command line, so the file is the record's copy and can live next
    to the results it produced.

    Args:
        settings: The json body, as `DesignSettings.body` returns it.
        path: Destination file, whose parents are created.

    Returns:
        `path`.
    """
    io.write_json(path, settings)
    return path


# Where a free port is drawn from, above the services a node listens on and below the kernel's ephemeral range
# => not a bind to port 0, whose ephemeral range every outgoing connection shares, and a server binds its ports seconds to minutes after they are picked
FREE_PORT_FLOOR = 20000

# The kernel's ephemeral range, and Linux's default when the host does not say
EPHEMERAL_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")
EPHEMERAL_DEFAULT = (32768, 60999)

# The ports sglang derives from `--port`: +1 the gRPC sidecar, +13 the dp-attention handshake, +233 to +239
# dp attention's zmq ports and +10000 legacy gRPC, each taken below the port instead when above would pass 65535
DERIVED_PORT_OFFSETS = (1, 13, *range(233, 240), 10000)

# torch's rendezvous port, which sglang draws from the ephemeral range unless given one
NCCL_PORT_FLAG = "--nccl-port"


def ephemeral_floor() -> int:
    """Return the lowest port the kernel hands out on its own.

    Returns:
        The start of `ip_local_port_range`, or of Linux's default range when the host does not say.
    """
    try:
        return int(EPHEMERAL_RANGE.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return EPHEMERAL_DEFAULT[0]


def derived_ports(port: int) -> set[int]:
    """Return the ports sglang may take for itself from a server's http port.

    Args:
        port: The server's `--port`.

    Returns:
        The port and every one sglang derives from it, either way round.
    """
    return {port} | {port + sign * offset for offset in DERIVED_PORT_OFFSETS for sign in (1, -1)}


def free_port(avoid: Collection[int] = ()) -> int:
    """Return a port nothing on this host is bound to, outside the range the kernel hands out on its own.

    A port drawn at random between `FREE_PORT_FLOOR` and the ephemeral range is bound once on every
    interface, the way the server binds it, and returned when that bind succeeds.

    Args:
        avoid: Ports not to return although free, such as the ones sglang derives from a port already picked.

    Returns:
        The port, free again by the time it is returned.

    Raises:
        RuntimeError: Every port tried was taken.
    """
    ceiling = ephemeral_floor()
    if ceiling <= FREE_PORT_FLOOR:
        ceiling = 65536
    for _ in range(100):
        port = random.randrange(FREE_PORT_FLOOR, ceiling)
        if port in avoid:
            continue
        with socket.socket() as probe:
            try:
                probe.bind(("", port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"no free port between {FREE_PORT_FLOOR} and {ceiling} after 100 tries")


def names_flag(serve_args: Sequence[str], flag: str) -> bool:
    """Return whether a server's arguments already give a flag, as `--flag value` or `--flag=value`.

    Args:
        serve_args: The server's command line, as a run gives it.
        flag: The flag to look for.

    Returns:
        True when the flag is there in either form.
    """
    return any(argument.partition("=")[0] == flag for argument in serve_args)


@contextmanager
def serve(backend: SglangConfig, model: str, settings: Path | None, log: Path) -> Iterator[str]:
    """Run the server for the body of the block and stop it afterwards.

    A `backend.port` of 0 is resolved here, on the host the server starts on, so workers sharing a node each
    take a port of their own. A server on one node is given torch's rendezvous port the same way unless its
    `serve_args` name one, clear of the ports sglang derives from the http port.

    Args:
        backend: The launch block.
        model: The checkpoint to serve.
        settings: The written indexer settings, which the server is handed in its model config override, or None to run
            the checkpoint's own indexer.
        log: Where the server's own output is written, whose parents are created.

    Yields:
        The server's http root, whose port is the resolved one.

    Raises:
        NotNamed: `backend.image` is empty.
        TimeoutError: The server did not answer `/health` within `backend.startup_minutes`.
        RuntimeError: The server exited before it was ready, or came up without the plugin armed.
    """
    require_image(backend.image)
    log.parent.mkdir(parents=True, exist_ok=True)
    if not backend.port:
        backend = replace(backend, port=free_port())
        print(f"run.port is 0 - serving on the free port {backend.port}", flush=True)
    if backend.pp == 1 and not names_flag(backend.serve_args, NCCL_PORT_FLAG):
        # sglang would draw it from the ephemeral range, where another server on the node can take it first
        nccl = free_port(avoid=derived_ports(backend.port))
        backend = replace(backend, serve_args=[*backend.serve_args, NCCL_PORT_FLAG, str(nccl)])
        print(f"torch's rendezvous on the free port {nccl}", flush=True)
    if settings is not None:
        backend = replace(backend, serve_args=with_settings(backend.serve_args, io.read_json(settings)))
    argv = container_command(backend, launch_line(backend, model))
    print(f"serving: {' '.join(argv)}", flush=True)
    with log.open("w") as sink:
        process = subprocess.Popen(
            argv,
            env=serve_environment(backend.pp, backend.environment),
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_ready(process, backend, log, armed=settings is not None, model=model)
            yield f"http://127.0.0.1:{backend.port}"
        finally:
            stop(process)


def plugin_armed(log: Path) -> bool:
    """Return whether the plugin attached and was then actually reached.

    Two banners rather than one, because the first run showed the difference. The hooks registered, the
    model asked for a class other than the one that was replaced, and the server printed the first banner
    and served the checkpoint's own indexer. Only the second says an LSA indexer was built.

    Args:
        log: The server's own output.

    Returns:
        True once every required banner is in the log.
    """
    if not log.is_file():
        return False
    text = log.read_text(errors="replace")
    return Banner.READY in text and Banner.BUILT in text


def serving_checkpoint(port: int) -> str:
    """Return the checkpoint the server on this port reports, or an empty string.

    Args:
        port: The port to ask.

    Returns:
        Its `model_path`, or "" when nothing answered or the answer had no such field.
    """
    url = f"http://127.0.0.1:{port}/model_info"
    try:
        with urllib.request.urlopen(url, timeout=Server.POLL_TIMEOUT_SECONDS) as answer:
            return str(json.loads(answer.read()).get("model_path") or "")
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return ""


def wait_ready(
    process: subprocess.Popen[bytes],
    backend: SglangConfig,
    log: Path,
    armed: bool = False,
    model: str = "",
) -> None:
    """Block until the server answers, is the one this run started, and has the hooks it claims.

    A healthy port is not the whole condition, for two reasons.

    sglang logs a hook that failed to apply rather than raising it, so an LSA run whose hooks did not land
    would serve the checkpoint's own indexer and score as LSA. That is what the two banners are for.

    And the port is the whole address. `port` defaults to one number, so two runs on a node answer at the
    same place - the second finds the first's server healthy, and scores against it. An LSA configuration
    escapes that by accident, because the banner check reads its own log and fails; two native
    configurations do not, and would quietly score one server twice under two names. So the checkpoint is
    compared as well.

    Args:
        process: The running server, watched so a crashed load fails here rather than at the deadline.
        backend: The launch block, whose `startup_minutes` is the deadline.
        log: The server's own output, whose tail goes into the message.
        armed: Whether this run needs the plugin, which is every run but the native baseline.
        model: The checkpoint this run asked for, or "" not to check it.

    Raises:
        TimeoutError: The deadline passed with no healthy answer.
        RuntimeError: The server exited first, answered for another checkpoint, or came up unarmed.
    """
    url = f"http://127.0.0.1:{backend.port}/health"
    deadline = time.monotonic() + backend.startup_minutes * 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the sglang server exited with {process.returncode}\n{tail(log)}")
        if healthy(url):
            answered = serving_checkpoint(backend.port)
            if model and answered and answered.rstrip("/") != model.rstrip("/"):
                raise RuntimeError(
                    f"port {backend.port} is answering for {answered} rather than {model} - another run "
                    f"holds it, so give this one its own run.port rather than scoring against theirs"
                )
            if armed and not plugin_armed(log):
                raise RuntimeError(
                    f"the server is up but never printed both {Banner.READY!r} and {Banner.BUILT!r} - its "
                    f"hooks did not apply or were never reached, so it is running the checkpoint's own "
                    f"indexer\n{tail(log)}"
                )
            print(f"server up on port {backend.port}", flush=True)
            return
        time.sleep(Server.POLL_SECONDS)
    raise TimeoutError(f"the sglang server was not ready in {backend.startup_minutes} minutes\n{tail(log)}")


def healthy(url: str) -> bool:
    """Return whether one health poll answered.

    Args:
        url: The health endpoint.

    Returns:
        True when the server answered 200.
    """
    try:
        with urllib.request.urlopen(url, timeout=Server.POLL_TIMEOUT_SECONDS) as answer:
            return bool(answer.status == 200)
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def stop(process: subprocess.Popen[bytes]) -> None:
    """Take down the server and every worker it started.

    Args:
        process: The running server.
    """
    if process.poll() is not None:
        return
    group = os.getpgid(process.pid)
    os.killpg(group, signal.SIGTERM)
    try:
        process.wait(timeout=Server.SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(group, signal.SIGKILL)
        process.wait()


def tail(log: Path, lines: int = 40) -> str:
    """Return the end of a log, for a message that says what went wrong.

    Args:
        log: The file to read.
        lines: How many lines to keep.

    Returns:
        The tail, or a note that there is nothing to read.
    """
    if not log.is_file():
        return f"no log at {log}"
    return "\n".join(log.read_text(errors="replace").splitlines()[-lines:])
