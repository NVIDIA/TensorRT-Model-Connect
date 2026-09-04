# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "models" / "cosmos3" / "dual_spark"
LAUNCHER = EXAMPLE_ROOT / "run_dual_spark.py"
sys.path.insert(0, str(LAUNCHER.parent))

import run_dual_spark  # noqa: E402


EXPECTED_SCENES = (
    ("showcase-high-speed-racing", 21001, "showcase-high-speed-racing-720p.mp4"),
    ("showcase-mars-robots", 21003, "showcase-mars-robots-720p.mp4"),
    ("showcase-delivery-robot", 21004, "showcase-delivery-robot-720p.mp4"),
    ("showcase-apple-to-plate", 21006, "showcase-apple-to-plate-720p.mp4"),
    ("showcase-humanoid-sprint", 21011, "showcase-humanoid-sprint-720p.mp4"),
    ("showcase-cake-cutting", 21012, "showcase-cake-cutting-720p.mp4"),
)


def _clean_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("COSMOS3_")
    }
    environment["PATH"] = ""
    return environment


def _dry_run(
    tmp_path: Path,
    scene: str,
    extra_arguments: Sequence[str] = (),
) -> dict[str, Any]:
    work_root = tmp_path / "must-not-exist"
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "all",
            "--scene",
            scene,
            "--peer-host",
            "peer.example",
            "--peer-user",
            "tester",
            "--run-id",
            "contract-run",
            "--work-root",
            str(work_root),
            "--remote-root",
            "/var/tmp/cosmos3-contract",
            *extra_arguments,
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_clean_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert not work_root.exists()
    return json.loads(completed.stdout)


def _option_value(argv: Sequence[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def _docker_environment(argv: Sequence[str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for index, value in enumerate(argv[:-1]):
        if value == "-e":
            name, setting = argv[index + 1].split("=", 1)
            environment[name] = setting
    return environment


def _generation_command(argv: Sequence[str]) -> list[str]:
    return list(argv[argv.index("generate-video") :])


def test_showcase_prompt_is_built_in_and_compact_in_dry_run(
    tmp_path: Path,
) -> None:
    scene_id = "showcase-high-speed-racing"
    plan = _dry_run(tmp_path, scene_id)
    scene_record = plan["run"]["scenes"][0]
    scene = run_dual_spark.SCENE_BY_SLUG[scene_id]
    command = _generation_command(scene_record["ranks"][0]["docker_argv"])
    compact_prompt = run_dual_spark._prompt_text(scene)

    assert scene_record["seed"] == 21001
    assert scene_record["prompt"] == {
        "positive": dict(scene.prompt),
        "negative": scene.negative_prompt,
    }
    assert scene_record["prompt"]["positive"]["duration"] == "7.875s"
    assert json.loads(compact_prompt) == scene.prompt
    assert "\n" not in compact_prompt
    assert ": " not in compact_prompt
    assert _option_value(command, "--prompt") == compact_prompt
    assert _option_value(
        command, "--negative-prompt"
    ) == run_dual_spark._negative_prompt_text(scene)


@pytest.mark.parametrize("scene_id,seed,filename", EXPECTED_SCENES)
def test_launcher_selects_one_native_720p_cp2_scene(
    tmp_path: Path,
    scene_id: str,
    seed: int,
    filename: str,
) -> None:
    plan = _dry_run(tmp_path, scene_id)
    scene = plan["run"]["scenes"][0]

    assert plan["schema_version"] == 1
    assert plan["action"] == "all"
    assert plan["dry_run"] is True
    assert plan["scene_selection"] == scene_id
    assert plan["selected_scene_slugs"] == [scene_id]
    assert plan["run"]["order"] == [scene_id]
    assert len(plan["run"]["scenes"]) == 1
    assert (scene["slug"], scene["seed"], Path(scene["mp4"]).name) == (
        scene_id,
        seed,
        filename,
    )
    assert plan["profile"] == {
        "width": 1280,
        "height": 720,
        "frames": 189,
        "fps": 24,
        "steps": 35,
        "guidance_scale": 6.0,
        "flow_shift": 10.0,
        "context_parallel_size": 2,
    }

    bundle = plan["prepare"]["bundle_build_argv"]
    assert plan["prepare"]["image"] == "trtmc-cosmos3-dual-spark:local"
    assert plan["prepare"]["image_source"] == (
        "prebuilt locally from this example's Dockerfile"
    )
    assert "image_build_argv" not in plan["prepare"]
    assert _option_value(bundle, "--context-parallel-size") == "2"
    assert "--family" not in bundle
    assert "--task" not in bundle
    assert _option_value(bundle, "--image-height") == "720"
    assert _option_value(bundle, "--image-width") == "1280"
    assert Path(plan["prepare"]["bundle"]).name == "cosmos3-nano-cp2-1280x720.bundle"

    assert [(rank["host"], rank["rank"]) for rank in scene["ranks"]] == [
        ("primary", 0),
        ("tester@peer.example", 1),
    ]
    for rank_number, rank in enumerate(scene["ranks"]):
        environment = _docker_environment(rank["docker_argv"])
        assert environment["OMPI_COMM_WORLD_SIZE"] == "2"
        assert environment["OMPI_COMM_WORLD_RANK"] == str(rank_number)
        assert environment["OMPI_COMM_WORLD_LOCAL_RANK"] == "0"
        assert environment["CUDA_VISIBLE_DEVICES"] == "0"
        command = _generation_command(rank["docker_argv"])
        assert _option_value(command, "--runtime-root") == "/opt/trtmc/lib"
        assert _option_value(command, "--height") == "720"
        assert _option_value(command, "--width") == "1280"
        assert _option_value(command, "--seed") == str(seed)
    assert _generation_command(scene["ranks"][0]["docker_argv"]) == (
        _generation_command(scene["ranks"][1]["docker_argv"])
    )


def test_launcher_uses_strict_ssh_roce_and_declares_run_telemetry(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "identity"
    known_hosts = tmp_path / "known_hosts"
    plan = _dry_run(
        tmp_path,
        EXPECTED_SCENES[0][0],
        ("--ssh-key", str(identity), "--known-hosts", str(known_hosts)),
    )

    for argv in (plan["ssh"]["argv"], plan["ssh"]["scp_argv"]):
        joined = " ".join(argv)
        assert "BatchMode=yes" in joined
        assert "StrictHostKeyChecking=yes" in joined
        assert "IdentitiesOnly=yes" in joined
        assert f"UserKnownHostsFile={known_hosts.resolve()}" in joined
        assert "StrictHostKeyChecking=no" not in joined
    assert plan["roce"]["hca"] == "rocep1s0f0:1"
    assert plan["roce"]["network_interface"] == "enp1s0f0np0"
    assert plan["roce"]["gid_index"] == 3
    assert plan["roce"]["uverbs_resolution"]["resolved_at_runtime"] is True
    assert plan["run"]["telemetry"] == {
        "performance": "parsed from the rank-0 [cosmos3.perf] record",
        "memory": "sampled from each rank's cgroup-v2 byte counters",
    }


def test_launcher_parses_one_matching_cosmos3_performance_record(
    tmp_path: Path,
) -> None:
    scene = run_dual_spark.SCENE_BY_SLUG[EXPECTED_SCENES[0][0]]
    log = tmp_path / "rank0.log"
    fields = {
        "prompt_conditioning_ms": 10.0,
        "denoise_prep_ms": 20.0,
        "denoiser_engine_load_ms": 30.0,
        "denoiser_ms": 40.0,
        "scheduler_cfg_ms": 50.0,
        "vae_decoder_ms": 60.0,
        "generation_excluding_denoiser_load_ms": 80_000.0,
        "total_ms": 82_710.0,
        "cp_size": 2,
        "seed": scene.seed,
    }
    record = " ".join(f"{name}={value}" for name, value in fields.items())
    log.write_text(f"noise\n[cosmos3.perf] {record}\n", encoding="utf-8")

    performance = run_dual_spark._parse_cosmos3_performance(log, scene, 85.25)

    assert performance["inference_seconds"] == pytest.approx(82.71)
    assert performance["total_seconds"] == pytest.approx(82.71)
    assert performance["rank0_wall_seconds"] == pytest.approx(85.25)
    assert performance["cp_size"] == 2
    assert performance["seed"] == scene.seed


def test_launcher_tolerates_cgroup_teardown_after_confirmed_container_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = run_dual_spark.Settings(
        action="run",
        scene=EXPECTED_SCENES[1][0],
        peer_host="peer.example",
        peer_user="tester",
        ssh_key=None,
        known_hosts=None,
        work_root=Path("/tmp/cosmos3-contract"),
        remote_root="/var/tmp/cosmos3-contract",
        image="cosmos3-contract:latest",
        run_id="contract-run",
        ib_hca="rocep1s0f0:1",
        net_iface="enp1s0f0np0",
        gid_index=3,
        runtime_timeout=60,
        build_timeout=60,
        dry_run=False,
    )
    tracker = run_dual_spark.MemoryTracker(
        container="rank1",
        remote=True,
        cgroup_path="/sys/fs/cgroup/disappeared",
        samples=1,
    )
    observed_states = iter(
        (
            {"Running": True, "ExitCode": 0},
            {"Running": False, "ExitCode": 0},
        )
    )
    observed_queries: list[tuple[str, bool]] = []

    def container_state(
        _settings: run_dual_spark.Settings,
        name: str,
        *,
        remote: bool,
    ) -> dict[str, Any]:
        observed_queries.append((name, remote))
        return next(observed_states)

    monkeypatch.setattr(
        run_dual_spark,
        "_container_state",
        container_state,
    )

    def missing_cgroup(*_args: Any, **_kwargs: Any) -> None:
        raise run_dual_spark.DualSparkError("cgroup counters disappeared")

    monkeypatch.setattr(run_dual_spark, "_sample_memory_tracker", missing_cgroup)

    states = run_dual_spark._wait_for_containers(
        settings,
        ((tracker.container, tracker.remote),),
        {tracker.container: tracker},
    )

    assert states == {"rank1": {"Running": False, "ExitCode": 0}}
    assert observed_queries == [(tracker.container, True), (tracker.container, True)]


def test_launcher_stops_sampling_a_finished_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = run_dual_spark.Settings(
        action="run",
        scene=EXPECTED_SCENES[1][0],
        peer_host="peer.example",
        peer_user="tester",
        ssh_key=None,
        known_hosts=None,
        work_root=Path("/tmp/cosmos3-contract"),
        remote_root="/var/tmp/cosmos3-contract",
        image="cosmos3-contract:latest",
        run_id="contract-run",
        ib_hca="rocep1s0f0:1",
        net_iface="enp1s0f0np0",
        gid_index=3,
        runtime_timeout=60,
        build_timeout=60,
        dry_run=False,
    )
    rank0 = run_dual_spark.MemoryTracker(
        container="rank0",
        remote=False,
        cgroup_path="/sys/fs/cgroup/rank0",
        samples=1,
    )
    rank1 = run_dual_spark.MemoryTracker(
        container="rank1",
        remote=True,
        cgroup_path="/sys/fs/cgroup/disappeared",
        samples=1,
    )
    observed_states = iter(
        (
            {"Running": True, "ExitCode": 0},
            {"Running": False, "ExitCode": 0},
            {"Running": False, "ExitCode": 0},
            {"Running": False, "ExitCode": 0},
        )
    )
    sampled: list[str] = []
    messages: list[str] = []

    def container_state(
        _settings: run_dual_spark.Settings,
        _name: str,
        *,
        remote: bool,
    ) -> dict[str, Any]:
        del remote
        return next(observed_states)

    def sample_memory(
        _settings: run_dual_spark.Settings,
        tracker: run_dual_spark.MemoryTracker,
    ) -> None:
        sampled.append(tracker.container)
        if tracker.container == rank1.container:
            raise run_dual_spark.DualSparkError("cgroup counters disappeared")

    monkeypatch.setattr(run_dual_spark, "_container_state", container_state)
    monkeypatch.setattr(run_dual_spark, "_sample_memory_tracker", sample_memory)
    monkeypatch.setattr(run_dual_spark, "_log", messages.append)
    monkeypatch.setattr(run_dual_spark.time, "sleep", lambda _seconds: None)

    states = run_dual_spark._wait_for_containers(
        settings,
        ((rank0.container, rank0.remote), (rank1.container, rank1.remote)),
        {rank0.container: rank0, rank1.container: rank1},
    )

    assert states == {
        "rank0": {"Running": False, "ExitCode": 0},
        "rank1": {"Running": False, "ExitCode": 0},
    }
    assert sampled == [rank0.container, rank1.container, rank0.container]
    assert messages == [
        "WARNING: final cgroup memory sample was unavailable "
        "for rank1: cgroup counters disappeared"
    ]


def test_launcher_removes_rank1_verified_empty_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = run_dual_spark.Settings(
        action="run",
        scene=EXPECTED_SCENES[1][0],
        peer_host="peer.example",
        peer_user="tester",
        ssh_key=None,
        known_hosts=None,
        work_root=Path("/tmp/cosmos3-contract"),
        remote_root="/var/tmp/cosmos3-contract",
        image="cosmos3-contract:latest",
        run_id="contract-run",
        ib_hca="rocep1s0f0:1",
        net_iface="enp1s0f0np0",
        gid_index=3,
        runtime_timeout=60,
        build_timeout=60,
        dry_run=False,
    )
    observed: list[tuple[list[str], bool]] = []

    def remote_run(
        _settings: run_dual_spark.Settings,
        argv: Sequence[str],
        *,
        check: bool = True,
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        observed.append((list(argv), check))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(run_dual_spark, "_remote_run", remote_run)

    run_dual_spark._remove_rank1_empty_frames(
        settings,
        "/var/tmp/cosmos3-contract/runs/run/scene/rank1",
    )

    assert observed == [
        (
            [
                "rmdir",
                "--",
                "/var/tmp/cosmos3-contract/runs/run/scene/rank1/frames",
            ],
            False,
        )
    ]


def test_prepare_requires_the_readme_image_to_exist_locally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = run_dual_spark.Settings(
        action="prepare",
        scene=EXPECTED_SCENES[0][0],
        peer_host="peer.example",
        peer_user="tester",
        ssh_key=None,
        known_hosts=None,
        work_root=tmp_path / "work",
        remote_root="/var/tmp/cosmos3-contract",
        image="trtmc-cosmos3-dual-spark:local",
        run_id="contract-run",
        ib_hca="rocep1s0f0:1",
        net_iface="enp1s0f0np0",
        gid_index=3,
        runtime_timeout=60,
        build_timeout=60,
        dry_run=False,
    )
    monkeypatch.setattr(run_dual_spark, "_preflight", lambda _settings: {})
    monkeypatch.setattr(
        run_dual_spark,
        "_image_available",
        lambda _settings, *, remote: "",
    )

    with pytest.raises(
        run_dual_spark.DualSparkError,
        match="build it from this example's Dockerfile first",
    ):
        run_dual_spark.prepare(settings)


def test_remote_root_helper_restricts_directory_and_rejects_symlink(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir(mode=0o755)
    completed = subprocess.run(
        [sys.executable, "-c", run_dual_spark.REMOTE_ROOT_HELPER, str(remote_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert stat.S_IMODE(remote_root.stat().st_mode) == 0o700

    target = tmp_path / "target"
    target.mkdir()
    redirected_child = remote_root / "models"
    redirected_child.symlink_to(target, target_is_directory=True)
    polluted = subprocess.run(
        [sys.executable, "-c", run_dual_spark.REMOTE_ROOT_HELPER, str(remote_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert polluted.returncode != 0
    assert "must not contain symbolic links" in polluted.stderr

    symlink_root = tmp_path / "symlink-root"
    symlink_root.symlink_to(target, target_is_directory=True)
    rejected = subprocess.run(
        [sys.executable, "-c", run_dual_spark.REMOTE_ROOT_HELPER, str(symlink_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert rejected.returncode != 0
    assert "real directory" in rejected.stderr


def test_readme_leads_with_the_cli_and_one_time_image_build() -> None:
    readme = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    assert "# Cosmos3 dual-Spark video generation example" in readme
    assert "does not start a server" in normalized
    assert "## Build the image once" in readme
    assert "--platform linux/arm64" in readme
    assert "--file examples/models/cosmos3/dual_spark/Dockerfile" in readme
    assert "python3 examples/models/cosmos3/dual_spark/run_dual_spark.py all" in readme
    assert "--peer-host <SECOND_SPARK>" in readme
    assert "--image trtmc-cosmos3-dual-spark:local" in readme
    assert "one native 1280x720 Cosmos3-Nano video" in normalized
    assert "1280x720" in readme
    assert "CP=2" in readme
    assert "primary" in readme and "peer" in readme
    assert "can take hours" in readme
    assert "showcase-delivery-robot" in readme
    assert "checkpoint, TensorRT bundle, generated video, SSH key" in normalized
    assert "http://localhost" not in readme
    assert "browser app" not in readme.lower()


def test_dockerfile_ships_a_pinned_isolated_cosmos3_environment() -> None:
    dockerfile = (EXAMPLE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (EXAMPLE_ROOT / "Dockerfile.dockerignore").read_text(
        encoding="utf-8"
    )

    assert "nvcr.io/nvidia/tensorrt:26.07-py3@sha256:" in dockerfile
    assert "ARG TRTMC_CUDA_ARCHITECTURES=121-real" in dockerfile
    assert "-DTRTMC_BUILD_TESTS=OFF" in dockerfile
    assert "-DTRTMC_BUILD_EXAMPLES=OFF" in dockerfile
    assert "-DTRTMC_ENABLE_BYOK=OFF" in dockerfile
    assert "trtmc_model_cosmos3" in dockerfile
    assert "TRTMC_SOURCE_REVISION" not in dockerfile
    assert 'ENTRYPOINT ["/opt/trtmc/bin/trtmc"]' in dockerfile
    assert dockerignore.startswith("# SPDX-FileCopyrightText:")
    assert "\n*\n" in dockerignore
    assert "!ASSET_LICENSES.md" in dockerignore
    assert "!families/__init__.py" in dockerignore
    assert "!families/cosmos3/**" in dockerignore
    assert "!core/**" in dockerignore
    assert "website/**" in dockerignore
    assert "nemotron_voicechat" not in dockerignore


def test_sample_sources_do_not_contain_private_machine_defaults() -> None:
    sources = LAUNCHER.read_text(encoding="utf-8")
    for private_value in (
        "spark-8363",
        "spark-7bc1",
        "169.254.",
        "/home/rajerao",
        "/mnt/scratch/.ssh",
        "id_ed25519_spark_peer",
    ):
        assert private_value not in sources
    for unsafe_pattern in ("shell=True", "eval(", "StrictHostKeyChecking=no"):
        assert unsafe_pattern not in sources
