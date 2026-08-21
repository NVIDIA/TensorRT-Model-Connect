# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "cosmos3_story_scene"
APP_ROOT = EXAMPLE_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))

from story_scene.config import AppConfig  # noqa: E402
from story_scene.jobs import JobManager  # noqa: E402
from story_scene.prompts import (  # noqa: E402
    PUBLIC_PRESET_IDS,
    ValidationError,
    compile_prompt,
    parse_json_body,
    preset_catalog,
    validate_submission,
)
from story_scene.runtime import (  # noqa: E402
    FRAME_COUNT,
    StoryScenePipeline,
    build_ffmpeg_commands,
    build_generate_command,
    run_command,
)
from story_scene.server import _safe_static_file  # noqa: E402
from story_scene.server import create_server  # noqa: E402


UI_PRESETS = (
    "impossible-asmr",
    "pocket-universe",
    "product-metamorphosis",
    "plot-twist",
    "nature-glitch",
    "custom",
)


UI_PRESET_NAMES = {
    "impossible-asmr": "Impossible ASMR",
    "pocket-universe": "Pocket Universe",
    "product-metamorphosis": "Product Metamorphosis",
    "plot-twist": "Plot Twist",
    "nature-glitch": "Nature Glitch",
    "custom": "Custom Direction",
}


def test_container_build_declares_x86_and_native_gb10_paths() -> None:
    dockerfile = (EXAMPLE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    readme = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")

    assert "121-real" in dockerfile
    assert 'target_triplet="$(gcc -dumpmachine)"' in dockerfile
    assert "/opt/trtmc/platform/lib" in dockerfile
    assert "/usr/lib/x86_64-linux-gnu" not in dockerfile
    assert "OPAL_PREFIX=" in dockerfile
    assert "/usr/bin/mpirun -np 1 /usr/bin/true" in dockerfile
    assert "mpirun --version" not in dockerfile
    assert (
        "sha256:f794a79e8b996d16dbc2e5884e19d8e2269a51c960106c9b49b0061a6926c541"
        in readme
    )


def test_container_preserves_python_only_editable_install() -> None:
    dockerfile = (EXAMPLE_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "pip install --no-deps -e . -C py-only=true" in dockerfile
    assert "COPY --from=trtmc-builder /src/python /src/python" in dockerfile
    assert "pip install --no-deps . -C py-only=true" not in dockerfile


def test_container_context_includes_cmake_contract_sources() -> None:
    dockerfile = (EXAMPLE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignore_rules = (EXAMPLE_ROOT / "Dockerfile.dockerignore").read_text(
        encoding="utf-8"
    ).splitlines()

    native_build = dockerfile.index("RUN pip install --no-deps -e .")
    source_copies = (
        "COPY tests/cpp/models ./tests/cpp/models",
        "COPY examples/byok/identity_copy_kernel.cpp "
        "./examples/byok/identity_copy_kernel.cpp",
    )
    for source_copy in source_copies:
        assert source_copy in dockerfile
        assert dockerfile.index(source_copy) < native_build
    assert "-DTRTMC_BUILD_TESTS=OFF" in dockerfile
    required_rules = (
        "!tests/",
        "!tests/cpp/",
        "!tests/cpp/models/",
        "!tests/cpp/models/**",
        "!examples/byok/",
        "!examples/byok/identity_copy_kernel.cpp",
    )
    assert all(rule in ignore_rules for rule in required_rules)


def test_hugging_face_token_directory_is_gitignored() -> None:
    ignore_rules = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/examples/cosmos3_story_scene/secrets/" in ignore_rules


def test_compose_defaults_to_tokenless_public_checkpoint() -> None:
    compose = (EXAMPLE_ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "file: \"${HF_TOKEN_FILE:-/dev/null}\"" in compose
    assert "./secrets/hf_token" not in compose


def test_entrypoint_builds_public_checkpoint_without_token(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        """#!/usr/bin/env bash
case "$*" in
  *compute_cap*) printf "12.1\n" ;;
  *memory.total*) printf "122880\n" ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    trtmc = fake_bin / "trtmc"
    trtmc.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf "%s|%s\n" "${HF_TOKEN+x}" "${HUGGING_FACE_HUB_TOKEN+x}" > "${ENV_CAPTURE}"
output=""
while (( $# )); do
  if [[ "$1" == "-o" && $# -ge 2 ]]; then
    output="$2"
    break
  fi
  shift
done
[[ -n "${output}" ]]
printf "bundle" > "${output}"
""",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    trtmc.chmod(0o755)

    bundle = tmp_path / "models" / "cosmos3.trtfb"
    env_capture = tmp_path / "build-environment"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": os.pathsep.join((str(fake_bin), environment["PATH"])),
            "TRTMC_BIN": str(trtmc),
            "COSMOS3_BUNDLE": str(bundle),
            "COSMOS3_CP_SIZE": "1",
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "HF_HUB_CACHE": str(tmp_path / "hf-cache"),
            "ENV_CAPTURE": str(env_capture),
        }
    )
    environment.pop("HF_TOKEN", None)
    environment.pop("HUGGING_FACE_HUB_TOKEN", None)

    result = subprocess.run(
        [str(EXAMPLE_ROOT / "scripts" / "entrypoint.sh"), "true"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert bundle.read_bytes() == b"bundle"
    assert bundle.with_suffix(".trtfb.build-spec").is_file()
    assert env_capture.read_text(encoding="utf-8") == "|\n"
    assert "without a Hugging Face token" in result.stderr


def _payload(preset: str = "pocket-universe") -> dict[str, object]:
    return {
        "preset": preset,
        "subject": "A chipped coffee mug on a quiet kitchen table",
        "hook": "The coffee begins reflecting a different night sky",
        "visual_twist": "A tiny comet escapes the mug and loops around its handle",
        "camera": "Slow centered orbit",
        "lighting": "Soft window light with a nebula glow",
        "cta": "What is hiding in your cup?",
        "seed": 31415,
    }


@pytest.mark.parametrize("preset", UI_PRESETS)
def test_all_ui_presets_validate_and_public_catalog_is_exact(preset: str) -> None:
    submission = validate_submission(_payload(preset))

    assert submission.preset == preset
    assert tuple(PUBLIC_PRESET_IDS) == UI_PRESETS
    assert tuple(item["id"] for item in preset_catalog()) == UI_PRESETS
    assert f"Format: {UI_PRESET_NAMES[preset]}." in compile_prompt(submission)
    assert preset_catalog()[UI_PRESETS.index(preset)]["name"] == UI_PRESET_NAMES[preset]


def test_blank_optional_fields_are_omitted_but_required_controls_are_strict() -> None:
    payload = _payload()
    payload.update({"camera": "  ", "lighting": "", "cta": "", "seed": None})

    submission = validate_submission(payload)

    assert submission.camera
    assert submission.lighting
    assert submission.cta
    assert 0 <= submission.seed <= 2**31 - 1

    for field in ("subject", "hook", "visual_twist"):
        invalid = _payload()
        invalid[field] = " "
        with pytest.raises(ValidationError, match=field):
            validate_submission(invalid)


def test_json_and_submission_validation_reject_ambiguous_or_unsafe_input() -> None:
    with pytest.raises(ValidationError, match="Duplicate field"):
        parse_json_body(b'{"preset":"custom","preset":"plot-twist"}')
    with pytest.raises(ValidationError, match="finite"):
        parse_json_body(b'{"seed":NaN}')
    with pytest.raises(ValidationError, match="JSON object"):
        parse_json_body(b"[]")
    with pytest.raises(ValidationError, match="Unknown field"):
        validate_submission({**_payload(), "admin": True})
    with pytest.raises(ValidationError, match="integer"):
        validate_submission({**_payload(), "seed": True})
    with pytest.raises(ValidationError, match="control"):
        validate_submission({**_payload(), "cta": "hello\nworld"})


def test_idea_and_caption_aliases_remain_argv_safe() -> None:
    payload = _payload("plot-twist")
    payload["idea"] = payload.pop("subject")
    payload["caption"] = "%{eif\\:1}'; movie=secret"
    payload["cta"] = ""
    submission = validate_submission(payload)
    prompt = compile_prompt(submission)
    config = AppConfig(
        trtmc_bin="trtmc",
        cosmos3_bundle=Path("/models/cosmos3.trtfb"),
        output_root=Path("/outputs"),
    )

    command = build_generate_command(
        config,
        prompt=prompt,
        frames_dir=Path("/outputs/job/frames"),
        seed=submission.seed,
    )

    assert command[command.index("--prompt") + 1] == prompt
    assert submission.subject in prompt
    assert submission.caption not in " ".join(command)


def test_generate_command_matches_cli_contract_and_context_parallel_prefix(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        trtmc_bin="/opt/trtmc/bin/trtmc",
        cosmos3_bundle=Path("/models/cosmos3.trtfb"),
        cosmos3_cp_size=1,
        output_root=tmp_path,
    )
    frames = tmp_path / "uuid with spaces" / "frames"
    command = build_generate_command(
        config,
        prompt="one prompt; still one argument",
        frames_dir=frames,
        seed=7,
    )

    assert command == (
        "/opt/trtmc/bin/trtmc",
        "generate-video",
        "/models/cosmos3.trtfb",
        "--prompt",
        "one prompt; still one argument",
        "--output",
        str(frames),
        "--seed",
        "7",
    )
    assert build_generate_command(
        replace(config, cosmos3_cp_size=4),
        prompt="prompt",
        frames_dir=frames,
        seed=8,
    )[:4] == ("mpirun", "-np", "4", "/opt/trtmc/bin/trtmc")
    assert "--num-frames" not in command
    assert "--output-dir" not in command


def test_ffmpeg_commands_fix_frame_contract_and_keep_caption_out_of_filter() -> None:
    horizontal, social = build_ffmpeg_commands()
    social_filter = social[social.index("-filter_complex") + 1]

    for command in (horizontal, social):
        assert command[command.index("-i") + 1] == "frames/frame_%04d.png"
        assert command[command.index("-frames:v") + 1] == str(FRAME_COUNT)
        assert command[command.index("-framerate") + 1] == "24"
        assert command[command.index("-r") + 1] == "24"
        assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert horizontal[-1] == "horizontal.mp4"
    assert social[-1] == "social.mp4"
    assert "scale=720:1280" in social_filter
    assert "textfile=caption.txt:expansion=none" in social_filter
    assert "%{" not in social_filter


def test_pipeline_runner_is_injectable_and_caption_never_enters_ffmpeg_argv(
    tmp_path: Path,
) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_runner(argv: Sequence[str], cwd: Path) -> None:
        command = tuple(argv)
        commands.append(command)
        if "generate-video" in command:
            frames = Path(command[command.index("--output") + 1])
            for index in range(FRAME_COUNT):
                (frames / f"frame_{index:04d}.png").write_bytes(b"png")
        else:
            (cwd / command[-1]).write_bytes(b"mp4")

    payload = _payload()
    payload["cta"] = "100%: share this; [still plain text]"
    submission = validate_submission(payload)
    config = AppConfig(output_root=tmp_path)
    job_dir = tmp_path / "3bea468f-7333-4d0a-ad11-f1ca9b107a0f"
    job_dir.mkdir()
    progress: list[int] = []

    outputs = StoryScenePipeline(config, runner=fake_runner).run(
        job_dir,
        submission,
        compile_prompt(submission),
        progress.append,
    )

    assert outputs == {"horizontal": "horizontal.mp4", "social": "social.mp4"}
    assert progress == [70, 85, 95]
    caption_lines = submission.caption.splitlines()
    assert 1 <= len(caption_lines) <= 3
    assert all(len(line) <= 26 for line in caption_lines)
    assert (job_dir / "caption.txt").read_text(encoding="utf-8") == submission.caption
    assert submission.caption not in " ".join(commands[1] + commands[2])


def test_default_runner_uses_no_shell_and_strips_secret_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> object:
        captured["argv"] = argv
        captured.update(kwargs)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setenv("HF_TOKEN", "must-not-reach-child")
    monkeypatch.setenv("UNRELATED", "kept")
    monkeypatch.setattr(subprocess, "run", fake_run)

    run_command(("tool;not-a-shell", "$(not-expanded)"), tmp_path)

    assert captured["argv"] == ["tool;not-a-shell", "$(not-expanded)"]
    assert captured["shell"] is False
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert "HF_TOKEN" not in environment
    assert environment["UNRELATED"] == "kept"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert captured["stdout"] is subprocess.DEVNULL
    assert captured["stderr"] is subprocess.DEVNULL


def _wait_for_status(manager: JobManager, job_id: str, status: str) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if manager.get(job_id).status == status:
            return
        time.sleep(0.005)
    raise AssertionError(f"job {job_id} did not reach {status}")


def test_job_manager_uses_unique_dirs_and_one_serial_worker(tmp_path: Path) -> None:
    entered: list[str] = []
    first_started = threading.Event()
    release_first = threading.Event()

    def processor(
        job_dir: Path,
        submission: object,
        _prompt: str,
        update_progress: object,
    ) -> dict[str, str]:
        subject = getattr(submission, "subject")
        entered.append(subject)
        if len(entered) == 1:
            first_started.set()
            assert release_first.wait(2)
        update_progress(70)  # type: ignore[operator]
        (job_dir / "horizontal.mp4").write_bytes(b"clean")
        (job_dir / "social.mp4").write_bytes(b"social")
        return {"horizontal": "horizontal.mp4", "social": "social.mp4"}

    manager = JobManager(tmp_path, processor)
    try:
        first_submission = validate_submission(_payload())
        second_payload = _payload("custom")
        second_payload["subject"] = "A second subject that must wait"
        second_submission = validate_submission(second_payload)
        first = manager.submit(first_submission)
        assert first_started.wait(2)
        second = manager.submit(second_submission)

        assert first.job_id != second.job_id
        assert manager.get(second.job_id).status == "queued"
        assert entered == [first_submission.subject]
        assert (tmp_path / first.job_id).is_dir()
        assert (tmp_path / second.job_id).is_dir()

        release_first.set()
        first_done = manager.wait(first.job_id)
        second_done = manager.wait(second.job_id)

        assert first_done.status == "succeeded"
        assert first_done.progress == 100
        assert second_done.status == "succeeded"
        assert entered == [first_submission.subject, second_submission.subject]
        assert first_done.clean_video_url == (
            f"/outputs/{first.job_id}/horizontal.mp4"
        )
        assert first_done.social_video_url == f"/outputs/{first.job_id}/social.mp4"
    finally:
        release_first.set()
        manager.close()


def test_job_failure_is_sanitized_and_worker_survives(tmp_path: Path) -> None:
    def fail(
        _job_dir: Path,
        _submission: object,
        _prompt: str,
        _update_progress: object,
    ) -> dict[str, str]:
        raise RuntimeError("hf_private-token-must-not-escape")

    manager = JobManager(tmp_path, fail)
    try:
        job = manager.submit(validate_submission(_payload()))
        failed = manager.wait(job.job_id)

        assert failed.status == "failed"
        assert failed.progress == 100
        assert "hf_private" not in (failed.error or "")
        assert manager.is_alive
    finally:
        manager.close()


def test_static_path_resolution_stays_inside_root(tmp_path: Path) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("ok", encoding="utf-8")

    assert _safe_static_file(static, "index.html") == static / "index.html"
    assert _safe_static_file(static, "../secret") is None
    assert _safe_static_file(static, "%2e%2e/secret") is None
    assert _safe_static_file(static, "..%5csecret") is None


def test_config_parses_documented_environment_and_rejects_invalid_cp(
    tmp_path: Path,
) -> None:
    config = AppConfig.from_env(
        {
            "TRTMC_BIN": "/opt/trtmc/bin/trtmc",
            "COSMOS3_BUNDLE": "/models/custom.trtfb",
            "COSMOS3_CP_SIZE": "8",
            "OUTPUT_ROOT": str(tmp_path),
            "HOST": "127.0.0.1",
            "PORT": "9090",
        }
    )

    assert config.cosmos3_cp_size == 8
    assert config.port == 9090
    assert config.output_root == tmp_path
    with pytest.raises(ValueError, match="one of"):
        AppConfig.from_env({"COSMOS3_CP_SIZE": "3"})


def test_health_submission_status_and_range_output_api(tmp_path: Path) -> None:
    def fake_runner(argv: Sequence[str], cwd: Path) -> None:
        command = tuple(argv)
        if "generate-video" in command:
            frames = Path(command[command.index("--output") + 1])
            for index in range(FRAME_COUNT):
                (frames / f"frame_{index:04d}.png").write_bytes(b"png")
        else:
            (cwd / command[-1]).write_bytes(b"mp4")

    config = AppConfig(
        output_root=tmp_path,
        host="127.0.0.1",
        port=0,
    )
    server, manager = create_server(config, runner=fake_runner)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        server.server_address[1],
        timeout=2,
    )
    try:
        connection.request("GET", "/healthz")
        health = connection.getresponse()
        assert health.status == 200
        assert json.loads(health.read())["worker_ready"] is True

        body = json.dumps(_payload()).encode("utf-8")
        connection.request(
            "POST",
            "/api/jobs",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        submitted_response = connection.getresponse()
        assert submitted_response.status == 202
        submitted = json.loads(submitted_response.read())
        assert submitted["compiled_prompt"]

        manager.wait(submitted["job_id"])
        connection.request("GET", f"/api/jobs/{submitted['job_id']}")
        status_response = connection.getresponse()
        completed = json.loads(status_response.read())
        assert completed["status"] == "succeeded"
        assert completed["social_video_url"].endswith("/social.mp4")

        connection.request(
            "GET",
            completed["social_video_url"],
            headers={"Range": "bytes=1-2"},
        )
        output_response = connection.getresponse()
        assert output_response.status == 206
        assert output_response.getheader("Content-Range") == "bytes 1-2/3"
        assert output_response.read() == b"p4"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        manager.close()
        thread.join(2)
