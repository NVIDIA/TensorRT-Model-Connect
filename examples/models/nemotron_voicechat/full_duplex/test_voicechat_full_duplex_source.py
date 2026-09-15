# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-only contracts for the local Nemotron VoiceChat microphone example."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = REPO_ROOT / "examples" / "models" / "nemotron_voicechat" / "full_duplex"


def _text(name: str) -> str:
    return (EXAMPLE / name).read_text(encoding="utf-8")


def _docker_from_images(source: str) -> list[str]:
    arguments: dict[str, str] = {}
    images: list[str] = []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if line.startswith("ARG ") and "=" in line:
            name, value = line.removeprefix("ARG ").split("=", 1)
            arguments[name] = value
        if not line.startswith("FROM "):
            continue
        image = line.split()[1]
        variable = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", image)
        images.append(arguments.get(variable.group(1), "") if variable else image)
    return images


def _shell_blocks(markdown: str) -> list[str]:
    return re.findall(r"```(?:bash|sh)\n(.*?)```", markdown, flags=re.DOTALL)


def test_docker_contract_is_pinned_and_starts_the_example() -> None:
    dockerfile = _text("Dockerfile")
    images = _docker_from_images(dockerfile)

    assert images
    assert all(re.fullmatch(r"\S+@sha256:[0-9a-f]{64}", image) for image in images)

    entrypoint_match = re.search(r"(?m)^ENTRYPOINT\s+(\[[^\n]+\])$", dockerfile)
    command_match = re.search(r"(?m)^CMD\s+(\[[^\n]+\])$", dockerfile)
    assert entrypoint_match is not None
    assert command_match is not None
    assert json.loads(entrypoint_match.group(1)) == ["/opt/trtmc/bin/trtmc_voicechat_full_duplex"]
    assert json.loads(command_match.group(1)) == ["/models/model.bundle"]


def test_readme_documents_one_off_build_and_offline_device_scoped_run() -> None:
    readme = _text("README.md")
    blocks = _shell_blocks(readme)
    build_blocks = [block for block in blocks if "docker build" in block]
    run_blocks = [block for block in blocks if "docker run" in block]
    conversation_blocks = [block for block in run_blocks if "--gpus" in block]

    assert build_blocks
    assert conversation_blocks
    assert readme.index("docker build") < readme.index("docker run")
    for block in run_blocks:
        assert "--privileged" not in block
    for block in conversation_blocks:
        assert "--network none" in block
        assert "--device /dev/snd:/dev/snd" in block
        assert "readonly" in block or ":ro" in block
    assert "headset" in readme.lower()
    assert "python -m tensorrt_model_connect build" in readme
    assert "--revision 359ada7b1c60851e40ff08065f9b0340244f27e0" in readme
    assert "--quantization int8" in readme
    assert "--max-sequence-length 512" in readme
    assert "--playback-gain-db" in readme
    assert "160 ms" in readme
    assert "rolls recurrent generation state" in readme
    assert "clipped-sample count" in readme


def test_application_wires_alsa_capture_session_events_and_barge_in_flush() -> None:
    cmake = _text("CMakeLists.txt")
    source = _text("main.cpp")

    assert "find_package(ALSA REQUIRED)" in cmake
    assert "find_package(Threads REQUIRED)" in cmake
    assert "ALSA::ALSA" in cmake
    assert "Threads::Threads" in cmake
    assert "TRTMC_BUILD_EXAMPLES OFF" in cmake

    for symbol in (
        "snd_pcm_open",
        "SND_PCM_NONBLOCK",
        "snd_pcm_set_params",
        "snd_pcm_readi",
        "snd_pcm_wait",
        "snd_pcm_writei",
        "snd_pcm_recover",
        "-ESTRPIPE",
        "snd_pcm_drop",
        "snd_pcm_prepare",
        "ISpeechSessionProvider",
        "create_speech_session",
        "SpeechSessionEventKind::kYielded",
        "SpeechSessionEventKind::kContextRolled",
    ):
        assert symbol in source
    assert source.count("std::thread") >= 2
    assert source.count("joinable()") >= 2
    assert re.search(
        r"SpeechSessionEventKind::kYielded[\s\S]{0,800}request_flush\s*\(",
        source,
    )
    assert re.search(
        r"PlaybackQueueItemKind::kFlush[\s\S]{0,800}playback\.flush_playback\s*\(",
        source,
    )
    rollover_case = re.search(
        r"case SpeechSessionEventKind::kContextRolled:([\s\S]*?)break;", source
    )
    assert rollover_case is not None
    assert "request_flush" not in rollover_case.group(1)
    flush_method = re.search(
        r"void\s+flush_playback\s*\(\)\s*\{([\s\S]{0,1200}?)\n    \}",
        source,
    )
    assert flush_method is not None
    assert "snd_pcm_drop" in flush_method.group(1)
    assert "snd_pcm_prepare" in flush_method.group(1)

    read_method = re.search(
        r"std::size_t\s+read_frames\s*\([^)]*\)\s*\{([\s\S]{0,2400}?)\n    \}",
        source,
    )
    assert read_method is not None
    read_body = read_method.group(1)
    for symbol in ("snd_pcm_state", "SND_PCM_STATE_PREPARED", "snd_pcm_start"):
        assert symbol in read_body
    assert read_body.index("snd_pcm_state") < read_body.index("snd_pcm_start")
    assert read_body.index("snd_pcm_start") < read_body.index("snd_pcm_readi")

    capacity = re.search(r"constexpr int kPlaybackQueueSeconds = (\d+);", source)
    assert capacity is not None
    # The model-owned response limit is 256 x 80 ms = 20.48 seconds. Keep the
    # event consumer non-blocking for barge-in while bounding buffered PCM.
    assert int(capacity.group(1)) >= 21
    assert "four-second bound" not in source

    for event_serializer in (
        (REPO_ROOT / "apps" / "cli" / "cli.cpp").read_text(encoding="utf-8"),
        (
            REPO_ROOT
            / "families"
            / "nemotron_voicechat"
            / "tests"
            / "cpp"
            / "native_lifecycle_probe.cpp"
        ).read_text(encoding="utf-8"),
    ):
        assert re.search(
            r"case (?:SpeechSessionEventKind|EventKind)::kContextRolled:\s*"
            r'return "context_rolled";',
            event_serializer,
        )

    assert re.search(
        r"AlsaPcm\s+capture\([^;]*SND_PCM_STREAM_CAPTURE[^;]*,\s*1U\s*,",
        source,
        flags=re.DOTALL,
    )
    assert re.search(
        r"AlsaPcm\s+playback\([^;]*SND_PCM_STREAM_PLAYBACK[^;]*,\s*2U\s*,",
        source,
        flags=re.DOTALL,
    )
    assert re.search(r"stereo\[frame\s*\*\s*2U\]\s*=\s*mono\[frame\]", source)
    assert re.search(r"stereo\[frame\s*\*\s*2U\s*\+\s*1U\]\s*=\s*mono\[frame\]", source)
    playback_loop = re.search(
        r"void\s+playback_loop\s*\([^)]*\)\s*noexcept\s*\{([\s\S]{0,7000}?)\n\}",
        source,
    )
    assert playback_loop is not None
    playback_body = playback_loop.group(1)
    assert "queue.try_pop()" in playback_body
    assert "queue.wait_pop()" not in playback_body
    assert "std::fill(mono.begin(), mono.end(), 0)" in playback_body
    assert "while (written < period_frames)" in playback_body


def test_playback_gain_cli_is_bounded_and_applied_once() -> None:
    source = _text("main.cpp")

    assert re.search(r"float\s+playback_gain_db\s*\{\s*0(?:\.0)?F?\s*\}", source)
    assert re.search(r"--playback-gain-db[^\n]*default:\s*0", source)

    parser = re.search(
        r'if\s*\(argument\s*==\s*"--playback-gain-db"\)\s*\{([\s\S]{0,800}?)\n\s*\}',
        source,
    )
    assert parser is not None
    parser_body = parser.group(1)
    assert "playback_gain_db" in parser_body
    assert re.search(r"-24(?:\.0)?F?", parser_body)
    assert re.search(r"(?<!-)24(?:\.0)?F?", parser_body)

    float_parser = re.search(
        r"float\s+parse_float\s*\([^)]*\)\s*\{([\s\S]{0,1200}?)\n\}",
        source,
    )
    assert float_parser is not None
    assert "std::isfinite" in float_parser.group(1)
    assert "parsed < minimum" in float_parser.group(1)
    assert "parsed > maximum" in float_parser.group(1)

    gain_resolution = re.findall(
        r"playback_gain_from_db\s*\(\s*options\.playback_gain_db\s*\)", source
    )
    assert len(gain_resolution) == 1
    assert re.search(
        r"float_to_pcm16\s*\(\s*sample\s*,\s*playback_gain\s*\)", source
    )
    assert re.search(
        r"consume_event\s*\([^;]*playback_gain[^;]*\)", source, flags=re.DOTALL
    )


def test_playback_queue_is_pure_cpp_and_runs_without_alsa(tmp_path: Path) -> None:
    header = _text("playback_queue.h")
    test_source = _text("test_playback_queue.cpp")
    for source in (header, test_source):
        assert "alsa/" not in source
        assert "snd_pcm_" not in source

    compiler_command = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler_command or shutil.which(compiler_command[0]) is None:
        pytest.skip("a C++ compiler is not installed")

    executable = tmp_path / "test_playback_queue"
    compile_result = subprocess.run(
        [
            *compiler_command,
            "-std=c++17",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(EXAMPLE),
            str(EXAMPLE / "test_playback_queue.cpp"),
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
