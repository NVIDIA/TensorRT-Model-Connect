# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local Cosmos3 and FFmpeg execution with argv-only subprocesses."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
import subprocess

from .config import AppConfig
from .prompts import Submission


FRAME_COUNT = 189
FRAME_RATE = 24
Command = tuple[str, ...]
CommandRunner = Callable[[Sequence[str], Path], None]
ProgressCallback = Callable[[int], None]


class PipelineError(RuntimeError):
    """A local pipeline failure with a message safe for job status."""


class CommandFailed(PipelineError):
    def __init__(self, returncode: int):
        super().__init__(f"A local media command failed with exit code {returncode}.")
        self.returncode = returncode


def build_generate_command(
    config: AppConfig,
    *,
    prompt: str,
    frames_dir: Path,
    seed: int,
) -> Command:
    """Build the recovered Cosmos3 CLI contract without a command shell."""

    command = (
        config.trtmc_bin,
        "generate-video",
        str(config.cosmos3_bundle),
        "--prompt",
        prompt,
        "--output",
        str(frames_dir),
        "--seed",
        str(seed),
    )
    if config.cosmos3_cp_size > 1:
        return ("mpirun", "-np", str(config.cosmos3_cp_size), *command)
    return command


def build_ffmpeg_commands() -> tuple[Command, Command]:
    """Return horizontal and captioned 9:16 packaging commands.

    Commands run inside a UUID job directory. All filter-graph paths are fixed,
    so neither user text nor an environment-provided output path is interpreted
    as FFmpeg filter syntax.
    """

    common_input = (
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(FRAME_RATE),
        "-start_number",
        "0",
        "-i",
        "frames/frame_%04d.png",
        "-frames:v",
        str(FRAME_COUNT),
    )
    encoding = (
        "-r",
        str(FRAME_RATE),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    )
    horizontal = (*common_input, *encoding, "horizontal.mp4")
    social_filter = (
        "[0:v]split=2[bgsrc][fgsrc];"
        "[bgsrc]scale=720:1280:force_original_aspect_ratio=increase,"
        "crop=720:1280,gblur=sigma=24[bg];"
        "[fgsrc]scale=720:1280:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        "drawtext=textfile=caption.txt:expansion=none:fontcolor=white:"
        "fontsize=46:borderw=4:bordercolor=black@0.85:"
        "box=1:boxcolor=black@0.28:boxborderw=20:"
        "x=(w-text_w)/2:y=h-text_h-96[v]"
    )
    social = (
        *common_input,
        "-filter_complex",
        social_filter,
        "-map",
        "[v]",
        *encoding,
        "social.mp4",
    )
    return horizontal, social


def _offline_environment(source: Mapping[str, str]) -> dict[str, str]:
    environment = {
        name: value
        for name, value in source.items()
        if not any(
            marker in name.upper()
            for marker in ("TOKEN", "PASSWORD", "SECRET", "API_KEY")
        )
    }
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def run_command(argv: Sequence[str], cwd: Path) -> None:
    """Run one local command without a shell or secret-bearing environment."""

    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            env=_offline_environment(os.environ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            shell=False,
        )
    except OSError as exc:
        raise PipelineError("A required local executable is unavailable.") from exc
    if completed.returncode != 0:
        raise CommandFailed(completed.returncode)


def _verify_frames(frames_dir: Path) -> None:
    expected = {f"frame_{index:04d}.png" for index in range(FRAME_COUNT)}
    actual = {
        path.name
        for path in frames_dir.glob("frame_*.png")
        if path.is_file()
    }
    if actual != expected:
        raise PipelineError(
            "Cosmos3 did not produce exactly frame_0000.png through frame_0188.png."
        )


def _verify_video(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise PipelineError("FFmpeg did not produce the expected video output.")


class StoryScenePipeline:
    """Generate frames, then serialize both customer-ready MP4 variants."""

    def __init__(
        self,
        config: AppConfig,
        runner: CommandRunner = run_command,
    ) -> None:
        self._config = config
        self._runner = runner

    def run(
        self,
        job_dir: Path,
        submission: Submission,
        compiled_prompt: str,
        update_progress: ProgressCallback,
    ) -> dict[str, str]:
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(mode=0o700)
        (job_dir / "caption.txt").write_text(
            submission.caption,
            encoding="utf-8",
        )

        generate = build_generate_command(
            self._config,
            prompt=compiled_prompt,
            frames_dir=frames_dir,
            seed=submission.seed,
        )
        self._runner(generate, job_dir)
        _verify_frames(frames_dir)
        update_progress(70)

        horizontal, social = build_ffmpeg_commands()
        self._runner(horizontal, job_dir)
        _verify_video(job_dir / "horizontal.mp4")
        update_progress(85)
        self._runner(social, job_dir)
        _verify_video(job_dir / "social.mp4")
        update_progress(95)
        return {
            "horizontal": "horizontal.mp4",
            "social": "social.mp4",
        }
