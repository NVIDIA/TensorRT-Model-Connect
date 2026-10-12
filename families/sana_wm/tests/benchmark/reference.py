#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the family-owned SANA-WM reference."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from qualification_tests.benchmark_qualification.performance import reference_harness


def _execute(
    arguments: Any,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> reference_harness.Measurement:
    options = dict(options)
    testcase_name = getattr(arguments, "testcase_name", None)
    if testcase_name is not None:
        manifest_value = json.loads(arguments.manifest.read_text(encoding="utf-8"))
        matches = [
            value
            for value in manifest_value.get("testcases", [])
            if isinstance(value, Mapping) and value.get("name") == testcase_name
        ]
        if len(matches) != 1:
            raise ValueError(f"SANA-WM requires exactly one testcase named {testcase_name!r}")
        for name in (
            "translation_speed",
            "rotation_speed_deg",
            "fps",
            "flow_shift",
            "no_action_overlay",
        ):
            if name in matches[0]:
                options.setdefault(name, matches[0][name])
    reference_repo = Path(str(options.get("reference_repo", ""))).resolve()
    if not reference_repo.is_dir():
        raise ValueError("SANA-WM reference requires reference_repo")
    model_root = arguments.manifest.resolve().parent.parent

    def model_input(value: Any) -> Path:
        path = Path(str(value))
        return path.resolve() if path.is_absolute() else (model_root / path).resolve()

    image = model_input(request["image_path"])
    intrinsics = model_input(
        options.get("intrinsics_path", options.get("intrinsics", "assets/demo_0_intrinsics.npy"))
    )
    for label, path in (("image", image), ("intrinsics", intrinsics)):
        if not path.is_file():
            raise FileNotFoundError(f"SANA-WM {label} input does not exist: {path}")
    prompt_text = str(request.get("prompt", ""))
    if not prompt_text:
        raise ValueError("SANA-WM request requires prompt")
    action = request.get("action")
    if not isinstance(action, str) or not action:
        raise ValueError("SANA-WM request requires a non-empty action")
    if "model_dir" not in options:
        raise ValueError("SANA-WM reference requires model_dir")
    model_dir = Path(str(options["model_dir"])).resolve()
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    num_frames = int(
        request["num_frames"]
        if "num_frames" in request
        else manifest["video_num_frames"]
    )
    with tempfile.TemporaryDirectory(prefix="trtmc-perf-sana-wm-") as temporary:
        root = Path(temporary)
        output = root / "benchmark.json"
        prompt = root / "prompt.txt"
        prompt.write_text(prompt_text, encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).with_name("upstream_reference.py")),
            "--reference-repo",
            str(reference_repo),
            "--image",
            str(image),
            "--model-dir",
            str(model_dir),
            "--prompt",
            str(prompt),
            "--action",
            action,
            "--intrinsics",
            str(intrinsics),
            "--translation_speed",
            str(
                request["translation_speed"]
                if "translation_speed" in request
                else options["translation_speed"]
            ),
            "--rotation_speed_deg",
            str(
                request["rotation_speed_deg"]
                if "rotation_speed_deg" in request
                else options["rotation_speed_deg"]
            ),
            "--num_frames",
            str(num_frames),
            "--fps",
            str(request["fps"] if "fps" in request else options["fps"]),
            "--step",
            str(request["num_steps"]),
            "--cfg_scale",
            str(request["cfg_scale"]),
            "--flow_shift",
            str(
                request["flow_shift"] if "flow_shift" in request else options["flow_shift"]
            ),
            "--seed",
            str(request["seed"]),
            "--refiner_seed",
            str(request["seed"]),
            "--warmup",
            str(arguments.warmup),
            "--iterations",
            str(arguments.iterations),
            "--output",
            str(output),
        ]
        no_overlay = (
            request["no_action_overlay"]
            if "no_action_overlay" in request
            else options["no_action_overlay"]
        )
        if bool(no_overlay):
            command.append("--no_action_overlay")
        completed = subprocess.run(
            command,
            cwd=reference_repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"SANA-WM reference failed with rc={completed.returncode}: "
                f"{completed.stderr[-4000:]}"
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
        summary = dict(payload.get("output_summary", {}))
        artifacts = summary.get("frame_artifacts", [])
        if artifacts:
            destination = arguments.output.with_suffix(".media").resolve()
            destination.mkdir(parents=True, exist_ok=True)
            copied = []
            for source_value in artifacts:
                source = Path(str(source_value))
                target = destination / source.name
                shutil.copyfile(source, target)
                copied.append(str(target))
            summary["frame_artifacts"] = copied
    return reference_harness.Measurement(
        [float(value) for value in payload.get("samples_ms", [])],
        summary,
        "sana-wm-pytorch",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
        asset_loading_included=False,
    )


def _run_compat(
    arguments: Any,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any], str, str, bool, bool]:
    """Compatibility shape for family-local unit tests written before the move."""
    value = _execute(arguments, request, options)
    return (
        list(value.samples_ms),
        dict(value.output_summary),
        value.backend,
        value.timing_scope,
        value.input_preparation_included,
        value.asset_loading_included,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return reference_harness.run_premeasured(
        list(sys.argv[1:] if argv is None else argv),
        description=__doc__,
        execute=_execute,
    )


if __name__ == "__main__":
    raise SystemExit(main())
