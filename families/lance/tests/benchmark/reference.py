#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the family-owned Lance reference."""

from __future__ import annotations

import json
from pathlib import Path
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
    reference_repo = Path(str(options.get("reference_repo", ""))).resolve()
    if not reference_repo.is_dir():
        raise ValueError("Lance reference requires reference_repo")
    if arguments.mode != "pytorch-eager" or arguments.precision != "bf16":
        raise ValueError("Lance reference requires pytorch-eager bf16")
    with tempfile.TemporaryDirectory(prefix="trtmc-perf-lance-") as temporary:
        output = Path(temporary) / "result.json"
        command = [
            sys.executable,
            str(Path(__file__).with_name("upstream_reference.py")),
            "--reference-repo",
            str(reference_repo),
            "--model",
            arguments.model,
            "--model-subdir",
            str(options.get("model_subdir", "Lance_3B")),
            "--vit-subdir",
            str(options.get("vit_subdir", "Qwen2.5-VL-ViT")),
            "--image",
            str(request["image_path"]),
            "--prompt",
            str(request.get("prompt", "")),
            "--instruction",
            str(options.get("instruction", "Look at the image carefully and answer the question.")),
            "--max-new-tokens",
            str(request.get("max_new_tokens", 16)),
            "--warmup",
            str(arguments.warmup),
            "--iterations",
            str(arguments.iterations),
            "--resolution",
            str(options.get("resolution", "image_768res")),
            "--height",
            str(options.get("height", 768)),
            "--width",
            str(options.get("width", 768)),
            "--output",
            str(output),
        ]
        if arguments.revision:
            command.extend(("--revision", arguments.revision))
        if arguments.local_files_only:
            command.append("--local-files-only")
        completed = subprocess.run(
            command,
            cwd=reference_repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Lance reference failed with rc={completed.returncode}: {completed.stderr[-2000:]}"
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
    return reference_harness.Measurement(
        [float(value) for value in payload.get("samples_ms", [])],
        {"text": str(payload.get("text", "")), "output_tokens": None},
        "lance-pytorch",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
        asset_loading_included=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return reference_harness.run_premeasured(
        list(sys.argv[1:] if argv is None else argv),
        description=__doc__,
        execute=_execute,
    )


if __name__ == "__main__":
    raise SystemExit(main())
