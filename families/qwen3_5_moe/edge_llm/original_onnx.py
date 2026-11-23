# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release 0.12 ONNX producer for the original fused Qwen MoE experts."""

import json
from pathlib import Path
import subprocess
import tempfile


def compile_engines(python, package, source, staging, log_path, limit, task, variant=None, draft=None):
    """Export the unmodified checkpoint and compile on the executing SDK/device."""
    if package.get("onnx") is not True:
        raise ValueError("The selected Edge SDK was built without the ONNX compilers")
    prefix = Path(package["install_prefix"])

    def tool(key):
        relative = Path(package[key])
        exe = (prefix / relative).resolve()
        if relative.is_absolute() or not exe.is_relative_to(prefix) or not exe.is_file():
            raise ValueError(f"Invalid Edge SDK compiler path: {key}")
        return str(exe)

    engine = staging / "edge_llm/engine"
    vision = task == "vision_language_generation"
    # The release compiler loads its plugin from its own SDK prefix.
    with tempfile.TemporaryDirectory(dir=staging) as tmp, log_path.open("a", encoding="utf-8") as log:
        onnx = Path(tmp)
        exporter = [python, "-I", "-c", "from tensorrt_edgellm.onnx.cli import main; main()",
                    str(source.resolve()), str(onnx), "--dtype", "float16", "--tp-size", "1"]
        compiler = tool("onnx_builder")
        limits = ["--engine-dir", str(engine), "--max-batch-size", "1",
                  "--max-input-len", str(limit), "--max-kv-cache-capacity", str(limit)]
        base_export = [*exporter, "--components", "thinker,visual" if vision else "thinker",
                       "--externalize-weights", "all"]
        base_build = [compiler, "--onnx-dir", str(onnx / "llm"), *limits]
        commands = [base_export, base_build]
        if variant is not None:
            if variant not in {"dflash", "jetspec"} or draft is None:
                raise ValueError("Original Qwen MoE needs a matching DFlash or JetSpec draft")
            companion = [f"--{variant}-draft-dir", str(draft.resolve())]
            base_export.extend([f"--{variant}-base", *companion])
            base_build.extend(["--spec-base", "--max-spec-verify-tokens", "16"])
            # The released draft-only exporter writes a separate graph, not a
            # second target. Both compiler roles publish into one engine directory.
            commands.insert(1, [*exporter, f"--{variant}-draft", *companion])
            commands.append([compiler, "--onnx-dir", str(onnx / f"{variant}_draft"), *limits,
                             "--spec-draft", "--max-spec-draft-tokens", "16"])
        if vision:
            commands.append([tool("visual_builder"), "--onnx-dir", str(onnx / "visual"),
                             "--engine-dir", str(engine), "--min-image-tokens", "4",
                             "--max-image-tokens", "1024", "--max-image-tokens-per-image", "512"])
        for command in commands:
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, check=True, cwd=staging,
                           stdout=log, stderr=subprocess.STDOUT)
