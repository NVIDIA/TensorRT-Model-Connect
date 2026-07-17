#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble the durable Wan2.2 native SDPA proof summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _lane(root: Path, capture_dir: Path, native_dir: Path) -> dict[str, object]:
    capture = _load(root / capture_dir / "manifest.json")
    build = _load(root / native_dir / "build_report.json")
    run = _load(root / native_dir / "run_report.json")
    qualification = _load(root / native_dir / "qualification.json")
    official_log = (root / capture_dir / "cudnn_frontend.log").read_text()
    native_log = (root / native_dir / "cudnn_frontend_run.log").read_text()
    official_configs = re.findall(
        r"Heuristic query for mode 3 has (\d+) configurations", official_log
    )
    official_selected = re.findall(
        r"Check support for index 0 passed with cfg ([^\n]+)", official_log
    )
    native_selected = re.findall(r"get_plan_name_at_index\(0\) is ([^\n]+)", native_log)
    return {
        "capture_kind": capture["kind"],
        "torch_default_vs_forced_cudnn": capture["comparisons"]["default_vs_forced_cudnn"],
        "wan_wrapper_vs_forced_cudnn": capture["comparisons"]["wan_source_vs_forced_cudnn"],
        "official_heuristic_candidate_count": int(official_configs[-1]),
        "official_selected_plan": official_selected[-1],
        "native_selected_plan": native_selected[-1],
        "workspace_bytes": 0,
        "mean_ms": run["mean_ms"],
        "build": build,
        "qualification": qualification,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ldd_path = args.artifact_root / "native" / "ldd_plugin.txt"
    ldd_text = ldd_path.read_text()
    dependencies = [
        match.group(1)
        for line in ldd_text.splitlines()
        if (match := re.match(r"\s*([^\s]+)\s+=>", line)) is not None
    ]
    forbidden = [
        dependency
        for dependency in dependencies
        if dependency.startswith(("libtorch", "libc10", "libpython"))
    ]
    report = {
        "kind": "wan2_2_ti2v_native_cudnn_sdpa_proof_summary",
        "scope": "isolated_probe_not_integrated_into_production_dit",
        "hardware": "NVIDIA GB300 SM103",
        "software": {
            "torch_reference": "2.12.0+cu130 git 7661cd9c6b841b62b7f411aa52ec51f05457263b",
            "tensorrt": "11.0.0.114",
            "cudnn": "9.20.0",
            "cudnn_frontend": "1.22.1 tag v1.22.1 commit a91f0e04dcea10515f0f776fc5a89535e316a9c8",
        },
        "native_apis": [
            "TensorRT IPluginV2DynamicExt",
            "cuDNN frontend Graph SDPA",
            "cuDNN backend pinned execution plan",
            "CUDA Runtime streams/events/memory",
        ],
        "self_attention": _lane(args.artifact_root, Path("capture"), Path("native")),
        "cross_attention": _lane(args.artifact_root, Path("cross/capture"), Path("cross/native")),
        "plugin_dependencies": dependencies,
        "forbidden_runtime_dependencies": forbidden,
        "no_torch_aten_python_dependency": not forbidden,
        "ldd_proof": str(ldd_path),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not forbidden else 2


if __name__ == "__main__":
    raise SystemExit(main())
