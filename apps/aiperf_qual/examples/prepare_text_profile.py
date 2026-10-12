# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fill a checked-in text-profile example with local paths; do not build or serve."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml


def absolute(value: str | Path) -> Path:
    # Resolving a venv's Python symlink would select its base interpreter instead.
    return Path(os.path.abspath(Path(value).expanduser()))


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path,
                        default=repo / "apps/aiperf_qual/config/text/qwen3-0.6b-fp16.yaml")
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--serve-python", default=sys.executable)
    parser.add_argument("--aiperf-python", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--include-streaming", action="store_true")
    args = parser.parse_args()
    if args.requests < 1 or args.warmup_requests < 0:
        parser.error("requests must be positive and warmup-requests must be nonnegative")

    build, bundle, out = map(absolute, (args.build_dir, args.bundle, args.out))
    serve_python, aiperf_python = map(absolute, (args.serve_python, args.aiperf_python))
    config = yaml.safe_load(absolute(args.template).read_text())
    replacements = {
        config["server"]["bundle"]: str(bundle),
        config["server"]["runtime_root"]: str(build),
        "/path/trtmc-python": str(serve_python),
    }
    config["validation_commands"] = [
        [next((replacement + argument[len(prefix):]
               for prefix, replacement in replacements.items()
               if argument == prefix or argument.startswith(prefix + "/")), argument)
         for argument in command]
        for command in config["validation_commands"]
    ]
    config["server"].update(bundle=str(bundle), binary=str(build / "trtmc-server"),
                            runtime_root=str(build))
    config["workload"].update(requests=args.requests, warmup_requests=args.warmup_requests)
    if args.include_streaming:
        config["workload"]["streaming"] = [False, True]

    for path in [bundle, serve_python, aiperf_python,
                 build / "trtmc", build / "trtmc-server",
                 *(Path(command[0]) for command in config["validation_commands"])]:
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if out.exists() and any(out.iterdir()):
        parser.error(f"configuration directory must be new or empty: {out}")

    cache = repo / ".ci/aiperf-text"
    cache.mkdir(parents=True, exist_ok=True)
    hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not hub_cache:
        hub_cache = str(Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub")
    environment = {
        "repo": str(repo), "serve_python": str(serve_python),
        "aiperf_python": str(aiperf_python), "aiperf": str(aiperf_python.parent / "aiperf"),
        "hf_hub_cache": str(absolute(hub_cache)),
        "hf_datasets_cache": str(cache / "datasets"), "gpu_lock": str(cache / "gpu.lock"),
        "build_env": {"TRTMC_BINARY": str(build / "trtmc"), "TRTMC_RUNTIME_ROOT": str(build)},
        "ports": {"candidate": config["server"]["port"], "reference": 8001},
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (("environment.yaml", environment), ("model.yaml", config)):
        path = out / name
        path.write_text(yaml.safe_dump(value, sort_keys=False))
        print(path)


if __name__ == "__main__":
    main()
