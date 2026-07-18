# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-language contract: Python bundle writer -> C++ generic loader."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from tensorrt_model_connect.optimized_runtime.build_adapter import ProbeResult, run_build
from tensorrt_model_connect.optimized_runtime.bundle import write_optimized_bundle
from tensorrt_model_connect.optimized_runtime.manifest import (
    ImplementationRequest,
    load_implementation_manifest,
)


IMPLEMENTATION_ID = "example-optimized-runtime"
MODEL_ID = "Example/Optimized-Model"
RUNTIME_LIBRARY = "libtrtmc_impl_example_optimized_runtime.so"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.provider.name != RUNTIME_LIBRARY:
        raise RuntimeError(
            f"fake provider must be named {RUNTIME_LIBRARY}, got {arguments.provider.name}"
        )

    with tempfile.TemporaryDirectory(prefix="trtmc-writer-cpp-contract-") as temporary:
        root = Path(temporary)
        capsule = root / "capsule"
        (capsule / "builder").mkdir(parents=True)
        adapter_path = capsule / "builder" / "adapter.py"
        adapter_path.write_text(
            f"""import json
import pathlib
import shutil
import sys

request = json.loads(pathlib.Path(sys.argv[sys.argv.index("--request") + 1]).read_text())
output = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
artifacts = output / "artifacts"
(artifacts / "engine.dir").mkdir(parents=True)
(artifacts / "engine.dir" / "engine.plan").write_bytes(b"contract-engine")
shutil.copy2(request["parameters"]["provider"], artifacts / {RUNTIME_LIBRARY!r})
descriptor = {{
    "schema_version": 1,
    "build_binding": request["build_binding"],
    "private": {{"owned_by": {IMPLEMENTATION_ID!r}}},
}}
(output / "descriptor.json").write_text(json.dumps(descriptor, sort_keys=True))
print(json.dumps({{
    "schema_version": 1,
    "descriptor": "descriptor.json",
    "artifacts": "artifacts",
}}))
""",
            encoding="utf-8",
        )

        manifest_path = capsule / "IMPLEMENTATION.toml"
        manifest_path.write_text(
            f'''schema_version = 1
implementation_id = "{IMPLEMENTATION_ID}"
downstream_runtime = "test-optimized-runtime"
downstream_version = "test-runtime-1.0"
downstream_commit = "test-runtime-commit"

[model]
id = "{MODEL_ID}"
revisions = ["0123456789abcdef"]

[target]
os = "linux"

[build]
entrypoint = "builder/adapter.py"
timeout_seconds = 30

[runtime]
library = "{RUNTIME_LIBRARY}"
abi = 1
''',
            encoding="utf-8",
        )
        manifest = load_implementation_manifest(manifest_path)
        request = ImplementationRequest(
            model_id=MODEL_ID,
            model_revision="0123456789abcdef",
            target={"os": "linux"},
            parameters={
                "provider": str(arguments.provider),
            },
        )
        probe = ProbeResult(
            supported=True,
            profile_id="a100-fp16-b4",
        )
        build = run_build(manifest, request, root / "build", probe=probe)
        bundle = root / "writer-output.trtfb"
        write_optimized_bundle(bundle, manifest, request, build)

        completed = subprocess.run(
            [
                str(arguments.loader),
                "--load-writer-bundle",
                str(bundle),
                str(root / "runtime-cache"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "C++ loader rejected Python-writer bundle:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
