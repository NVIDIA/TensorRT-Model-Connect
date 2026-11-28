# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select complete-network offload without changing model or precision."""

import logging
import subprocess
import tempfile
from pathlib import Path

from tensorrt_model_connect.build import detect_local_platform

from . import builder


def build(request, writer, input_limit, capacity, native):
    logfile = request.output_path.with_suffix(request.output_path.suffix + ".edge.log")
    residency = getattr(request, "edge_ple_residency", "resident")
    if residency not in ("resident", "cached"):
        raise ValueError("Unknown Flash-Next Edge PLE residency")
    with tempfile.TemporaryDirectory(prefix="flash-next-", dir=request.output_path.parent) as work:
        engines = Path(work)
        try:
            target = detect_local_platform()
            package = builder.sdk(target, getattr(request, "edge_builder_python", None))
            with logfile.open("a", encoding="utf-8") as log:
                builder.run_build(
                    request.model_dir,
                    engines,
                    package,
                    input_limit,
                    capacity,
                    request.task == "vision_language_generation",
                    log,
                )
            assets = builder.prepare_assets(
                engines, request.model_dir, request.task, input_limit, capacity
            )
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            logging.getLogger(__name__).warning(
                "Flash-Next Edge preparation failed: %s. See %s. Falling back to native TensorRT.",
                exc,
                logfile,
            )
        else:
            # Do not retry into a writer that has already accepted Edge sections.
            builder.publish(
                writer, assets, request.task, target, package[2], input_limit, capacity, residency
            )
            return
    native(request, writer)
