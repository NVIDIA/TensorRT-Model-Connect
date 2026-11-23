# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bolt owns complete-network selection. Never change the requested draft or precision."""

import logging
import subprocess
from pathlib import Path
import tempfile

from tensorrt_model_connect.build import detect_local_platform
from . import builder


def build(request, writer, draft, input_limit, capacity, native):
    logfile = request.output_path.with_suffix(request.output_path.suffix + ".edge.log")
    with tempfile.TemporaryDirectory(
        prefix="bolt-engines-", dir=request.output_path.parent
    ) as work:
        staging = Path(work)
        try:
            target = detect_local_platform()
            if draft is None or target.get("sm") == 120:
                pkg = builder.sdk(target, getattr(request, "edge_builder_python", None))
                with logfile.open("a", encoding="utf-8") as log:
                    builder.run_build(
                        request.model_dir,
                        draft,
                        staging,
                        pkg,
                        input_limit,
                        capacity,
                        request.task == "vision_language_generation",
                        log,
                    )
                prepared = builder.prepare_assets(
                    request,
                    staging,
                    request.model_dir,
                    draft,
                    target,
                    pkg[2],
                    input_limit,
                    capacity,
                )
            else:
                # The released paired Edge engine is only supported on SM120.
                prepared = None
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            logging.getLogger(__name__).warning(
                "Bolt Edge preparation failed: %s. See %s. Falling back to native TensorRT.",
                exc,
                logfile,
            )
        else:
            if prepared is not None:
                builder.publish(writer, *prepared)
                return
    native(request, writer)
