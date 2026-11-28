# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transactional family-owned complete-network selection."""

import logging
import os
from pathlib import Path
import sys
import tempfile
import traceback

from . import builder

_LOG = logging.getLogger(__name__)


def build(request, writer, native, capacity: int):
    """Use compatible local Edge FP16 tools, otherwise the unchanged native request.

    There is no family GPU restriction. The optional SDK advertises its own
    compiled architectures; all other targets retain pure TensorRT construction.
    Inference never silently retries a failed Edge request on another backend.
    """
    # Release video placeholder admission fails for real timestamped inputs.
    # Never disguise a video as images or drop audio from an aligned request.
    edge_tasks = {
        "text_generation",
        "vision_language_generation",
        "text_continuation",
        "images_text_to_text",
    }
    if (
        request.task not in edge_tasks | builder.ACOUSTIC_TASKS
        or request.precision.lower() != "fp16"
        or sys.platform != "linux"
        or not builder.package_present()
    ):
        native(request, writer)
        return
    descriptor, name = tempfile.mkstemp(
        prefix=f".{request.output_path.name}.edge-", suffix=".log", dir=request.output_path.parent
    )
    os.close(descriptor)
    log_path = Path(name)
    failure = None
    with tempfile.TemporaryDirectory(
        prefix=f".{request.output_path.name}.edge-", dir=request.output_path.parent
    ) as directory:
        try:
            target = builder.local_target()
            acoustic = request.task in builder.ACOUSTIC_TASKS
            package = builder.installed_package(target, acoustic=acoustic)
            eligible = not acoustic or (
                package["version"] == "0.12.0" and package.get("onnx") is True
            )
            if eligible:
                files, marker = builder.prepare(
                    request, target, Path(directory), log_path, capacity, package
                )
        except Exception as error:
            failure = error
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning(
                "Nemotron Omni build failed: %s. Diagnostics: %s. "
                "Retrying native once with the unchanged request.",
                error,
                log_path,
                exc_info=True,
            )
        except BaseException:
            log_path.unlink(missing_ok=True)
            raise
        else:
            # Do not put publication inside the preparation try block.
            if eligible:
                builder.publish(request, writer, files, marker)
                log_path.unlink()
                return
            log_path.unlink()
    try:
        native(request, writer)
    except Exception as error:
        if failure is not None:
            raise error from failure
        raise
