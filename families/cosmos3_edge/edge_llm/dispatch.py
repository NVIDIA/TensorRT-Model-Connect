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
    if request.precision.lower() != "fp16" or sys.platform != "linux" or not builder.package_present():
        native(request, writer)
        return
    descriptor, name = tempfile.mkstemp(
        prefix=f".{request.output_path.name}.edge-", suffix=".log", dir=request.output_path.parent)
    os.close(descriptor)
    log_path = Path(name)
    failure = None
    with tempfile.TemporaryDirectory(
            prefix=f".{request.output_path.name}.edge-", dir=request.output_path.parent) as directory:
        try:
            target = builder.local_target()
            files, marker = builder.prepare(request, target, Path(directory), log_path, capacity)
        except Exception as error:
            failure = error
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning(
                "Cosmos3-Edge build failed: %s. Diagnostics: %s. "
                "Retrying native once with the unchanged request.", error, log_path, exc_info=True)
        except BaseException:
            log_path.unlink(missing_ok=True)
            raise
        else:
            # Do not put publication inside the preparation try block.
            builder.publish(request, writer, files, marker)
            log_path.unlink()
            return
    try:
        native(request, writer)
    except Exception as error:
        if failure is not None:
            raise error from failure
        raise
