# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Errors raised by the TensorRT-Model-Connect serving control plane."""

from __future__ import annotations

from typing import Any


class ServeError(RuntimeError):
    """Base class for expected serving failures."""

    code = "serve_error"


class WorkerError(ServeError):
    """Base class for native worker failures."""

    code = "worker_error"


class WorkerStartupError(WorkerError):
    """A worker did not complete its ready handshake."""

    code = "worker_startup_failed"


class WorkerCrashedError(WorkerError):
    """A worker exited or lost its protocol stream."""

    code = "worker_crashed"


class WorkerTimeoutError(WorkerError):
    """A worker operation exceeded its deadline."""

    code = "worker_timeout"


class WorkerSaturatedError(WorkerError):
    """No native worker replica is immediately available."""

    code = "server_busy"


class WorkerProtocolError(WorkerError):
    """A worker emitted malformed or unexpected JSONL."""

    code = "worker_protocol_error"


class WorkerRequestTooLargeError(WorkerError):
    """A serialized request would exceed the native JSONL line limit."""

    code = "request_too_large"


class WorkerRemoteError(WorkerError):
    """A worker returned a structured operation error."""

    code = "worker_operation_failed"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.details = details


class ModelNotFoundError(ServeError):
    """The requested model name is not registered."""

    code = "model_not_found"


class ModelCapabilityError(ServeError):
    """The requested model does not implement the required API capability."""

    code = "model_capability_mismatch"
