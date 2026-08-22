# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the local realtime speech host."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Sequence

from .server import RealtimeServerConfig, serve
from .worker import NativeJsonlWorker, WorkerError, find_native_worker


DEFAULT_TOKEN_ENV = "TRTMC_REALTIME_TOKEN"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tensorrt_model_connect.realtime",
        description="Serve a native TRTMC speech worker over a local Realtime WebSocket.",
    )
    parser.add_argument("--worker", type=Path, help="native JSONL worker executable")
    parser.add_argument("--bundle", required=True, type=Path, help="speech model bundle")
    parser.add_argument(
        "--backend-dir",
        action="append",
        default=[],
        type=Path,
        help="extra native backend directory; repeatable",
    )
    parser.add_argument(
        "--model-plugin-dir",
        action="append",
        default=[],
        type=Path,
        help="extra model plugin directory; repeatable",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--token-env",
        default=DEFAULT_TOKEN_ENV,
        help="environment variable containing the bearer token",
    )
    parser.add_argument("--input-queue-size", default=32, type=int)
    parser.add_argument("--message-queue-size", default=32, type=int)
    parser.add_argument("--output-queue-size", default=128, type=int)
    parser.add_argument("--max-message-bytes", default=1 << 20, type=int)
    parser.add_argument("--max-audio-bytes", default=4_800, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    token = os.environ.get(arguments.token_env)
    if not token:
        parser.error(f"bearer token environment variable {arguments.token_env!r} is not set")
    config = RealtimeServerConfig(
        bearer_token=token,
        host=arguments.host,
        port=arguments.port,
        input_queue_size=arguments.input_queue_size,
        message_queue_size=arguments.message_queue_size,
        output_queue_size=arguments.output_queue_size,
        max_message_bytes=arguments.max_message_bytes,
        max_audio_bytes=arguments.max_audio_bytes,
    )
    try:
        worker_path = find_native_worker(arguments.worker)
    except WorkerError as exc:
        parser.error(str(exc))

    worker_arguments = ["--bundle", str(arguments.bundle)]
    for backend_dir in arguments.backend_dir:
        worker_arguments.extend(("--backend-dir", str(backend_dir)))
    for model_plugin_dir in arguments.model_plugin_dir:
        worker_arguments.extend(("--model-plugin-dir", str(model_plugin_dir)))

    def worker_factory() -> NativeJsonlWorker:
        return NativeJsonlWorker(
            worker_path,
            worker_arguments,
            max_line_bytes=config.max_message_bytes,
        )

    try:
        asyncio.run(serve(config, worker_factory))
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        parser.error(str(exc))
    return 0
