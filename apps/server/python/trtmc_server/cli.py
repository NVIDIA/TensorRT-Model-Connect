# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line for the local text inference control plane."""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path


def packaged_runtime_root(control_plane_file: Path = Path(__file__)) -> Path | None:
    """Locate native DSOs installed beside the wheel's builder package."""

    candidate = control_plane_file.resolve().parent.parent / "tensorrt_model_connect" / "bin"
    required = (
        candidate / "trtmc-server",
        candidate / "libtrtmc_runtime.so",
        candidate / "libtrtmc_backend_trt.so",
    )
    return candidate if all(path.is_file() for path in required) else None


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="trtmc-server",
        description="Serve text-generation bundles through persistent native workers",
    )
    result.add_argument("bundle", nargs="?", help="single .bundle artifact")
    result.add_argument("--model-name", help="API model name for the positional bundle")
    result.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="register another text model; repeatable",
    )
    result.add_argument("--replicas", type=positive, default=1)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8000)
    result.add_argument("--runtime-root")
    result.add_argument("--kv-cache-size", type=positive)
    result.add_argument("--runtime-cache")
    result.add_argument("--cuda-graphs", action="store_true")
    result.add_argument("--startup-timeout", type=float, default=120.0)
    result.add_argument("--request-timeout", type=float, default=120.0)
    result.add_argument("--max-body-bytes", type=positive, default=1024 * 1024)
    result.add_argument("--max-prompt-bytes", type=positive, default=256 * 1024)
    result.add_argument("--max-new-tokens", type=positive, default=4096)
    result.add_argument("--access-log", action="store_true")
    result.add_argument("--worker-binary", type=Path, help=argparse.SUPPRESS)
    return result


def model_assignment(value: str, replicas: int) -> tuple[str, Path, int]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError("--model must use NAME=PATH")
    bundle = Path(path).expanduser().resolve()
    if not bundle.is_file():
        raise ValueError(f"bundle for model {name!r} does not exist")
    return name, bundle, replicas


def main(argv: list[str] | None = None) -> int:
    command = parser()
    args = command.parse_args(argv)
    try:
        try:
            import uvicorn

            from .app import ServerConfig, create_app
            from .registry import ModelRegistry, ModelSpec
            from .worker import WorkerLoadOptions
        except ImportError:
            command.error(
                "optional serving dependencies are unavailable; install "
                "'tensorrt-model-connect[serve]'"
            )
        host = ipaddress.ip_address(args.host)
        if not host.is_loopback:
            raise ValueError("--host must be a loopback IP address")
        if not 0 <= args.port <= 65535:
            raise ValueError("--port must be between 0 and 65535")
        if args.startup_timeout <= 0 or args.request_timeout <= 0:
            raise ValueError("worker timeouts must be positive")
        specs = [ModelSpec(*model_assignment(value, args.replicas)) for value in args.model]
        if args.bundle:
            if not args.model_name:
                raise ValueError("--model-name is required with the positional bundle")
            bundle = Path(args.bundle).expanduser().resolve()
            if not bundle.is_file():
                raise ValueError("bundle does not exist")
            specs.insert(0, ModelSpec(args.model_name, bundle, args.replicas))
        elif args.model_name:
            raise ValueError("--model-name requires the positional bundle")
        if not specs:
            raise ValueError("provide a bundle or at least one --model NAME=PATH")
        packaged_root = packaged_runtime_root()
        if args.worker_binary is None and packaged_root is not None:
            args.worker_binary = packaged_root / "trtmc-server"
        if args.worker_binary is None or not args.worker_binary.is_file():
            raise ValueError("native worker executable is unavailable")
        runtime_root: str | None = None
        if args.runtime_root:
            root = Path(args.runtime_root).expanduser().resolve()
            if not root.is_dir():
                raise ValueError("--runtime-root must be a directory")
            runtime_root = str(root)
        else:
            if packaged_root is not None:
                runtime_root = str(packaged_root)
        registry = ModelRegistry(
            specs,
            worker_binary=args.worker_binary.resolve(),
            load_options=WorkerLoadOptions(
                runtime_root=runtime_root,
                kv_cache_size_bytes=args.kv_cache_size,
                runtime_cache=args.runtime_cache,
                cuda_graphs=args.cuda_graphs,
            ),
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
        )
        app = create_app(
            registry,
            ServerConfig(
                max_body_bytes=args.max_body_bytes,
                max_prompt_bytes=args.max_prompt_bytes,
                max_generation_tokens=args.max_new_tokens,
            ),
        )
        uvicorn_config = uvicorn.Config(
            app,
            host=str(host),
            port=args.port,
            access_log=args.access_log,
            log_level="info",
        )
        server = uvicorn.Server(uvicorn_config)
        server.run()
        return 0 if server.started else 1
    except (OSError, ValueError) as error:
        command.error(str(error))
    return 2
