# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the Python serving control plane."""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .registry import ModelRegistry, ModelSpec
from .worker import WorkerLoadOptions


_REDACTED = "<redacted>"
_SENSITIVE_LOG_KEYS = frozenset({"access_token", "authorization", "cookie"})
_SECRET_PATTERNS = (
    re.compile(r"([?&]access_token=)[^&\s\"']+", re.IGNORECASE),
    re.compile(
        r"(\baccess_token\b\s*(?:=|:)\s*(?:\"|')?)[^&,\s\"'}]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\bauthorization\b\s*(?:=|:)\s*(?:\"|')?)(?:bearer\s+)?[^,\r\n\"'}]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\bcookie\b\s*(?:=|:)\s*(?:\"|')?)[^,\r\n\"'}]+",
        re.IGNORECASE,
    ),
)


def _redact_text(value: str) -> str:
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub(rf"\1{_REDACTED}", value)
    return value


def _is_sensitive_log_key(value: object) -> bool:
    if isinstance(value, bytes):
        normalized = value.decode("ascii", errors="ignore")
    elif isinstance(value, str):
        normalized = value
    else:
        return False
    return normalized.strip().lower().replace("-", "_") in _SENSITIVE_LOG_KEYS


def _redacted_like(value: object) -> str | bytes | bytearray:
    if isinstance(value, bytes):
        return _REDACTED.encode("ascii")
    if isinstance(value, bytearray):
        return bytearray(_REDACTED, "ascii")
    return _REDACTED


def _redact_log_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, bytes):
        return _redact_text(value.decode("utf-8", errors="replace")).encode("utf-8")
    if isinstance(value, bytearray):
        redacted = _redact_text(bytes(value).decode("utf-8", errors="replace"))
        return bytearray(redacted, "utf-8")
    if isinstance(value, Mapping):
        return {
            key: _redacted_like(item) if _is_sensitive_log_key(key) else _redact_log_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence):
        items = list(value)
        if len(items) == 2 and _is_sensitive_log_key(items[0]):
            items[1] = _redacted_like(items[1])
        else:
            items = [_redact_log_value(item) for item in items]
        if isinstance(value, tuple):
            return tuple(items)
        if isinstance(value, list):
            return items
        return items
    return value


class _RedactAccessToken(logging.Filter):
    """Recursively remove transport credentials from structured log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _redact_log_value(record.msg)
        record.args = _redact_log_value(record.args)
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trtmc serve",
        description="Serve TensorRT-Model-Connect bundles over local HTTP and Realtime APIs",
    )
    parser.add_argument(
        "--chat-model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Register a text-generation bundle (repeatable)",
    )
    parser.add_argument(
        "--transcription-model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Register an offline/realtime transcription bundle (repeatable)",
    )
    parser.add_argument(
        "--default-chat-model",
        metavar="NAME",
        help="Default model for the chat endpoint (first registered chat model otherwise)",
    )
    parser.add_argument(
        "--default-transcription-model",
        metavar="NAME",
        help="Default model for audio endpoints (first registered transcription model otherwise)",
    )
    parser.add_argument(
        "--trtmc-binary",
        metavar="PATH",
        help="Native trtmc executable (injected by `trtmc serve`; PATH lookup otherwise)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind loopback IP literal (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=8000,
        help="Bind port; 0 selects a free port (default: 8000)",
    )
    parser.add_argument(
        "--api-key",
        help="Bearer token; falls back to TRTMC_SERVE_TOKEN",
    )
    parser.add_argument(
        "--require-streaming-transcription",
        action="append",
        default=[],
        metavar="MODEL",
        help="Fail startup unless MODEL passes a native streaming probe (repeatable)",
    )
    parser.add_argument(
        "--model-replicas",
        action="append",
        default=[],
        metavar="NAME=N",
        help="Run N native worker replicas for MODEL (repeatable; default: 1)",
    )
    parser.add_argument("--backend-dir", action="append", default=[], metavar="PATH")
    parser.add_argument("--model-plugin-dir", action="append", default=[], metavar="PATH")
    parser.add_argument("--runtime-cache", metavar="PATH")
    parser.add_argument("--kernel-bindings", metavar="PATH")
    parser.add_argument("--config", metavar="PATH")
    parser.add_argument("--set", action="append", default=[], dest="set_values", metavar="K=V")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--startup-timeout", type=_positive_float, default=120.0)
    parser.add_argument("--request-timeout", type=_positive_float, default=120.0)
    parser.add_argument("--max-generation-tokens", type=_positive_int, default=4096)
    parser.add_argument("--realtime-idle-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--realtime-max-session-seconds", type=_positive_float, default=4 * 60 * 60)
    parser.add_argument("--realtime-max-audio-bytes", type=_positive_int, default=512 * 1024 * 1024)
    parser.add_argument(
        "--parent-liveness-stdin",
        action="store_true",
        help="Gracefully stop when the parent-owned stdin pipe reaches EOF",
    )
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info"),
        default="info",
    )
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Enable HTTP access logs (disabled by default to avoid logging WS tokens)",
    )
    return parser


def parse_model_assignment(value: str, *, kind: str) -> ModelSpec:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"--{kind}-model must use NAME=PATH")
    bundle = Path(raw_path).expanduser().resolve()
    if not bundle.is_file():
        raise ValueError(f"bundle for model {name!r} does not exist or is not a file")
    model_kind = "chat" if kind == "chat" else "transcription"
    return ModelSpec(name=name, bundle=bundle, kind=model_kind)  # type: ignore[arg-type]


def parse_replica_assignment(value: str) -> tuple[str, int]:
    name, separator, raw_replicas = value.partition("=")
    if not separator or not name or not raw_replicas:
        raise ValueError("--model-replicas must use NAME=N")
    try:
        replicas = int(raw_replicas)
    except ValueError as exc:
        raise ValueError(f"replicas for model {name!r} must be a positive integer") from exc
    if replicas <= 0:
        raise ValueError(f"replicas for model {name!r} must be a positive integer")
    return name, replicas


def validate_bind_policy(host: str) -> None:
    if not is_loopback_host(host):
        raise ValueError(f"refusing host {host!r}; --host must be a loopback IP literal")


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def resolve_trtmc_binary(value: str | None) -> Path:
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        discovered = shutil.which(value)
        if discovered:
            return Path(discovered).resolve()
        raise ValueError("trtmc executable does not exist or is not a file")
    discovered = shutil.which("trtmc")
    if discovered:
        return Path(discovered).resolve()
    raise ValueError("cannot find trtmc on PATH; native forwarding must pass --trtmc-binary")


def bind_socket(host: str, port: int) -> socket.socket:
    """Pre-bind the listening socket so port 0 can be reported without a race."""

    errors: list[OSError] = []
    addresses = socket.getaddrinfo(
        host,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        flags=socket.AI_PASSIVE,
    )
    for family, socktype, protocol, _canonical_name, sockaddr in addresses:
        listener = socket.socket(family, socktype, protocol)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(sockaddr)
            listener.listen(2048)
            listener.setblocking(False)
            return listener
        except OSError as exc:
            errors.append(exc)
            listener.close()
    detail = "; ".join(str(error) for error in errors) or "no addresses resolved"
    raise OSError(f"cannot bind {host}:{port}: {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        specs = [
            *(parse_model_assignment(value, kind="chat") for value in args.chat_model),
            *(
                parse_model_assignment(value, kind="transcription")
                for value in args.transcription_model
            ),
        ]
        if not specs:
            raise ValueError("at least one --chat-model or --transcription-model is required")

        replica_assignments = [parse_replica_assignment(value) for value in args.model_replicas]
        replica_names = [name for name, _replicas in replica_assignments]
        if len(replica_names) != len(set(replica_names)):
            raise ValueError("--model-replicas may be specified only once per model")
        model_replicas = dict(replica_assignments)

        trtmc_binary = resolve_trtmc_binary(args.trtmc_binary)
        api_key = args.api_key
        if api_key is None:
            api_key = os.environ.get("TRTMC_SERVE_TOKEN")
        if api_key is not None:
            api_key = api_key.strip()
            if not api_key:
                api_key = None
        validate_bind_policy(args.host)
        load_options = WorkerLoadOptions(
            backend_dirs=tuple(args.backend_dir),
            model_plugin_dirs=tuple(args.model_plugin_dir),
            runtime_cache=args.runtime_cache,
            kernel_bindings=args.kernel_bindings,
            config=args.config,
            set_values=tuple(args.set_values),
            cuda_graphs=args.cuda_graphs,
        )
        registry = ModelRegistry(
            specs,
            trtmc_binary=trtmc_binary,
            default_chat_model=args.default_chat_model,
            default_transcription_model=args.default_transcription_model,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
            load_options=load_options,
            model_replicas=model_replicas,
            required_streaming_transcription=args.require_streaming_transcription,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        import uvicorn

        from .app import ServerConfig, create_app
    except ModuleNotFoundError as exc:
        if exc.name in {"fastapi", "multipart", "uvicorn"}:
            parser.error("serve dependencies are missing; install 'tensorrt-model-connect[serve]'")
        raise
    try:
        app = create_app(
            registry,
            config=ServerConfig(
                api_key=api_key,
                max_generation_tokens=args.max_generation_tokens,
                max_realtime_session_bytes=args.realtime_max_audio_bytes,
                realtime_idle_timeout_seconds=args.realtime_idle_timeout,
                realtime_max_session_seconds=args.realtime_max_session_seconds,
            ),
        )
    except RuntimeError as exc:
        if "python-multipart" in str(exc):
            parser.error("serve dependencies are missing; install 'tensorrt-model-connect[serve]'")
        raise
    try:
        listener = bind_socket(args.host, args.port)
    except OSError as exc:
        parser.error(str(exc))
    actual_port = int(listener.getsockname()[1])

    try:

        class ReadyServer(uvicorn.Server):
            _ready_emitted = False

            async def startup(self, sockets: list[socket.socket] | None = None) -> None:
                await super().startup(sockets=sockets)
                if self.started and not self._ready_emitted:
                    self._ready_emitted = True
                    ready = {
                        "event": "ready",
                        "host": args.host,
                        "port": actual_port,
                        "models": registry.names,
                    }
                    print(
                        json.dumps(ready, separators=(",", ":"), ensure_ascii=False),
                        file=sys.stdout,
                        flush=True,
                    )

        _install_secret_log_filter()
        uvicorn_config = uvicorn.Config(
            app,
            host=args.host,
            port=actual_port,
            log_level=args.log_level,
            access_log=args.access_log,
        )
        server = ReadyServer(uvicorn_config)
        if args.parent_liveness_stdin:
            threading.Thread(
                target=_watch_parent_stdin,
                args=(server,),
                name="trtmc-parent-liveness",
                daemon=True,
            ).start()
        server.run(sockets=[listener])
        return 0
    finally:
        listener.close()


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be from 0 to 65535")
    return port


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _watch_parent_stdin(server: Any) -> None:
    try:
        while sys.stdin.buffer.read(8192):
            pass
    except (OSError, ValueError):
        pass
    server.should_exit = True


def _install_secret_log_filter() -> None:
    redactor = _RedactAccessToken()
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.addFilter(redactor)
        for handler in logger.handlers:
            handler.addFilter(redactor)


if __name__ == "__main__":
    raise SystemExit(main())
