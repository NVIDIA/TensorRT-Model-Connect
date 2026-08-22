#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only executable fixture for the trtmc serve JSONL protocol."""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "_serve-worker":
        print("expected _serve-worker BUNDLE", file=sys.stderr, flush=True)
        return 2
    bundle = Path(sys.argv[2])
    worker_args = sys.argv[3:]
    mode = bundle.stem
    if "startup-timeout" in mode:
        print(f"startup detail for {bundle.resolve()}", file=sys.stderr, flush=True)
        time.sleep(60)
        return 1
    if "bad-ready" in mode:
        print("not-json", flush=True)
        return 1

    kind = (
        "transcription"
        if any(token in mode for token in ("asr", "transcription", "stream"))
        else "chat"
    )
    ready = {
        "event": "ready",
        "protocol_version": 1 if "protocol-v1" in mode else 2,
        "model_id": mode,
        "pipeline_type": kind,
        "default_max_new_tokens": 64,
        "runtime_strategy": "fake",
        "max_cache_length": 2048,
        "serve_token_present": "TRTMC_SERVE_TOKEN" in os.environ,
        "allowed_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "secret_environment_present": any(
            name in os.environ
            for name in (
                "ACCESS_TOKEN",
                "AUTHORIZATION",
                "AWS_SECRET_ACCESS_KEY",
                "COOKIE",
                "GITHUB_TOKEN",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "NVIDIA_API_KEY",
                "TRTMC_SERVE_TOKEN",
                "UNLISTED_ENVIRONMENT",
            )
        ),
        "worker_args": worker_args,
    }
    if "legacy-ready" in mode:
        ready.pop("event")
        ready["ready"] = True
    emit(ready)
    print(f"fake worker ready: {mode}", file=sys.stderr, flush=True)

    active_samples: int | None = None
    chunk_failed = False
    for line in sys.stdin:
        request = json.loads(line)
        request_id = request["id"]
        op = request["op"]
        if "slow-request" in mode:
            if op == "generate":
                time.sleep(2)
        elif "saturation" in mode and op == "generate":
            time.sleep(0.5)
        elif "slow" in mode and op != "shutdown":
            time.sleep(2)
        if "delayed-start" in mode and op == "stream_start":
            time.sleep(0.25)
        if "delayed-reset" in mode and op == "stream_reset":
            time.sleep(0.25)
        if "crash-request" in mode and op == "generate":
            print(
                f"intentional fake crash at {bundle.resolve()} with access_token=worker-secret",
                file=sys.stderr,
                flush=True,
            )
            return 17
        if "crash" in mode and "crash-request" not in mode and op != "shutdown":
            print(
                f"intentional fake crash at {bundle.resolve()} with access_token=worker-secret",
                file=sys.stderr,
                flush=True,
            )
            return 17
        if "bad-response" in mode:
            emit({"id": "not-the-request-id", "ok": True, "result": {}})
            continue
        if "non-bool-ok" in mode:
            emit({"id": request_id, "ok": "true", "result": {}})
            continue
        if "missing-result" in mode:
            emit({"id": request_id, "ok": True})
            continue

        try:
            if op == "generate":
                if "invalid-request" in mode:
                    raise ValueError("invalid fake generation request")
                prompt = request.get("prompt", "")
                thinking = request.get("config", {}).get("enable_thinking")
                text = f"generated:{prompt}"
                if isinstance(thinking, bool):
                    text = f"enable_thinking={str(thinking).lower()}:{text}"
                result = {
                    "text": text,
                    "usage": {
                        "prompt_tokens": len(str(prompt).split()),
                        "completion_tokens": 1,
                    },
                }
            elif op == "transcribe":
                audio_path = Path(request["audio_path"])
                try:
                    with wave.open(str(audio_path), "rb") as audio:
                        supported = audio.getsampwidth() == 2 and audio.getnframes() > 0
                except FileNotFoundError as exc:
                    raise RuntimeError("read_wav: cannot open input file") from exc
                except (EOFError, wave.Error):
                    supported = False
                if not supported:
                    raise ValueError("read_wav: WAV samples must be PCM16 or IEEE float32")
                result = {"text": f"transcribed {audio_path.stat().st_size} bytes"}
            elif op == "stream_start":
                if active_samples is not None:
                    raise ValueError("another stream is already active")
                active_samples = 0
                result = {}
            elif op == "probe_transcription_stream":
                if kind != "transcription" or "no-stream" in mode:
                    raise RuntimeError("streaming transcription is unavailable")
                result = {"supported": True}
            elif op == "stream_chunk":
                if "chunk-error" in mode and not chunk_failed:
                    chunk_failed = True
                    raise ValueError("invalid fake audio chunk")
                if active_samples is None:
                    raise ValueError("no active transcription stream")
                audio = base64.b64decode(request["audio"], validate=True)
                active_samples += len(audio) // 2
                result = {"text": f"{active_samples} samples"}
            elif op == "stream_finish":
                if active_samples is None:
                    raise ValueError("no active transcription stream")
                result = {"text": f"{active_samples} samples"}
                active_samples = None
            elif op == "stream_reset":
                if active_samples is None:
                    raise ValueError("no active transcription stream")
                active_samples = None
                result = {}
            elif op == "shutdown":
                emit({"id": request_id, "ok": True, "result": {"status": "shutting_down"}})
                return 0
            else:
                raise ValueError(f"unsupported op: {op}")
            emit({"id": request_id, "ok": True, "result": result})
        except Exception as exc:  # noqa: BLE001 - fixture reports protocol errors
            error_type = (
                "runtime_error" if isinstance(exc, RuntimeError) else "invalid_request_error"
            )
            error = {"type": error_type, "message": str(exc)}
            if op == "transcribe" and error_type == "invalid_request_error":
                error.update(code="unsupported_media_type", param="file")
            emit(
                {
                    "id": request_id,
                    "ok": False,
                    "error": error,
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
