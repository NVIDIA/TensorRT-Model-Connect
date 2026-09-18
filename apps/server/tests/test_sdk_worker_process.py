# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real server executable with existing CPU-only SDK fixture DSOs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def write_bundle(path: Path, task: str, family: str = "api_fixture") -> None:
    header = json.dumps({
        "format": 1, "family": family, "task": task, "backend": "fake",
        "sections": {"engine.plan": {"offset": 0, "length": 4}},
    }).encode("utf-8")
    path.write_bytes(b"BUNDLE\x01\x00" + struct.pack("<Q", len(header)) + header + b"PLAN")


def invoke(binary: Path, runtime: Path, bundle: Path, requests: list[dict]):
    completed = subprocess.run(
        [str(binary), "_serve-worker", str(bundle), "--runtime-root", str(runtime),
         "--kv-cache-size", "7"],
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True, text=True, timeout=30, check=False,
    )
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    return completed, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="trtmc-server-sdk-") as temporary:
        bundle = Path(temporary) / "model.bundle"
        write_bundle(bundle, "text_continuation")
        requests = [
            {"id": "first", "op": "generate", "prompt": "hello"},
            {"id": "bad", "op": "generate", "prompt": "hello", "config": {"unknown": 1}},
            {"id": "last", "op": "generate", "prompt": "hi", "config": {"suffix": "?"}},
            {"id": "stop", "op": "shutdown"},
        ]
        completed, records = invoke(args.binary, args.runtime_root, bundle, requests)
        assert completed.returncode == 0, completed.stderr
        assert len(records) == 5
        assert records[0] == {
            "event": "ready", "protocol_version": 1,
            "capabilities": ["text_generation"], "default_max_new_tokens": 4,
        }
        assert records[1]["id"] == "first" and records[1]["ok"] is True
        assert records[1]["result"] == {
            "text": "hello!|eos", "completion_tokens": 2,
            "setup_ms": 7.0, "prefill_ms": 0.75, "decode_ms": 4.0,
        }
        assert records[2]["ok"] is False
        assert records[2]["error"]["type"] == "invalid_request_error"
        assert records[3]["id"] == "last" and records[3]["result"]["text"] == "hi?|eos"
        assert records[4]["result"]["status"] == "shutting_down"

        for task, expected in (
            ("conditional_text_generation", "conditional:hello!"),
            ("text_translation", "translation:fixed-src->en:hello!"),
        ):
            write_bundle(bundle, task, "text_fixture")
            completed, records = invoke(args.binary, args.runtime_root, bundle, [requests[0]])
            assert completed.returncode == 0, completed.stderr
            assert records[0]["default_max_new_tokens"] == 128
            assert records[1]["result"]["text"] == expected

        write_bundle(bundle, "must_not_run")
        completed, records = invoke(args.binary, args.runtime_root, bundle, requests)
        assert completed.returncode == 1
        assert len(records) == 2
        assert records[1]["id"] == "first" and records[1]["ok"] is False
        assert records[1]["error"] == {
            "type": "runtime_error", "message": "native worker operation failed",
        }
        assert "the fixture execution must not be reached" not in completed.stdout

        write_bundle(bundle, "disabled")
        completed, records = invoke(args.binary, args.runtime_root, bundle, requests)
        assert completed.returncode != 0 and not records


if __name__ == "__main__":
    main()
