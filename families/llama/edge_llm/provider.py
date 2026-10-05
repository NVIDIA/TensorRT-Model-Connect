# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Llama's versioned Edge provider, usable from an installed wheel or source build.

Only this standalone module is installed alongside the provider DSOs. No Model
Connect, Torch or Edge imports occur until the selected operation needs them.
The descriptor is local administrator configuration, never executable bundle data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import socket
import sys
import traceback

REVISIONS = {
    "0.10.0": "71dd1bae032e70771265917ec74d3ff4cad07a10",
    "0.11.0": "95515c2f87fba8982db5a519f9022277667b3cc9",
}
MAX_MESSAGE = 16 * 1024 * 1024


def _installation():
    """Load generic SDK mechanics from the source tree or adjacent installed file."""
    local = Path(__file__).with_name("installation.py")
    if not local.is_file():
        local = Path(__file__).resolve().parents[3] / "cmake/edge_llm/provider/installation.py"
    if not local.is_file():
        local = (
            Path(__file__).resolve().parents[3]
            / "tensorrt_model_connect/bin/edge_llm/llama/installation.py"
        )
    spec = importlib.util.spec_from_file_location("trtmc_edge_installation", local)
    if spec is None or spec.loader is None:
        raise RuntimeError("Edge installation mechanics are not installed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def descriptor(path: Path) -> dict:
    data = _installation().descriptor(path)
    if data["version"] not in REVISIONS:
        raise ValueError("Llama provider requires Edge 0.10.0 or 0.11.0")
    if data["version"] == "0.10.0" and not data.get("native_module"):
        raise ValueError("Edge 0.10.0 requires its source-built Python runtime module")
    return data


def command(path: Path, operation: str, *arguments: str) -> tuple[list[str], dict]:
    descriptor(path)
    return _installation().command(Path(__file__), path, operation, *arguments)


def load_native(data: dict):
    native, identity, plugin = _installation().load_native(data)
    identity["revision"] = REVISIONS[data["version"]]
    return native, identity, plugin


def build(data: dict, inputs: dict) -> None:
    """Translate the family request to the release's complete upstream flow."""
    native, identity, plugin = load_native(data)
    if identity != inputs["identity"]:
        raise ValueError("Edge provider changed between discovery and build")
    model, engine = inputs["checkpoint"], inputs["engine"]
    limit = inputs["limit"]
    if data["version"] == "0.10.0":
        # This binding has no checkpoint_dir; use the supported embedded ONNX flow.
        from tensorrt_edgellm.scripts.export import main

        output = str(Path(engine).parent / "onnx")
        sys.argv = [
            "tensorrt-edgellm-export",
            model,
            output,
            "--components",
            "thinker",
            "--dtype",
            "float16",
        ]
        main()
        config = native.LLMBuilderConfig()
        config.max_input_len = limit
        config.max_kv_cache_capacity = limit
        config.max_batch_size = 1
        if not native.LLMBuilder(str(Path(output) / "llm"), engine, config).build():
            raise RuntimeError("Edge 0.10.0 ONNX builder returned failure")
    else:
        from experimental.builder.cli import main

        main(
            [
                "--model-dir",
                model,
                "--engine-dir",
                engine,
                "--components",
                "llm",
                "--plugin-path",
                str(plugin),
                "--dense",
                "fp16",
                "--max-input-len",
                str(limit),
                "--max-kv-cache-capacity",
                str(limit),
                "--max-batch-size",
                "1",
                "--externalize-weights",
                "all",
            ]
        )


def validate_capacity(counts, marker: dict, generated: int) -> None:
    """Reject clipping; do not change the family's total-token capacity contract."""
    if (
        len(counts) != 1
        or counts[0] <= 0
        or counts[0] > marker["max_input_length"]
        or generated <= 0
        or counts[0] + generated > marker["max_sequence_length"]
    ):
        raise ValueError("Llama Edge prompt and generation exceed bundle capacity")


def serve(data: dict, version: str, fd: int) -> None:
    """Keep one runtime alive, with protocol traffic separate from native logs."""
    if data["version"] != version:
        raise ValueError("Descriptor does not match the selected provider DSO")
    with socket.socket(fileno=fd) as channel, channel.makefile("rwb") as stream:
        runtime = native = marker = None
        for line in iter(lambda: stream.readline(MAX_MESSAGE + 1), b""):
            try:
                if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
                    raise ValueError("Oversized provider message")
                message = json.loads(line)
                if message["op"] == "open":
                    if runtime is not None:
                        raise ValueError("Provider runtime already loaded")
                    native, identity, _ = load_native(data)
                    marker = message["marker"]
                    if identity != marker["provider"]:
                        raise ValueError("Installed Edge provider does not match the bundle")
                    if version == "0.10.0":
                        runtime = native.LLMRuntime(message["engine"])
                    else:
                        runtime = native.LLMRuntime(
                            message["engine"], checkpoint_dir=message["checkpoint"]
                        )
                    result = {"ready": True}
                elif message["op"] == "generate":
                    if runtime is None:
                        raise ValueError("Provider runtime is not loaded")
                    length = message["max_new_tokens"]
                    request = native.create_generation_request(
                        [[native.create_text_message("user", message["prompt"])]],
                        temperature=message["temperature"],
                        top_p=message["top_p"],
                        top_k=message["top_k"],
                        max_generate_length=length,
                        apply_chat_template=message["use_chat_template"],
                        enable_thinking=message["enable_thinking"],
                    )
                    if version == "0.11.0":
                        validate_capacity(runtime.count_prompt_tokens(request), marker, length)
                    response = runtime.handle_request(request)
                    # 0.10 reports counts only after execution: reject a clipped result.
                    validate_capacity(response.prompt_token_counts, marker, length)
                    if (
                        len(response.output_ids) != 1
                        or not response.output_ids[0]
                        or len(response.output_ids[0]) > length
                        or len(response.output_texts) != 1
                        or len(response.finish_reasons) != 1
                        or response.finish_reasons[0]
                        not in (native.FinishReason.END_ID, native.FinishReason.LENGTH)
                    ):
                        raise RuntimeError("Edge generation did not complete successfully")
                    result = {"text": response.output_texts[0], "token_ids": response.output_ids[0]}
                else:
                    raise ValueError("Unknown provider operation")
                response_data = {"result": result}
            except Exception as error:
                traceback.print_exc()
                response_data = {"error": str(error)}
            payload = json.dumps(response_data, ensure_ascii=True).encode() + b"\n"
            if len(payload) > MAX_MESSAGE:
                raise ValueError("Oversized provider response")
            stream.write(payload)
            stream.flush()
            if "error" in response_data:
                return  # Do not reuse vendor state after a failed request.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    sub = parser.add_subparsers(dest="operation", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--output", type=Path, required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--input", type=Path, required=True)
    worker = sub.add_parser("serve")
    worker.add_argument("--version", required=True, choices=REVISIONS)
    worker.add_argument("--fd", type=int, required=True)
    args = parser.parse_args()
    data = descriptor(args.provider)
    if args.operation == "probe":
        _, identity, _ = load_native(data)
        args.output.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    elif args.operation == "build":
        build(data, json.loads(args.input.read_text(encoding="utf-8")))
    else:
        serve(data, args.version, args.fd)


if __name__ == "__main__":
    main()
