# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``trtmc-perf-serve``: serve one operation over HTTP, or write load-generator payloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import platform
from .files import inline_files


def _json_object(raw: str, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _overrides(items: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        name, separator, raw = item.partition("=")
        if not separator or not name:
            raise ValueError(f"--set expects FIELD=VALUE, got {item!r}")
        try:
            result[name] = json.loads(raw)
        except json.JSONDecodeError:
            result[name] = raw
    return result


def _profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True, help="benchmark catalog model profile, e.g. qwen3-0.6b-fp16")
    parser.add_argument("--testcase", help="catalog testcase name (default: the profile's first testcase)")
    parser.add_argument("--operation", help="operation override (default: resolved from the testcase)")
    parser.add_argument("--task", dest="selected_task", help="select a Task bound by the model")
    parser.add_argument("--manifest-root", type=Path, default=Path("families"))
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="FIELD=VALUE",
                        help="override a base-request field (same semantics as trtmc-bench --set)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trtmc-perf-serve", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve one operation of one backend")
    _profile_arguments(serve)
    serve.add_argument("--backend", required=True, choices=("trtmc", "reference"),
                       help="trtmc: the TRTMC bundle; reference: the generic Hugging Face adapters")
    serve.add_argument("--bundle", type=Path, help="TRTMC bundle (trtmc backend)")
    serve.add_argument("--runtime-root", type=Path, help="directory with libtrtmc_runtime and family DSOs")
    serve.add_argument("--worker", type=Path, help="trtmc_benchmark_worker executable")
    serve.add_argument("--mode", choices=("eager", "compile"), default="eager", help="reference execution mode")
    serve.add_argument("--precision", choices=("fp16", "bf16", "fp32"), help="reference precision")
    serve.add_argument("--reference-model", help="override the profile checkpoint for the reference")
    serve.add_argument("--reference-revision", help="revision of the reference checkpoint (default: --revision)")
    serve.add_argument("--revision", help="the profile checkpoint's revision when the catalog does not pin one")
    serve.add_argument("--reference-options", default="{}", help="adapter options JSON")
    serve.add_argument("--reference-adapter", help="a family's native pipeline file (defines Adapter)")
    serve.add_argument("--trust-remote-code", action="store_true")
    serve.add_argument("--deterministic", action="store_true",
                       help="reference only: disable TF32 and use deterministic kernels (golden generation)")
    serve.add_argument("--model-name", help="model id reported by /v1/models (default: profile name)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--records", type=Path, required=True, help="per-request JSONL record file")
    serve.add_argument("--scratch", type=Path, required=True, help="per-request input/output directory")
    serve.add_argument("--max-queue", type=int, default=64, help="waiting requests before 429 (0: none)")
    serve.add_argument("--keep-artifacts", action="store_true", help="keep per-request inputs and outputs")
    serve.add_argument("--memory-probe", action="store_true",
                       help="report each call's peak GPU memory (NVML; valid while the server is alone on the GPU)")
    serve.add_argument("--isolate-requests", action="store_true",
                       help="trtmc backend: restart the worker before every request after the first")
    serve.add_argument("--full-observations", action="store_true",
                       help="report whole observations instead of compacting arrays longer than 64 items")
    serve.add_argument("--request-timeout", type=float, default=900.0)

    commands.add_parser("platform", help="print the platform fingerprint and host details as JSON")

    env = commands.add_parser("reference-env",
                              help="prepare a reference Python environment and print its interpreter")
    env.add_argument("--requirements", type=Path, help="requirements file layered on this interpreter (none: it as is)")
    env.add_argument("--root", type=Path, required=True, help="directory holding the environments")
    env.add_argument("--no-build-isolation", action="store_true", help="pip install --no-build-isolation")
    env.add_argument("--prepare", type=Path, help="a script run once with the environment's interpreter after "
                                                    "installing (for example an upstream checkout)")

    payload = commands.add_parser("payload", help="write aiperf mooncake_trace payloads for /v1/tasks")
    _profile_arguments(payload)
    payload.add_argument("--output", type=Path, required=True)
    return parser


def _reference_precision(profile_precision: str, requested: str | None) -> str:
    if requested:
        return requested
    return profile_precision if profile_precision in ("fp16", "bf16", "fp32") else "fp16"


def _chat_renderer(profile: Any, revision: str | None):
    """Chat-template renderer from the profile checkpoint's tokenizer, or None if it has no template."""
    from transformers import AutoTokenizer

    kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(profile.model.hf_id, **kwargs)
    if not getattr(tokenizer, "chat_template", None):
        return None

    def render(messages: list[dict[str, str]], enable_thinking: bool) -> str:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                             enable_thinking=enable_thinking)

    return render


def serve(arguments: argparse.Namespace) -> int:
    import uvicorn

    from .app import ServingConfig, create_app
    from .profiles import resolve_profile

    profile = resolve_profile(
        arguments.profile, manifest_root=arguments.manifest_root, testcase=arguments.testcase,
        operation=arguments.operation, selected_task=arguments.selected_task, bundle=arguments.bundle,
        runtime_root=arguments.runtime_root, overrides=_overrides(arguments.sets))
    revision = profile.model.hf_revision or arguments.revision or None
    # The baseline must be taken before the backend loads its model.
    memory_probe = None
    if arguments.memory_probe:
        from .gpu_memory import open_probe

        memory_probe = open_probe()
    if arguments.backend == "trtmc":
        from .backends.trtmc import TrtmcWorkerBackend

        if arguments.bundle is None or arguments.runtime_root is None or arguments.worker is None:
            raise ValueError("the trtmc backend requires --bundle, --runtime-root, and --worker")
        backend = TrtmcWorkerBackend(worker=arguments.worker, session=profile.worker_request,
                                     scratch=arguments.scratch / "_worker",
                                     request_timeout_s=arguments.request_timeout,
                                     full_observations=arguments.full_observations,
                                     isolate_requests=arguments.isolate_requests)
    else:
        from .backends.reference import ReferenceBackend
        from .backends.reference.common import ReferenceSpec

        backend = ReferenceBackend(ReferenceSpec(
            operation=profile.operation,
            model=arguments.reference_model or profile.model.hf_id,
            revision=arguments.reference_revision or (None if arguments.reference_model else revision),
            precision=_reference_precision(profile.model.precision, arguments.precision),
            mode=arguments.mode,
            trust_remote_code=arguments.trust_remote_code,
            deterministic=arguments.deterministic,
            options=_json_object(arguments.reference_options, "--reference-options"),
            adapter=arguments.reference_adapter))
    latent_replay = None
    if profile.operation == "generate_image":
        from .latents import Replay, read_checkpoint, snapshot

        source = ((arguments.reference_model, arguments.reference_revision)
                  if arguments.backend == "reference" and arguments.reference_model else (profile.model.hf_id, revision))
        latent_replay = Replay(lambda: read_checkpoint(snapshot(*source)))
    chat_renderer = None
    if profile.operation == "generate":
        try:
            chat_renderer = _chat_renderer(profile, revision)
        except Exception as error:  # noqa: BLE001 - only the OpenAI chat route needs it
            print(f"trtmc-perf-serve: chat template unavailable ({type(error).__name__}: {error}); "
                  "/v1/chat/completions will reject multi-message requests", file=sys.stderr)
    config = ServingConfig(
        chat_renderer=chat_renderer,
        base_request=profile.base_request, records=arguments.records, scratch=arguments.scratch,
        model_name=arguments.model_name or profile.model.name, max_queue=arguments.max_queue,
        keep_artifacts=arguments.keep_artifacts, full_observations=arguments.full_observations,
        memory_probe=memory_probe, latent_replay=latent_replay, info={"profile": profile.model.name, "testcase": profile.testcase, "family": profile.model.family,
              "platform": platform.fingerprint(), "host": platform.host_details()})
    try:
        uvicorn.run(create_app(backend, config), host=arguments.host, port=arguments.port, log_level="warning")
    finally:
        backend.close()
    return 0


def write_payload(arguments: argparse.Namespace) -> int:
    from .profiles import resolve_profile

    profile = resolve_profile(
        arguments.profile, manifest_root=arguments.manifest_root, testcase=arguments.testcase,
        operation=arguments.operation, selected_task=arguments.selected_task, overrides=_overrides(arguments.sets))
    record = {"payload": {"request": inline_files(dict(profile.base_request))}}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record) + "\n")
    print(json.dumps({"output": str(arguments.output), "operation": profile.operation,
                      "testcase": profile.testcase, "route": f"/v1/tasks/{profile.operation}"}))
    return 0


def reference_env(arguments: argparse.Namespace) -> int:
    from .environments import reference_python

    python = (reference_python(arguments.requirements, arguments.root, build_isolation=not arguments.no_build_isolation,
                               prepare=arguments.prepare)
              if arguments.requirements else Path(sys.executable))
    print(json.dumps({"python": str(python), "requirements": str(arguments.requirements or "") or None}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "reference-env":
            return reference_env(arguments)
        if arguments.command == "platform":
            print(json.dumps({"fingerprint": platform.fingerprint(), "host": platform.host_details()}))
            return 0
        return serve(arguments) if arguments.command == "serve" else write_payload(arguments)
    except (ValueError, RuntimeError) as error:
        print(f"trtmc-perf-serve: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
