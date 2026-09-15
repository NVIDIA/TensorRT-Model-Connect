# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Individual component commands for CosyVoice3 development."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .artifacts import write_component
from .config import MODEL_ID, MODEL_REVISION, SOURCE_REVISION, ShapeProfile, read_config


def build(args):
    from . import trt_compat
    from .checkpoint_mapper import load_flow_weights
    from .flow_builder import build_flow_engine

    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    cfg = read_config(args.model_dir)
    profile = ShapeProfile(args.min_frames, args.opt_frames, args.max_frames)
    weights = load_flow_weights(args.model_dir, cfg)
    plan = build_flow_engine(weights, cfg, profile, workspace_mib=args.workspace_mib)
    metadata = {
        "schema_version": 1, "component": "cosyvoice3_flow_estimator",
        "status": "component_built",
        "target_model_id": MODEL_ID, "target_model_revision": MODEL_REVISION,
        "equations_source_revision": SOURCE_REVISION,
        "local_checkpoint_revision_verified": False,
        "architecture": asdict(cfg), "profile": asdict(profile),
        "precision": "fp32", "tf32": False, "streaming": False,
        "tensorrt_version": trt_compat.module_version(),
        "workspace_mib": args.workspace_mib,
    }
    write_component(output, "flow.plan", plan, metadata)
    print(json.dumps(metadata, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="CosyVoice3 component tools. Individual component builds are not full TTS bundles; use the standard build API for native per-request voice conditioning.")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Read the model config safely, without executing YAML constructors")
    inspect.add_argument("--model-dir", type=Path, required=True)
    build_parser = commands.add_parser("build-flow", help="Build a native FP32 offline DiT component engine")
    build_parser.add_argument("--model-dir", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--min-frames", type=int, default=4)
    build_parser.add_argument("--opt-frames", type=int, default=64)
    build_parser.add_argument("--max-frames", type=int, default=256)
    build_parser.add_argument("--workspace-mib", type=int, default=512)
    conditioner = commands.add_parser("build-conditioner", help="Build offline token/speaker preprocessing (not full TTS)")
    conditioner.add_argument("--model-dir", type=Path, required=True)
    conditioner.add_argument("--output", type=Path, required=True)
    conditioner.add_argument("--min-tokens", type=int, default=2)
    conditioner.add_argument("--opt-tokens", type=int, default=32)
    conditioner.add_argument("--max-tokens", type=int, default=128)
    conditioner.add_argument("--workspace-mib", type=int, default=64)
    for component in ("llm", "hift", "campplus", "speech-tokenizer"):
        command = commands.add_parser(f"build-{component}", help=f"Build native offline {component} component")
        command.add_argument("--model-dir", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--workspace-mib", type=int, default=256)
        if component == "llm":
            command.add_argument("--max-context", type=int, default=1024)
            command.add_argument("--opt-tokens", type=int, default=64)
        else:
            command.add_argument("--min-frames", type=int, default=4)
            command.add_argument("--opt-frames", type=int, default=64 if component == "hift" else 500)
            command.add_argument("--max-frames", type=int, default=256 if component == "hift" else 3000)
    voice = commands.add_parser("prepare-voice", help="Reference WAV -> native TensorRT frontend -> voice NPZ")
    for name in ("audio", "campplus", "speech-tokenizer", "output"):
        voice.add_argument(f"--{name}", type=Path, required=True)
    tts = commands.add_parser("synthesize", help="Offline text + prepared voice NPZ -> WAV (experimental, unqualified)")
    for name in ("model-dir", "llm", "conditioner", "flow", "hift", "voice", "output"):
        tts.add_argument(f"--{name}", type=Path, required=True)
    tts.add_argument("--text", required=True)
    tts.add_argument("--instruction", default="You are a helpful assistant.")
    tts.add_argument("--prompt-text", default="", help="Exact reference transcript for zero-shot; omit for instruction mode")
    tts.add_argument("--max-tokens", type=int, default=100)
    tts.add_argument("--seed", type=int, default=2512)
    tts.add_argument("--greedy", action="store_true", help="Deterministic argmax, not the official default RAS sampler")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        print(json.dumps({"target": MODEL_ID, "flow": asdict(read_config(args.model_dir)), "status": "component_only"}, indent=2))
    elif args.command == "build-flow":
        build(args)
    elif args.command == "build-conditioner":
        build_conditioner(args)
    elif args.command == "synthesize":
        from .tts import synthesize

        synthesize(args)
    elif args.command == "prepare-voice":
        from .tts import prepare_voice

        prepare_voice(args)
    else:
        build_speech_component(args)


def build_conditioner(args):
    from . import trt_compat
    from .conditioning import TokenProfile, build_engine, load_weights

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    read_config(args.model_dir)
    profile = TokenProfile(args.min_tokens, args.opt_tokens, args.max_tokens)
    plan = build_engine(load_weights(args.model_dir), profile, workspace_mib=args.workspace_mib)
    manifest = {
        "schema_version": 1, "component": "cosyvoice3_conditioner",
        "status": "component_built",
        "target_model_id": MODEL_ID, "target_model_revision": MODEL_REVISION,
        "equations_source_revision": SOURCE_REVISION,
        "local_checkpoint_revision_verified": False,
        "precision": "fp32", "tf32": False, "streaming": False,
        "profile": asdict(profile),
        "tensorrt_version": trt_compat.module_version(), "workspace_mib": args.workspace_mib,
    }
    write_component(output, "conditioning.plan", plan, manifest)
    print(json.dumps(manifest, indent=2))


def build_speech_component(args):
    from . import trt_compat

    component = args.command.removeprefix("build-").replace("-", "_")
    frontend = component in ("campplus", "speech_tokenizer")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    metadata = {}
    if component == "llm":
        from .llm import build_engine, load_weights

        cfg, weights = load_weights(args.model_dir)
        plan = build_engine(weights, cfg, max_context=args.max_context, opt_tokens=args.opt_tokens,
                            workspace_mib=args.workspace_mib)
        metadata.update(architecture=asdict(cfg), max_context=args.max_context, opt_tokens=args.opt_tokens,
                        compact_kv_cache=True, checkpoint="llm.pt")
    elif frontend:
        from .frontend import FrontendProfile, build_engine

        profile = FrontendProfile(args.min_frames, args.opt_frames, args.max_frames)
        plan = build_engine(args.model_dir, component, profile, workspace_mib=args.workspace_mib)
        metadata.update(profile=asdict(profile), batch_size=1, padded_input=False,
                        learned_execution="native_tensorrt", checkpoint_structure_validated=True)
    else:
        from .hift import build_engine, load_weights

        profile = ShapeProfile(args.min_frames, args.opt_frames, args.max_frames)
        plan = build_engine(load_weights(args.model_dir), profile, workspace_mib=args.workspace_mib)
        metadata.update(profile=asdict(profile), sample_rate=24000, samples_per_frame=480,
                        f0_precision="fp32_reference_uses_fp64", finalize=True, noise="explicit_uniform_0_1",
                        phase_accumulation="chronological_fp32_recurrence")
    metadata.update(schema_version=1, component=f"cosyvoice3_{component}",
                    status="component_built",
                    target_model_id=MODEL_ID, target_model_revision=MODEL_REVISION,
                    equations_source_revision=SOURCE_REVISION, local_checkpoint_revision_verified=False,
                    precision="fp32", tf32=False, streaming=False,
                    tensorrt_version=trt_compat.module_version(), workspace_mib=args.workspace_mib)
    write_component(output, f"{component}.plan", plan, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
