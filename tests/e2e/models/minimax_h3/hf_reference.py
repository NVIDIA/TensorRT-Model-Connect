# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Hugging Face eager/torch.compile MiniMax-H3 reference and timing receipt."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from tensorrt_model_connect.families.minimax_h3.provenance import (
    CHECKPOINT_REVISION,
    atomic_write_json,
    checkpoint_snapshot_record,
    file_identity,
    stable_file_record,
    validate_file_identity,
    validate_git_archive_source_unchanged,
    validate_source_revision,
    validated_git_archive_source_record,
    validated_git_source_record,
)

DIFFUSERS_REVISION = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
TRANSFORMERS_COMPAT_REVISION = "bed02e1faee69e866e382f835b4f7b0a3c7b8431"
BASE_TRANSFORMERS_VERSION = "5.2.0"
BASE_TRANSFORMERS_ENTRYPOINT = Path(
    "/opt/venv/lib/python3.12/site-packages/transformers/__init__.py"
)
BASE_TRANSFORMERS_ENTRYPOINT_RECORD = {
    "bytes": 38424,
    "sha256": "91b2c544c6848f4ce8213c770aaa705ce682ee656c995f4ce58352c4b7368ee7",
}


def _has_git_checkout(entrypoint: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(entrypoint.resolve(strict=True).parent),
                "rev-parse",
                "--show-toplevel",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def qualified_diffusers_source(entrypoint: Path, evidence_path: Path | None) -> dict:
    if evidence_path is not None:
        return validated_git_archive_source_record(
            entrypoint,
            evidence_path=evidence_path,
            label="Diffusers source",
        )
    return {
        "qualification": "clean_git_checkout",
        **validated_git_source_record(
            entrypoint,
            expected_revision=DIFFUSERS_REVISION,
            label="Diffusers source",
        ),
    }


def qualified_transformers_source(entrypoint: Path, version: str) -> dict:
    entrypoint = entrypoint.resolve(strict=True)
    if _has_git_checkout(entrypoint):
        return {
            "qualification": "clean_git_checkout",
            **validated_git_source_record(
                entrypoint,
                expected_revision=TRANSFORMERS_COMPAT_REVISION,
                label="Transformers source",
            ),
        }
    if entrypoint != BASE_TRANSFORMERS_ENTRYPOINT.resolve():
        raise ValueError(
            "MiniMax-H3 Transformers source is neither the pinned Git checkout "
            "nor the qualified immutable CI base"
        )
    if version != BASE_TRANSFORMERS_VERSION:
        raise ValueError(
            "MiniMax-H3 immutable base Transformers version mismatch: "
            f"expected {BASE_TRANSFORMERS_VERSION}, got {version}"
        )
    entrypoint_record, _ = stable_file_record(entrypoint, "Transformers base entrypoint")
    if entrypoint_record != BASE_TRANSFORMERS_ENTRYPOINT_RECORD:
        raise ValueError("MiniMax-H3 immutable base Transformers entrypoint mismatch")
    return {
        "qualification": "immutable_base_5_2_plus_local_shim",
        "version": version,
        "entrypoint": str(entrypoint),
        "entrypoint_record": entrypoint_record,
    }


def create_mm_token_type_ids(processor, input_ids):
    def modality_ids(name):
        plural = getattr(processor, f"{name}_token_ids", None)
        return plural if plural is not None else [getattr(processor, f"{name}_token_id", None)]

    result = []
    for tokenizer_input in input_ids:
        if not isinstance(tokenizer_input, list):
            tokenizer_input = tokenizer_input.tolist()
        tokenizer_input = np.asarray(tokenizer_input)
        token_types = np.zeros_like(tokenizer_input)
        token_types[np.isin(tokenizer_input, modality_ids("image"))] = 1
        token_types[np.isin(tokenizer_input, modality_ids("video"))] = 2
        token_types[np.isin(tokenizer_input, modality_ids("audio"))] = 3
        result.append(token_types.tolist())
    return result


def _processor_method_identity(processor) -> tuple[object, object]:
    method = getattr(processor, "create_mm_token_type_ids", None)
    if not callable(method):
        raise ValueError("MiniMax-H3 processor has no callable create_mm_token_type_ids")
    return getattr(method, "__self__", None), getattr(method, "__func__", method)


def prepare_processor_compat(processor, transformers_source: dict) -> tuple[str | None, tuple]:
    qualification = transformers_source.get("qualification")
    if qualification == "immutable_base_5_2_plus_local_shim":
        if hasattr(processor, "create_mm_token_type_ids"):
            raise ValueError(
                "MiniMax-H3 immutable Transformers base unexpectedly provides "
                "create_mm_token_type_ids"
            )
        processor.create_mm_token_type_ids = MethodType(create_mm_token_type_ids, processor)
        identity = _processor_method_identity(processor)
        if identity != (processor, create_mm_token_type_ids):
            raise ValueError("MiniMax-H3 could not bind its local processor compatibility helper")
        return "local-create-mm-token-type-ids-for-transformers-5.2.0", identity
    if qualification != "clean_git_checkout":
        raise ValueError("MiniMax-H3 Transformers source has an unknown qualification")
    return None, _processor_method_identity(processor)


def validate_processor_method_unchanged(processor, expected: tuple[object, object]) -> None:
    current = _processor_method_identity(processor)
    if current[0] is not expected[0] or current[1] is not expected[1]:
        raise ValueError("MiniMax-H3 processor compatibility helper changed during the run")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--diffusers-evidence")
    parser.add_argument("--compile", action="store_true", dest="use_compile")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output-type", default="np", choices=("np", "latent"))
    args = parser.parse_args()
    source_revision = validate_source_revision(args.source_revision)
    if args.warmup < 0 or args.measure < 1 or args.steps < 1:
        raise ValueError("warmup must be non-negative; measure and steps must be positive")

    model_path = Path(args.model_path)
    snapshot_record = checkpoint_snapshot_record(model_path)
    prompt_path = Path(args.prompt_file)
    prompt_identity = file_identity(prompt_path)
    prompt_spec = json.loads(prompt_path.read_text())
    prompt_record, prompt_hashed_identity = stable_file_record(prompt_path, "prompt file")
    if prompt_hashed_identity != prompt_identity:
        raise ValueError("MiniMax-H3 prompt file changed while it was being read")
    if not isinstance(prompt_spec.get("prompt"), str) or not prompt_spec["prompt"]:
        raise ValueError("MiniMax-H3 prompt file must contain a non-empty prompt")
    if not isinstance(prompt_spec.get("seed"), int) or isinstance(prompt_spec["seed"], bool):
        raise ValueError("MiniMax-H3 prompt file must contain an integer seed")
    script_path = Path(__file__).resolve()
    script_record, script_identity = stable_file_record(script_path, "HF reference helper")
    request = {
        "prompt": prompt_spec["prompt"],
        "seed": int(prompt_spec["seed"]),
        "height": 768,
        "width": 1344,
        "num_frames": 124,
        "num_inference_steps": args.steps,
        "output_type": args.output_type,
        "warmup": args.warmup,
        "measure": args.measure,
    }

    import diffusers
    from diffusers import ComponentsManager, ModularPipeline

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the lexical evidence path intact. The provenance validator owns
    # existence, stability, and no-symlink enforcement.
    diffusers_evidence = (
        Path(args.diffusers_evidence).absolute() if args.diffusers_evidence else None
    )
    diffusers_source = qualified_diffusers_source(
        Path(diffusers.__file__),
        diffusers_evidence,
    )
    manager = ComponentsManager()
    started = time.perf_counter()
    pipe = ModularPipeline.from_pretrained(args.model_path, components_manager=manager)
    # The published modular index records the Hub repo for each component. Override
    # that field so a pinned local snapshot remains fully offline and revision exact.
    pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=args.model_path)
    import transformers

    transformers_source = qualified_transformers_source(
        Path(transformers.__file__),
        transformers.__version__,
    )
    processor_compat, processor_method_identity = prepare_processor_compat(
        pipe.processor, transformers_source
    )
    pipe.to("cuda:0")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    phase = "compile"
    try:
        if args.use_compile:
            compiled_transformer = torch.compile(
                pipe.transformer, mode=args.compile_mode, dynamic=False
            )
            pipe.update_components(transformer=compiled_transformer)

        def run():
            generator = torch.Generator().manual_seed(int(prompt_spec["seed"]))
            torch.cuda.synchronize()
            begin = time.perf_counter()
            state = pipe(
                prompt=prompt_spec["prompt"],
                height=768,
                width=1344,
                num_frames=124,
                num_inference_steps=args.steps,
                generator=generator,
                output_type=args.output_type,
            )
            torch.cuda.synchronize()
            return state, time.perf_counter() - begin

        phase = "warmup"
        for _ in range(args.warmup):
            run()
        phase = "measure"
        timings, state = [], None
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.measure):
            state, elapsed = run()
            timings.append(elapsed)
    except Exception as error:
        failure_receipt = {
            "backend": "hf_diffusers_torch_compile" if args.use_compile else "hf_diffusers_eager",
            "status": "failed",
            "failure_phase": phase,
            "error": f"{type(error).__name__}: {error}",
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_inventory_sha256": snapshot_record["inventory_sha256"],
            "source_revision": source_revision,
            "builder_source": script_record,
            "checkpoint_snapshot": snapshot_record,
            "inputs": {"prompt_file": prompt_record},
            "request": request,
            "diffusers_revision": DIFFUSERS_REVISION,
            "diffusers_version": diffusers.__version__,
            "diffusers_source": diffusers_source,
            "transformers_source": transformers_source,
            "compile_mode": args.compile_mode if args.use_compile else None,
            "load_s": load_s,
            "torch": torch.__version__,
            "processor_compat": processor_compat,
            "gpu": torch.cuda.get_device_name(0),
            "host": platform.node(),
        }
        atomic_write_json(output_dir / "hf_receipt.json", failure_receipt)
        print(json.dumps(failure_receipt, indent=2))
        raise
    videos = state.get("videos")
    if isinstance(videos, torch.Tensor):
        frames = videos.detach().float().cpu().numpy()
    else:
        frames = np.asarray(videos[0])
    frames_path = output_dir / "hf_frames.npy"
    np.save(frames_path, frames)
    frames_record, _ = stable_file_record(frames_path, "HF decoded frames")
    validate_file_identity(prompt_path, prompt_hashed_identity, "prompt file")
    validate_file_identity(script_path, script_identity, "HF reference helper")
    if diffusers_evidence is not None:
        validate_git_archive_source_unchanged(
            Path(diffusers.__file__),
            evidence_path=diffusers_evidence,
            expected_record=diffusers_source,
            label="Diffusers source",
        )
    elif qualified_diffusers_source(Path(diffusers.__file__), None) != diffusers_source:
        raise ValueError("MiniMax-H3 Diffusers source changed during the HF reference run")
    if (
        qualified_transformers_source(Path(transformers.__file__), transformers.__version__)
        != transformers_source
    ):
        raise ValueError("MiniMax-H3 Transformers source changed during the HF reference run")
    validate_processor_method_unchanged(pipe.processor, processor_method_identity)
    if checkpoint_snapshot_record(model_path) != snapshot_record:
        raise ValueError("MiniMax-H3 checkpoint snapshot changed during the HF reference run")
    receipt = {
        "backend": "hf_diffusers_torch_compile" if args.use_compile else "hf_diffusers_eager",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_inventory_sha256": snapshot_record["inventory_sha256"],
        "source_revision": source_revision,
        "builder_source": script_record,
        "checkpoint_snapshot": snapshot_record,
        "inputs": {"prompt_file": prompt_record},
        "request": request,
        "diffusers_revision": DIFFUSERS_REVISION,
        "diffusers_version": diffusers.__version__,
        "diffusers_source": diffusers_source,
        "transformers_source": transformers_source,
        "compile_mode": args.compile_mode if args.use_compile else None,
        "status": "passed",
        "load_s": load_s,
        "request_s": timings,
        "median_request_s": float(np.median(timings)),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "torch": torch.__version__,
        "processor_compat": processor_compat,
        "gpu": torch.cuda.get_device_name(0),
        "host": platform.node(),
        "shape": list(frames.shape),
        "frames": frames_record,
    }
    atomic_write_json(output_dir / "hf_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
