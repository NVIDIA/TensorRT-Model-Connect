# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build pinned MiniMax-H3 native plans one at a time and emit a receipt."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

from tensorrt_model_connect.families.minimax_h3.checkpoint import (
    load_component_state_dict,
    load_selected_component_state_dict,
    numpy_state,
)
from tensorrt_model_connect.families.minimax_h3.config import (
    SOL_ENGINE_1344X768_124F,
)
from tensorrt_model_connect.families.minimax_h3.provenance import (
    CHECKPOINT_REVISION,
    atomic_write_bytes,
    atomic_write_json,
    builder_source_sha256,
    checkpoint_snapshot_record,
    file_record,
    serialized_profile,
    sha256_file,
    validate_record,
    validate_source_revision,
)


def _write(output: Path, name: str, payload: bytes, elapsed: float, receipt: dict) -> None:
    path = output / name
    digest = hashlib.sha256(payload).hexdigest()
    atomic_write_bytes(path, payload)
    receipt["components"][name] = {
        "bytes": len(payload),
        "sha256": digest,
        "build_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--cp-size", type=int, default=1, choices=(1,))
    parser.add_argument(
        "--component",
        action="append",
        choices=("text_encoder", "adaln_precompute", "denoiser", "vae_decoder"),
        help="Build only the selected component(s); may be repeated.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep valid existing plans and build only missing selected components.",
    )
    args = parser.parse_args()
    model = Path(args.model_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_revision = validate_source_revision(args.source_revision)
    profile_values = {
        **SOL_ENGINE_1344X768_124F.__dict__,
        "context_parallel_size": args.cp_size,
    }
    profile = SOL_ENGINE_1344X768_124F.__class__(**profile_values)
    receipt_path = output / "build_receipt.json"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    receipt = {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_snapshot": checkpoint_snapshot_record(model),
        "source_revision": source_revision,
        "builder_source_sha256": builder_source_sha256(),
        "build_helper_sha256": sha256_file(Path(__file__).resolve()),
        "profile": serialized_profile(profile),
        "assets": {"tokenizer.json": file_record(tokenizer)},
        "components": {},
    }
    if args.resume and receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        for key in (
            "checkpoint_revision",
            "source_revision",
            "builder_source_sha256",
            "build_helper_sha256",
            "checkpoint_snapshot",
            "profile",
            "assets",
        ):
            if previous.get(key) != receipt[key]:
                raise ValueError(f"Cannot resume: existing receipt has different {key}")
        validate_record(
            tokenizer,
            previous["assets"]["tokenizer.json"],
            "tokenizer.json",
            hash_file=True,
        )
        receipt["components"].update(previous.get("components", {}))
    selected = set(
        args.component or ("text_encoder", "adaln_precompute", "denoiser", "vae_decoder")
    )

    def should_build(component: str, filename: str) -> bool:
        if component not in selected:
            return False
        path = output / filename
        recorded = receipt["components"].get(filename)
        if not args.resume or not path.is_file() or not recorded:
            return True
        try:
            validate_record(path, recorded, filename, hash_file=True)
        except ValueError:
            return True
        return False

    def checkpoint_receipt() -> None:
        atomic_write_json(receipt_path, receipt)

    if should_build("text_encoder", "text_encoder.plan"):
        from tensorrt_model_connect.families.minimax_h3.text_encoder_builder import (
            build_text_encoder_engine,
        )
        from tensorrt_model_connect.families.minimax_h3.text_encoder_builder import (
            checkpoint_keys as text_keys,
        )

        state = load_selected_component_state_dict(model / "text_encoder", text_keys())
        weights = numpy_state(state)
        del state
        started = time.perf_counter()
        plan = build_text_encoder_engine(weights, sequence_length=profile.text_rows)
        _write(output, "text_encoder.plan", plan, time.perf_counter() - started, receipt)
        checkpoint_receipt()
        del weights, plan
        gc.collect()

    from tensorrt_model_connect.families.minimax_h3.adaln_builder import (
        build_adaln_precompute_engine,
    )
    from tensorrt_model_connect.families.minimax_h3.dit_builder import build_dit_engine

    build_adaln = should_build("adaln_precompute", "adaln_precompute.plan")
    build_denoiser = should_build("denoiser", "denoiser.plan")
    if build_adaln or build_denoiser:
        state = load_component_state_dict(model / "transformer")
        weights = numpy_state(state)
        del state
        if build_adaln:
            started = time.perf_counter()
            plan = build_adaln_precompute_engine(weights, profile)
            _write(output, "adaln_precompute.plan", plan, time.perf_counter() - started, receipt)
            checkpoint_receipt()
            del plan
            gc.collect()
        if build_denoiser:
            started = time.perf_counter()
            plan = build_dit_engine(weights, profile)
            _write(output, "denoiser.plan", plan, time.perf_counter() - started, receipt)
            checkpoint_receipt()
            del plan
            gc.collect()
        del weights
        gc.collect()

    from tensorrt_model_connect.families.minimax_h3.vae_builder import (
        build_vae_tile_decoder_engine,
    )
    from tensorrt_model_connect.families.minimax_h3.vae_builder import (
        checkpoint_keys as vae_keys,
    )

    if should_build("vae_decoder", "vae_tile_decoder.plan"):
        state = load_selected_component_state_dict(model / "vae", vae_keys())
        weights = numpy_state(state)
        del state
        started = time.perf_counter()
        plan = build_vae_tile_decoder_engine(weights)
        _write(output, "vae_tile_decoder.plan", plan, time.perf_counter() - started, receipt)
        checkpoint_receipt()
    checkpoint_receipt()
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
