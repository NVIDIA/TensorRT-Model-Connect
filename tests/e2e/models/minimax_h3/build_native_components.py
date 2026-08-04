# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build pinned MiniMax-H3 native plans one at a time and emit a receipt."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from tensorrt_model_connect.families.minimax_h3.checkpoint import (
    load_component_state_dict,
    load_selected_component_state_dict,
    numpy_state,
)
from tensorrt_model_connect.families.minimax_h3.config import (
    SOL_ENGINE_1344X768_124F,
)


def _write(output: Path, name: str, payload: bytes, elapsed: float, receipt: dict) -> None:
    path = output / name
    path.write_bytes(payload)
    receipt["components"][name] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "build_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cp-size", type=int, default=4, choices=(4, 8))
    parser.add_argument(
        "--diagnostic-num-layers",
        type=int,
        help="Build a reduced-depth denoiser plan for stage-local parity diagnosis.",
    )
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
    profile_values = {
        **SOL_ENGINE_1344X768_124F.__dict__,
        "context_parallel_size": args.cp_size,
    }
    if args.diagnostic_num_layers is not None:
        if args.diagnostic_num_layers < 0:
            raise ValueError("--diagnostic-num-layers must be non-negative")
        profile_values["num_layers"] = args.diagnostic_num_layers
    profile = SOL_ENGINE_1344X768_124F.__class__(**profile_values)
    receipt_path = output / "build_receipt.json"
    serialized_profile = json.loads(json.dumps(profile.__dict__))
    receipt = {
        "checkpoint_revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
        "profile": serialized_profile,
        "components": {},
    }
    if args.resume and receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        if previous.get("profile") != receipt["profile"]:
            raise ValueError("Cannot resume: existing receipt uses a different build profile")
        receipt["components"].update(previous.get("components", {}))
    selected = set(
        args.component or ("text_encoder", "adaln_precompute", "denoiser", "vae_decoder")
    )

    def should_build(component: str, filename: str) -> bool:
        if component not in selected:
            return False
        path = output / filename
        recorded = receipt["components"].get(filename)
        if not args.resume or not path.is_file():
            return True
        if not recorded:
            # A build interrupted after atomically writing a plan but before
            # writing its receipt is still resumable. The output directory is
            # profile-specific, and TensorRT will validate the plan again at
            # deserialize time before parity/performance qualification.
            receipt["components"][filename] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "build_s": None,
                "adopted_on_resume": True,
            }
            checkpoint_receipt()
            return False
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest != recorded.get("sha256") or path.stat().st_size != recorded.get("bytes")

    def checkpoint_receipt() -> None:
        receipt_path.write_text(json.dumps(receipt, indent=2))

    if should_build("text_encoder", "text_encoder.plan"):
        from tensorrt_model_connect.families.minimax_h3.text_encoder_builder import (
            build_text_encoder_engine,
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
    build_denoiser = should_build("denoiser", "denoiser_cp.plan")
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
            _write(output, "denoiser_cp.plan", plan, time.perf_counter() - started, receipt)
            checkpoint_receipt()
            del plan
            gc.collect()
        del weights
        gc.collect()

    from tensorrt_model_connect.families.minimax_h3.vae_builder import (
        build_vae_tile_decoder_engine,
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
