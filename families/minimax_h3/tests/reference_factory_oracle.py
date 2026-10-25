# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual native Ref2VA factory with guarded fixture plans.

Use real original checkpoint configuration/index/tokenizer bytes, but never
load model weights or construct a TensorRT engine. Plan bytes are explicit
fixtures and the native backend guard always throws before deserialization.
This qualifies factory wiring only, not models, reference encoding or output.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter
from families.minimax_h3 import reference_build as owner

CASES = (
    ("minimal-1344x768", "minimal", 1344, 768, None),
    ("original-1344x768", "original", 1344, 768, None),
    ("minimal-1536x864", "minimal", 1536, 864, None),
    ("original-1536x864", "original", 1536, 864, None),
    ("reject-text-only-weights", "minimal", 1344, 768, "variant"),
    ("reject-residency-batch-mismatch", "minimal", 1344, 768, "batch"),
    ("reject-missing-audio-plan", "minimal", 1344, 768, "missing"),
    ("reject-invalid-statistics", "minimal", 1344, 768, "stats"),
    ("reject-unknown-schema", "minimal", 1344, 768, "schema"),
    ("reject-wrong-primary-task", "minimal", 1344, 768, "task"),
)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def build_fixtures(checkpoint, root):
    from families.minimax_h3 import adaln_builder, audio_vae_builder, dit_builder
    from families.minimax_h3 import reference_posterior_builder, text_encoder_builder
    from families.minimax_h3 import vae_builder, vae_encoder_builder, vision_builder

    bindings = (
        (vision_builder, "build_vision_encoder_engine", "vision.plan"),
        (text_encoder_builder, "build_text_encoder_engine", "text_encoder.plan"),
        (vae_encoder_builder, "build_vae_tile_encoder_engine", "vae_encoder.plan"),
        (
            reference_posterior_builder,
            "build_reference_posterior_engine",
            "vae_reference_posterior.plan",
        ),
        (adaln_builder, "build_adaln_precompute_engine", "adaln.plan"),
        (dit_builder, "build_dit_engine", "denoiser.plan"),
        (vae_builder, "build_vae_tile_decoder_engine", "vae.plan"),
        (audio_vae_builder, "build_audio_vae_decoder_engine", "audio_vae.plan"),
    )
    manifest = []

    def marker(name):
        return lambda *_args, **_kwargs: ("fixture:" + name).encode()

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(owner, "load_selected_component_state_dict", return_value={})
        )
        stack.enter_context(patch.object(owner, "numpy_state", lambda value: value))
        for module, function, name in bindings:
            stack.enter_context(patch.object(module, function, marker(name)))
        for case, preset, height, width, mutation in CASES:
            path = root / (case + ".bundle")
            writer = BundleWriter(path)
            add, add_json, header = writer.add_bytes, writer.add_json, writer.set_header

            def stage(name, data, *, _mutation=mutation, _add=add):
                if _mutation == "missing" and name == "audio_vae.plan":
                    return
                _add(name, data)

            def config(name, data, *, _mutation=mutation, _json=add_json):
                assert name == "runtime.json"
                if _mutation == "variant":
                    data["weights_variant"] = "transformer"
                elif _mutation == "batch":
                    data["vae_tile_batch"] = 28
                elif _mutation == "stats":
                    data["latents_std"][0] = -1.0
                elif _mutation == "schema":
                    data["schema_version"] = "unknown/v2"
                _json(name, data)

            def set_header(*, _mutation=mutation, _header=header, **kwargs):
                if _mutation == "task":
                    kwargs["task"] = "image_generation"
                _header(**kwargs)

            writer.add_bytes, writer.add_json, writer.set_header = stage, config, set_header
            request = BuildRequest(
                model_dir=checkpoint,
                output_path=path,
                family="minimax_h3",
                task=owner.TASK,
                precision="bf16",
                image_height=height,
                image_width=width,
            )
            try:
                owner.build(request, writer, memory_preset=preset)
                writer.finish()
            except BaseException:
                writer.abort()
                raise
            manifest.append(dict(case=case, bundle=str(path), constructs=mutation is None))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.checkpoint.name == "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "factory-only oracle must hide all GPUs"
    probe = args.runtime_root / "minimax_h3_reference_factory_probe"
    library = args.runtime_root / "libtrtmc_model_minimax_h3.so"
    assert probe.is_file() and library.is_file()
    args.output.mkdir(mode=0o700)
    manifest = build_fixtures(args.checkpoint, args.output)
    manifest_path = args.output / "fixture-manifest.json"
    with manifest_path.open("x") as file:
        json.dump(manifest, file, indent=2)
    result = subprocess.run(
        [str(probe), str(library), str(manifest_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    (args.output / "native.stdout.log").write_text(result.stdout)
    (args.output / "native.stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert [row["case"] for row in report["cases"]] == [case[0] for case in CASES]
    assert sum(row["constructed"] for row in report["cases"]) == 4
    assert sum(row["guarded_lazy_loads"] for row in report["cases"]) == 8
    assert report["engine_execution_count"] == 0 and report["full_ref2va_qualified"] is False
    report.update(
        family_dso_sha256=sha(library),
        probe_sha256=sha(probe),
        original_tokenizer_sha256=sha(args.checkpoint / "tokenizer/tokenizer.json"),
    )
    with (args.output / "report.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    files = {
        p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(args.output.iterdir())
        if p.is_file()
    }
    with (args.output / "files.json").open("x") as file:
        json.dump(files, file, indent=2, allow_nan=False)
    print(json.dumps(dict(factory_cases=10, engine_execution_count=0, full_ref2va_qualified=False)))


if __name__ == "__main__":
    main()
