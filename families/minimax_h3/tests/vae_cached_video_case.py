# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in reuse of the exact full trained-video oracle, never a weaker gate.

Keep both complete candidate bundles and all four cold/warm FP32 videos.
Building both bundles before inference bounds disk staging independently of
video outputs. The default owning test still generates its own reference.
"""

import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from families.minimax_h3.tests.test_trained_video_decoder import (
    _ATOL,
    _RTOL,
    _SHAPE,
    _SHARDS,
    _compare_frames,
    _large_inventory,
    _sha,
)
from families.minimax_h3.tests.vae_decoder_prefix_localize import ROWS_SHA, UPSTREAM_SHA
from families.minimax_h3.tests.vae_encoder_checkpoint_oracle import CONFIG_SHA256, REVISION


PIXEL_BYTES = 1535901696
ROWS_BYTES = 14321664
PLAN_MARGIN = 64 << 20
DISK_RESERVE = 2 << 30
CACHE_FILES = {
    "case.json": (1381, "bc23c4081769e20c5d7a3e09a9eb766552c9133537491dc3c47fa24bd420fb91"),
    "rows.f32.bin": (ROWS_BYTES, ROWS_SHA),
    "upstream.f32.bin": (
        PIXEL_BYTES,
        "a6e92952a4fe22e39a53d0744dd09c2bc014d4a7e6baad7a28a1a3d87f71eabe",
    ),
    "minimal.component.bundle": (
        4852903817,
        "d2dd5689d4f2c5a6c1187b80f20abd996fc7f1a61cc4582ac9e03497ead3133e",
    ),
    "original.component.bundle": (
        4858158137,
        "0d836cb3e4ab311a0ca991899d5155a870ae02533a9652e22b17fc033c1e7daf",
    ),
}


def verify_file(path, size, digest):
    assert path.is_file() and not path.is_symlink(), f"missing or indirect frozen reference: {path}"
    assert path.stat().st_size == size and _sha(path) == digest, f"frozen reference changed: {path}"


def storage_budget(plan_cap):
    # A component bundle has one plan and a small fixed JSON header. Retain
    # all four complete videos; no byte-equality deduplication is assumed.
    bundle_cap = plan_cap + 4096
    staging = 3 * bundle_cap  # first bundle + second staged plan + atomic output
    final = 2 * bundle_cap + 4 * PIXEL_BYTES + ROWS_BYTES + (64 << 10)
    return dict(
        plan_cap=plan_cap,
        bundle_cap=bundle_cap,
        staging_peak=staging,
        final_payload=final,
        reserve=DISK_RESERVE,
        required_free_bytes=max(staging, final) + DISK_RESERVE,
    )


def write_json(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)


def run_cached_case(output, record_property, cache):
    import torch
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    from tensorrt_model_connect.bundle_writer import BundleWriter

    from families.minimax_h3.checkpoint import load_selected_component_state_dict, numpy_state
    from families.minimax_h3.vae_builder import build_vae_tile_decoder_engine, checkpoint_keys

    assert importlib.metadata.version("diffusers") == "0.40.0"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert _sha(Path(inspect.getfile(AutoencoderKLMiniMaxH3))) == UPSTREAM_SHA
    for name, (size, digest) in CACHE_FILES.items():
        verify_file(cache / name, size, digest)
    case = json.loads((cache / "case.json").read_text())
    assert case["hf_revision"] == REVISION and case["checkpoint_shards"] == _SHARDS
    assert case["rtol"] == _RTOL and case["atol"] == _ATOL
    assert case["output_shape"] == list(_SHAPE) and case["comparison_dtype"] == "float32"
    assert case["input_seed"] == 831 and case["TF32"] is False
    plan_cap = (
        max(CACHE_FILES[name][0] for name in CACHE_FILES if name.endswith(".bundle")) + PLAN_MARGIN
    )
    budget = storage_budget(plan_cap)
    budget["observed_free_bytes"] = shutil.disk_usage(output).free
    write_json(output / "storage-budget.json", budget)
    assert budget["observed_free_bytes"] >= budget["required_free_bytes"], (
        "cached full-video proof needs its derived disk reserve; do not delete old assets"
    )
    component = Path(os.environ["TRTMC_MINIMAX_H3_MODEL_DIR"]) / "vae"
    assert component.parent.name == REVISION and _sha(component / "config.json") == CONFIG_SHA256
    assert set(p.name for p in component.glob("*.safetensors")) == set(_SHARDS)
    for name, digest in _SHARDS.items():
        # Read-only Hugging Face snapshots legitimately link into their blob store.
        assert _sha(component / name) == digest
    config = json.loads((component / "config.json").read_text())
    assert all(config[key] == value for key, value in case["architecture"].items())
    index = json.loads(next(component.glob("*.safetensors.index.json")).read_text())["weight_map"]
    assert {name for name in index if name.startswith(("decoder.", "post_quant_conv."))} == set(
        checkpoint_keys()
    )
    state = load_selected_component_state_dict(component, checkpoint_keys())
    assert len(state) == 585 and all(value.dtype == torch.float32 for value in state.values())
    with torch.device("meta"):
        model = AutoencoderKLMiniMaxH3.from_config(config)
    expected_shapes = model.state_dict()
    assert {
        name for name in expected_shapes if name.startswith(("decoder.", "post_quant_conv."))
    } == set(state)
    assert all(
        tuple(value.shape) == tuple(expected_shapes[name].shape) for name, value in state.items()
    )
    del expected_shapes, model
    torch.set_num_threads(4)
    normalized = np.random.default_rng(831).normal(0, 1, (1, 24, 37, 84, 48)).astype(np.float32)
    packed = patchify_video_latents(torch.from_numpy(normalized), (1, 2, 2)).numpy()
    assert packed.shape == (37296, 96) and hashlib.sha256(packed.tobytes()).hexdigest() == ROWS_SHA
    np.testing.assert_array_equal(
        packed, np.fromfile(cache / "rows.f32.bin", np.float32).reshape(37296, 96)
    )
    with (output / "rows.f32.bin").open("xb") as file:
        packed.tofile(file)
    del packed, normalized
    write_json(output / "case.json", case)
    write_json(
        output / "reference-reuse.json",
        dict(
            exact_cached_reference=True,
            files={
                name: dict(bytes=size, sha256=digest)
                for name, (size, digest) in CACHE_FILES.items()
            },
            complete_frames_compared=124,
            comparison_dtype="float32",
            rtol=_RTOL,
            atol=_ATOL,
            oracle_regenerated=False,
            both_bundles_built_before_any_native_video=True,
            candidate_plans_rebuilt=True,
            old_candidate_bundles_used_for_storage_bound_only=True,
            full_ref2va_qualified=False,
        ),
    )
    weights = numpy_state(state)
    root = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    probe = root / "minimax_h3_video_decoder_probe"
    assert probe.is_file() and (root / "libtrtmc_backend_trt.so").is_file()
    presets = (("minimal", 1, "staged"), ("original", 28, "resident"))
    failures, hashes = [], {}
    try:
        for preset, batch, _mode in presets:
            write_json(
                output / (preset + ".config.json"),
                dict(
                    latent_frames=37,
                    latent_height=84,
                    latent_width=48,
                    tile_batch=batch,
                    latents_mean=config["latents_mean"],
                    latents_std=config["latents_std"],
                ),
            )
            plan = build_vae_tile_decoder_engine(weights, batch_size=batch, workspace_bytes=8 << 30)
            assert len(plan) <= plan_cap, "candidate plan exceeds frozen storage bound"
            writer = BundleWriter(output / (preset + ".component.bundle"))
            try:
                writer.set_header(family="minimax_h3", task="image_generation", backend="trt")
                writer.add_bytes("vae.plan", plan)
                writer.finish()
            except BaseException:
                writer.abort()
                raise
            assert (output / (preset + ".component.bundle")).stat().st_size <= budget["bundle_cap"]
            del plan, writer
            gc.collect()
        # Both candidate plans are now retained. No parent copy of the trained
        # parameter tables is needed while the native decoder owns its output.
        del weights, state
        gc.collect()
        for preset, batch, mode in presets:
            prefix = output / preset
            run = subprocess.run(
                [
                    str(probe),
                    str(root),
                    str(output / (preset + ".component.bundle")),
                    str(output / "rows.f32.bin"),
                    str(output / (preset + ".config.json")),
                    str(prefix),
                    mode,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
            )
            with (output / (preset + ".probe.log")).open("x") as file:
                file.write(run.stdout + run.stderr)
            if run.returncode:
                failures.append(preset + ": native probe failed: " + run.stdout + run.stderr)
                continue
            report = json.loads(run.stdout)
            assert (report["frames"], report["height"], report["width"], report["tile_batch"]) == (
                *_SHAPE[:3],
                batch,
            )
            assert report["loads"] == (2 if mode == "staged" else 1)
            assert report["cold_forwards"] == report["warm_forwards"] == (196 if batch == 1 else 7)
            cold, warm = Path(str(prefix) + ".cold.f32.bin"), Path(str(prefix) + ".warm.f32.bin")
            assert cold.stat().st_size == warm.stat().st_size == PIXEL_BYTES
            assert _sha(cold) == _sha(warm)
            comparison, errors = _compare_frames(cold, cache / "upstream.f32.bin", _SHAPE)
            hashes[preset] = _sha(cold)
            result = dict(
                **report,
                **comparison,
                errors=errors,
                rtol=_RTOL,
                atol=_ATOL,
                decoder_component_only=True,
                native_ref2va_video_qualified=False,
            )
            write_json(output / (preset + ".comparison.json"), result)
            record_property(preset, json.dumps(result))
            failures.extend(preset + ": " + error for error in errors)
        write_json(
            output / "cross-preset.json",
            dict(
                byte_exact=hashes.get("minimal") == hashes.get("original")
                if len(hashes) == 2
                else None,
                passing_requires_both_presets_against_upstream=True,
            ),
        )
        verify_file(cache / "upstream.f32.bin", *CACHE_FILES["upstream.f32.bin"])
        assert not failures, "\n".join(failures)
    finally:
        _large_inventory(output)
