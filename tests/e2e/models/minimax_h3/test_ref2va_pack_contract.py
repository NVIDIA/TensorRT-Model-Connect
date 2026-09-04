# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import sys

from tensorrt_model_connect.families.minimax_h3.ref2va_checkpoint import (
    CHECKPOINT_REVISION,
    COMPONENT_NAME,
    MODEL_ID,
    TOTAL_TENSOR_BYTES,
    TransformerRefIdentity,
)
from tests.e2e.models.minimax_h3 import pack_native_bundle


def test_external_packer_adds_ref2va_only_with_strict_checkpoint_and_four_plans(
    tmp_path: Path, monkeypatch
) -> None:
    plans = tmp_path / "plans"
    plans.mkdir()
    selected = {**pack_native_bundle.PLAN_SECTIONS, **pack_native_bundle.REF2VA_PLAN_SECTIONS}
    recorded = {}
    for filename in selected.values():
        payload = filename.encode()
        (plans / filename).write_bytes(payload)
        recorded[filename] = {"bytes": len(payload), "sha256": "a" * 64}
    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}", encoding="utf-8")
    audio_config = model / "audio_vae" / "config.json"
    audio_config.parent.mkdir(parents=True)
    audio_config.write_text(
        json.dumps(
            {
                "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
                "sampling_rate": 32_000,
                "latents_mean": [0.0] * 32,
                "latents_std": [1.0] * 32,
            }
        ),
        encoding="utf-8",
    )
    transformer_ref = model / COMPONENT_NAME
    transformer_ref.mkdir()
    identity = TransformerRefIdentity(
        model_id=MODEL_ID,
        revision=CHECKPOINT_REVISION,
        component=COMPONENT_NAME,
        tensor_bytes=TOTAL_TENSOR_BYTES,
        tensor_count=638,
        inventory_sha256="1" * 64,
        files={},
    )
    (plans / "build_receipt.json").write_text(
        json.dumps(
            {
                "build_helper_sha256": "b" * 64,
                "workspace_limit_bytes": {filename: 8 << 30 for filename in selected.values()},
                "denoiser_mode": "monolithic",
                "fast_h3": None,
                "transformer_ref": identity.bundle_metadata(),
            }
        ),
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(
        pack_native_bundle,
        "validate_transformer_ref_checkpoint",
        lambda _path: identity,
    )
    monkeypatch.setattr(
        pack_native_bundle,
        "validate_build_receipt",
        lambda *_args, **_kwargs: (
            "c" * 64,
            recorded,
            {"sha256": "d" * 64},
            {"inventory_sha256": "e" * 64},
        ),
    )
    monkeypatch.setattr(
        pack_native_bundle,
        "_target_metadata",
        lambda: ("1.6.1.120", "1.6", "RTX"),
    )

    def capture(_output, _info, sections):
        config = next(section for section in sections if section.name == "config.json")
        captured.update(json.loads(config.data))

    monkeypatch.setattr(pack_native_bundle, "write_bundle", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pack_native_bundle.py",
            "--plans-dir",
            str(plans),
            "--model-path",
            str(model),
            "--output",
            str(tmp_path / "h3.bundle"),
            "--source-revision",
            "2" * 40,
            "--transformer-ref",
            str(transformer_ref),
        ],
    )
    assert pack_native_bundle.main() == 0
    assert captured["public_workflows"] == ["t2va", "fl2va", "ref2va"]
    assert captured["ref2va_transformer_ref"]["revision"] == CHECKPOINT_REVISION
    assert captured["ref2va_scheduler"] == {
        "sigma_grid_points": 50,
        "transformer_forwards": 49,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "guidance_scale": 1.0,
        "guidance_distilled": True,
    }
    assert captured["ref2va_limits"]["requires_image_or_video"] is True
    assert captured["conditioning"]["vision_patch_profile"] == [2_040, 4_032, 65_536]
    assert set(pack_native_bundle.REF2VA_PLAN_SECTIONS).issubset(
        captured["bundle_loading"]["lazy_sections"]
    )
