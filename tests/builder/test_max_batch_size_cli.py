# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``--max-batch-size`` CLI flag plumbing.

Trace: ARCH-DIFF-BATCH-001, UD-DIFF-BATCH-CLI
Intent: Verify that ``trtmc build --max-batch-size N`` reaches the family
plugin's ``build_components`` and that the resulting ``.bundle`` bundle
records the expected per-component batch envelope on disk.
Preconditions: tensorrt_model_connect is importable; tests monkeypatch the
actual TRT engine builds so no GPU or TRT runtime is required.
Postconditions: N=1 path stays byte-compatible with PR 1 readers (no
``max_batch_size`` block in the bundle header) and N=4 produces
``{"dit": 4, "text_encoder": 8, "vae": 1}``.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import pytest

try:
    import tensorrt_model_connect.build_cli as cli
    import tensorrt_model_connect.engine_builder as eb
except (ImportError, ModuleNotFoundError):  # pragma: no cover - dependency-only
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _read_bundle_header(bundle_path: Path) -> dict:
    """Decode the JSON header from a .bundle file written by ``write_bundle``."""
    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        assert magic == b"BUNDLE\x01\x00", f"bad magic: {magic!r}"
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len).decode("utf-8"))


def _make_fake_diffusion_model_dir(tmp_path: Path) -> Path:
    """Create a minimal diffusers-format directory that engine_builder accepts.

    Returns the directory path. The pipeline class points to a stub plugin
    registered via ``_install_stub_plugin``.
    """
    model_dir = tmp_path / "fake_diffusion_model"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "FakeBatchPipeline",
            }
        )
    )
    return model_dir


class _FakeDiffusionPlugin:
    """Plugin stub that mimics a diffusion family plugin contract."""

    name = "fake_batch"
    runtime_strategy = "diffusion"
    pipeline_classes = ["FakeBatchPipeline"]

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.tokenizer_add_special_calls = 0
        self.tokenizer_section_calls = 0

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in ("fake_batch", "fakebatchpipeline")

    def load_weights(self, model_dir, config):  # noqa: ARG002
        return {}

    def build_components(
        self,
        model_dir,
        config,
        weights,
        *,
        precision="fp32",
        verbose=False,
        fp8_scales=None,
        build_timing=None,
        parallel_config=None,
        max_batch_size: int = 1,
    ):
        # Mirror the per-component batch policy that the three real plugins
        # apply (Decisions C / E):
        #   DiT honours N; text encoder caps at min(2N, 8); VAE always = 1.
        dit_mbs = int(max_batch_size)
        te_mbs = min(dit_mbs * 2, 8)
        vae_mbs = 1
        self.calls.append(
            {
                "max_batch_size": max_batch_size,
                "dit_mbs": dit_mbs,
                "te_mbs": te_mbs,
                "vae_mbs": vae_mbs,
                "family_build_options": config.raw.get("_family_build_options", {}),
            }
        )
        out = {
            "text_encoders": [("fake_te", b"te-plan")],
            "denoiser": b"dit-plan",
            "vae_decoder": b"vae-plan",
            "preprocessor_weights": b"pp",
        }
        if max_batch_size > 1:
            out["max_batch_size_envelope"] = {
                "dit": dit_mbs,
                "text_encoder": te_mbs,
                "vae": vae_mbs,
            }
        return out

    def get_diffusion_config(self, config):  # noqa: ARG002
        return {"diffusion_backend_type": "fake"}

    def diffusion_bundle_config(self, config, *, components):  # noqa: ARG002
        cfg = self.get_diffusion_config(config)
        cfg["num_text_encoders"] = len(components["text_encoders"])
        return cfg

    def diffusion_bundle_sections(self, components, *, parallel_config=None):  # noqa: ARG002
        sections = []
        for index, (_name, plan) in enumerate(components["text_encoders"]):
            sections.append((f"text_encoder_{index}_plan", plan))
        sections.append(("denoiser_plan", components["denoiser"]))
        sections.append(("vae_decoder_plan", components["vae_decoder"]))
        sections.append(("preprocessor_weights", components["preprocessor_weights"]))
        return sections

    def diffusion_tokenizer_add_special_tokens(
        self,
        model_dir_path,
        *,
        detect_tokenizer_add_special_tokens,
    ):  # noqa: ARG002
        self.tokenizer_add_special_calls += 1
        return False

    def diffusion_tokenizer_bundle_sections(
        self,
        model_dir_path,
        *,
        ensure_tokenizer_json,
    ):  # noqa: ARG002
        self.tokenizer_section_calls += 1
        return []


def _install_stub_plugin(monkeypatch) -> _FakeDiffusionPlugin:
    """Patch ``find_diffusion_plugin`` to return a fresh stub plugin instance."""
    plugin = _FakeDiffusionPlugin()
    monkeypatch.setattr(
        eb,
        "find_diffusion_plugin",
        lambda pipeline_class: plugin if pipeline_class == "FakeBatchPipeline" else None,
    )
    monkeypatch.setattr(
        eb,
        "find_plugin",
        lambda _model_type: None,
    )
    # Skip TRT version probing for the fake bundle.
    monkeypatch.setattr(eb, "_get_trt_version", lambda: "10.0.0")
    monkeypatch.setattr(eb, "_trt_abi_from_version", lambda _v: "trt10")
    monkeypatch.setattr(eb, "_get_gpu_name", lambda: "stub-gpu")
    monkeypatch.setattr(eb, "_setup_trt_import", lambda _rtx: None)
    # The diffusion bundle path calls trt_compat.resolved_summary() before
    # invoking the plugin; stub it so the test stays GPU-/TRT-free.
    from tensorrt_model_connect import trt_compat

    monkeypatch.setattr(
        trt_compat,
        "resolved_summary",
        lambda: "stub",
        raising=False,
    )
    return plugin


def _build_args(model_dir: Path, output: Path, max_batch_size: int) -> argparse.Namespace:
    return argparse.Namespace(
        model=str(model_dir),
        output=str(output),
        max_cache_length=32,
        precision="fp32",
        method="trt",
        quantize=None,
        quant_scales=None,
        quant_calibration_samples=512,
        verbose=False,
        fp8=False,
        fp8_scales=None,
        save_fp8_scales=None,
        rtx=False,
        triattention_stats=None,
        triattention_kv_budget=None,
        triattention_divide_length=128,
        triattention_recent_window=128,
        triattention_score_aggregation="mean",
        triattention_count_prompt_tokens=True,
        triattention_protect_prefill=True,
        triattention_disable_mlr=False,
        triattention_disable_trig=False,
        decoder_engine_layout="split",
        dynamic_kv_cache=False,
        dynamic_kv_profile_rows=None,
        image_height=None,
        image_width=None,
        video_height=None,
        video_width=None,
        video_num_frames=None,
        num_inference_steps=None,
        tensor_parallel_size=1,
        build_timing_json=None,
        config=None,
        set_flags=None,
        max_batch_size=max_batch_size,
        _skip_profile_resolution=True,
    )


def test_default_max_batch_size_omits_envelope(monkeypatch, tmp_path):
    """``--max-batch-size 1`` (the default) must keep the bundle byte-compatible
    with PR 1 readers: no ``max_batch_size`` block in the JSON header."""
    plugin = _install_stub_plugin(monkeypatch)
    model_dir = _make_fake_diffusion_model_dir(tmp_path)
    output = tmp_path / "out.bundle"

    rc = cli._cmd_build(_build_args(model_dir, output, max_batch_size=1))
    assert rc == 0, "CLI should succeed for the default batch size"

    header = _read_bundle_header(output)
    assert "max_batch_size" not in header, (
        "Default build (N=1) must not emit max_batch_size in the bundle header"
    )

    # Plugin should still have been called, with max_batch_size=1.
    assert len(plugin.calls) == 1
    assert plugin.calls[0]["max_batch_size"] == 1
    assert plugin.tokenizer_add_special_calls == 1
    assert plugin.tokenizer_section_calls == 1


def test_max_batch_size_four_records_envelope(monkeypatch, tmp_path):
    """``--max-batch-size 4`` should record the per-component envelope:
    ``{"dit": 4, "text_encoder": 8, "vae": 1}`` (Decision C / E)."""
    plugin = _install_stub_plugin(monkeypatch)
    model_dir = _make_fake_diffusion_model_dir(tmp_path)
    output = tmp_path / "out.bundle"

    rc = cli._cmd_build(_build_args(model_dir, output, max_batch_size=4))
    assert rc == 0

    header = _read_bundle_header(output)
    assert header.get("max_batch_size") == {
        "dit": 4,
        "text_encoder": 8,
        "vae": 1,
    }, f"unexpected envelope in bundle header: {header.get('max_batch_size')!r}"

    assert plugin.calls[0]["max_batch_size"] == 4
    assert plugin.calls[0]["te_mbs"] == 8  # encoder cap policy
    assert plugin.calls[0]["vae_mbs"] == 1  # VAE always slices
    assert plugin.tokenizer_add_special_calls == 1
    assert plugin.tokenizer_section_calls == 1


def test_diffusion_family_build_options_reach_plugin(monkeypatch, tmp_path):
    """Schema-resolved ``--set`` values must survive diffusion dispatch."""
    plugin = _install_stub_plugin(monkeypatch)
    model_dir = _make_fake_diffusion_model_dir(tmp_path)
    output = tmp_path / "out.bundle"
    args = _build_args(model_dir, output, max_batch_size=1)
    args.set_flags = ["minimax_h3.first_block_cache=true"]

    assert cli._cmd_build(args) == 0

    assert plugin.calls[0]["family_build_options"]["minimax_h3"] == {
        "workflow": "t2va",
        "first_block_cache": True,
        "first_block_cache_threshold": 0.025,
    }
