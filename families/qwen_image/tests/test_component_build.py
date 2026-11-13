# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edit build process boundaries preserve the original component arguments."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from families.qwen_image import component_build


def test_all_edit_components_finish_in_separate_checked_processes(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 3)))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    settings = {name: {"marker": name, "shape": [1, 2, 3]} for name in
                ("text_encoder", "denoiser", "vision", "vae_decoder", "vae_encoder")}
    calls = []
    real_run = subprocess.run

    def run(command, *, check):
        assert check is True and command[0] == sys.executable
        assert command[4] == component_build.__spec__.name
        if not calls:
            probe = real_run([*command[:5], "--help"], check=True,
                             capture_output=True, text=True, timeout=30)
            assert "separate TensorRT allocation lifetimes" in probe.stdout
        kind, model, output, device, options = command[5:]
        assert Path(model) == tmp_path / "model with spaces" and device == "3"
        assert json.loads(options) == settings[kind]
        calls.append((kind, Path(output)))
        Path(output).write_bytes(kind.encode())
        if kind == "vae_decoder":
            Path(output).with_suffix(".weights").write_bytes(b"original preprocessor")

    monkeypatch.setattr(component_build.subprocess, "run", run)
    result = component_build.build_edit_components(tmp_path / "model with spaces", b"original config", settings)
    assert [kind for kind, _ in calls] == list(settings)
    assert result == {"config_json": b"original config", "text_encoders": [("qwen2_5_vl_lm", b"text_encoder")],
        "denoiser": b"denoiser", "vision_engine": b"vision", "vae_decoder": b"vae_decoder",
        "vae_encoder": b"vae_encoder", "preprocessor_weights": b"original preprocessor"}
    assert all(not path.exists() for _, path in calls)


def test_failed_component_cannot_produce_a_partial_bundle(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0)))

    def fail(command, *, check):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(component_build.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        component_build.build_edit_components(tmp_path, b"config", {"text_encoder": {}})


@pytest.mark.parametrize("kind", ["text_encoder", "denoiser", "vision", "vae_decoder", "vae_encoder"])
def test_child_uses_original_graphs_loaders_and_explicit_settings(monkeypatch, tmp_path, kind):
    from families.qwen_image import model

    config, weights = object(), object()
    loads, builds = [], []

    def load(path, **kwargs):
        loads.append((path, kwargs))
        return config, weights

    def build(cfg, values, output, **kwargs):
        assert cfg is config and values is weights
        builds.append(kwargs)
        output.write_bytes(b"original plan")

    def vision(cfg, values, **kwargs):
        assert cfg == {"hidden": 128} and values is weights
        builds.append(kwargs)
        return b"original vision"

    monkeypatch.setitem(sys.modules, "families.qwen_image.qwen25_vl_text_encoder_builder", SimpleNamespace(
        load_qwen25vl_text_encoder_weights=load, build_qwen25vl_text_encoder_engine=build))
    monkeypatch.setitem(sys.modules, "families.qwen_image.qwen_image_dit_builder", SimpleNamespace(
        load_qwen_image_dit_weights=load, build_qwen_image_dit_engine=build))
    monkeypatch.setitem(sys.modules, "families.qwen_image.qwen_vl_vision_builder", SimpleNamespace(
        build_qwen_vl_vision_engine=vision))
    monkeypatch.setitem(sys.modules, "families.qwen_image.qwen_image_vae_builder", SimpleNamespace(
        load_qwen_image_vae_weights=load, build_qwen_image_vae_decoder_engine=build,
        build_qwen_image_vae_encoder_engine=build))
    monkeypatch.setitem(sys.modules, "families.qwen_image.qwen_image_preprocessor", SimpleNamespace(
        extract_preprocessor_source=lambda cfg: cfg, pack_qwen_image_preprocessor_weights=lambda cfg: b"preprocessor"))
    monkeypatch.setattr(model, "_load_qwen25vl_visual_weights", lambda path: weights)
    settings = {
        "text_encoder": {"max_seq_len": 1536, "apply_final_norm": True,
                         "image_grid_thw": [1, 20, 36], "enable_image_inputs": True, "image_token_start": 70},
        "denoiser": {"h_lat": 64, "w_lat": 64, "n_text": 1536,
                     "image_token_shapes": [[64, 64], [50, 82]], "max_batch_size": 1, "opt_batch_size": 1},
        "vision": {"vision_config": {"hidden": 128}, "fixed_image_height": 280, "fixed_image_width": 504},
        "vae_decoder": {"h_lat": 128, "w_lat": 128},
        "vae_encoder": {"image_h": 800, "image_w": 1312},
    }
    output = tmp_path / f"{kind}.plan"
    component_build._build_component(kind, tmp_path, output, settings[kind])
    assert output.read_bytes().startswith(b"original") and len(builds) == 1
    expected = dict(settings[kind])
    if kind == "text_encoder":
        expected.pop("max_seq_len")
        expected.pop("apply_final_norm")
        expected["image_grid_thw"] = (1, 20, 36)
        assert loads == [(tmp_path / "text_encoder", {"max_seq_len": 1536, "apply_final_norm": True})]
    elif kind == "denoiser":
        expected["image_token_shapes"] = [(64, 64), (50, 82)]
        assert loads == [(tmp_path / "transformer", {})]
    elif kind == "vision":
        expected.pop("vision_config")
    else:
        assert loads == [(tmp_path / "vae", {})]
    assert builds[0] == expected
    if kind == "vae_decoder":
        assert output.with_suffix(".weights").read_bytes() == b"preprocessor"
